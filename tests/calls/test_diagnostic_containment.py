"""A raising handler on ``llmkit.calls`` or ``llmkit._litellm`` never replaces a call's outcome.

The call lanes and the LiteLLM transport report problems they have already
contained (a schema whose dump fails for the log, a malformed tool call
dropped from a round, a provider that could not be resolved for the record, a
refused ``temperature``, a cost or diagnostic that could not be read) on their
module loggers. A host :class:`logging.Handler` runs inline inside that
report, so a handler whose ``emit`` raises used to turn the answer, or the
provider's error, into the handler's exception. These tests install such a
handler and check that the original result or exception type survives, that
each lost record is counted exactly once in
:func:`llmkit.get_diagnostics_health`, and that the seam emits nothing further
after the swallowed failure (no recursion). A handler that raises cancellation
or ``KeyboardInterrupt`` is not contained.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from litellm import exceptions as litellm_errors
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, model_serializer
from pydantic_core import PydanticSerializationError

import llmkit._litellm as llm_litellm
from llmkit import (
    NO_RETRY,
    LLMClientConfig,
    Provider,
    ResultValidationError,
    RetryPolicy,
    ToolDefinition,
    capture_llm_records,
    configure_llm_client,
    make_provider,
    structured_llm_call,
    text_llm_call,
    tool_llm_call,
)
from llmkit.exceptions import OutputLimitDiagnostics
from llmkit.providers import LLMProviderInterface
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from tests._support import OkSchema
from tests._support.diagnostics import (
    DiagnosticHandlerError,
    RaisingHandler,
    lost_diagnostics,
    raising_handler,
)

_CALLS = "llmkit.calls"
_LITELLM = "llmkit._litellm"
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


def _messages(handler: RaisingHandler) -> list[tuple[str, int]]:
    return [(record.getMessage(), record.levelno) for record in handler.records]


def _openai() -> LLMProviderInterface:
    """A real provider that supports compose; the fake answers before any network call."""
    return make_provider(Provider.OPENAI, api_key="k")


# --- calls/structured.py: the log-serialization warning ----------------------


class _UndumpableAnswer(BaseModel):
    ok: bool

    @model_serializer
    def _boom(self) -> dict[str, object]:
        raise ValueError("serializer exploded")


#: What the dump sites' ``except`` block sees, and so what ``exc_info=True``
#: must carry: pydantic wraps the serializer's ``ValueError`` in its own error.
_DUMP_ERROR = PydanticSerializationError


def _assert_exc_info(handler: RaisingHandler, expected: type[BaseException]) -> None:
    """Every record carries the exception its site was handling, not a bare message."""
    assert handler.records
    for record in handler.records:
        assert record.exc_info is not None
        assert record.exc_info[0] is expected


_STRUCTURED_DUMP = (
    "Failed to serialize LLM response for log f/None; recording response as None",
    logging.WARNING,
)


async def test_structured_success_survives_a_raising_handler_on_the_dump_warning() -> None:
    before = lost_diagnostics()
    with (
        raising_handler(_CALLS) as handler,
        install(lambda _request: Reply.text('{"ok": true}')),
        capture_llm_records() as records,
    ):
        result = await structured_llm_call("q", _UndumpableAnswer, feature="f")

    assert type(result) is _UndumpableAnswer
    assert result.ok is True
    assert lost_diagnostics() - before == 1
    assert _messages(handler) == [_STRUCTURED_DUMP]
    _assert_exc_info(handler, _DUMP_ERROR)
    [record] = records
    assert record.error is None
    assert cast("object", record.response) is None


async def test_structured_error_path_keeps_the_original_error_not_the_handlers() -> None:
    """The dump also runs in the ``finally`` of a rejected attempt; its lost
    warning must not "mask the real provider error"."""

    def _reject(_result: _UndumpableAnswer) -> None:
        raise ResultValidationError("never good enough")

    before = lost_diagnostics()
    with (
        raising_handler(_CALLS) as handler,
        install(lambda _request: Reply.text('{"ok": true}')) as installed,
        pytest.raises(ResultValidationError, match="never good enough"),
    ):
        _ = await structured_llm_call(
            "q", _UndumpableAnswer, feature="f", retry=_NO_BACKOFF, on_result=_reject
        )

    # The default validation budget is two attempts; each one's dump warning is lost.
    assert len(installed.requests) == 2
    assert lost_diagnostics() - before == 2
    assert _messages(handler) == [_STRUCTURED_DUMP, _STRUCTURED_DUMP]
    _assert_exc_info(handler, _DUMP_ERROR)


# --- calls/tool.py: the compose dump warning ---------------------------------


async def test_compose_answer_survives_a_raising_handler_on_the_dump_warning() -> None:
    before = lost_diagnostics()
    with (
        raising_handler(_CALLS) as handler,
        install(lambda _request: Reply.text('{"ok": true}')),
    ):
        result = await tool_llm_call(
            "q", [], output_schema=_UndumpableAnswer, feature="f", provider=_openai()
        )

    assert type(result.parsed) is _UndumpableAnswer
    assert lost_diagnostics() - before == 1
    assert _messages(handler) == [
        (
            "Could not dump the f/None compose result for the log; recording response as null",
            logging.WARNING,
        )
    ]
    _assert_exc_info(handler, _DUMP_ERROR)


# --- calls/_shared.py: the malformed tool call warning -----------------------


class _WeatherArgs(BaseModel):
    city: str


def _partly_malformed(_request: TransportRequest) -> Reply:
    return Reply(
        tool_calls=(
            FakeToolCall("weather", '{"city":"Ottawa"}', id="call_good"),
            FakeToolCall("weather", "{not json", id="call_bad"),
        )
    )


async def test_malformed_tool_call_warning_is_contained_first_and_repeated() -> None:
    """No latch: every round with a dropped call warns, and every lost warning counts."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    before = lost_diagnostics()
    with raising_handler(_CALLS) as handler, install(_partly_malformed):
        first = await tool_llm_call("weather?", [definition], feature="a", retry=NO_RETRY)
        repeated = await tool_llm_call("weather?", [definition], feature="a", retry=NO_RETRY)

    for result in (first, repeated):
        assert [call.id for call in result.tool_calls] == ["call_good"]
        assert [error.call_id for error in result.invalid_calls] == ["call_bad"]
    assert lost_diagnostics() - before == 2
    assert [level for _message, level in _messages(handler)] == [logging.WARNING] * 2
    assert all(
        message.startswith("Dropped 1 malformed tool call(s) from a round of 2: ")
        for message, _level in _messages(handler)
    )


async def test_cancellation_from_the_handler_propagates_out_of_the_call() -> None:
    """``asyncio.CancelledError`` is a ``BaseException``: the seam must not swallow it."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    before = lost_diagnostics()
    with (
        raising_handler(_CALLS, error=asyncio.CancelledError) as handler,
        install(_partly_malformed),
        pytest.raises(asyncio.CancelledError),
    ):
        _ = await tool_llm_call("weather?", [definition], feature="a", retry=NO_RETRY)

    assert lost_diagnostics() == before
    assert len(handler.records) == 1


# --- calls/_shared.py: the two provider-resolution debugs --------------------


class _ConfigBroken(Exception):
    """The configuration error the caller must see."""


def _broken_config() -> LLMClientConfig:
    raise _ConfigBroken("no provider configured")


async def test_provider_resolution_debugs_never_replace_the_configuration_error() -> None:
    previous = configure_llm_client(_broken_config)
    before = lost_diagnostics()
    try:
        with raising_handler(_CALLS) as handler, pytest.raises(_ConfigBroken):
            _ = await text_llm_call("hi", feature="f", retry=NO_RETRY)
    finally:
        _ = configure_llm_client(previous)

    assert _messages(handler) == [
        ("Could not pre-resolve provider for LLM call", logging.DEBUG),
        ("Could not resolve provider for LLM log", logging.DEBUG),
    ]
    _assert_exc_info(handler, _ConfigBroken)
    assert lost_diagnostics() - before == 2


class _FailsOnceWhenArmed:
    """A config source whose next build fails once after :meth:`arm`, then works."""

    def __init__(self) -> None:
        self.armed: bool = False

    def arm(self) -> None:
        self.armed = True

    def __call__(self) -> LLMClientConfig:
        if self.armed:
            self.armed = False
            raise _ConfigBroken("transient config read failure")
        return LLMClientConfig(provider=Provider.OLLAMA, model="fake-model")


async def test_pre_resolve_debug_keeps_a_successful_answer_first_and_repeated() -> None:
    """The pre-resolution fails, the attempt re-resolves and answers: the lost
    debug costs nothing, the second time as much as the first (no latch)."""
    source = _FailsOnceWhenArmed()
    previous = configure_llm_client(source)
    before = lost_diagnostics()
    try:
        with raising_handler(_CALLS) as handler, install(lambda _request: Reply.text("answer")):
            source.arm()
            first = await text_llm_call("hi", feature="f", retry=NO_RETRY)
            source.arm()
            repeated = await text_llm_call("hi", feature="f", retry=NO_RETRY)
    finally:
        _ = configure_llm_client(previous)

    assert first == repeated == "answer"
    assert (
        _messages(handler) == [("Could not pre-resolve provider for LLM call", logging.DEBUG)] * 2
    )
    assert lost_diagnostics() - before == 2


# --- _litellm.py: the refused-temperature notice (latched WARNING, then DEBUG) -


def _temperature_refusal() -> Exception:
    return litellm_errors.BadRequestError(
        message="AnthropicException - `temperature` is not supported for this model.",
        model="claude-sonnet-5-5",
        llm_provider="anthropic",
    )


def _response(content: str) -> ModelResponse:
    response = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )
    response._hidden_params = {}  # mirror litellm's cost-lookup attribute
    return response


class _ScriptedLiteLLM:
    """Raise the scripted errors in order, then answer; count every request."""

    def __init__(self, *errors: Exception) -> None:
        self.errors: list[Exception] = list(errors)
        self.requests: int = 0

    async def __call__(self, **_request: object) -> object:
        self.requests += 1
        if self.errors:
            raise self.errors.pop(0)
        return _response("ok")


async def _text_over(fake: _ScriptedLiteLLM) -> str:
    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake.__call__):
        return await text_llm_call(
            "hi", feature="t", provider=_openai(), temperature=0.0, retry=NO_RETRY
        )


def _is_temperature_notice(record: logging.LogRecord) -> bool:
    return "refused temperature" in record.getMessage()


async def test_temperature_notice_is_contained_first_warning_and_repeated_debug() -> None:
    before = lost_diagnostics()
    with raising_handler(_LITELLM, raise_on=_is_temperature_notice) as handler:
        first = await _text_over(_ScriptedLiteLLM(_temperature_refusal()))
        repeated = await _text_over(_ScriptedLiteLLM(_temperature_refusal()))

    assert first == repeated == "ok"
    notices = [r for r in handler.records if _is_temperature_notice(r)]
    assert [r.levelno for r in notices] == [logging.WARNING, logging.DEBUG]
    assert lost_diagnostics() - before == 2
    # The seam re-logged nothing about its own failure.
    assert [r for r in handler.records if not _is_temperature_notice(r)] == []


class _RefusesTemperature:
    """Refuse every request that still carries ``temperature``; answer the re-sent one."""

    def __init__(self, content: str) -> None:
        self.content: str = content
        self.requests: int = 0

    async def __call__(self, **request: object) -> object:
        self.requests += 1
        if "additional_drop_params" not in request:
            raise _temperature_refusal()
        return _response(self.content)


async def _structured_over(fake: _RefusesTemperature) -> OkSchema:
    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake.__call__):
        return await structured_llm_call(
            "hi", OkSchema, feature="t", provider=_openai(), temperature=0.0, retry=NO_RETRY
        )


async def test_temperature_notice_never_replaces_a_later_provider_failure() -> None:
    """The notice fires once the re-sent request answers; when that answer then
    fails the call, the caller sees the same error type as with no handler."""
    with pytest.raises(Exception) as baseline:
        _ = await _structured_over(_RefusesTemperature("not json"))
    expected = type(baseline.value)
    assert expected is not DiagnosticHandlerError
    llm_litellm.reset_temperature_warnings()

    before = lost_diagnostics()
    with raising_handler(_LITELLM, raise_on=_is_temperature_notice) as handler:
        first = _RefusesTemperature("not json")
        with pytest.raises(expected):
            _ = await _structured_over(first)
        repeated = _RefusesTemperature("not json")
        with pytest.raises(expected):
            _ = await _structured_over(repeated)

    notices = [r for r in handler.records if _is_temperature_notice(r)]
    # Each call sends twice (the answer and its in-call repair re-ask), each
    # send is refused and re-sent, and each re-send reports: four notices, of
    # which only the very first is a WARNING.
    assert first.requests == repeated.requests == 4
    assert [r.levelno for r in notices] == [logging.WARNING] + [logging.DEBUG] * 3
    assert lost_diagnostics() - before == 4
    assert [r for r in handler.records if not _is_temperature_notice(r)] == []


async def test_temperature_notice_keeps_a_structured_answer() -> None:
    before = lost_diagnostics()
    with raising_handler(_LITELLM, raise_on=_is_temperature_notice) as handler:
        answer = await _structured_over(_RefusesTemperature('{"ok": true}'))

    assert answer == OkSchema(ok=True)
    assert [r.levelno for r in handler.records if _is_temperature_notice(r)] == [logging.WARNING]
    assert lost_diagnostics() - before == 1


def test_keyboard_interrupt_from_the_handler_propagates_out_of_the_notice() -> None:
    before = lost_diagnostics()
    with (
        raising_handler(_LITELLM, error=KeyboardInterrupt) as handler,
        pytest.raises(KeyboardInterrupt),
    ):
        llm_litellm._report_dropped_temperature("gpt-5-mini", 0.0)

    assert lost_diagnostics() == before
    assert len(handler.records) == 1


def test_temperature_notice_record_names_the_real_call_site() -> None:
    with raising_handler(_LITELLM) as handler:
        llm_litellm._report_dropped_temperature("gpt-5-mini", 0.0)

    [record] = handler.records
    assert record.funcName == "_report_dropped_temperature"
    assert record.pathname == llm_litellm.__file__


# --- _litellm.py: the best-effort debugs and the instructor-mismatch warning ---


def _assert_one_lost(handler: RaisingHandler, before: int, expected: tuple[str, int]) -> None:
    assert lost_diagnostics() - before == 1
    assert _messages(handler) == [expected]


def test_stream_cost_debug_is_contained() -> None:
    before = lost_diagnostics()
    with (
        raising_handler(_LITELLM) as handler,
        patch("llmkit._litellm.ModelResponse", side_effect=RuntimeError("no table")),
    ):
        cost = llm_litellm._stream_cost(object(), "openai/gpt-5-mini")

    assert cost is None
    _assert_one_lost(
        handler, before, ("Could not price streamed response for openai/gpt-5-mini", logging.DEBUG)
    )
    _assert_exc_info(handler, RuntimeError)


class _Unreadable:
    def __getattr__(self, name: str) -> object:
        raise RuntimeError(f"cannot read {name}")


def test_output_limit_diagnostics_debug_is_contained() -> None:
    before = lost_diagnostics()
    with raising_handler(_LITELLM) as handler:
        diagnostics = llm_litellm._output_limit_diagnostics(_Unreadable())

    assert diagnostics == OutputLimitDiagnostics()
    _assert_one_lost(handler, before, ("Could not read output-limit diagnostics", logging.DEBUG))
    _assert_exc_info(handler, RuntimeError)


@pytest.mark.parametrize(
    ("flush", "expected", "exc_type"),
    [
        (
            AsyncMock(side_effect=TimeoutError),
            "LiteLLM async-logging drain timed out after 0.5s",
            None,
        ),
        (
            AsyncMock(side_effect=RuntimeError("worker gone")),
            "LiteLLM async-logging drain failed",
            RuntimeError,
        ),
    ],
    ids=["timeout", "failure"],
)
async def test_async_logging_drain_debugs_are_contained(
    flush: AsyncMock, expected: str, exc_type: type[BaseException] | None
) -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    before = lost_diagnostics()
    with (
        raising_handler(_LITELLM) as handler,
        patch.object(GLOBAL_LOGGING_WORKER, "flush", flush),
    ):
        await llm_litellm.drain_async_logging(timeout=0.5)

    _assert_one_lost(handler, before, (expected, logging.DEBUG))
    if exc_type is None:
        # The timeout is an expected outcome, reported without a traceback.
        assert handler.records[0].exc_info is None
    else:
        _assert_exc_info(handler, exc_type)


class _Strict(BaseModel):
    ok: bool


def test_instructor_mismatch_warning_is_contained() -> None:
    before = lost_diagnostics()
    with raising_handler(_LITELLM) as handler:
        answer = llm_litellm._as_output_schema(_Strict(ok=True), _Strict, '{"ok": "yes"}')

    assert answer == _Strict(ok=True)
    _assert_one_lost(
        handler,
        before,
        (
            "Structured answer for _Strict validated in instructor but not on its own "
            + "text (ValidationError); returning instructor's answer",
            logging.WARNING,
        ),
    )
