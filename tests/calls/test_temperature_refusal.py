"""A refused ``temperature`` is omitted once, and the record says so.

Some models take no sampling temperature: LiteLLM refuses ``temperature=0.0``
client-side on GPT-5 models whose reasoning effort is not ``none`` and on Claude
5.5-era models, and a provider can refuse it server-side with an HTTP 400 when
LiteLLM's model map does not know the model. The transport funnel
(``llmkit._litellm._acompletion``) re-issues such a request exactly once with
LiteLLM's per-call ``additional_drop_params=["temperature"]``, and every lane's
call record then shows ``temperature: None`` — what was sent, not what was asked.

The provider seam ``llmkit._litellm.litellm.acompletion`` is patched, so each
test runs the real lane, transport and funnel over a scripted LiteLLM.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import final
from unittest.mock import MagicMock, patch

import instructor
import pytest
from litellm import exceptions as litellm_errors
from litellm.types.utils import ModelResponse

from llmkit import (
    NO_RETRY,
    UNSET,
    InMemoryLLMCache,
    RetryPolicy,
    ServiceUnavailableError,
    Unset,
    capture_llm_records,
    configure_llm_cache,
)
from llmkit import calls as llm_calls
from llmkit.logging import LLMCallRecord
from tests._support import OkSchema

_CLIENT_REFUSAL = (
    "gpt-5-mini doesn't support temperature=0.0 while reasoning is active. Only "
    "temperature=1 is supported. To drop unsupported params set `litellm.drop_params = True`"
)


def _client_refusal() -> Exception:
    return litellm_errors.UnsupportedParamsError(
        message=_CLIENT_REFUSAL, llm_provider="openai", model="gpt-5-mini"
    )


def _server_refusal() -> Exception:
    return litellm_errors.BadRequestError(
        message="AnthropicException - `temperature` is not supported for this model.",
        model="claude-sonnet-5-5",
        llm_provider="anthropic",
    )


def _response(content: str) -> ModelResponse:
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )
    resp._hidden_params = {}  # mirror litellm's cost-lookup attribute
    return resp


@final
class _Chunk:
    """A stream frame carrying one text delta, in the loose shape LiteLLM yields."""

    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        self.choices = [choice]
        self.usage = None
        self._hidden_params: dict[str, object] = {}


@final
class _Stream:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks
        self.usage = None
        self._hidden_params: dict[str, object] = {}

    def __aiter__(self) -> AsyncIterator[_Chunk]:
        async def _gen() -> AsyncIterator[_Chunk]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _provider(**completion_kwargs: object) -> MagicMock:
    """A provider double with every attribute the five lanes and the cache read."""
    provider = MagicMock()
    provider.name = "OpenAI"
    provider.model = "gpt-5-mini"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", **completion_kwargs})
    provider.native_kwargs = MagicMock(return_value={})
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.litellm_model = MagicMock(return_value="openai/gpt-5-mini")
    provider.reasoning_effort = None
    provider.strict_json_schema = False
    provider.single_tool_call_per_schema = False
    provider.supports_tool_choice = True
    provider.reasoning_kwargs = MagicMock(return_value={})
    return provider


@final
class _ScriptedLiteLLM:
    """Raise the scripted errors in order, then answer; record every request."""

    def __init__(self, *errors: Exception, stream: bool = False, content: str = "ok") -> None:
        self.errors: list[Exception] = list(errors)
        self.requests: list[dict[str, object]] = []
        self._stream: bool = stream
        self._content: str = content

    async def __call__(self, **request: object) -> object:
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        if self._stream:
            return _Stream([_Chunk(self._content), _Chunk(None, finish_reason="stop")])
        return _response(self._content)


_Temperature = float | None | Unset


async def _text(
    provider: MagicMock, temperature: _Temperature = UNSET, retry: RetryPolicy | Unset = UNSET
) -> None:
    _ = await llm_calls.text_llm_call(
        "hi", feature="t", provider=provider, temperature=temperature, retry=retry
    )


async def _structured(
    provider: MagicMock, temperature: _Temperature = UNSET, retry: RetryPolicy | Unset = UNSET
) -> None:
    _ = await llm_calls.structured_llm_call(
        "hi", OkSchema, feature="t", provider=provider, temperature=temperature, retry=retry
    )


async def _tool(
    provider: MagicMock, temperature: _Temperature = UNSET, retry: RetryPolicy | Unset = UNSET
) -> None:
    _ = await llm_calls.tool_llm_call(
        "hi", [], feature="t", provider=provider, temperature=temperature, retry=retry
    )


async def _text_stream(
    provider: MagicMock, temperature: _Temperature = UNSET, retry: RetryPolicy | Unset = UNSET
) -> None:
    async for _chunk in llm_calls.text_llm_call_stream(
        "hi", feature="t", provider=provider, temperature=temperature, retry=retry
    ):
        pass


async def _tool_stream(
    provider: MagicMock, temperature: _Temperature = UNSET, retry: RetryPolicy | Unset = UNSET
) -> None:
    async for _event in llm_calls.tool_llm_call_stream(
        "hi", [], feature="t", provider=provider, temperature=temperature, retry=retry
    ):
        pass


_Lane = Callable[..., Awaitable[None]]

_LANES: list[tuple[str, _Lane, bool, str]] = [
    ("text", _text, False, "ok"),
    ("structured", _structured, False, '{"ok": true}'),
    ("tool", _tool, False, "ok"),
    ("text_stream", _text_stream, True, "ok"),
    ("tool_stream", _tool_stream, True, "ok"),
]
_LANE_PARAMS = [
    pytest.param(lane, stream, content, id=name) for name, lane, stream, content in _LANES
]


async def _run(
    lane: _Lane, fake: _ScriptedLiteLLM, provider: MagicMock | None = None, **kwargs: object
) -> list[LLMCallRecord]:
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake.__call__),
        capture_llm_records() as records,
    ):
        await lane(provider if provider is not None else _provider(), **kwargs)
    return records


# --- (a)/(b) the refusal is absorbed on every lane --------------------------


@pytest.mark.parametrize("refusal", [_client_refusal, _server_refusal], ids=["client", "server"])
@pytest.mark.parametrize(("lane", "stream", "content"), _LANE_PARAMS)
async def test_refused_temperature_is_dropped_once_and_recorded_as_none(
    lane: _Lane, stream: bool, content: str, refusal: Callable[[], Exception]
) -> None:
    fake = _ScriptedLiteLLM(refusal(), stream=stream, content=content)

    [record] = await _run(lane, fake, temperature=0.0)

    refused, reissued = fake.requests
    assert "additional_drop_params" not in refused
    assert reissued["additional_drop_params"] == ["temperature"]
    # Only the named parameter differs: LiteLLM, not llmkit, removes the key.
    assert reissued["temperature"] == 0.0
    assert {k: v for k, v in reissued.items() if k != "additional_drop_params"} == refused
    assert record.error is None
    assert record.temperature is None


@pytest.mark.parametrize(("lane", "stream", "content"), _LANE_PARAMS)
async def test_accepted_temperature_is_recorded_as_sent(
    lane: _Lane, stream: bool, content: str
) -> None:
    fake = _ScriptedLiteLLM(stream=stream, content=content)

    [record] = await _run(lane, fake, temperature=0.0)

    assert len(fake.requests) == 1
    assert record.temperature == 0.0


# --- (c) anything else is not re-issued -------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            litellm_errors.BadRequestError(
                message="max_tokens is too large", model="m", llm_provider="openai"
            ),
            id="400-other-param",
        ),
        pytest.param(
            litellm_errors.RateLimitError(
                message="temperature-heavy traffic", llm_provider="openai", model="m"
            ),
            id="429",
        ),
    ],
)
async def test_other_errors_are_not_reissued(error: Exception) -> None:
    fake = _ScriptedLiteLLM(error)

    with pytest.raises(type(error)):
        _ = await _run(_text, fake, temperature=0.0, retry=NO_RETRY)

    assert len(fake.requests) == 1


async def test_a_503_naming_temperature_is_not_reissued_and_is_still_normalised() -> None:
    fake = _ScriptedLiteLLM(
        litellm_errors.ServiceUnavailableError(
            message="temperature service down", llm_provider="openai", model="m"
        )
    )

    with pytest.raises(ServiceUnavailableError):
        _ = await _run(_text, fake, temperature=0.0, retry=NO_RETRY)

    assert len(fake.requests) == 1


async def test_a_request_without_temperature_is_not_reissued() -> None:
    fake = _ScriptedLiteLLM(_client_refusal())

    with pytest.raises(litellm_errors.UnsupportedParamsError):
        _ = await _run(_text, fake, retry=NO_RETRY)

    assert len(fake.requests) == 1
    assert "temperature" not in fake.requests[0]


async def test_a_host_drop_list_is_extended_not_replaced() -> None:
    fake = _ScriptedLiteLLM(_client_refusal())

    _ = await _run(_text, fake, _provider(additional_drop_params=["top_p"]), temperature=0.0)

    assert fake.requests[1]["additional_drop_params"] == ["top_p", "temperature"]


async def test_a_request_already_dropping_temperature_is_not_reissued() -> None:
    fake = _ScriptedLiteLLM(_client_refusal())

    with pytest.raises(litellm_errors.UnsupportedParamsError):
        _ = await _run(
            _text,
            fake,
            _provider(additional_drop_params=["temperature"]),
            temperature=0.0,
            retry=NO_RETRY,
        )

    assert len(fake.requests) == 1


# --- (d) a failing re-issue surfaces the original refusal --------------------


@pytest.mark.parametrize(("lane", "stream", "content"), _LANE_PARAMS)
async def test_a_failing_reissue_raises_the_original_refusal(
    lane: _Lane, stream: bool, content: str
) -> None:
    original = _client_refusal()
    second = litellm_errors.BadRequestError(message="still no", model="m", llm_provider="openai")
    fake = _ScriptedLiteLLM(original, second, stream=stream, content=content)

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake.__call__),
        capture_llm_records() as records,
        pytest.raises(Exception) as raised,
    ):
        await lane(_provider(), temperature=0.0, retry=NO_RETRY)

    # instructor wraps a permanent error on the structured lane; the refusal
    # it wraps is still the original one, carrying the second as context.
    surfaced: BaseException | None = raised.value
    while surfaced is not None and surfaced is not original:
        surfaced = surfaced.__cause__ or surfaced.__context__
    assert surfaced is original
    assert original.__context__ is second
    assert len(fake.requests) == 2
    [record] = records
    assert record.error is not None
    assert record.temperature == 0.0


# --- (e) a cache hit records the requested value -----------------------------


@pytest.mark.parametrize(
    "lane", [pytest.param(_text, id="text"), pytest.param(_text_stream, id="text_stream")]
)
async def test_a_cache_hit_after_a_dropped_call_records_the_requested_value(
    lane: _Lane,
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    try:
        provider = _provider()
        fake = _ScriptedLiteLLM(_client_refusal(), stream=lane is _text_stream)
        with (
            patch("llmkit._litellm.litellm.acompletion", side_effect=fake.__call__),
            capture_llm_records() as records,
        ):
            await lane(provider, temperature=0.0)
            await lane(provider, temperature=0.0)
    finally:
        configure_llm_cache(None)

    paid, hit = records
    assert (paid.cache_hit, paid.temperature) == (False, None)
    assert (hit.cache_hit, hit.temperature) == (True, 0.0)
    assert len(fake.requests) == 2


# --- The stamp never leaks across calls --------------------------------------


async def test_a_concurrent_call_does_not_see_another_calls_drop() -> None:
    """Each task has its own context: a drop in one never reaches the other's record."""
    dropping = _ScriptedLiteLLM(_client_refusal())
    accepting = _ScriptedLiteLLM()

    async def _route(**request: object) -> object:
        fake = dropping if request["temperature"] == 0.0 else accepting
        return await fake(**request)

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=_route),
        capture_llm_records() as records,
    ):
        _ = await asyncio.gather(
            _text(_provider(), temperature=0.0), _text(_provider(), temperature=0.7)
        )

    assert sorted((r.temperature for r in records), key=lambda t: t is not None) == [None, 0.7]
