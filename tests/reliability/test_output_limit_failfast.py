"""Fail-fast on output-limit truncation in structured calls.

A structured completion truncated by the output-token limit
(``finish_reason='length'``) is never worth re-asking with an identical
budget: the motivating production failure was a degenerate repetition loop
that burned to the provider ceiling on the original ask *and* on every blind
re-ask. These tests pin the whole contract:

* the in-call instructor loop refuses the length re-ask — one generation,
  surfaced as :class:`~llmkit.OutputLimitError` with the cause chain and
  diagnostics (``model`` / ``max_tokens`` / ``completion_tokens``);
* the outer retry layer (:func:`~llmkit.retry.with_retries`) propagates it
  immediately under the default policy *and* under ``retry_on=None`` ("retry
  anything") — only an explicit ``retry_on`` listing the type opts back in;
* the taxonomy holds: catchable via ``LLM_RECOVERABLE_ERRORS``, outside both
  retry-budget sets;
* no collateral damage: the one schema-repair re-ask (and the
  ``InstructorRetryException`` wrap on exhaustion) keeps working;
* the tenacity behaviour the design leans on — a *declined* retry predicate
  propagates the original exception bare, never a ``RetryError`` — is
  asserted empirically, as a tripwire for future instructor/tenacity bumps.

Unlike the rest of the offline suite these tests patch the **provider seam**
(``llmkit._litellm.litellm.acompletion``) rather than the transport
coroutines, so instructor's real ``process_response`` / retry machinery runs
over the faked ``ModelResponse``.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import pytest
from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from litellm.types.utils import ModelResponse, Usage

from llmkit import (
    LLM_OUTPUT_LIMIT_ERRORS,
    LLM_RECOVERABLE_ERRORS,
    LLM_SCHEMA_ERRORS,
    LLM_TRANSPORT_ERRORS,
    OutputLimitError,
    _litellm,
    structured_output,
)
from llmkit.retry import with_retries
from tests._support import OkSchema, quiet_logging


def _response(
    content: str, *, finish_reason: str = "stop", completion_tokens: int = 5
) -> ModelResponse:
    """A real litellm ``ModelResponse`` (so instructor's parsing runs) with
    the given first-choice content, finish reason, and usage."""
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage=Usage(
            prompt_tokens=7,
            completion_tokens=completion_tokens,
            total_tokens=7 + completion_tokens,
        ),
    )
    resp._hidden_params = {}  # mirror litellm's cost-lookup attribute
    return resp


def _truncated_response() -> ModelResponse:
    """A runaway generation cut off at the cap: length-truncated mid-JSON."""
    return _response('{"ok": tru', finish_reason="length", completion_tokens=64)


def _fake_provider() -> MagicMock:
    """A provider whose ``instructor_mode`` is a *real* mode, so
    ``from_litellm`` builds a genuine instructor client over the patched
    ``litellm.acompletion`` (unlike ``tests._support._transport_provider``,
    which pairs with a fully-faked client)."""
    provider = MagicMock()
    provider.name = "fake-provider"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None
    return provider


async def _drive_truncated(*, max_tokens: int | None) -> tuple[OutputLimitError, int]:
    """Run ``acompletion_structured`` against an always-truncating provider;
    return the raised error and how many generations were attempted."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _truncated_response()

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        pytest.raises(OutputLimitError) as excinfo,
    ):
        _ = await _litellm.acompletion_structured(
            "hi",
            OkSchema,
            temperature=0.0,
            model=None,
            max_tokens=max_tokens,
            provider=_fake_provider(),
        )
    return excinfo.value, calls


async def test_truncation_fails_in_one_generation() -> None:
    """The headline acceptance criterion: a length-truncated completion raises
    ``OutputLimitError`` after exactly one generation — the in-call re-ask
    that doubled every production failure is gone."""
    _err, calls = await _drive_truncated(max_tokens=64)
    assert calls == 1


async def test_truncation_error_carries_cause_and_diagnostics() -> None:
    """The raised error chains the instructor original and carries the
    diagnostics that make the fix legible from the error alone."""
    err, _calls = await _drive_truncated(max_tokens=64)
    assert isinstance(err.__cause__, IncompleteOutputException)
    assert err.model == "fake/model"
    assert err.max_tokens == 64
    assert err.completion_tokens == 64


async def test_truncation_without_cap_reports_none() -> None:
    """With no caller cap (provider-ceiling truncation, the production case),
    ``max_tokens`` is ``None`` and the request carried no cap key."""
    err, calls = await _drive_truncated(max_tokens=None)
    assert calls == 1
    assert err.max_tokens is None
    assert err.completion_tokens == 64


async def test_public_call_default_policy_adds_no_attempts_or_backoff() -> None:
    """End-to-end through ``structured_llm_call`` under its default
    ``RetryPolicy``: the error propagates with zero additional attempts and
    zero backoff sleeps — seconds, not minutes, to a legible failure."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _truncated_response()

    sleep_mock = AsyncMock()
    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        patch("llmkit.retry.asyncio.sleep", sleep_mock),
        pytest.raises(OutputLimitError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", provider=_fake_provider()
        )
    assert calls == 1
    assert sleep_mock.await_count == 0


async def test_with_retries_retry_on_none_fails_fast() -> None:
    """The regression test for the bare-exception hole: ``retry_on=None``
    ("retry on any Exception") must still propagate an ``OutputLimitError``
    immediately instead of charging it to the transport budget."""
    attempts = 0

    async def boom() -> None:
        nonlocal attempts
        attempts += 1
        raise OutputLimitError(model="fake/model", max_tokens=None, completion_tokens=None)

    with pytest.raises(OutputLimitError):
        await with_retries(boom, max_attempts=3)
    assert attempts == 1


async def test_with_retries_explicit_opt_in_still_retries() -> None:
    """A caller that explicitly lists the type in ``retry_on`` gets its
    retries — the fail-fast is a default, not a prohibition."""
    attempts = 0

    async def boom() -> None:
        nonlocal attempts
        attempts += 1
        raise OutputLimitError(model="fake/model", max_tokens=8, completion_tokens=8)

    with pytest.raises(OutputLimitError):
        await with_retries(boom, max_attempts=2, retry_on=(OutputLimitError,))
    assert attempts == 2


def test_taxonomy_membership() -> None:
    """Catchable via the recoverable union (hosts' existing except-nets keep
    degrading gracefully), excluded from both retry-budget sets."""
    err = OutputLimitError(model="m", max_tokens=8, completion_tokens=8)
    assert isinstance(err, LLM_OUTPUT_LIMIT_ERRORS)
    assert isinstance(err, LLM_RECOVERABLE_ERRORS)
    assert not isinstance(err, LLM_TRANSPORT_ERRORS)
    assert not isinstance(err, LLM_SCHEMA_ERRORS)
    with pytest.raises(LLM_RECOVERABLE_ERRORS):
        raise OutputLimitError(model="m", max_tokens=8, completion_tokens=8)


async def test_schema_repair_reask_still_works() -> None:
    """No collateral damage: a malformed-JSON response still earns the one
    in-call repair re-ask, and the second (valid) response succeeds."""
    calls = 0
    responses = [_response('{"ok": nope}'), _response('{"ok": true}')]

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return responses[calls - 1]

    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion):
        parsed, _cost = await _litellm.acompletion_structured(
            "hi", OkSchema, temperature=0.0, model=None, provider=_fake_provider()
        )
    assert parsed.ok is True
    assert calls == 2


async def test_schema_exhaustion_still_wraps() -> None:
    """No collateral damage: two malformed responses exhaust the in-call
    budget after exactly two generations and surface as the usual
    ``InstructorRetryException`` — the outer layer's wrapped-exception
    classification stays intact."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _response('{"ok": nope}')

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        pytest.raises(InstructorRetryException),
    ):
        _ = await _litellm.acompletion_structured(
            "hi", OkSchema, temperature=0.0, model=None, provider=_fake_provider()
        )
    assert calls == 2


async def test_declined_predicate_propagates_bare_exception() -> None:
    """The tenacity-behaviour tripwire, asserted empirically rather than
    trusted from source: when :func:`llmkit._litellm._schema_repair_retrying`'s
    predicate declines the retry, instructor's machinery propagates the
    original ``IncompleteOutputException`` *bare* — never wrapped in
    ``InstructorRetryException`` — after a single generation. If a future
    instructor/tenacity version changes its exception flow, this fails first."""
    calls = 0

    async def fake_completion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _truncated_response()

    client = cast(
        "instructor.AsyncInstructor",
        cast("object", instructor.from_litellm(fake_completion, mode=instructor.Mode.JSON_SCHEMA)),
    )
    with pytest.raises(IncompleteOutputException) as excinfo:
        _ = await client.chat.completions.create_with_completion(
            model="fake/model",
            messages=[{"role": "user", "content": "hi"}],
            response_model=OkSchema,
            max_retries=_litellm._schema_repair_retrying(),
        )
    assert type(excinfo.value) is IncompleteOutputException
    assert calls == 1
