"""instructor's in-call retry is restricted to genuine parse failures.

The in-call loop pinned by :func:`llmkit._litellm._schema_repair_retrying` used
to re-ask *every* exception except a length truncation — so a transport error
(429/5xx/network) or a permanent one (401/400/403) was blindly re-sent inside
the single ``GlobalRateLimiter`` slot, with no backoff and ignoring
``Retry-After``, doubling the real requests one structured call makes. The
predicate now retries only the parse failures in ``REPAIRABLE_PARSE_ERRORS``
(``reraise=True``, mirroring instructor's own int path).

These tests pin the restricted contract by driving the **real** instructor loop
— patching the provider seam ``llmkit._litellm.litellm.acompletion`` so
instructor's genuine ``process_response`` / retry machinery runs, exactly like
``test_output_limit_failfast.py`` — rather than hand-building an
``InstructorRetryException``:

* a transport error (429) is **not** re-asked (one generation) and surfaces
  wrapped in ``InstructorRetryException`` whose ``__cause__`` is the raw error,
  which :func:`~llmkit.exceptions.underlying_provider_error` digs back out under
  the 1.15.x wrap shape (kills the amplification bug *and* the dead-unwrap-path
  finding in one test);
* a permanent error (401) is **not** re-asked and fails fast through the default
  :class:`~llmkit.RetryPolicy` with no cross-call retry and no backoff sleep;
* a genuine parse failure **is** still re-asked once (two generations) and, on
  exhaustion, routes to the lower *validation* budget in
  :func:`~llmkit.retry.with_retries` — not the transport budget;
* ``Retry-After`` on a wrapped 429 is still recovered by
  :func:`~llmkit.retry._retry_after_seconds` under the new wrap shape.

The predicate itself (parse-in, everything-else-out) and the v1-compat
hand-built-wrapper budget routing are pinned in ``test_retry_policy.py``; this
file is the end-to-end real-loop complement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import instructor
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException
from litellm.types.utils import ModelResponse, Usage

from llmkit import RetryPolicy, _litellm
from llmkit import calls as llm_calls
from llmkit.exceptions import underlying_provider_error
from llmkit.retry import _retry_after_seconds
from tests._support import OkSchema, quiet_logging


def _fake_provider() -> MagicMock:
    """A provider whose ``instructor_mode`` is a *real* mode, so ``from_litellm``
    builds a genuine instructor client over the patched ``litellm.acompletion``
    (same shape as ``test_output_limit_failfast._fake_provider``)."""
    provider = MagicMock()
    provider.name = "fake-provider"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None
    provider.strict_json_schema = False
    return provider


def _invalid_response() -> ModelResponse:
    """A real litellm ``ModelResponse`` whose content fails ``OkSchema`` parsing,
    so instructor raises a genuine (re-askable) parse error."""
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"ok": nope}'},
            }
        ],
        usage=Usage(prompt_tokens=7, completion_tokens=5, total_tokens=12),
    )
    resp._hidden_params = {}
    return resp


def _http_error(
    cls: type[openai.APIStatusError], status: int, *, headers: dict[str, str] | None = None
) -> openai.APIStatusError:
    """An ``openai`` status error carrying a real ``httpx`` response, so the
    provider raises the same shape LiteLLM surfaces in production."""
    request = httpx.Request("POST", "https://fake/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("boom", response=response, body=None)


async def test_transport_error_is_not_reasked_and_unwraps() -> None:
    """A 429 raised inside the structured call is **not** re-asked — one
    generation, not two — and surfaces wrapped in ``InstructorRetryException``
    whose raw ``RateLimitError`` cause ``underlying_provider_error`` recovers.
    This single test kills both problems: the 2x in-slot amplification and the
    dead ``underlying_provider_error`` unwrap path (``__cause__`` is now the raw
    error, not a tenacity ``RetryError``)."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        raise _http_error(openai.RateLimitError, 429)

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await _litellm.acompletion_structured(
            "hi", OkSchema, temperature=0.0, model=None, provider=_fake_provider()
        )

    assert calls == 1
    root = underlying_provider_error(excinfo.value)
    assert isinstance(root, openai.RateLimitError)


async def test_permanent_error_fails_fast_through_default_policy() -> None:
    """A permanent 401 is **not** re-asked in-call (one generation) and, through
    ``structured_llm_call``'s default policy, propagates immediately: no
    cross-call retry, no backoff sleep — the fail-fast contract the bare-4xx
    text path already honours, now matched on the structured path."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        raise _http_error(openai.AuthenticationError, 401)

    sleep_mock = AsyncMock()
    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        patch("llmkit.retry.asyncio.sleep", sleep_mock),
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="test", provider=_fake_provider()
        )

    assert calls == 1
    assert sleep_mock.await_count == 0
    assert isinstance(underlying_provider_error(excinfo.value), openai.AuthenticationError)


async def test_parse_failure_uses_validation_budget_through_real_loop() -> None:
    """A persistent parse failure keeps the in-call repair re-ask (two
    generations per outer attempt) and, on exhaustion, routes to the *lower*
    validation budget in ``with_retries`` — not the transport one. Proven with a
    policy whose two budgets are deliberately far apart (5 transport / 2
    validation): validation routing makes ``2 x 2 = 4`` provider generations,
    where transport routing would make ``5 x 2 = 10``. So ``calls == 4`` pins
    the routing unambiguously — the exhaustion ``InstructorRetryException``
    (``__cause__`` now the raw parse error) is classified as validation-shaped,
    not transport."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        return _invalid_response()

    policy = RetryPolicy(max_attempts=5, validation_max_attempts=2, backoff_base_seconds=0.0)
    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        patch("llmkit.retry.asyncio.sleep", AsyncMock()),
        pytest.raises(InstructorRetryException),
    ):
        _ = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="test", retry=policy, provider=_fake_provider()
        )

    assert calls == 4


async def test_retry_after_recovered_from_wrapped_429() -> None:
    """``Retry-After`` on a wrapped 429 is still found under the new wrap shape:
    ``_retry_after_seconds`` unwraps the ``InstructorRetryException`` (whose
    ``__cause__`` is now the raw ``RateLimitError``, not a ``RetryError``) and
    reads the header — so the structured path keeps honouring server-directed
    waits."""
    calls = 0

    async def fake_acompletion(**_kwargs: object) -> ModelResponse:
        nonlocal calls
        calls += 1
        raise _http_error(openai.RateLimitError, 429, headers={"retry-after": "5"})

    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=fake_acompletion),
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await _litellm.acompletion_structured(
            "hi", OkSchema, temperature=0.0, model=None, provider=_fake_provider()
        )

    assert calls == 1
    assert _retry_after_seconds(excinfo.value) == 5.0
