"""Tests for the ``RetryPolicy`` value object and the recoverable-error set.

These pin the *policy* side of retries, complementing the call-surface
behaviour in ``test_default_retries.py``:

* the shipped default budget (3 transport attempts / 2 validation, 0.5s base
  backoff) and the jittered exponential-ceiling backoff;
* ``LLM_RECOVERABLE_ERRORS`` membership — 429 / 5xx / network are transient,
  auth and other 4xx are not — and that it's the union of the transport and
  schema subsets;
* the ``with_retries`` primitive's ``max_attempts`` naming and the nested-retry
  guard that stops an outer wrapper from multiplying the inner budget;
* the validation/transport budget split (a deterministic schema failure can't
  burn the full transport budget).

A canonical transient error is :class:`TimeoutError`; a canonical
non-recoverable one is an ``openai`` auth/4xx status error.
"""

from __future__ import annotations

import asyncio
import random
import warnings
from pathlib import Path
from unittest.mock import patch

import httpx
import openai
import pytest
from pydantic import ValidationError

from llmkit import (
    DEFAULT_RETRY_POLICY,
    LLM_RECOVERABLE_ERRORS,
    NO_RETRY,
    LocalYamlLogSink,
    RetryPolicy,
    configure_llm_logging,
    structured_output,
)
from llmkit.exceptions import LLM_SCHEMA_ERRORS, LLM_TRANSPORT_ERRORS
from llmkit.retry import with_retries
from llmkit.structured_output import capture_llm_log_paths
from tests._support import OkSchema

_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


def _make_validation_error() -> ValidationError:
    """Build a real pydantic ``ValidationError`` for offline tests.

    ``OkSchema.ok`` is a required bool; constructing it from a non-bool
    payload raises the genuine article rather than a hand-rolled stand-in.
    """
    try:
        _ = OkSchema.model_validate({"ok": "not-a-bool-or-coercible"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")  # pragma: no cover


# --- RetryPolicy defaults & validation -----------------------------------


def test_default_retry_policy_values() -> None:
    """The shipped default budget is three attempts with 0.5s base backoff."""
    assert DEFAULT_RETRY_POLICY.max_attempts == 3
    assert DEFAULT_RETRY_POLICY.backoff_base_seconds == 0.5


@pytest.mark.asyncio
async def test_default_backoff_sleeps_with_jittered_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the default policy, a transient-then-success run sleeps once
    before the retry, bounded by the exponential ceiling
    ``base * 2**(attempt-1)``. Pin the jitter to its max to assert the
    exact ceiling (``0.5 * 2**0 == 0.5``)."""
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    configure_llm_logging(None)
    try:
        with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
            result = await structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", retry=DEFAULT_RETRY_POLICY
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert result.ok is True
    assert slept == [0.5]


def test_max_attempts_zero_raises_value_error() -> None:
    """``RetryPolicy(max_attempts=0)`` is rejected at construction."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        _ = RetryPolicy(max_attempts=0)


# --- transient set excludes auth / non-429 4xx ---------------------------


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """Build a minimal ``openai`` status error of *cls* for offline tests."""
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def test_recoverable_set_excludes_auth_and_other_4xx() -> None:
    """Permanent 4xx errors are not in the transient set; 429 / 5xx / network are.

    This is the core of the auth/4xx fix: the set names specific transient
    ``openai`` subclasses rather than the broad ``openai.APIError`` base, so a
    bad key or malformed request can't be mistaken for something worth retrying.
    """
    assert not isinstance(_status_error(openai.AuthenticationError, 401), LLM_RECOVERABLE_ERRORS)
    assert not isinstance(_status_error(openai.BadRequestError, 400), LLM_RECOVERABLE_ERRORS)
    assert not isinstance(_status_error(openai.PermissionDeniedError, 403), LLM_RECOVERABLE_ERRORS)

    assert isinstance(_status_error(openai.RateLimitError, 429), LLM_RECOVERABLE_ERRORS)
    assert isinstance(_status_error(openai.InternalServerError, 503), LLM_RECOVERABLE_ERRORS)
    conn = openai.APIConnectionError(message="down", request=httpx.Request("POST", "https://x"))
    assert isinstance(conn, LLM_RECOVERABLE_ERRORS)


@pytest.mark.asyncio
async def test_structured_auth_error_is_not_retried() -> None:
    """A 401 ``AuthenticationError`` fails fast: a single attempt, no retry."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _status_error(openai.AuthenticationError, 401)

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(openai.AuthenticationError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_structured_rate_limit_error_is_retried_then_succeeds() -> None:
    """A 429 ``RateLimitError`` is transient: retried once, then succeeds."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise _status_error(openai.RateLimitError, 429)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert calls[0] == 2


# --- unified naming: max_attempts is the only name ------------------------


def test_recoverable_set_is_union_of_transport_and_schema() -> None:
    """``LLM_RECOVERABLE_ERRORS`` is exactly the union of the two subsets, so
    the documented single catch-set contract is preserved after the split."""
    assert set(LLM_RECOVERABLE_ERRORS) == set(LLM_TRANSPORT_ERRORS) | set(LLM_SCHEMA_ERRORS)
    # The subsets are disjoint and each carries its expected members.
    assert set(LLM_TRANSPORT_ERRORS).isdisjoint(LLM_SCHEMA_ERRORS)
    assert ValidationError in LLM_SCHEMA_ERRORS
    assert TimeoutError in LLM_TRANSPORT_ERRORS
    assert ValidationError not in LLM_TRANSPORT_ERRORS


@pytest.mark.asyncio
async def test_with_retries_max_attempts_runs_exactly_n_attempts() -> None:
    """``with_retries(max_attempts=3)`` makes exactly three attempts before
    re-raising — the canonical name, total attempts including the first."""
    calls = [0]

    async def _fn() -> str:
        calls[0] += 1
        raise TimeoutError("always")

    with pytest.raises(TimeoutError, match="always"):
        _ = await with_retries(_fn, max_attempts=3, retry_on=(TimeoutError,))

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_with_retries_max_retries_kwarg_raises_type_error() -> None:
    """The ``max_retries`` alias is hard-cut: passing it is an unexpected
    keyword argument and raises ``TypeError`` (no shim, no warning)."""
    calls = [0]

    async def _fn() -> str:
        calls[0] += 1
        raise TimeoutError("always")

    with pytest.raises(TypeError, match="max_retries"):
        # Deliberately passing the hard-cut ``max_retries`` alias: it is no longer
        # a parameter, so the type checker rightly flags it; the test pins the
        # runtime ``TypeError``.
        await with_retries(_fn, max_retries=3, retry_on=(TimeoutError,))  # pyright: ignore[reportCallIssue]

    # The callable was never invoked: the bad kwarg is rejected at the call.
    assert calls[0] == 0


def test_retry_policy_and_with_retries_agree_on_max_attempts() -> None:
    """Both surfaces name the attempt count ``max_attempts`` with identical
    semantics — ``max_attempts=3`` means three total attempts."""
    policy = RetryPolicy(max_attempts=3)
    assert policy.max_attempts == 3
    # ``with_retries`` accepts ``max_attempts`` directly (no alias needed).
    assert "max_attempts" in (with_retries.__doc__ or "")


# --- nested-retry guard: no budget multiplication -------------------------


@pytest.mark.asyncio
async def test_nested_with_retries_around_call_function_does_not_multiply() -> None:
    """Wrapping a call function (which already retries 3x) in
    ``with_retries(max_attempts=3)`` must NOT yield 3x3=9 attempts. The inner
    layer collapses to a single pass, so the outer loop owns the retries —
    a persistent transient error costs the *outer* budget, not the product."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("always")

    async def _wrapped() -> OkSchema:
        return await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.warns(RuntimeWarning, match="nested"),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await with_retries(_wrapped, max_attempts=3, retry_on=(TimeoutError,))

    # Outer budget of 3, inner collapsed to a single pass each -> 3 total,
    # NOT 9 (3 x 3). The guard prevented the multiplication.
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_no_retry_inner_drives_retries_from_outer_wrapper_without_warning() -> None:
    """The documented escape hatch: drive retries from an outer ``with_retries``
    by opting the inner call out with ``retry=NO_RETRY``. The inner collapses to
    a single pass (so the *outer* budget owns the retries — 3 transport calls,
    not 9) and, because the inner explicitly opted out, the nested guard stays
    silent: no ``RuntimeWarning`` fires on the user-intended path."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("always")

    async def _wrapped() -> OkSchema:
        return await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=NO_RETRY
        )

    with warnings.catch_warnings():
        # Any nested-guard RuntimeWarning on this intended path would fail the test.
        warnings.simplefilter("error", RuntimeWarning)
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            pytest.raises(TimeoutError, match="always"),
        ):
            _ = await with_retries(_wrapped, max_attempts=3, retry_on=(TimeoutError,))

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_nested_guard_preserves_one_log_per_attempt(tmp_path: Path) -> None:
    """The nested guard preserves the one-log-per-attempt contract: an outer
    ``with_retries(max_attempts=2)`` around a transient-then-success call
    function writes one log per real attempt (here: 2)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    async def _wrapped() -> OkSchema:
        return await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            capture_llm_log_paths() as paths,
            pytest.warns(RuntimeWarning, match="nested"),
        ):
            result = await with_retries(_wrapped, max_attempts=2, retry_on=(TimeoutError,))
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert result.ok is True
    assert calls[0] == 2
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


@pytest.mark.asyncio
async def test_with_retries_around_bare_awaitable_still_retries() -> None:
    """The guard must NOT penalize the legitimate advanced use: wrapping a
    plain (non-llmkit) awaitable has no active inner policy, so it retries
    normally up to ``max_attempts`` with no warning."""
    calls = [0]

    async def _bare() -> str:
        calls[0] += 1
        if calls[0] < 3:
            raise TimeoutError("transient")
        return "ok"

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here would fail the test
        result = await with_retries(_bare, max_attempts=3, retry_on=(TimeoutError,))

    assert result == "ok"
    assert calls[0] == 3


# --- validation split: lower budget for schema-validation failures --------


@pytest.mark.asyncio
async def test_persistent_validation_error_uses_lower_validation_budget() -> None:
    """A persistent ``ValidationError`` is retried only on the lower
    validation budget (default 2 = one retry), NOT the transport budget (3).
    A deterministic schema failure can't burn the full transport budget."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _make_validation_error()

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(ValidationError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    # validation_max_attempts default is 2 -> two attempts, not three.
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_transport_error_still_uses_full_transport_budget() -> None:
    """A persistent transport error still gets the full transport budget (3)
    under the same default policy — the split lowered *only* validation."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("always")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_instructor_wrapped_transport_error_uses_full_transport_budget() -> None:
    """instructor wraps an exhausted *transport* attempt in
    ``InstructorRetryException`` (which lives in ``LLM_SCHEMA_ERRORS``). The
    retry layer must unwrap it and charge the transport budget (3), not the
    lower validation budget (2) — otherwise a structured call is less resilient
    to a 429/network blip than the identical plain-text call."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        root = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        raise InstructorRetryException(root, n_attempts=1, total_usage=0)

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(InstructorRetryException),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_instructor_wrapped_schema_error_uses_validation_budget() -> None:
    """An ``InstructorRetryException`` wrapping a genuine ``ValidationError``
    still uses the lower validation budget (2) — unwrapping must not push a real
    schema failure onto the transport budget."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise InstructorRetryException(_make_validation_error(), n_attempts=1, total_usage=0)

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(InstructorRetryException),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 2


@pytest.mark.asyncio
async def test_transiently_malformed_json_gets_one_cross_call_validation_retry() -> None:
    """A transiently-malformed response (one ``ValidationError`` then success)
    is still recovered: the one validation retry yields a clean parse."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise _make_validation_error()
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert calls[0] == 2


def test_default_validation_budget_is_lower_than_transport() -> None:
    """The shipped default validation budget (2) is lower than transport (3)."""
    assert DEFAULT_RETRY_POLICY.validation_max_attempts == 2
    assert DEFAULT_RETRY_POLICY.max_attempts == 3


def test_no_retry_disables_both_budgets() -> None:
    """``NO_RETRY`` is a single attempt for *both* classes."""
    assert NO_RETRY.max_attempts == 1
    assert NO_RETRY.validation_max_attempts == 1


@pytest.mark.asyncio
async def test_no_retry_single_attempt_for_validation_error() -> None:
    """``retry=NO_RETRY`` makes a ``ValidationError`` propagate after one
    attempt (the validation budget is disabled too)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _make_validation_error()

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(ValidationError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=NO_RETRY
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_no_retry_single_attempt_for_transport_error() -> None:
    """``retry=NO_RETRY`` makes a transport error propagate after one
    attempt (mirrors the validation case for the other budget)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("transient")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TimeoutError, match="transient"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=NO_RETRY
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_validation_retries_are_each_their_own_logged_call(tmp_path: Path) -> None:
    """Each validation attempt is its own logged call: a persistent
    ``ValidationError`` under the default validation budget (2) writes two
    log records (one per attempt), so per-attempt logging survives the split."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _make_validation_error()

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            capture_llm_log_paths() as paths,
            pytest.raises(ValidationError),
        ):
            _ = await structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", retry=_NO_BACKOFF
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert calls[0] == 2
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_validation_max_attempts_zero_raises_value_error() -> None:
    """``RetryPolicy(validation_max_attempts=0)`` is rejected at construction."""
    with pytest.raises(ValueError, match="validation_max_attempts must be >= 1"):
        _ = RetryPolicy(validation_max_attempts=0)
