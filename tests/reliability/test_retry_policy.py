"""Tests for the ``RetryPolicy`` value object and the recoverable-error set.

These pin the *policy* side of retries, complementing the call-surface
behaviour in ``test_default_retries.py``:

* the shipped default budget (3 transport attempts / 2 validation, 0.5s base
  backoff, 30s per-sleep cap) and the jittered exponential-ceiling backoff;
* ``LLM_RECOVERABLE_ERRORS`` membership — 429 / 5xx / network are transient,
  auth and other 4xx are not — and that it's the union of the transport,
  schema, backpressure, and output-limit subsets (the last two carrying
  llmkit's fail-fast ``CircuitOpenError`` / ``OutputLimitError``, which must
  stay out of the retried transport set); including litellm's own
  ``ServiceUnavailableError`` (its 503, which does not subclass
  ``openai.InternalServerError``);
* instructor's in-call schema-repair budget (a per-call ``AsyncRetrying``
  stopping after two attempts total = one repair re-ask) as pinned by
  ``acompletion_structured``;
* the ``with_retries`` primitive's ``max_attempts`` naming and the nested-retry
  guard that stops an outer wrapper from multiplying the inner budget;
* the validation/transport budget split (a deterministic schema failure can't
  burn the full transport budget).

A canonical transient error is :class:`TimeoutError`; a canonical
non-recoverable one is an ``openai`` auth/4xx status error.
"""

from __future__ import annotations

import asyncio
import json
import random
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import litellm
import openai
import pytest
from instructor.core.exceptions import (
    AsyncValidationError,
    IncompleteOutputException,
    InstructorRetryException,
    ResponseParsingError,
)
from pydantic import ValidationError
from tenacity import AsyncRetrying

from llmkit import (
    DEFAULT_RETRY_POLICY,
    LLM_RECOVERABLE_ERRORS,
    NO_RETRY,
    CircuitOpenError,
    LocalYamlLogSink,
    OutputLimitError,
    RetryPolicy,
    configure_llm_logging,
    structured_output,
)
from llmkit.capture import capture_llm_log_paths
from llmkit.exceptions import (
    LLM_BACKPRESSURE_ERRORS,
    LLM_OUTPUT_LIMIT_ERRORS,
    LLM_SCHEMA_ERRORS,
    LLM_TRANSPORT_ERRORS,
    REPAIRABLE_PARSE_ERRORS,
)
from llmkit.retry import with_retries
from tests._support import OkSchema, capture_structured_provider_kwargs

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


def _wrap_v2(inner: BaseException) -> InstructorRetryException:
    """The real instructor 1.15.x wrap shape: a ``str`` message with the root
    error on ``__cause__`` (``raise InstructorRetryException(str) from inner``),
    as opposed to the pre-1.15 ``args[0]``-is-the-exception shape the other
    hand-built tests here use to exercise the compat unwrap branch."""
    try:
        raise InstructorRetryException("boom", n_attempts=2, total_usage=0) from inner
    except InstructorRetryException as exc:
        return exc


# --- RetryPolicy defaults & validation -----------------------------------


def test_default_retry_policy_values() -> None:
    """The shipped default budget is three attempts with 0.5s base backoff
    and a 30s cap on any single backoff sleep."""
    assert DEFAULT_RETRY_POLICY.max_attempts == 3
    assert DEFAULT_RETRY_POLICY.backoff_base_seconds == 0.5
    assert DEFAULT_RETRY_POLICY.max_backoff_seconds == 30.0


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


@pytest.mark.parametrize("bad_cap", [0.0, -1.0])
def test_non_positive_max_backoff_seconds_raises_value_error(bad_cap: float) -> None:
    """``RetryPolicy`` rejects a non-positive backoff cap at construction,
    like its other numeric fields (disable backoff via ``backoff_base_seconds=0``)."""
    with pytest.raises(ValueError, match="max_backoff_seconds must be > 0"):
        _ = RetryPolicy(max_backoff_seconds=bad_cap)


def test_default_retry_after_cap_is_sixty_seconds() -> None:
    """The shipped default caps a server-supplied ``Retry-After`` at 60s — a
    separate ceiling from the 30s ``max_backoff_seconds`` on the *computed*
    exponential, so a legitimate 45s server directive is still honoured."""
    assert DEFAULT_RETRY_POLICY.retry_after_cap == 60.0


@pytest.mark.parametrize("bad_cap", [0.0, -1.0])
def test_non_positive_retry_after_cap_raises_value_error(bad_cap: float) -> None:
    """``RetryPolicy`` rejects a non-positive ``retry_after_cap`` at construction."""
    with pytest.raises(ValueError, match="retry_after_cap must be > 0"):
        _ = RetryPolicy(retry_after_cap=bad_cap)


# --- transient set excludes auth / non-429 4xx ---------------------------


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """Build a minimal ``openai`` status error of *cls* for offline tests."""
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def _litellm_503() -> litellm.exceptions.ServiceUnavailableError:
    """Build litellm's 503 the way litellm itself constructs it.

    LiteLLM's exception mapping raises ``ServiceUnavailableError`` with
    keyword ``message`` / ``llm_provider`` / ``model`` (and optionally the
    underlying ``httpx.Response``) — not the ``openai`` SDK's
    ``(message, response=, body=)`` shape.
    """
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return litellm.exceptions.ServiceUnavailableError(
        message="service unavailable",
        llm_provider="openai",
        model="gpt-4o-mini",
        response=httpx.Response(503, request=request),
    )


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
    assert isinstance(_status_error(openai.InternalServerError, 500), LLM_RECOVERABLE_ERRORS)
    conn = openai.APIConnectionError(message="down", request=httpx.Request("POST", "https://x"))
    assert isinstance(conn, LLM_RECOVERABLE_ERRORS)


def test_litellm_service_unavailable_503_is_in_transport_set() -> None:
    """litellm's real 503 class is transport-recoverable.

    ``litellm.exceptions.ServiceUnavailableError`` does NOT subclass
    ``openai.InternalServerError`` (its MRO goes straight to
    ``openai.APIStatusError``), so the subclass-of-openai pattern the rest of
    the transport set relies on never covered it — it must be (and is) listed
    explicitly in ``LLM_TRANSPORT_ERRORS``, via a lazy stand-in that resolves
    litellm's class at ``isinstance`` time (litellm is imported in this test
    session, so the real class matches here; the litellm-free import path is
    pinned in ``tests/packaging/test_litellm_lazy_import.py``).
    """
    err = _litellm_503()
    # The premise that forces the explicit listing: no openai 5xx subclassing.
    assert not isinstance(err, openai.InternalServerError)
    assert isinstance(err, LLM_TRANSPORT_ERRORS)
    assert isinstance(err, LLM_RECOVERABLE_ERRORS)


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


@pytest.mark.asyncio
async def test_structured_litellm_503_is_retried_on_transport_budget() -> None:
    """A litellm-native 503 gets the full transport budget (3 attempts).

    Regression test: litellm raises ``ServiceUnavailableError`` for HTTP 503
    (never ``openai.InternalServerError``), so before the explicit listing in
    ``LLM_TRANSPORT_ERRORS`` a real 503 matched neither retry budget and
    propagated unretried after a single attempt.
    """
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _litellm_503()

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(litellm.exceptions.ServiceUnavailableError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_structured_litellm_503_is_retried_then_succeeds() -> None:
    """A transient litellm 503 is recovered: retried once, then succeeds."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise _litellm_503()
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert calls[0] == 2


# --- instructor's in-call schema-repair budget -----------------------------


def test_instructor_in_call_budget_is_two_attempts_one_parse_repair() -> None:
    """The structured call feeds instructor a per-call ``AsyncRetrying`` that
    keeps the historical budget (``stop_after_attempt(2)`` counts *total
    attempts*, so 2 = two attempts total = one schema-repair re-ask) but
    restricts the re-ask to *genuine parse failures* and sets ``reraise=True``,
    mirroring instructor's own int-based path.

    The predicate retries pydantic / JSON / instructor parse errors and
    **declines everything else**: a transport (429/5xx/network) or permanent
    (401/400/403) error is not re-asked in-call — re-asking it would only be a
    blind, backoff-free re-send inside the single rate-limiter slot, doubling
    the real requests a structured call makes. ``IncompleteOutputException``
    (length truncation, which a same-budget re-ask cannot fix) stays declined
    as defense in depth even though it is already outside the parse set today.
    The declined predicate's wrap / bare-propagation behaviour is pinned
    end-to-end in ``test_output_limit_failfast.py`` and
    ``test_in_call_retry_scope.py``.
    """
    seen = capture_structured_provider_kwargs()
    retrying = seen["max_retries"]
    assert isinstance(retrying, AsyncRetrying)
    assert getattr(retrying.stop, "max_attempt_number", None) == 2
    # ``reraise=True`` matches instructor's int path: on exhaustion the raw last
    # error rides ``__cause__`` (no tenacity ``RetryError`` hop).
    assert retrying.reraise is True
    predicate = cast("Callable[[BaseException], bool]", getattr(retrying.retry, "predicate", None))
    assert callable(predicate)
    # The mirrored parse set (pydantic + stdlib JSON + instructor parse signals)
    # is what instructor's own int path restricts its re-ask to.
    assert REPAIRABLE_PARSE_ERRORS == (
        ValidationError,
        json.JSONDecodeError,
        AsyncValidationError,
        ResponseParsingError,
    )
    # Genuine parse failures are re-asked...
    assert predicate(ValidationError.from_exception_data("Bad", [])) is True
    assert predicate(json.JSONDecodeError("bad", "{", 0)) is True
    # ...every non-parse error is declined (the fix — no blind in-call re-send
    # of transport/permanent errors). A plain ``ValueError`` is declined too:
    # only the *specific* parse types opt in, not their ``ValueError`` base.
    assert predicate(TimeoutError("transient")) is False
    assert predicate(RuntimeError("permanent")) is False
    assert predicate(ValueError("not a parse signal")) is False
    # Defense in depth: length truncation stays declined.
    assert predicate(IncompleteOutputException(last_completion=None)) is False


# --- unified naming: max_attempts is the only name ------------------------


def test_recoverable_set_union_invariant_still_holds() -> None:
    """``LLM_RECOVERABLE_ERRORS`` is exactly the union of the *four* subsets
    (transport + schema + backpressure + output-limit), so the documented
    single catch-set contract is preserved as each fail-fast type
    (``CircuitOpenError``, ``OutputLimitError``) joins it."""
    assert set(LLM_RECOVERABLE_ERRORS) == set(LLM_TRANSPORT_ERRORS) | set(LLM_SCHEMA_ERRORS) | set(
        LLM_BACKPRESSURE_ERRORS
    ) | set(LLM_OUTPUT_LIMIT_ERRORS)
    # The four subsets are pairwise disjoint and each carries its expected members.
    assert set(LLM_TRANSPORT_ERRORS).isdisjoint(LLM_SCHEMA_ERRORS)
    assert set(LLM_TRANSPORT_ERRORS).isdisjoint(LLM_BACKPRESSURE_ERRORS)
    assert set(LLM_SCHEMA_ERRORS).isdisjoint(LLM_BACKPRESSURE_ERRORS)
    assert set(LLM_OUTPUT_LIMIT_ERRORS).isdisjoint(LLM_TRANSPORT_ERRORS)
    assert set(LLM_OUTPUT_LIMIT_ERRORS).isdisjoint(LLM_SCHEMA_ERRORS)
    assert set(LLM_OUTPUT_LIMIT_ERRORS).isdisjoint(LLM_BACKPRESSURE_ERRORS)
    assert ValidationError in LLM_SCHEMA_ERRORS
    assert TimeoutError in LLM_TRANSPORT_ERRORS
    assert ValidationError not in LLM_TRANSPORT_ERRORS
    # The breaker error is fail-fast: it is the backpressure subset, and crucially
    # is NOT in the transport set (the default ``retry_on``) — so the retry layer
    # never re-asks a circuit the breaker already knows is open.
    assert CircuitOpenError in LLM_BACKPRESSURE_ERRORS
    assert CircuitOpenError not in LLM_TRANSPORT_ERRORS
    assert CircuitOpenError not in LLM_SCHEMA_ERRORS
    assert CircuitOpenError in LLM_RECOVERABLE_ERRORS
    # Same fail-fast contract for the output-limit subset: catchable via the
    # union, retried on neither budget.
    assert OutputLimitError in LLM_OUTPUT_LIMIT_ERRORS
    assert OutputLimitError not in LLM_TRANSPORT_ERRORS
    assert OutputLimitError not in LLM_SCHEMA_ERRORS
    assert OutputLimitError in LLM_RECOVERABLE_ERRORS


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
@pytest.mark.parametrize(
    "make_inner",
    [
        pytest.param(_make_validation_error, id="pydantic-ValidationError"),
        pytest.param(lambda: ResponseParsingError("unparseable"), id="ResponseParsingError"),
        pytest.param(lambda: AsyncValidationError("async-validator"), id="AsyncValidationError"),
    ],
)
async def test_v2_wrapped_parse_error_uses_validation_budget(
    make_inner: Callable[[], BaseException],
) -> None:
    """The **real** 1.15.x wrap shape (``raise InstructorRetryException(str) from
    inner``, ``__cause__`` = raw error) routes *every* genuine parse failure to
    the lower validation budget (2) — not just pydantic ``ValidationError`` but
    instructor's own ``ResponseParsingError`` (e.g. a blocked/empty Gemini
    ``Mode.JSON`` generation) and ``AsyncValidationError`` (a failing async field
    validator). Before ``_SCHEMA_SHAPED_CAUSES`` was aligned with the in-call
    re-ask set, the latter two hit the permanent-cause guard and failed fast
    after a single attempt — a type instructor re-asks in-call being denied its
    cross-call validation retry."""
    calls = [0]

    async def _fail(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise _wrap_v2(make_inner())

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fail),
        pytest.raises(InstructorRetryException),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 2


def test_in_call_reask_set_and_schema_shaped_causes_are_one_tuple() -> None:
    """The in-call re-ask predicate set and the outer validation-budget
    ``_SCHEMA_SHAPED_CAUSES`` are the *same object* — the single source of truth
    in ``llmkit.exceptions``. This is the structural guard against the two
    drifting apart: a type instructor re-asks in-call is, by construction, also
    charged the validation budget across calls rather than mis-routed to
    fail-fast."""
    from llmkit.retry import _SCHEMA_SHAPED_CAUSES

    assert _SCHEMA_SHAPED_CAUSES is REPAIRABLE_PARSE_ERRORS
    assert set(REPAIRABLE_PARSE_ERRORS) == {
        ValidationError,
        json.JSONDecodeError,
        AsyncValidationError,
        ResponseParsingError,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "status"),
    [
        pytest.param(openai.AuthenticationError, 401, id="401"),
        pytest.param(openai.BadRequestError, 400, id="400"),
        pytest.param(openai.PermissionDeniedError, 403, id="403"),
    ],
)
async def test_instructor_wrapped_permanent_4xx_fails_fast(
    cls: type[openai.APIStatusError], status: int
) -> None:
    """An ``InstructorRetryException`` wrapping a *permanent* 4xx (bad key,
    malformed request) fails fast: exactly one outer attempt, no retry on
    either budget — the same fail-fast contract the bare-4xx text path
    honours. Regression test: the wrapper itself matched
    ``validation_retry_on``, so a wrapped 401 used to burn a validation-budget
    retry (a whole extra instructor call pair) on a request that could never
    succeed."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise InstructorRetryException(_status_error(cls, status), n_attempts=1, total_usage=0)

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(InstructorRetryException),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_transport_cause",
    [
        pytest.param(lambda: _status_error(openai.RateLimitError, 429), id="429"),
        pytest.param(_litellm_503, id="litellm-503"),
    ],
)
async def test_retry_on_none_wrapped_transport_cause_uses_transport_budget(
    make_transport_cause: Callable[[], Exception],
) -> None:
    """``retry_on=None`` ("retry on any Exception") must not break the budget
    *routing*: an ``InstructorRetryException`` wrapping a transport cause
    (429, litellm 503) is classified against ``LLM_TRANSPORT_ERRORS`` after
    unwrapping and charged the full transport budget (4), not the lower
    validation budget (2). Regression test: the cause check used to require
    an explicit ``retry_on``, so direct ``with_retries`` callers using the
    documented ``None`` default got the exact misclassification the unwrap
    exists to prevent."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _fn() -> str:
        calls[0] += 1
        raise InstructorRetryException(make_transport_cause(), n_attempts=1, total_usage=0)

    with pytest.raises(InstructorRetryException):
        _ = await with_retries(
            _fn,
            max_attempts=4,
            retry_on=None,
            validation_max_attempts=2,
            validation_retry_on=LLM_SCHEMA_ERRORS,
        )

    assert calls[0] == 4


@pytest.mark.asyncio
async def test_retry_on_none_wrapped_validation_cause_uses_validation_budget() -> None:
    """Under the same ``retry_on=None`` setup, an ``InstructorRetryException``
    wrapping a genuine ``ValidationError`` still lands on the lower validation
    budget (2) — the transport fallback must not swallow real schema failures."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _fn() -> str:
        calls[0] += 1
        raise InstructorRetryException(_make_validation_error(), n_attempts=1, total_usage=0)

    with pytest.raises(InstructorRetryException):
        _ = await with_retries(
            _fn,
            max_attempts=4,
            retry_on=None,
            validation_max_attempts=2,
            validation_retry_on=LLM_SCHEMA_ERRORS,
        )

    assert calls[0] == 2


@pytest.mark.asyncio
async def test_retry_on_none_wrapped_permanent_4xx_fails_fast() -> None:
    """Even under ``retry_on=None`` ("retry on any Exception"), an
    ``InstructorRetryException`` wrapping a permanent 401 propagates after a
    single attempt: the unwrapped cause is neither transport- nor
    schema-shaped, so the fail-fast contract overrides the blanket retry for
    the wrapper, exactly as under the curated default sets."""
    from instructor.core import InstructorRetryException

    calls = [0]

    async def _fn() -> str:
        calls[0] += 1
        raise InstructorRetryException(
            _status_error(openai.AuthenticationError, 401), n_attempts=1, total_usage=0
        )

    with pytest.raises(InstructorRetryException):
        _ = await with_retries(
            _fn,
            max_attempts=4,
            retry_on=None,
            validation_max_attempts=2,
            validation_retry_on=LLM_SCHEMA_ERRORS,
        )

    assert calls[0] == 1


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
