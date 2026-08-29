"""Offline tests for the opt-in per-provider circuit breaker.

These never touch the network. They pin the aggregate guard that stops doomed
work when a provider is *down* — the ``0.4.0`` follow-on to adaptive concurrency
(AIMD), reading the *same* per-provider unwrapped outcome stream:

* the breaker state machine on ``CircuitBreaker`` (count-based ring, trip
  threshold + minimum sample, rolling window, cooldown → single HALF_OPEN probe,
  close-on-success-and-clear / reopen-on-failure-and-re-arm) — driven with a
  frozen clock;
* the end-to-end wiring through ``acquire_async``: a full window of throttles
  **opens** the breaker, after which a call raises ``CircuitOpenError`` *fast*
  with no provider call and no slot/RPM held; a structured-call throttle (wrapped
  in ``InstructorRetryException``) feeds the window (D0);
* the HALF_OPEN probe through the real acquire path — closes on success, re-opens
  on a throttle, and a **cancelled** probe (in the body *or* parked mid-acquire)
  reverts to OPEN and refunds its RPM token, so the breaker never wedges;
* per-provider isolation, the breaker being one object shared across event loops
  (single process-wide probe), and ``breaker=False`` being a complete no-op;
* ``CircuitOpenError`` is **not** retried by ``with_retries`` — not even under
  its ``retry_on=None`` default, whether the error arrives bare or wrapped in
  ``InstructorRetryException`` — the structured call function, or the streaming
  loop; an explicit ``retry_on`` listing the type still opts back in.

Clock-driven state reads ``rate_limiting._now`` (monkeypatched) so the cooldown
is deterministic and nothing sleeps for real.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import patch

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import (
    CircuitOpenError,
    backpressure_callback,
    configure_rate_limit,
    structured_output,
)
from llmkit.exceptions import LLM_SCHEMA_ERRORS, LLM_TRANSPORT_ERRORS
from llmkit.rate_limiting import BackpressureEvent, GlobalRateLimiter, _tuning
from llmkit.rate_limiting._breaker import Admit, CircuitBreaker, CircuitState
from llmkit.rate_limiting._tuning import BREAKER_COOLDOWN, BREAKER_MIN_SAMPLES
from llmkit.retry import RetryPolicy, with_retries
from tests._support import OkSchema, quiet_logging

_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def _wrapped(status: int) -> InstructorRetryException:
    """A throttle wrapped as instructor surfaces it on the structured path."""
    return InstructorRetryException(
        _status_error(openai.RateLimitError, status), n_attempts=1, total_usage=0
    )


def _fill_throttles(breaker: CircuitBreaker, n: int) -> None:
    """Feed *n* throttle outcomes straight into a breaker's window."""
    for _ in range(n):
        _ = breaker.on_record(throttled=True)


# --- CircuitBreaker state machine (frozen clock) ------------------------


def test_full_window_of_throttles_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full window at/above the threshold opens; below the min sample it can't."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    breaker = CircuitBreaker("openai", ceiling=8)
    # Below the minimum sample, even all-throttles cannot trip the breaker.
    for _ in range(BREAKER_MIN_SAMPLES - 1):
        assert breaker.on_record(throttled=True) is None
        assert breaker._state is CircuitState.CLOSED
    # The outcome that fills the window to the minimum sample opens it.
    event = breaker.on_record(throttled=True)
    assert event == BackpressureEvent("openai", 8, 0, "breaker_open")
    assert breaker._state is CircuitState.OPEN
    # Within the cooldown, every admission is rejected (no event re-fired).
    assert breaker.admit() == (Admit.REJECT, None)


def test_sub_threshold_window_does_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full window below the throttle threshold leaves the breaker closed."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    breaker = CircuitBreaker("openai", ceiling=8)
    # 9 throttles + 11 successes over a 20-wide window = 0.45 < 0.5.
    for i in range(BREAKER_MIN_SAMPLES):
        _ = breaker.on_record(throttled=i < 9)
    assert breaker._state is CircuitState.CLOSED
    assert breaker.admit()[0] is Admit.NORMAL


def test_rolling_window_opens_once_throttles_dominate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A healthy window then a storm: the breaker opens when throttles reach half
    of the *rolling* window, not from the start of time."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    breaker = CircuitBreaker("openai", ceiling=4)
    # Fill the window with successes — healthy, stays closed.
    for _ in range(BREAKER_MIN_SAMPLES):
        assert breaker.on_record(throttled=False) is None
    assert breaker._state is CircuitState.CLOSED
    # Now a throttle storm. The window is 20 wide, so the 10th throttle makes the
    # rolling fraction exactly 0.5 (10 throttles evicting 10 of the old successes).
    for _ in range(BREAKER_MIN_SAMPLES // 2 - 1):
        assert breaker.on_record(throttled=True) is None
    assert breaker._state is CircuitState.CLOSED
    event = breaker.on_record(throttled=True)
    assert event == BackpressureEvent("openai", 4, 0, "breaker_open")


def test_cooldown_then_single_probe_closes_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the cooldown, HALF_OPEN admits exactly one probe; a clean success
    closes the breaker and clears the window."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    breaker = CircuitBreaker("openai", ceiling=8)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)
    assert breaker._state is CircuitState.OPEN

    # Still cooling down: rejected, no transition.
    assert breaker.admit() == (Admit.REJECT, None)
    # Cooldown elapsed: the first admission becomes the single probe.
    clock["t"] += BREAKER_COOLDOWN
    decision, event = breaker.admit()
    assert decision is Admit.PROBE
    assert event == BackpressureEvent("openai", 0, 1, "breaker_half_open")
    assert breaker._state is CircuitState.HALF_OPEN
    # A second concurrent admission while the probe is out is rejected.
    assert breaker.admit() == (Admit.REJECT, None)

    # The probe succeeds: close and clear the window.
    close_event = breaker.on_probe_success()
    assert close_event == BackpressureEvent("openai", 1, 8, "breaker_closed")
    assert breaker._state is CircuitState.CLOSED
    assert len(breaker._window) == 0
    assert breaker.admit()[0] is Admit.NORMAL


def test_probe_failure_reopens_and_rearms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe re-opens the breaker and re-arms the cooldown; the claim is
    always released so the breaker cannot wedge HALF_OPEN."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    breaker = CircuitBreaker("openai", ceiling=8)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)
    clock["t"] += BREAKER_COOLDOWN
    assert breaker.admit()[0] is Admit.PROBE

    reopen = breaker.on_probe_failure()
    assert reopen == BackpressureEvent("openai", 8, 0, "breaker_open")
    assert breaker._state is CircuitState.OPEN
    assert breaker._probe_in_flight is False
    # The cooldown was re-armed from now: an immediate retry is rejected...
    assert breaker.admit() == (Admit.REJECT, None)
    # ...and only after another full cooldown does a fresh probe go.
    clock["t"] += BREAKER_COOLDOWN
    assert breaker.admit()[0] is Admit.PROBE


# --- end-to-end wiring through acquire_async -----------------------------


async def test_opens_after_threshold_then_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full window of (wrapped, structured-path) throttles through the real
    acquire path opens the breaker; the next call raises ``CircuitOpenError``
    *before* the provider call, holding no slot (the call counter stays frozen)."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # frozen: still cooling
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "openai"
    calls = [0]
    events: list[BackpressureEvent] = []

    with backpressure_callback(events.append):
        # Sequential (so unsaturated — AIMD stays put) wrapped 429s fill the window.
        for _ in range(BREAKER_MIN_SAMPLES):
            with pytest.raises(InstructorRetryException):
                async with GlobalRateLimiter.acquire_async(key):
                    calls[0] += 1
                    raise _wrapped(429)  # D0: a structured-call throttle, wrapped

        assert calls[0] == BREAKER_MIN_SAMPLES  # every attempt reached the provider
        # The breaker is now OPEN: a fast fail, before the body and any slot.
        with pytest.raises(CircuitOpenError) as excinfo:
            async with GlobalRateLimiter.acquire_async(key):
                calls[0] += 1  # must not run

    assert excinfo.value.provider == key
    assert calls[0] == BREAKER_MIN_SAMPLES  # frozen: no provider call while OPEN
    # Exactly one transition event fired — the open at the full window.
    assert events == [BackpressureEvent(key, 8, 0, "breaker_open")]
    # No concurrency slot was taken by the rejected call.
    assert GlobalRateLimiter._get_async_gate(key)._in_flight == 0


async def test_open_breaker_deducts_no_rpm_and_holds_no_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected (OPEN) call deducts no RPM token and occupies no concurrency
    slot — it fast-fails before either gate."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=8, rpm=600, breaker=True)
    key = "openai"
    breaker = GlobalRateLimiter._get_breaker(key)
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 1_000.0  # within the cooldown at the frozen now

    with pytest.raises(CircuitOpenError):
        async with GlobalRateLimiter.acquire_async(key):
            pass

    assert GlobalRateLimiter._rpm_buckets[key]._level == 8.0  # full: nothing deducted
    assert GlobalRateLimiter._get_async_gate(key)._in_flight == 0  # no slot held


async def test_half_open_probe_closes_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real acquire path: after the cooldown a clean probe closes the
    breaker, and normal calls flow again."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "openai"
    breaker = GlobalRateLimiter._get_breaker(key)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)

    # Within the cooldown: fast fail.
    with pytest.raises(CircuitOpenError):
        async with GlobalRateLimiter.acquire_async(key):
            pass

    clock["t"] += BREAKER_COOLDOWN
    events: list[BackpressureEvent] = []
    with backpressure_callback(events.append):
        async with GlobalRateLimiter.acquire_async(key):  # admitted as the probe
            pass  # clean success

    assert breaker._state is CircuitState.CLOSED
    assert events == [
        BackpressureEvent(key, 0, 1, "breaker_half_open"),
        BackpressureEvent(key, 1, 8, "breaker_closed"),
    ]
    # Closed again: a normal call is admitted (no CircuitOpenError).
    async with GlobalRateLimiter.acquire_async(key):
        pass


async def test_half_open_probe_reopens_on_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe that throttles re-opens the breaker and re-arms the cooldown."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "openai"
    breaker = GlobalRateLimiter._get_breaker(key)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)
    clock["t"] += BREAKER_COOLDOWN

    with pytest.raises(InstructorRetryException):
        async with GlobalRateLimiter.acquire_async(key):  # the probe
            raise _wrapped(429)  # ...throttles

    assert breaker._state is CircuitState.OPEN
    # Re-armed cooldown: an immediate call is rejected again.
    with pytest.raises(CircuitOpenError):
        async with GlobalRateLimiter.acquire_async(key):
            pass


async def test_cancelled_probe_in_body_reverts_to_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe cancelled while its body runs reverts to OPEN (claim released) —
    no wedged HALF_OPEN — and other calls stay rejected meanwhile (single probe)."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "openai"
    breaker = GlobalRateLimiter._get_breaker(key)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)
    clock["t"] += BREAKER_COOLDOWN

    started = asyncio.Event()

    async def probe() -> None:
        async with GlobalRateLimiter.acquire_async(key):  # admitted as the probe
            started.set()
            await asyncio.sleep(3_600)  # park in the body until cancelled

    task = asyncio.create_task(probe())
    _ = await started.wait()
    assert breaker._state is CircuitState.HALF_OPEN
    assert breaker._probe_in_flight is True
    # The single probe is out: every other call is rejected.
    with pytest.raises(CircuitOpenError):
        async with GlobalRateLimiter.acquire_async(key):
            pass

    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker._state is CircuitState.OPEN  # reverted, not wedged HALF_OPEN
    assert breaker._probe_in_flight is False  # claim released


async def test_cancelled_probe_mid_acquire_reverts_and_refunds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe cancelled *during acquisition* (parked on the TPM gate, after its
    RPM token is deducted) reverts to OPEN and refunds the RPM token — the probe
    claim shares the rpm-refund-on-cancel leak class and is released coherently."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])  # frozen: no refill
    configure_rate_limit(max_concurrent=8, rpm=480, tpm=6_000, breaker=True)
    key = "openai"

    # Exhaust TPM (while CLOSED) so the probe will park on the TPM gate mid-acquire.
    async with GlobalRateLimiter.acquire_async(key) as slot:
        slot.record_tokens(9_000)
    rpm_bucket = GlobalRateLimiter._rpm_buckets[key]
    assert rpm_bucket._level == 7.0  # capacity min(8, 480) == 8, minus the one call

    # Force OPEN with an already-elapsed cooldown (no clock advance, so TPM stays dry).
    breaker = GlobalRateLimiter._get_breaker(key)
    breaker._state = CircuitState.OPEN
    breaker._opened_at = clock["t"] - BREAKER_COOLDOWN - 1.0

    async def probe() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            pass

    task = asyncio.create_task(probe())
    for _ in range(10):
        await asyncio.sleep(0)
    # Admitted as the probe, RPM debited, now parked on the exhausted TPM gate.
    assert breaker._state is CircuitState.HALF_OPEN
    assert breaker._probe_in_flight is True
    assert rpm_bucket._level == 6.0  # RPM debited by the probe

    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker._state is CircuitState.OPEN  # reverted
    assert breaker._probe_in_flight is False  # claim released
    assert rpm_bucket._level == 7.0  # RPM refunded, not silently lost


async def test_breakers_are_per_provider_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening one provider's breaker leaves every other provider unaffected."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=8, breaker=True)
    _fill_throttles(GlobalRateLimiter._get_breaker("openai"), BREAKER_MIN_SAMPLES)

    # openai is open → fast fail.
    with pytest.raises(CircuitOpenError):
        async with GlobalRateLimiter.acquire_async("openai"):
            pass
    # google is untouched → admits normally.
    async with GlobalRateLimiter.acquire_async("google"):
        pass
    assert GlobalRateLimiter._breaker_states["google"]._state is CircuitState.CLOSED


async def test_breaker_off_is_a_complete_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``breaker=False`` (the default) nothing changes: no breaker is created,
    even a flood of throttles never raises ``CircuitOpenError``."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=8)  # breaker defaults off
    key = "openai"
    for _ in range(BREAKER_MIN_SAMPLES * 2):
        with pytest.raises(InstructorRetryException):
            async with GlobalRateLimiter.acquire_async(key):
                raise _wrapped(429)
    # No breaker object was ever created, and a fresh call still admits normally.
    assert key not in GlobalRateLimiter._breaker_states
    async with GlobalRateLimiter.acquire_async(key):
        pass


# --- cross-loop: one breaker, one process-wide probe ---------------------


def test_breaker_is_shared_across_event_loops() -> None:
    """One breaker object per provider name, shared across loops (it has no loop
    affinity — only a lock-guarded window), so an open observed on any loop is
    seen on every loop."""
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "google"
    seen: list[CircuitBreaker] = []

    async def grab() -> None:
        seen.append(GlobalRateLimiter._get_breaker(key))

    asyncio.run(grab())  # loop A
    asyncio.run(grab())  # loop B
    assert seen[0] is seen[1]


def test_probe_is_a_single_process_wide_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once HALF_OPEN admits the probe, every further admission — on this loop or
    any other, since the breaker is one shared, lock-guarded object — is rejected
    until the probe resolves."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    breaker = CircuitBreaker("openai", ceiling=8)
    _fill_throttles(breaker, BREAKER_MIN_SAMPLES)
    clock["t"] += BREAKER_COOLDOWN

    decision, event = breaker.admit()
    assert decision is Admit.PROBE
    assert event == BackpressureEvent("openai", 0, 1, "breaker_half_open")
    # Two more admissions (standing in for two other loops) are both rejected.
    assert breaker.admit() == (Admit.REJECT, None)
    assert breaker.admit() == (Admit.REJECT, None)


# --- CircuitOpenError is never retried -----------------------------------


async def test_circuit_open_error_not_retried_by_with_retries() -> None:
    """``with_retries`` does not retry ``CircuitOpenError``: it is outside both the
    transport and validation sets, so it propagates on the first raise."""
    calls = [0]

    async def fn() -> None:
        calls[0] += 1
        raise CircuitOpenError("openai")

    with pytest.raises(CircuitOpenError):
        _ = await with_retries(
            fn,
            max_attempts=3,
            retry_on=LLM_TRANSPORT_ERRORS,
            validation_max_attempts=2,
            validation_retry_on=LLM_SCHEMA_ERRORS,
        )
    assert calls[0] == 1


async def test_with_retries_retry_on_none_does_not_retry_circuit_open_error() -> None:
    """The regression test for the bare-``CircuitOpenError`` hole: with the
    documented default ``retry_on=None`` ("retry on any Exception"),
    ``with_retries`` must still fail fast on a bare ``CircuitOpenError`` —
    re-asking a circuit the breaker already knows is open would defeat its
    purpose. Fails pre-fix with ``calls[0] == 3`` (charged to the transport
    budget); passes with the zero-budget carve-out."""
    calls = [0]

    async def fn() -> None:
        calls[0] += 1
        raise CircuitOpenError("openai")

    with pytest.raises(CircuitOpenError):
        _ = await with_retries(fn, max_attempts=3)
    assert calls[0] == 1


async def test_with_retries_explicit_opt_in_still_retries_circuit_open_error() -> None:
    """A caller that explicitly lists the type in ``retry_on`` gets its retries:
    the fail-fast is a default, not a prohibition (the ``OutputLimitError``
    precedent). ``max_attempts=2`` -> two attempts."""
    calls = [0]

    async def fn() -> None:
        calls[0] += 1
        raise CircuitOpenError("openai")

    with pytest.raises(CircuitOpenError):
        _ = await with_retries(fn, max_attempts=2, retry_on=(CircuitOpenError,))
    assert calls[0] == 2


async def test_with_retries_retry_on_none_wrapped_circuit_open_fails_fast() -> None:
    """A ``CircuitOpenError`` wrapped in ``InstructorRetryException`` fails fast
    under ``retry_on=None`` — one attempt. This pins the *wrapped form's*
    fail-fast behaviour, not which guard owns it: both the zero-budget carve-out
    and the pre-existing ``wraps_permanent_cause`` guard (a wrapped cause that is
    neither transport- nor schema-shaped) catch it, so it passed before this
    change too and stays green as a regression guard on the wrapped path. The
    *bare*-``CircuitOpenError`` test above is the discriminating regression for
    the new carve-out (it fails pre-fix with ``calls == 3``)."""
    calls = [0]

    async def fn() -> None:
        calls[0] += 1
        raise InstructorRetryException(CircuitOpenError("openai"), n_attempts=1, total_usage=0)

    with pytest.raises(InstructorRetryException):
        _ = await with_retries(fn, max_attempts=3, retry_on=None, validation_max_attempts=2)
    assert calls[0] == 1


async def test_structured_call_does_not_retry_circuit_open_error() -> None:
    """The structured call function fails fast on a ``CircuitOpenError`` raised by
    the limiter — one attempt, no retry (it is not in the transport set)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise CircuitOpenError("openai")

    with (
        quiet_logging(),
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(CircuitOpenError),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )
    assert calls[0] == 1


async def test_streaming_call_does_not_retry_circuit_open_error() -> None:
    """The streaming loop does not retry a pre-first-chunk ``CircuitOpenError``:
    it is outside the transport/validation sets the loop keys off."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        raise CircuitOpenError("openai")
        # Unreachable by design: the bare ``yield`` makes this an async generator;
        # the ``raise`` above precedes it on every call.
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    with quiet_logging(), patch("llmkit._litellm.astream_text", _transport):
        with pytest.raises(CircuitOpenError):
            _ = [
                chunk
                async for chunk in structured_output.text_llm_call_stream(
                    "hi", feature="test", retry=_NO_BACKOFF
                )
            ]
    assert calls[0] == 1  # a single pre-first-chunk attempt, never retried
