"""Breaker outage classification, neutral probes and caller-owned cooldown.

Exercise the real acquire boundaries with explicit outcome classes. All clocks
are controlled; no provider requests or real cooldown sleeps are involved.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException
from litellm.exceptions import Timeout

from llmkit import (
    BackpressureEvent,
    backpressure_callback,
    configure_rate_limit,
    get_rate_limit_config,
)
from llmkit._litellm import _within
from llmkit.exceptions import CircuitOpenError, ResultValidationError
from llmkit.rate_limiting import GlobalRateLimiter, _tuning
from llmkit.rate_limiting._breaker import Admit, CircuitState
from llmkit.rate_limiting._tuning import BREAKER_MIN_SAMPLES
from tests._support.diagnostics import lost_diagnostics, raising_handler


def _error(kind: str) -> BaseException:
    request = httpx.Request("POST", "https://provider.invalid")
    if kind.startswith("status-"):
        return openai.APIStatusError(
            "provider error",
            response=httpx.Response(int(kind.removeprefix("status-")), request=request),
            body=None,
        )
    if kind in {"litellm-gateway-timeout", "litellm-timeout"}:
        return Timeout(
            message="timeout",
            model="test",
            llm_provider="openai",
            exception_status_code=504 if kind == "litellm-gateway-timeout" else 408,
        )
    if kind == "sdk-connection":
        return openai.APIConnectionError(request=request)
    if kind == "http-connection":
        return httpx.ConnectError("connection refused", request=request)
    if kind == "builtin-connection":
        return ConnectionRefusedError("connection refused")
    if kind == "sdk-timeout":
        return openai.APITimeoutError(request=request)
    if kind == "http-timeout":
        return httpx.ConnectTimeout("connection deadline", request=request)
    if kind == "deadline":
        return TimeoutError("no provider response within 0.01s")
    if kind == "validation":
        return ResultValidationError("answer rejected")
    if kind == "cancel":
        return asyncio.CancelledError()
    if kind == "interrupt":
        return KeyboardInterrupt()
    if kind == "wrapped-connection":
        wrapped = InstructorRetryException("connection failed", n_attempts=1, total_usage=0)
        wrapped.__cause__ = openai.APIConnectionError(request=request)
        return wrapped
    if kind == "wrapped-deadline":
        wrapped = InstructorRetryException("deadline expired", n_attempts=1, total_usage=0)
        wrapped.__cause__ = TimeoutError("no provider response within 0.01s")
        return wrapped
    raise AssertionError(kind)


_OUTAGES = (
    "status-429",
    "status-500",
    "status-502",
    "status-503",
    "status-504",
    "status-529",
    "sdk-connection",
    "http-connection",
    "builtin-connection",
    "wrapped-connection",
    "litellm-gateway-timeout",
)
_NEUTRAL = (
    "status-400",
    "status-401",
    "status-403",
    "status-501",
    "status-505",
    "sdk-timeout",
    "http-timeout",
    "deadline",
    "validation",
    "cancel",
    "interrupt",
    "wrapped-deadline",
    "litellm-timeout",
)


async def _attempt(sync: bool, error: BaseException | None) -> None:
    if sync:
        with GlobalRateLimiter.acquire_sync("OpenAI"):
            if error is not None:
                raise error
    else:
        async with GlobalRateLimiter.acquire_async("OpenAI"):
            if error is not None:
                raise error


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("kind", _OUTAGES + _NEUTRAL)
async def test_only_outages_fill_breaker_window(
    monkeypatch: pytest.MonkeyPatch, sync: bool, kind: str
) -> None:
    monkeypatch.setattr(_tuning, "now", lambda: 100.0)
    configure_rate_limit(max_concurrent=4, breaker=True, adaptive=True)
    breaker = GlobalRateLimiter._get_breaker("openai")
    for _ in range(BREAKER_MIN_SAMPLES):
        error = _error(kind)
        with pytest.raises(type(error)) as caught:
            await _attempt(sync, error)
        assert caught.value is error
    if kind in _OUTAGES:
        assert breaker._state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await _attempt(sync, None)
    else:
        assert breaker._state is CircuitState.CLOSED
        assert not breaker._window
        await _attempt(sync, None)
    # Calls were sequential/unsaturated: outage detection must not force an
    # adaptive reduction. Non-throttle outages also must not act as throttles.
    assert GlobalRateLimiter._adaptive_states["openai"].limit() == 4


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("kind", _OUTAGES + _NEUTRAL)
async def test_probe_uses_same_outage_classification_and_releases_claim(
    monkeypatch: pytest.MonkeyPatch, sync: bool, kind: str
) -> None:
    clock = [100.0]
    monkeypatch.setattr(_tuning, "now", lambda: clock[0])
    configure_rate_limit(breaker=True, breaker_cooldown=2.0)
    breaker = GlobalRateLimiter._get_breaker("openai")
    for _ in range(BREAKER_MIN_SAMPLES):
        _ = breaker.on_record(outage=True)
    clock[0] += 2.0
    error = _error(kind)
    with pytest.raises(type(error)) as caught:
        await _attempt(sync, error)
    assert caught.value is error
    assert not breaker._probe_in_flight
    if kind in _OUTAGES:
        assert breaker._state is CircuitState.OPEN
        assert breaker.admit()[0] is Admit.REJECT
        clock[0] += 2.0
    else:
        assert breaker._state is CircuitState.HALF_OPEN
        assert not breaker._window
    # A neutral result admits the next probe immediately; an outage admits it
    # after a fresh cooldown. Only genuine success declares recovery.
    await _attempt(sync, None)
    assert breaker._state is CircuitState.CLOSED
    assert not breaker._probe_in_flight


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_disabled_limiter_does_not_consult_or_record_breaker(sync: bool) -> None:
    configure_rate_limit(enabled=False, breaker=True)
    for _ in range(BREAKER_MIN_SAMPLES + 1):
        with pytest.raises(ConnectionRefusedError):
            await _attempt(sync, ConnectionRefusedError("down"))
    await _attempt(sync, None)
    assert not GlobalRateLimiter._breaker_states
    assert not GlobalRateLimiter._async_gates
    assert not GlobalRateLimiter._sync_gates


async def test_actual_caller_deadline_does_not_reopen_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(_tuning, "now", lambda: clock[0])
    configure_rate_limit(breaker=True, breaker_cooldown=2.0)
    breaker = GlobalRateLimiter._get_breaker("openai")
    for _ in range(BREAKER_MIN_SAMPLES):
        _ = breaker.on_record(outage=True)
    clock[0] += 2.0
    with pytest.raises(TimeoutError, match="no provider response within 0s"):
        async with GlobalRateLimiter.acquire_async("openai"):
            await _within(0.0, asyncio.sleep(3600))
    assert breaker._state is CircuitState.HALF_OPEN
    assert not breaker._probe_in_flight
    async with GlobalRateLimiter.acquire_async("openai"):
        pass
    assert breaker._state is CircuitState.CLOSED


def test_cooldown_is_configurable_retained_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(_tuning, "now", lambda: clock[0])
    assert get_rate_limit_config().breaker_cooldown == 30.0
    configure_rate_limit(breaker=True, breaker_cooldown=1.25)
    old = GlobalRateLimiter._get_breaker("openai")
    configure_rate_limit(rpm=120)
    assert get_rate_limit_config().breaker_cooldown == 1.25
    breaker = GlobalRateLimiter._get_breaker("openai")
    assert breaker is not old
    for _ in range(BREAKER_MIN_SAMPLES):
        _ = breaker.on_record(outage=True)
    clock[0] += 1.0
    assert breaker.admit()[0] is Admit.REJECT
    clock[0] += 0.25
    assert breaker.admit()[0] is Admit.PROBE
    GlobalRateLimiter._reset()
    assert get_rate_limit_config().breaker_cooldown == 30.0
    assert not GlobalRateLimiter._breaker_states


@pytest.mark.parametrize("cooldown", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_cooldown_does_not_change_configuration(cooldown: float) -> None:
    configure_rate_limit(breaker=True, breaker_cooldown=2.0)
    before = get_rate_limit_config()
    breaker = GlobalRateLimiter._get_breaker("openai")
    with pytest.raises(ValueError, match="breaker_cooldown"):
        configure_rate_limit(max_concurrent=1, breaker_cooldown=cooldown)
    assert get_rate_limit_config() == before
    assert GlobalRateLimiter._get_breaker("openai") is breaker


def test_sync_interrupted_probe_acquisition_refunds_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_tuning, "now", lambda: 100.0)
    configure_rate_limit(breaker=True, rpm=120)
    breaker = GlobalRateLimiter._get_breaker("openai")
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 0.0
    gate = GlobalRateLimiter._get_sync_gate("openai")
    original: Callable[[], None] = gate.acquire

    def interrupt() -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(gate, "acquire", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with GlobalRateLimiter.acquire_sync("openai"):
            pytest.fail("the provider must not run")
    assert GlobalRateLimiter._rpm_buckets["openai"]._level == 8.0
    assert breaker._state is CircuitState.HALF_OPEN
    assert not breaker._probe_in_flight
    monkeypatch.setattr(gate, "acquire", original)
    with GlobalRateLimiter.acquire_sync("openai"):
        pass
    assert breaker._state is CircuitState.CLOSED


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("kind", ["sdk-connection", "status-500", "status-429"])
async def test_only_throttles_reduce_saturated_adaptive_limit(
    monkeypatch: pytest.MonkeyPatch, sync: bool, kind: str
) -> None:
    monkeypatch.setattr(_tuning, "now", lambda: 100.0)
    configure_rate_limit(max_concurrent=2, breaker=True, adaptive=True)
    error = _error(kind)
    if sync:
        with GlobalRateLimiter.acquire_sync("openai"):
            with pytest.raises(type(error)):
                await _attempt(sync, error)
    else:
        async with GlobalRateLimiter.acquire_async("openai"):
            with pytest.raises(type(error)):
                await _attempt(sync, error)
    expected = 1 if kind == "status-429" else 2
    assert GlobalRateLimiter._adaptive_states["openai"].limit() == expected
    # All three are breaker outages; the holder's success follows that outcome.
    assert list(GlobalRateLimiter._get_breaker("openai")._window) == [True, False]


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_interrupted_transition_callback_releases_probe(
    monkeypatch: pytest.MonkeyPatch, sync: bool
) -> None:
    monkeypatch.setattr(_tuning, "now", lambda: 100.0)
    configure_rate_limit(breaker=True, rpm=120)
    breaker = GlobalRateLimiter._get_breaker("openai")
    breaker._state = CircuitState.OPEN
    breaker._opened_at = 0.0

    def interrupted(_event: BackpressureEvent) -> None:
        raise KeyboardInterrupt()

    with backpressure_callback(interrupted), pytest.raises(KeyboardInterrupt):
        await _attempt(sync, None)
    assert breaker._state is CircuitState.HALF_OPEN
    assert not breaker._probe_in_flight
    assert GlobalRateLimiter._rpm_buckets["openai"]._level == 8.0
    await _attempt(sync, None)
    assert breaker._state is CircuitState.CLOSED


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("outcome", ["throttle", "success"])
@pytest.mark.parametrize("interruption", ["cancel", "interrupt", "diagnostic"])
async def test_probe_resolves_before_outcome_callback_can_interrupt(
    monkeypatch: pytest.MonkeyPatch, sync: bool, outcome: str, interruption: str
) -> None:
    """The probe resolves before the outcome callback runs.

    ``cancel`` and ``interrupt`` raise from the callback and escape; under
    ``diagnostic`` the callback raises an ordinary exception and the host
    handler on ``llmkit.rate_limiting`` raises while logging it, and both are
    contained: the caller sees its own outcome and the loss is counted once.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 100.0)
    configure_rate_limit(max_concurrent=2, breaker=True, adaptive=True)
    breaker = GlobalRateLimiter._get_breaker("openai")
    # Create the shared adaptive state before configuring the recovery case.
    _ = GlobalRateLimiter._get_sync_gate("openai")
    state = GlobalRateLimiter._adaptive_states["openai"]
    seen: list[str] = []
    contained = interruption == "diagnostic"
    error = ValueError("callback failed") if contained else _error(interruption)

    def interrupted(event: BackpressureEvent) -> None:
        seen.append(event.reason)
        if event.reason in {"throttle", "recover"}:
            raise error

    async def probe() -> None:
        breaker._state = CircuitState.OPEN
        breaker._opened_at = 0.0
        provider_error = _error("status-429") if outcome == "throttle" else None
        if contained:
            # Installed after configure_rate_limit: its configure-time INFO
            # record is a plain log call, not a contained diagnostic.
            before = lost_diagnostics()
            with raising_handler("llmkit.rate_limiting") as handler:
                with backpressure_callback(interrupted):
                    if provider_error is None:
                        await _attempt(sync, None)
                    else:
                        with pytest.raises(openai.APIStatusError) as provider_caught:
                            await _attempt(sync, provider_error)
                        assert provider_caught.value is provider_error
            assert lost_diagnostics() == before + 1
            errors = [r for r in handler.records if r.levelno == logging.ERROR]
            assert len(errors) == 1
            assert errors[0].exc_info is not None
            assert errors[0].exc_info[1] is error
        else:
            with backpressure_callback(interrupted), pytest.raises(type(error)) as caught:
                await _attempt(sync, provider_error)
            assert caught.value is error
        assert not breaker._probe_in_flight
        expected = CircuitState.OPEN if outcome == "throttle" else CircuitState.CLOSED
        assert breaker._state is expected
        expected_seen = ["breaker_half_open", "throttle" if outcome == "throttle" else "recover"]
        if contained:
            # Nothing escaped the outcome callback, so the outcome path ran on
            # to report the breaker's own transition.
            expected_seen.append("breaker_open" if outcome == "throttle" else "breaker_closed")
        assert seen == expected_seen

    if outcome == "throttle":
        # A normal request admitted before OPEN occupies the other slot. The
        # recovery probe's throttle is therefore observed at saturation.
        if sync:
            with GlobalRateLimiter.acquire_sync("openai"):
                await probe()
        else:
            async with GlobalRateLimiter.acquire_async("openai"):
                await probe()
        assert state.limit() == 1
    else:
        state._limit = 1
        state._recovery_anchor = 0.0
        await probe()
        assert state.limit() == 2
    assert state._total_in_flight == 0
