"""Offline tests for the backpressure observability callback.

A host installs a :class:`~llmkit.BackpressureEvent` callback via
:func:`~llmkit.backpressure_callback`; the adaptive limiter fires it whenever a
provider's concurrency limit moves (a throttle-driven decrease or a time-driven
recovery). These pin:

* a decrease emits ``(provider, old, new, "throttle")`` and a recovery emits
  ``"recover"`` with the right payload;
* a ``None`` callback is a no-op and a raising callback is swallowed (never breaks
  a call);
* the contextvar is restored on scope exit (including nested scopes); and
* the callback **propagates across the ``run_sync`` boundary** — installed in the
  calling thread, it fires for limit changes observed on the persistent loop.

Driven with a frozen clock so decrease/recovery are deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import (
    BackpressureCallback,
    BackpressureEvent,
    backpressure_callback,
    configure_rate_limit,
    rate_limiting,
)
from llmkit.rate_limiting import GlobalRateLimiter, _backpressure_callback
from llmkit.sync import run_sync


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
    try:
        yield
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)


def _wrapped(status: int) -> InstructorRetryException:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    inner = openai.RateLimitError(
        "boom", response=httpx.Response(status, request=request), body=None
    )
    return InstructorRetryException(inner, n_attempts=1, total_usage=0)


async def _saturated_throttle(key: str) -> None:
    """Drive one saturated throttle (cap must be 2) so the limit halves 2 -> 1.

    Holds one slot in a background task, then throttles a second call while both
    are in flight (saturated), so :meth:`_AdaptiveState.on_throttle` decreases.
    """
    holder_in = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            holder_in.set()
            _ = await release_holder.wait()

    htask = asyncio.create_task(holder())
    _ = await holder_in.wait()
    try:
        async with GlobalRateLimiter.acquire_async(key):
            raise _wrapped(429)
    except InstructorRetryException:
        pass
    release_holder.set()
    _ = await htask


# --- events fire with the right payload ----------------------------------


async def test_throttle_event_fires_with_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saturated throttle emits a ``throttle`` event carrying old/new limits."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2)
    events: list[BackpressureEvent] = []
    with backpressure_callback(events.append):
        await _saturated_throttle("google")
    assert events == [BackpressureEvent("google", 2, 1, "throttle")]


async def test_recover_event_fires_with_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A success after a recovery interval emits a ``recover`` event."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiting, "_now", lambda: clock["t"])
    configure_rate_limit(max_concurrent=4)
    key = "google"
    gate = GlobalRateLimiter._get_async_gate(key)
    gate._state._limit = 1  # as if AIMD had driven it down
    gate._state._recovery_anchor = clock["t"]
    clock["t"] += rate_limiting._AIMD_RECOVERY_INTERVAL  # one interval elapsed

    events: list[BackpressureEvent] = []
    with backpressure_callback(events.append):
        async with GlobalRateLimiter.acquire_async(key):
            pass

    assert events == [BackpressureEvent("google", 1, 2, "recover")]


# --- robustness: None and raising callbacks ------------------------------


async def test_none_callback_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """No callback installed: the decrease still happens and nothing raises."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2)
    with backpressure_callback(None):
        await _saturated_throttle("google")
    assert GlobalRateLimiter._adaptive_states["google"].limit() == 1


async def test_raising_callback_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A callback that raises is logged, never allowed to break the call/limit."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2)

    def boom(_event: BackpressureEvent) -> None:
        raise RuntimeError("callback boom")

    with backpressure_callback(boom), caplog.at_level(logging.ERROR):
        await _saturated_throttle("google")

    assert GlobalRateLimiter._adaptive_states["google"].limit() == 1  # still decreased
    assert any("backpressure callback raised" in r.message for r in caplog.records)


# --- the contextvar is restored on exit ----------------------------------


def test_callback_resets_on_scope_exit() -> None:
    """The installer restores the previous callback on exit, including nesting."""
    outer: list[BackpressureEvent] = []
    inner: list[BackpressureEvent] = []
    cb_outer: BackpressureCallback = outer.append  # bind once (a fresh ``.append`` each access)
    cb_inner: BackpressureCallback = inner.append
    assert _backpressure_callback.get() is None
    with backpressure_callback(cb_outer):
        assert _backpressure_callback.get() is cb_outer
        with backpressure_callback(cb_inner):
            assert _backpressure_callback.get() is cb_inner
        assert _backpressure_callback.get() is cb_outer
    assert _backpressure_callback.get() is None


# --- propagation across run_sync -----------------------------------------


def test_event_propagates_across_run_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callback installed in the calling thread fires for a limit change observed
    on the persistent sync loop — the bridge copies the caller's context."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2)
    events: list[BackpressureEvent] = []

    with backpressure_callback(events.append):
        run_sync(_saturated_throttle("google"))

    assert events == [BackpressureEvent("google", 2, 1, "throttle")]
