"""Offline tests for adaptive per-provider concurrency (AIMD) and its FIFO gate.

These never touch the network. They pin the negative-feedback loop the rate
limiter grows under provider overload:

* the AIMD math on ``_AdaptiveState`` (saturated decrease, refractory cooldown so
  a correlated burst is one halving, floor at 1, wall-clock-paced recovery toward
  the ceiling, never exceeding it) — driven with a frozen clock;
* the throttle classifier (``_is_throttle_signal``) over 429/503/529, the 500/
  timeout exclusions, and the ``InstructorRetryException`` unwrap (the structured
  path);
* the end-to-end wiring through ``acquire_async``: a saturated structured-call
  throttle decreases the shared limit; an unsaturated one does not; ``adaptive=
  False`` pins the limit;
* the FIFO ``_AdaptiveGate`` (no barging, a grown limit admits several waiters,
  a shrunk limit never over-admits) and the gate enforcing a reduced limit;
* the cancellation rpm-token refund and reconfiguration/cross-loop behaviour.

Clock-driven state reads ``rate_limiting._now`` (monkeypatched) so cooldown and
recovery are deterministic and nothing sleeps for real.
"""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from llmkit import configure_rate_limit, rate_limiting
from llmkit.rate_limiting import (
    BackpressureEvent,
    GlobalRateLimiter,
    _AdaptiveGate,
    _AdaptiveState,
    _is_throttle_signal,
    _SyncAdaptiveGate,
)
from tests._support import OkSchema


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def _wrapped(status: int) -> InstructorRetryException:
    """A throttle wrapped as instructor surfaces it on the structured path."""
    return InstructorRetryException(
        _status_error(openai.RateLimitError, status), n_attempts=1, total_usage=0
    )


def _validation_error() -> ValidationError:
    try:
        _ = OkSchema.model_validate({"ok": "nope"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")  # pragma: no cover


class _Probe:
    """Records peak concurrency observed through ``acquire_async`` for a key."""

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0

    async def run(self, key: str, hold: asyncio.Event) -> None:
        async with GlobalRateLimiter.acquire_async(key):
            self.current += 1
            self.peak = max(self.peak, self.current)
            try:
                _ = await hold.wait()
            finally:
                self.current -= 1


# --- _is_throttle_signal -------------------------------------------------


def test_throttle_signal_classifies_status_codes() -> None:
    """429/503/529 are throttle signals; 500, timeouts, and validation are not."""
    assert _is_throttle_signal(_status_error(openai.RateLimitError, 429))
    assert _is_throttle_signal(_status_error(openai.InternalServerError, 503))
    assert _is_throttle_signal(_status_error(openai.InternalServerError, 529))
    assert not _is_throttle_signal(_status_error(openai.InternalServerError, 500))
    assert not _is_throttle_signal(TimeoutError("boom"))
    assert not _is_throttle_signal(_validation_error())


def test_throttle_signal_unwraps_instructor_wrapper() -> None:
    """A structured-call throttle wrapped in ``InstructorRetryException`` is seen."""
    assert _is_throttle_signal(_wrapped(429))
    # A wrapped permanent 4xx is not a throttle.
    assert not _is_throttle_signal(
        InstructorRetryException(
            _status_error(openai.AuthenticationError, 401), n_attempts=1, total_usage=0
        )
    )


# --- _AdaptiveState math (frozen clock) ----------------------------------


def _hold(state: _AdaptiveState, n: int) -> None:
    """Prime the state's provider-wide in-flight aggregate to *n* held slots.

    ``on_throttle`` now judges saturation on that aggregate, so a unit test that
    wants a *saturated* throttle holds ``ceiling`` slots up front — ``ceiling``
    stays >= every post-halving limit, so the same priming keeps every throttle in
    a sequence saturated. An unprimed state (aggregate 0) is the unsaturated case.
    """
    for _ in range(n):
        state.note_acquire()


def test_saturated_throttle_halves_and_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saturated throttle halves the limit and returns a throttle event."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    state = _AdaptiveState("openai", ceiling=8)
    _hold(state, 8)  # aggregate at the limit → saturated
    assert state.limit() == 8
    event = state.on_throttle()
    assert state.limit() == 4
    assert event == BackpressureEvent("openai", 8, 4, "throttle")


def test_unsaturated_throttle_does_not_decrease(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throttle received while below the limit changes nothing (not our fault)."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    state = _AdaptiveState("openai", ceiling=8)
    # Unprimed: the aggregate (0) is below the limit (8), so the throttle is noise.
    assert state.on_throttle() is None
    assert state.limit() == 8


def test_burst_within_cooldown_is_a_single_decrease(monkeypatch: pytest.MonkeyPatch) -> None:
    """Correlated throttles inside the refractory window cause one halving."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiting, "_now", lambda: clock["t"])
    state = _AdaptiveState("openai", ceiling=8)
    _hold(state, 8)  # aggregate stays >= every post-halving limit

    assert state.on_throttle() is not None  # 8 -> 4
    assert state.limit() == 4
    # A burst of further throttles within the cooldown is suppressed.
    assert state.on_throttle() is None
    assert state.on_throttle() is None
    assert state.limit() == 4
    # Past the cooldown, the next throttle halves again.
    clock["t"] += rate_limiting._AIMD_DECREASE_COOLDOWN
    assert state.on_throttle() is not None  # 4 -> 2
    assert state.limit() == 2


def test_decrease_floors_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated throttles drive the limit to 1 and no lower."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiting, "_now", lambda: clock["t"])
    state = _AdaptiveState("openai", ceiling=8)
    _hold(state, 8)  # aggregate stays >= every post-halving limit
    for expected in (4, 2, 1):
        assert state.on_throttle() is not None
        assert state.limit() == expected
        clock["t"] += rate_limiting._AIMD_DECREASE_COOLDOWN
    # At the floor, a further throttle is a no-op (max(1, int(1*0.5)) == 1).
    assert state.on_throttle() is None
    assert state.limit() == 1


def test_recovery_is_wall_clock_paced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery climbs one step per elapsed interval, catches up after a gap, and
    never exceeds the ceiling — and a success with no elapsed time is a no-op."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiting, "_now", lambda: clock["t"])
    interval = rate_limiting._AIMD_RECOVERY_INTERVAL
    state = _AdaptiveState("openai", ceiling=8)
    _hold(state, 8)  # saturate the aggregate for the drive-to-floor throttles
    # Drive to the floor.
    for _ in range(3):
        _ = state.on_throttle()
        clock["t"] += rate_limiting._AIMD_DECREASE_COOLDOWN
    assert state.limit() == 1

    # A success before an interval elapses does not recover.
    assert state.on_success() is None
    assert state.limit() == 1

    # One interval -> +1.
    clock["t"] += interval
    event = state.on_success()
    assert state.limit() == 2
    assert event == BackpressureEvent("openai", 1, 2, "recover")

    # A long gap catches up several steps at once (not one-per-success).
    clock["t"] += 3 * interval
    _ = state.on_success()
    assert state.limit() == 5  # 2 + 3

    # Far past the ceiling: clamps at the ceiling and then no-ops.
    clock["t"] += 100 * interval
    _ = state.on_success()
    assert state.limit() == 8
    assert state.on_success() is None
    assert state.limit() == 8


def test_throttle_restarts_the_recovery_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decrease resets the recovery anchor, so recovery is measured from it."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(rate_limiting, "_now", lambda: clock["t"])
    state = _AdaptiveState("openai", ceiling=8)
    _hold(state, 8)  # saturate the aggregate
    _ = state.on_throttle()  # 8 -> 4 at t=1000
    clock["t"] += rate_limiting._AIMD_RECOVERY_INTERVAL - 0.5  # just shy of a step
    assert state.on_success() is None  # not yet a full interval since the decrease
    assert state.limit() == 4


# --- saturation judged on the provider-wide aggregate --------------------


async def test_saturation_is_judged_on_the_aggregate_across_async_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two async gates on one loop share a state; a throttle on one fires a
    decrease because the *aggregate* (both gates) is at the limit, even though
    neither gate is alone at it. Fails pre-fix (per-gate saturation → no
    decrease)."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    state = _AdaptiveState("openai", ceiling=4)
    gate_a = _AdaptiveGate(state)
    gate_b = _AdaptiveGate(state)
    # Each gate holds 2 (fast path, never awaits): each is below the limit of 4
    # on its own, but together the aggregate is 4 == the limit.
    for _ in range(2):
        await gate_a.acquire()
        await gate_b.acquire()
    assert gate_a._in_flight == 2
    assert gate_b._in_flight == 2
    event = gate_b.on_throttle()
    assert event == BackpressureEvent("openai", 4, 2, "throttle")
    assert state.limit() == 2


async def test_saturation_counts_the_sync_gate_in_the_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aggregate spans populations: an async gate holding 2 and the sync gate
    holding 2 saturate a ceiling-4 state, so a throttle on the *async* gate
    decreases 4 -> 2. Fails pre-fix."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    state = _AdaptiveState("openai", ceiling=4)
    async_gate = _AdaptiveGate(state)
    sync_gate = _SyncAdaptiveGate(state)
    for _ in range(2):
        await async_gate.acquire()
        sync_gate.acquire()
    assert async_gate._in_flight == 2
    assert sync_gate._in_flight == 2
    event = async_gate.on_throttle()
    assert event == BackpressureEvent("openai", 4, 2, "throttle")
    assert state.limit() == 2


async def test_aggregate_below_limit_is_still_unsaturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the shared limit a throttle is noise; and ``note_release`` pairs
    correctly, so a held-then-released slot never leaves the aggregate inflated."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    state = _AdaptiveState("openai", ceiling=4)
    gate_a = _AdaptiveGate(state)
    gate_b = _AdaptiveGate(state)
    await gate_a.acquire()
    await gate_b.acquire()  # aggregate 2 < limit 4
    assert gate_a.on_throttle() is None  # noise, and _last_decrease untouched
    assert state.limit() == 4
    # Release everything: correct note_release drains the aggregate back to 0.
    gate_a.release()
    gate_b.release()
    # Re-prime one gate to the ceiling; a throttle now *does* fire — proving the
    # earlier holds did not leave the aggregate inflated (a leak would have kept
    # it saturated).
    for _ in range(4):
        await gate_a.acquire()
    assert gate_a.on_throttle() is not None
    assert state.limit() == 2


# --- end-to-end wiring through acquire_async ------------------------------


async def test_saturated_structured_throttle_decreases_shared_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structured-call throttle (wrapped) received at the limit halves it —
    proving the unwrap and the saturation gate on the real acquire path."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2)
    key = "google"
    holder_in = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            holder_in.set()
            _ = await release_holder.wait()

    htask = asyncio.create_task(holder())
    _ = await holder_in.wait()  # in_flight == 1

    with pytest.raises(InstructorRetryException):
        async with GlobalRateLimiter.acquire_async(key):  # in_flight == 2 == limit
            raise _wrapped(429)

    assert GlobalRateLimiter._adaptive_states[key].limit() == 1  # 2 -> 1

    release_holder.set()
    _ = await htask


async def test_unsaturated_throttle_via_acquire_does_not_decrease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throttle with only one call in flight (cap 8) does not move the limit."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=8)
    key = "google"
    with pytest.raises(InstructorRetryException):
        async with GlobalRateLimiter.acquire_async(key):  # in_flight 1 < limit 8
            raise _wrapped(503)
    assert GlobalRateLimiter._adaptive_states[key].limit() == 8


async def test_adaptive_false_pins_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``adaptive=False`` a saturated throttle leaves the limit at the cap."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=2, adaptive=False)
    key = "google"
    holder_in = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            holder_in.set()
            _ = await release_holder.wait()

    htask = asyncio.create_task(holder())
    _ = await holder_in.wait()

    with pytest.raises(InstructorRetryException):
        async with GlobalRateLimiter.acquire_async(key):
            raise _wrapped(429)

    assert GlobalRateLimiter._adaptive_states[key].limit() == 2  # unchanged

    release_holder.set()
    _ = await htask


async def test_success_does_not_grow_above_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A never-throttled provider stays exactly at the ceiling (no growth above)."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=4)
    key = "google"
    for _ in range(10):
        async with GlobalRateLimiter.acquire_async(key):
            pass
    assert GlobalRateLimiter._adaptive_states[key].limit() == 4


async def test_gate_enforces_a_reduced_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the shared limit is reduced, the gate admits only that many at once."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=4)
    key = "google"
    gate = GlobalRateLimiter._get_async_gate(key)
    gate._state._limit = 1  # as if AIMD had driven it down

    probe = _Probe()
    hold = asyncio.Event()
    tasks = [asyncio.create_task(probe.run(key, hold)) for _ in range(4)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 1  # only the reduced limit is admitted
    assert probe.peak == 1

    hold.set()
    _ = await asyncio.gather(*tasks)


# --- the FIFO gate -------------------------------------------------------


async def test_gate_serves_waiters_fifo() -> None:
    """Queued waiters are admitted strictly in arrival order (no barging)."""
    state = _AdaptiveState("k", ceiling=1)
    gate = _AdaptiveGate(state)
    await gate.acquire()  # holder takes the only slot
    order: list[int] = []

    async def waiter(i: int) -> None:
        await gate.acquire()
        order.append(i)
        gate.release()

    tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert order == []  # all three blocked behind the holder

    gate.release()  # frees the slot; admission cascades through the queue in order
    _ = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert order == [0, 1, 2]


async def test_release_after_grow_admits_multiple_waiters() -> None:
    """A grown limit admits several waiters at once on the next release (not one)."""
    state = _AdaptiveState("k", ceiling=3)
    state._limit = 1
    gate = _AdaptiveGate(state)
    await gate.acquire()  # holder, in_flight == 1 == limit
    admitted: list[int] = []

    async def waiter(i: int) -> None:
        await gate.acquire()
        admitted.append(i)

    tasks = [asyncio.create_task(waiter(i)) for i in range(2)]
    for _ in range(10):
        await asyncio.sleep(0)
    assert admitted == []  # both blocked at limit 1

    state._limit = 3  # grow (as a recovery step would)
    gate.release()  # one release, but the new limit has room for both waiters
    _ = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert sorted(admitted) == [0, 1]


async def test_shrink_never_admits_beyond_new_limit() -> None:
    """A shrunk limit holds back admission until in-flight drains below it."""
    state = _AdaptiveState("k", ceiling=8)
    state._limit = 3
    gate = _AdaptiveGate(state)
    await gate.acquire()
    await gate.acquire()  # in_flight == 2, limit 3

    state._limit = 1  # shrink below the current in-flight
    admitted: list[int] = []

    async def waiter() -> None:
        await gate.acquire()
        admitted.append(1)

    task = asyncio.create_task(waiter())
    for _ in range(10):
        await asyncio.sleep(0)
    assert admitted == []  # in_flight 2 > limit 1

    gate.release()  # in_flight 1, still not below limit 1
    for _ in range(10):
        await asyncio.sleep(0)
    assert admitted == []

    gate.release()  # in_flight 0 < limit 1 -> admit
    _ = await asyncio.wait_for(task, timeout=1.0)
    assert admitted == [1]
    gate.release()


async def test_cancelled_waiter_does_not_consume_a_slot() -> None:
    """A waiter cancelled while queued frees no phantom slot: the next waiter runs."""
    state = _AdaptiveState("k", ceiling=1)
    gate = _AdaptiveGate(state)
    await gate.acquire()  # holder

    first_started = asyncio.Event()

    async def cancellable() -> None:
        first_started.set()
        await gate.acquire()

    cancelled = asyncio.create_task(cancellable())
    _ = await first_started.wait()
    for _ in range(5):
        await asyncio.sleep(0)

    admitted = asyncio.Event()

    async def survivor() -> None:
        await gate.acquire()
        admitted.set()

    stask = asyncio.create_task(survivor())
    for _ in range(5):
        await asyncio.sleep(0)

    _ = cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    gate.release()  # holder releases its slot
    _ = await asyncio.wait_for(stask, timeout=1.0)
    assert admitted.is_set()  # survivor got the slot; the cancelled waiter ate nothing
    gate.release()


# --- cancellation refund of the RPM token --------------------------------


async def test_cancel_during_acquire_refunds_rpm_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call cancelled after its RPM token is deducted but before it holds a slot
    refunds the token — so a systematically-cancelled workload keeps its RPM."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)  # freeze: no refill
    configure_rate_limit(max_concurrent=8, rpm=480, tpm=6_000)
    key = "openai"

    # Exhaust the TPM budget so the *next* acquire debits RPM then blocks on TPM.
    async with GlobalRateLimiter.acquire_async(key) as slot:
        slot.record_tokens(9_000)
    rpm_bucket = GlobalRateLimiter._rpm_buckets[key]
    assert rpm_bucket._level == 7.0  # capacity min(8, 480) == 8, minus the one call

    async def blocked() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            pass

    task = asyncio.create_task(blocked())
    for _ in range(10):
        await asyncio.sleep(0)
    assert rpm_bucket._level == 6.0  # RPM debited; now parked on the TPM gate

    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rpm_bucket._level == 7.0  # refunded, not silently lost


# --- reconfiguration & cross-loop ----------------------------------------


async def test_configure_clears_adaptive_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """``configure`` clears the adaptive-state registry, so a reduced limit does
    not leak into the next configuration."""
    monkeypatch.setattr(rate_limiting, "_now", lambda: 1_000.0)
    configure_rate_limit(max_concurrent=4)
    key = "google"
    gate = GlobalRateLimiter._get_async_gate(key)
    gate._state._limit = 1
    assert GlobalRateLimiter._adaptive_states  # populated

    configure_rate_limit(max_concurrent=4)
    assert GlobalRateLimiter._adaptive_states == {}


def test_adaptive_state_is_shared_across_event_loops() -> None:
    """The per-provider adaptive state is one object across loops (only the gate
    is per-loop), so a throttle on one loop lowers the limit every loop sees."""
    configure_rate_limit(max_concurrent=4)
    key = "google"
    states: list[_AdaptiveState] = []

    async def grab() -> None:
        states.append(GlobalRateLimiter._get_async_gate(key)._state)

    asyncio.run(grab())  # loop A
    asyncio.run(grab())  # loop B (a different loop)
    assert states[0] is states[1]


def test_async_gate_not_reused_across_event_loops() -> None:
    """A saturated provider on one loop must not crash the next loop.

    The gate's waiter futures bind to the loop they are created on, so the
    process-global registry keys the gate per (provider, loop); a call on a second
    loop must get its own gate, not reuse the first loop's (now-closed) one. Plain
    ``def`` so it drives two separate loops via ``asyncio.run``, and deliberately
    does not ``configure`` between them (that would clear the registry and mask the
    bug). Replaces the old per-loop *semaphore* test for the new gate.
    """
    GlobalRateLimiter.configure(max_concurrent=1)

    async def contend(key: str) -> None:
        held = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with GlobalRateLimiter.acquire_async(key):
                held.set()
                _ = await release.wait()

        async def waiter() -> None:
            _ = await held.wait()
            async with GlobalRateLimiter.acquire_async(key):  # blocks: cap 1, held
                pass

        h = asyncio.create_task(holder())
        w = asyncio.create_task(waiter())
        _ = await held.wait()
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        _ = await asyncio.gather(h, w)

    asyncio.run(contend("openai"))  # loop A binds its gate's futures
    asyncio.run(contend("openai"))  # loop B: must not reuse loop A's gate
