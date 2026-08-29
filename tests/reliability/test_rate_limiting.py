"""Offline tests for the global, per-provider rate limiter.

These tests never touch the network and never read a provider credential.
They pin the behaviour the rate limiter promises:

* concurrency limiting is **on by default** with a per-provider cap of 8
  (zero config),
* each provider gets an **independent** budget (keyed by provider name,
  matched case-insensitively — ``"openai"`` and ``"OpenAI"`` are one budget),
* reconfiguring the cap / disabling takes effect for subsequent acquires,
* a :meth:`configure` swap never strands in-flight callers on the old
  semaphore (they release back onto their snapshot),
* the call layer accounts a slot under the *effective* provider's name —
  the same value logging records, and
* the opt-in **RPM** (requests/min) and **TPM** (tokens/min) token buckets
  are off by default, gate per provider when configured, and — for TPM —
  debit the bucket by each call's measured ``usage.total_tokens``.

The RPM/TPM tests freeze the monotonic clock (monkeypatching
``rate_limiting._now``) so token-bucket refill is deterministic and nothing
sleeps for real.

The limiter is a process-global; the shared ``reset_rate_limiter`` autouse
fixture in ``conftest.py`` restores a known default state around every test so
nothing leaks.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import override
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import CircuitOpenError, _litellm, structured_output
from llmkit.rate_limiting import (
    GlobalRateLimiter,
    RateLimitSlot,
    _tuning,
    configure_rate_limit,
    get_rate_limit_config,
    rate_limit_acquire_async,
    rate_limit_acquire_sync,
)
from llmkit.rate_limiting._adaptive import AdaptiveState, SyncAdaptiveGate
from llmkit.rate_limiting._breaker import CircuitState
from llmkit.rate_limiting._buckets import RateBucket
from llmkit.rate_limiting._tuning import BREAKER_MIN_SAMPLES
from tests._support import quiet_logging


class _ConcurrencyProbe:
    """Instrumented coroutine that records observed peak concurrency per key."""

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def run(self, key: str, hold: asyncio.Event) -> None:
        """Acquire a slot for ``key``, record peak concurrency, hold until set."""
        async with GlobalRateLimiter.acquire_async(key):
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                _ = await hold.wait()
            finally:
                async with self._lock:
                    self.current -= 1


async def test_default_on_caps_at_eight_with_zero_config() -> None:
    """With no configuration at all, one key caps concurrency at 8."""
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(11)]

    # Let the scheduler run; exactly 8 should get in, the rest block.
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8
    assert probe.peak == 8

    hold.set()
    _ = await asyncio.gather(*tasks)
    assert probe.peak == 8


async def test_providers_have_independent_budgets() -> None:
    """Saturating one key does not block a different key; each caps at 8."""
    probe = _ConcurrencyProbe()
    openai_hold = asyncio.Event()
    ollama_hold = asyncio.Event()

    # Saturate openai with 8 holders + 1 waiter.
    openai_tasks = [asyncio.create_task(probe.run("openai", openai_hold)) for _ in range(9)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8  # openai is full

    # An ollama acquire must proceed immediately despite openai being full.
    ollama_proceeded = asyncio.Event()

    async def ollama_call() -> None:
        async with GlobalRateLimiter.acquire_async("ollama"):
            ollama_proceeded.set()
            _ = await ollama_hold.wait()

    ollama_task = asyncio.create_task(ollama_call())
    for _ in range(20):
        await asyncio.sleep(0)
    assert ollama_proceeded.is_set(), "ollama was blocked by openai's full budget"

    # Independent caps: openai still holds exactly 8 (its ninth is queued).
    assert probe.current == 8

    openai_hold.set()
    ollama_hold.set()
    _ = await asyncio.gather(*openai_tasks, ollama_task)


async def test_reconfigure_raises_cap_for_new_acquires() -> None:
    """configure(max_concurrent=5) lets 5 run concurrently on one key."""
    configure_rate_limit(max_concurrent=5)
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(5)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 5
    assert probe.peak == 5

    hold.set()
    _ = await asyncio.gather(*tasks)


async def test_disabled_is_unbounded_no_op() -> None:
    """configure(enabled=False) makes acquire a no-op: no cap at all."""
    configure_rate_limit(enabled=False)
    assert GlobalRateLimiter.is_enabled() is False
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(7)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 7  # all in flight, unbounded
    assert probe.peak == 7

    hold.set()
    _ = await asyncio.gather(*tasks)


async def test_inflight_caller_not_stranded_by_configure_swap() -> None:
    """A held slot releases cleanly even after configure() clears the registry.

    The acquirer snapshots its semaphore at acquire time, so the swap that
    configure() performs (clearing the dict) cannot strand it; the block
    exits without deadlock and a fresh acquire works afterward.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async("openai"):
            entered.set()
            _ = await release.wait()
        finished.set()

    task = asyncio.create_task(holder())
    _ = await entered.wait()

    # Swap/clear the registry while the slot is held.
    configure_rate_limit(max_concurrent=3)

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert finished.is_set()

    # A fresh acquire after the swap still works (no deadlock, new cap in play).
    async with GlobalRateLimiter.acquire_async("openai"):
        pass


async def test_call_layer_accounts_under_effective_provider_name() -> None:
    """acompletion_text acquires under the running provider's ``.name``.

    Driven fully offline: the rate-limit acquire is monkeypatched to record
    the key, and ``litellm.acompletion`` is faked so no network call happens.
    A per-call ``provider=`` override must drive the recorded key.
    """
    recorded: list[str] = []

    @contextlib.asynccontextmanager
    async def _record(provider_key: str) -> AsyncGenerator[RateLimitSlot]:
        recorded.append(provider_key)
        yield RateLimitSlot()

    provider = MagicMock()
    provider.name = "ollama"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", "api_base": None})
    provider.litellm_model = MagicMock(return_value="ollama/fake")
    provider.reasoning_effort = None

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi"))],
            _hidden_params={},
        )

    with (
        patch.object(GlobalRateLimiter, "acquire_async", _record),
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        text, _cost = await _litellm.acompletion_text(
            "hi",
            temperature=0.0,
            model=None,
            provider=provider,
        )

    assert text == "hi"
    assert recorded == ["ollama"]


def test_get_rate_limit_config_reports_effective_values() -> None:
    """``get_rate_limit_config`` reflects the default and any reconfiguration,
    so a host can read its effective limits without touching ``_max_concurrent``."""
    default = get_rate_limit_config()
    assert default.enabled is True
    assert default.max_concurrent == 8
    assert default.rpm is None  # opt-in dimensions off by default
    assert default.tpm is None
    assert default.adaptive is True  # adaptive concurrency on by default
    assert default.breaker is False  # circuit breaker opt-in, off by default

    configure_rate_limit(max_concurrent=3, enabled=False, rpm=120, tpm=90_000)
    updated = get_rate_limit_config()
    assert updated.enabled is False
    assert updated.max_concurrent == 3
    assert updated.rpm == 120
    assert updated.tpm == 90_000
    assert updated.adaptive is True  # unset -> stays on
    assert updated.breaker is False  # unset -> stays off

    configure_rate_limit(adaptive=False)
    assert get_rate_limit_config().adaptive is False  # explicit opt-out reflected

    configure_rate_limit(breaker=True)
    assert get_rate_limit_config().breaker is True  # explicit opt-in reflected
    assert get_rate_limit_config().adaptive is True  # breaker opt-in doesn't disturb adaptive


class _AsyncFunctionProbe:
    """``_ConcurrencyProbe`` analogue that joins via ``rate_limit_acquire_async``.

    Exercises the module-level public function rather than
    ``GlobalRateLimiter.acquire_async`` directly, so the test pins the function
    path a host actually uses to join the global limit by hand.
    """

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def run(self, key: str, hold: asyncio.Event) -> None:
        """Join via the public async function, record peak, hold until set."""
        async with rate_limit_acquire_async(key):
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                _ = await hold.wait()
            finally:
                async with self._lock:
                    self.current -= 1


async def test_rate_limit_acquire_async_respects_configured_cap() -> None:
    """The public async function bounds concurrency at the configured cap."""
    probe = _AsyncFunctionProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(11)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8  # default per-provider cap
    assert probe.peak == 8

    hold.set()
    _ = await asyncio.gather(*tasks)
    assert probe.peak == 8


async def test_rate_limit_acquire_async_acquires_and_releases() -> None:
    """A held slot releases on block exit, so the next acquire proceeds.

    Saturate the cap by hand through the public function; a further acquire
    blocks until one holder exits, then runs.
    """
    configure_rate_limit(max_concurrent=1)
    entered_second = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with rate_limit_acquire_async("openai"):
            _ = await release_first.wait()

    async def second() -> None:
        async with rate_limit_acquire_async("openai"):
            entered_second.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())

    for _ in range(20):
        await asyncio.sleep(0)
    # The single slot is held by first(); second() must be blocked.
    assert not entered_second.is_set()

    release_first.set()  # free the slot
    _ = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1.0)
    assert entered_second.is_set()


async def test_rate_limit_acquire_async_disabled_bypass() -> None:
    """With limiting disabled, the public async function is an unbounded no-op."""
    configure_rate_limit(enabled=False)
    probe = _AsyncFunctionProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(7)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 7  # all in flight, unbounded
    assert probe.peak == 7

    hold.set()
    _ = await asyncio.gather(*tasks)


def test_rate_limit_acquire_sync_respects_configured_cap() -> None:
    """The public sync function bounds concurrency at the configured cap.

    Saturate a cap of 1 from one thread, then assert a second thread cannot
    enter until the first releases — proving the sync function holds the slot.
    """
    configure_rate_limit(max_concurrent=1)
    entered_second = threading.Event()
    release_first = threading.Event()
    first_holds = threading.Event()

    def first() -> None:
        with rate_limit_acquire_sync("openai"):
            first_holds.set()
            _ = release_first.wait(timeout=1.0)

    def second() -> None:
        with rate_limit_acquire_sync("openai"):
            entered_second.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_holds.wait(timeout=1.0)

    second_thread.start()
    # The single slot is held; second() must block.
    assert not entered_second.wait(timeout=0.1)

    release_first.set()  # free the slot
    second_thread.join(timeout=1.0)
    first_thread.join(timeout=1.0)
    assert entered_second.is_set()


def test_rate_limit_acquire_sync_disabled_bypass() -> None:
    """With limiting disabled, the public sync function is an unbounded no-op.

    A cap of 1 would otherwise serialise; disabled, two threads hold the
    "slot" simultaneously.
    """
    configure_rate_limit(max_concurrent=1, enabled=False)
    # If limiting were active (cap 1) the barrier would never reach 2 and time
    # out; disabled, both threads sit inside the acquire at once and trip it.
    both_in = threading.Barrier(2, timeout=1.0)
    release = threading.Event()
    reached_barrier = threading.Event()

    def worker() -> None:
        with rate_limit_acquire_sync("openai"):
            _ = both_in.wait()  # both threads must be inside at once
            reached_barrier.set()
            _ = release.wait(timeout=1.0)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert reached_barrier.wait(timeout=1.0), "disabled sync acquire serialised callers"

    release.set()
    for thread in threads:
        thread.join(timeout=1.0)


# --- sync/async parity: shared breaker + AIMD state and interrupt hardening ---
#
# The sync acquire path now shares the per-provider circuit breaker and AIMD
# ``AdaptiveState`` with the async path (only the in-flight *count* is its own
# population), and hardens the RPM-refund / permit-release the old fixed-semaphore
# path skipped. These tests are all offline and assert state transitions, not
# timing, to stay off the wall clock.


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def _wrapped_throttle(status: int) -> InstructorRetryException:
    """A throttle wrapped as instructor surfaces it on the structured path."""
    return InstructorRetryException(
        _status_error(openai.RateLimitError, status), n_attempts=1, total_usage=0
    )


def test_sync_acquire_raises_circuit_open_when_breaker_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the breaker OPEN, a sync acquire fast-fails with ``CircuitOpenError``
    before any gate — holding no slot — exactly like the async path.

    The breaker is driven OPEN through its own ``on_record`` API (a full window of
    throttles); the frozen clock keeps it inside the cooldown so it rejects.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # frozen: stays OPEN
    configure_rate_limit(max_concurrent=8, breaker=True)
    key = "openai"
    breaker = GlobalRateLimiter._get_breaker(key)
    for _ in range(BREAKER_MIN_SAMPLES):  # a full window of throttles opens it
        _ = breaker.on_record(throttled=True)
    assert breaker._state is CircuitState.OPEN

    # ``acquire_sync`` fetches the gate *before* consulting the breaker, so
    # pre-create it and assert the rejected call left that very object untouched.
    # (Fetching the gate only after the acquire builds a fresh, trivially-empty
    # gate that could not detect a slot wrongly taken before the breaker check.)
    gate = GlobalRateLimiter._get_sync_gate(key)
    assert gate._in_flight == 0

    with pytest.raises(CircuitOpenError) as excinfo:
        with rate_limit_acquire_sync(key):
            raise AssertionError("body must not run while the breaker is OPEN")

    assert excinfo.value.provider == key
    # No concurrency slot was taken by the rejected call — same gate, still empty.
    assert GlobalRateLimiter._get_sync_gate(key) is gate
    assert gate._in_flight == 0


def test_sync_throttle_while_saturated_halves_shared_limit_async_sees_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing parity test: a *sync* throttle-while-saturated halves the
    shared per-provider AIMD limit, and a subsequent *async* acquire enforces the
    halved cap — proving the state is genuinely SHARED, not mirrored per path.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # frozen: no recovery
    configure_rate_limit(max_concurrent=2)
    key = "google"
    holder_in = threading.Event()
    release_holder = threading.Event()

    def holder() -> None:
        with rate_limit_acquire_sync(key):
            holder_in.set()
            _ = release_holder.wait(timeout=2.0)

    hthread = threading.Thread(target=holder)
    hthread.start()
    assert holder_in.wait(timeout=1.0)  # sync gate in_flight == 1

    # A second sync acquire saturates the gate (in_flight == 2 == limit); its body
    # raises a (wrapped, structured-path) throttle, so the saturation gate fires.
    with pytest.raises(InstructorRetryException):
        with rate_limit_acquire_sync(key):
            raise _wrapped_throttle(429)

    # The SHARED per-provider AIMD limit dropped 2 -> 1 from the sync-side throttle.
    assert GlobalRateLimiter._adaptive_states[key].limit() == 1

    release_holder.set()
    hthread.join(timeout=1.0)
    assert GlobalRateLimiter._get_sync_gate(key)._in_flight == 0

    # The ASYNC path now enforces the shared, lowered cap of 1: three async
    # acquirers on a fresh loop, only one in flight at a time.
    async def _async_peak_under_shared_limit() -> int:
        current = 0
        peak = 0
        hold = asyncio.Event()

        async def run() -> None:
            nonlocal current, peak
            async with GlobalRateLimiter.acquire_async(key):
                current += 1
                peak = max(peak, current)
                try:
                    _ = await hold.wait()
                finally:
                    current -= 1

        tasks = [asyncio.create_task(run()) for _ in range(3)]
        for _ in range(30):
            await asyncio.sleep(0)
        observed = peak
        hold.set()
        _ = await asyncio.gather(*tasks)
        return observed

    assert asyncio.run(_async_peak_under_shared_limit()) == 1


def test_cross_population_saturated_throttle_decreases_shared_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing cross-population regression: a background thread holds one
    *sync* slot while an *async* call throttles. The async gate's own count is
    1 < 2, but the provider-wide aggregate (sync 1 + async 1) is 2 == the limit,
    so the shared AIMD limit must drop 2 -> 1. Fails pre-fix, where saturation was
    judged per gate and the async gate's local 1 < 2 classified the 429 as noise.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # frozen: no recovery
    configure_rate_limit(max_concurrent=2)
    key = "google"
    holder_in = threading.Event()
    release_holder = threading.Event()

    def holder() -> None:
        with rate_limit_acquire_sync(key):
            holder_in.set()
            _ = release_holder.wait(timeout=2.0)

    hthread = threading.Thread(target=holder)
    hthread.start()
    assert holder_in.wait(timeout=1.0)  # sync gate in_flight == 1, aggregate == 1

    # An async acquire whose body raises a wrapped 429. Its async gate holds only
    # 1 (< the limit of 2), but the aggregate is 2 == the limit, so the saturation
    # judgment fires and halves the SHARED per-provider limit.
    async def _throttled_async_call() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            raise _wrapped_throttle(429)

    with pytest.raises(InstructorRetryException):
        asyncio.run(_throttled_async_call())

    assert GlobalRateLimiter._adaptive_states[key].limit() == 1

    release_holder.set()
    hthread.join(timeout=1.0)
    assert GlobalRateLimiter._get_sync_gate(key)._in_flight == 0


def test_failed_sync_acquire_refunds_rpm_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sync acquire that fails *after* the RPM token is deducted but *before* the
    slot is granted refunds the token — the bucket level is restored, mirroring the
    async path's cancellation refund.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # frozen: no refill
    configure_rate_limit(max_concurrent=8, rpm=480)
    key = "openai"
    rpm_bucket = GlobalRateLimiter._get_rpm_bucket(key)
    assert rpm_bucket is not None
    full_level = rpm_bucket._level  # capacity min(8, 480) == 8, starts full

    # Force a failure between the RPM debit and the slot grant by making the
    # concurrency gate's acquire raise (the phase after RPM/TPM, before a slot).
    gate = GlobalRateLimiter._get_sync_gate(key)

    def _boom() -> None:
        raise RuntimeError("slot grant failed")

    monkeypatch.setattr(gate, "acquire", _boom)

    with pytest.raises(RuntimeError):
        with rate_limit_acquire_sync(key):
            raise AssertionError("body must not run when the grant fails")

    # The token debited before the failed grant was refunded: bucket full again.
    assert rpm_bucket._level == full_level


def test_exception_during_sync_body_leaks_no_permit() -> None:
    """An exception in the ``with`` body releases the concurrency slot (finally),
    so the full cap is available again — the permit-leak fix over the old
    ``sem.acquire()``-outside-``try`` path.
    """
    configure_rate_limit(max_concurrent=1)
    key = "openai"
    gate = GlobalRateLimiter._get_sync_gate(key)

    with pytest.raises(RuntimeError):
        with rate_limit_acquire_sync(key):
            raise RuntimeError("boom in body")

    assert gate._in_flight == 0  # slot released despite the exception

    # Full cap reclaimed: a fresh acquire from another thread still enters. A
    # leaked single permit (cap 1) would have wedged this forever.
    entered = threading.Event()

    def worker() -> None:
        with rate_limit_acquire_sync(key):
            entered.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(timeout=1.0), "permit leaked: the slot was never released"
    thread.join(timeout=1.0)
    assert gate._in_flight == 0


def test_sync_gate_queued_waiters_are_fifo_and_parked_not_spinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More-than-cap sync waiters queue in strict FIFO order, and a waiter woken
    without a free slot *parks* on the poll interval rather than busy-spinning.

    Regression for the never-cleared one-shot ``threading.Event`` ticket: a
    committing head wakes the next head unconditionally, so a head roused with no
    capacity would — with an un-cleared Event — see ``ticket.wait(timeout)`` return
    instantly every iteration and peg a core (measured >100k ``limit()`` reads in
    0.3s) instead of honouring the ~20Hz poll. Two holders saturate a cap-2 gate,
    three waiters queue in a pinned arrival order, and freeing exactly one slot
    admits the FIFO head while promoting the next waiter to a *set-but-uncommittable*
    head — the precise spin condition. Counting live ``AdaptiveState.limit()`` reads
    over a held real interval separates a bounded poll (a handful) from the spin.
    None of the pre-existing sync tests drove more than one genuinely-queued waiter,
    so this path — and the spin — was previously untested.
    """
    configure_rate_limit(max_concurrent=2)
    key = "openai"
    gate = GlobalRateLimiter._get_sync_gate(key)

    release_h0 = threading.Event()
    release_h1 = threading.Event()
    hold_waiters = threading.Event()
    order: list[int] = []
    order_lock = threading.Lock()

    def holder(release: threading.Event) -> None:
        with rate_limit_acquire_sync(key):
            _ = release.wait(timeout=5.0)

    def waiter(i: int) -> None:
        with rate_limit_acquire_sync(key):
            with order_lock:
                order.append(i)
            _ = hold_waiters.wait(timeout=5.0)

    h0 = threading.Thread(target=holder, args=(release_h0,), daemon=True)
    h1 = threading.Thread(target=holder, args=(release_h1,), daemon=True)
    h0.start()
    h1.start()
    _wait_until(lambda: gate._in_flight == 2)  # both slots held

    waiters: list[threading.Thread] = []
    for i in range(3):
        t = threading.Thread(target=waiter, args=(i,), daemon=True)
        t.start()
        waiters.append(t)
        # Pin deque order: don't start the next until this one has actually queued.
        _wait_until(lambda i=i: len(gate._waiters) == i + 1)

    orig_limit = AdaptiveState.limit
    try:
        # Free exactly one slot: the FIFO head (W0) commits and, per the wake path,
        # sets the NEXT head (W1) — which now has no capacity (H1 + W0 hold both
        # slots) and so must park, not spin.
        release_h0.set()
        _wait_until(lambda: order == [0])

        # W1 is now the head with a set-but-uncommittable ticket: the exact spin
        # condition. Count its live-limit reads over a real interval.
        calls = {"n": 0}

        def counting(self: AdaptiveState) -> int:
            calls["n"] += 1
            return orig_limit(self)

        monkeypatch.setattr(AdaptiveState, "limit", counting)
        time.sleep(0.2)  # a real interval (this test does not freeze the clock)
        monkeypatch.setattr(AdaptiveState, "limit", orig_limit)
        # ~4 for a 0.05s poll over 0.2s; the un-cleared-ticket spin was >100k.
        assert calls["n"] < 50, f"queued head busy-spun: {calls['n']} limit() reads in 0.2s"

        # Draining the rest admits W1 then W2 — strict FIFO end to end.
        release_h1.set()
        _wait_until(lambda: order == [0, 1])
        hold_waiters.set()  # W0, W1 exit → slots free → W2 admitted
        _wait_until(lambda: order == [0, 1, 2])
    finally:
        monkeypatch.setattr(AdaptiveState, "limit", orig_limit)
        release_h0.set()
        release_h1.set()
        hold_waiters.set()
        for t in (h0, h1, *waiters):
            t.join(timeout=2.0)

    assert order == [0, 1, 2]  # admitted in strict arrival order
    assert gate._in_flight == 0  # every slot released


def test_sync_gate_drop_waiter_keeps_queue_contiguous_and_advances_head() -> None:
    """Dropping an abandoned ticket (the interrupt path's baton hand-off) removes
    only that waiter and hands the head baton on when — and only when — the head left.

    Drives ``SyncAdaptiveGate._drop_waiter`` directly (the ``finally`` an interrupted
    ``acquire`` runs): a non-head waiter that leaves keeps the deque contiguous and
    signals nobody, so the real head keeps its turn; dropping the head wakes the new
    head so the queue can never wedge. The parity change's docstrings promise this
    but no thread-level test exercised it.
    """
    gate = SyncAdaptiveGate(AdaptiveState(provider="x", ceiling=2))
    t0, t1, t2 = (threading.Event() for _ in range(3))
    gate._waiters.extend([t0, t1, t2])

    # A non-head waiter leaving drops only itself and wakes no one (head keeps turn).
    gate._drop_waiter(t1)
    assert list(gate._waiters) == [t0, t2]
    assert not t0.is_set() and not t2.is_set()

    # The head leaving hands the baton to the new head.
    gate._drop_waiter(t0)
    assert list(gate._waiters) == [t2]
    assert t2.is_set()

    # Dropping the last waiter empties the queue and signals nobody (nothing to wake).
    gate._drop_waiter(t2)
    assert list(gate._waiters) == []


_CAP = 2
_FANOUT = 6  # deliberately > _CAP: the surplus is the whole point of the assertion


def _openai_async_gates() -> list[tuple[int, int]]:
    """``(in_flight, queued)`` for every async gate registered under ``openai``.

    Read from the *calling* thread, so it snapshots ``_async_gates`` directly
    rather than going through ``_get_async_gate`` (which needs a running loop).
    One entry means one loop is serving every sync call — the persistent-loop
    property itself; several entries is the fresh-loop-per-call regime.
    """
    return sorted(
        (gate._in_flight, len(gate._waiters))
        for (key, _loop), gate in list(GlobalRateLimiter._async_gates.items())
        if key == "openai"
    )


def test_sync_call_wrappers_share_the_cap_across_threads() -> None:
    """N threads fanning out the sync call wrappers never exceed the provider cap.

    Regression for the fresh-loop hole. Under the old fresh-loop-per-call bridge
    the per-(provider, loop) asyncio semaphore had a single user per call and
    bounded nothing across threads (six threads observed six concurrent in-flight
    calls under a cap of 2). Now every sync call runs on the *one* persistent
    loop, so that async semaphore is genuinely shared and bounds cross-thread
    fan-out from *inside* the async path — no separate calling-thread semaphore.

    The fake patches ``litellm.acompletion`` (deeper than the call functions) so
    the real ``acompletion_text`` — and its ``acquire_async`` on the shared loop
    — runs. The discriminating check is that the limiter is *holding* the
    surplus: with ``_FANOUT`` callers outstanding against a cap of ``_CAP``, the
    one shared ``openai`` async gate must read ``(in_flight, queued) == (_CAP,
    _FANOUT - _CAP)``. That target state is stable — nothing releases the
    in-flight callers until the test thread does — so the barrier's deadline
    bounds *arrival*, not the width of an observation window: a loaded runner
    takes longer to converge but never converges on a different answer.
    ``state['peak']`` backs it independently, measuring concurrency observed
    inside the transport rather than the gate's own bookkeeping.
    """
    configure_rate_limit(max_concurrent=_CAP)
    lock = threading.Lock()
    state: dict[str, int] = {"current": 0, "peak": 0}
    loop_box: list[asyncio.AbstractEventLoop] = []
    # Opened by the TEST thread alone (never by a caller), so nothing inside the
    # fan-out can impose the bound this test asserts — the flaw in the rendezvous
    # this replaces, where the second caller released the first at exactly two.
    release = asyncio.Event()

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        """One faked in-flight provider call: count in, park, count out."""
        if not loop_box:
            loop_box.append(asyncio.get_running_loop())
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        try:
            # A hang-guard, not a pacing device: on every path the test thread
            # sets ``release`` in its ``finally`` long before this fires.
            _ = await asyncio.wait_for(release.wait(), timeout=20.0)
        finally:
            with lock:
                state["current"] -= 1
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            _hidden_params={},
        )

    provider = MagicMock()
    provider.name = "openai"
    provider.model = "fake-model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.litellm_model = MagicMock(return_value="openai/fake")
    provider.reasoning_effort = None

    def _text_call() -> None:
        _ = structured_output.text_llm_call_sync("hi", feature="test", provider=provider)

    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        threads = [threading.Thread(target=_text_call) for _ in range(_FANOUT)]
        try:
            for thread in threads:
                thread.start()
            # THE discriminating assertion: every caller is outstanding and the
            # limiter must be holding the surplus on ONE shared gate. Stable state
            # (nothing releases the in-flight callers until this thread does), so
            # the deadline bounds arrival, not the width of an observation window.
            _wait_until(
                lambda: _openai_async_gates() == [(_CAP, _FANOUT - _CAP)],
                timeout=5.0,
                detail=lambda: f"openai (in_flight, queued) = {_openai_async_gates()}",
            )
        finally:
            if loop_box:
                _ = loop_box[0].call_soon_threadsafe(release.set)
            for thread in threads:
                thread.join(timeout=10.0)

    assert state["peak"] == _CAP, (
        f"cap={_CAP} but {state['peak']} sync calls were in flight at once"
    )


# ---------------------------------------------------------------------------
# Requests-per-minute (RPM) and tokens-per-minute (TPM): opt-in token buckets.
#
# These are pinned with a *frozen* monotonic clock (monkeypatching
# ``rate_limiting._now``) so refill is deterministic and nothing sleeps for
# real — the bucket arithmetic is exercised directly, and the limiter wiring is
# checked through the public acquire path.
# ---------------------------------------------------------------------------


def test_rate_bucket_acquire_drains_and_refills(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token bucket drains on acquire, refills at its rate, and caps at capacity."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    bucket = RateBucket(rate_per_sec=10.0, capacity=100.0)  # full at construction

    # Full bucket: draining the whole capacity succeeds with no wait.
    assert bucket._try_acquire(100.0) == 0.0
    # Now empty: the next unit isn't deducted; the gate reports the refill wait.
    assert bucket._try_acquire(1.0) == 1.0 / 10.0  # 1 token / 10 per sec

    # Advance 5s → +50 tokens refilled; 50 is now acquirable.
    clock["t"] += 5.0
    assert bucket._try_acquire(50.0) == 0.0

    # However long we wait, the level caps at capacity (no unbounded credit).
    clock["t"] += 10_000.0
    bucket._refill_locked()
    assert bucket._level == 100.0


def test_rate_bucket_record_drives_negative_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TPM accounting: a record over budget goes negative, then refills back up."""
    clock = {"t": 0.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    bucket = RateBucket(rate_per_sec=100.0, capacity=6_000.0)  # tpm 6000

    assert bucket._try_budget() == 0.0  # full → budget available
    bucket.record(6_500.0)  # one over-budget call → 500 into deficit
    assert bucket._level == -500.0
    # Exhausted: the gate now reports the wait until it climbs back above zero.
    assert bucket._try_budget() == (1.0 + 500.0) / 100.0

    clock["t"] += 6.0  # +600 tokens → back above zero
    assert bucket._try_budget() == 0.0


# ---------------------------------------------------------------------------
# Bucket waiter fairness: the cost-deducting acquire path admits in FIFO order.
#
# The RPM acquire path deducts a *scarce* token at admission, so under sustained
# saturation a barging newcomer could starve an older waiter indefinitely. These
# tests pin strict arrival-order admission on both the async (per-loop
# asyncio.Lock) and sync (Event ticket-queue) paths, and that an interrupted sync
# waiter drops only itself and never wedges the queue. Token availability is
# driven by a frozen, manually-advanced clock so that *ordering* — not wall-clock
# timing — is what is under test. (The TPM ``wait_for_budget`` path deducts
# nothing at admission, so it cannot barge and is deliberately not serialized.)
# ---------------------------------------------------------------------------


def _wait_until(
    predicate: Callable[[], bool],
    timeout: float = 2.0,
    detail: Callable[[], str] | None = None,
) -> None:
    """Poll ``predicate`` until true, raising if it stays false past ``timeout``.

    Uses the real monotonic clock (the rate-limiter's ``_now`` is monkeypatched
    to a frozen test clock, so it cannot be used for the timeout here). On
    timeout ``detail`` (when supplied) is called to describe the observed state,
    so a barrier that never converged reports *why* rather than a bare deadline.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            suffix = f" — {detail()}" if detail is not None else ""
            raise AssertionError(f"condition not met within {timeout}s{suffix}")
        time.sleep(0.001)


@contextlib.contextmanager
def _drained_on_exit(clock: dict[str, float]) -> Generator[list[threading.Thread]]:
    """Yield a list to register each *started* sync-waiter thread in, and on exit
    release and join them all — even when the body raised.

    The sync-waiter tests park daemon worker threads spinning on a frozen, empty
    bucket; only stepping ``clock`` frees them. Without this, an assertion that
    fails before a test's own joins would strand those threads spinning for the
    rest of the pytest session (perturbing later tests; the daemon flag only
    keeps a stray one from wedging interpreter shutdown). Stepping the clock one
    tick frees the current head — capacity caps the refill at a single token — so
    repeat until every worker has exited, then join. Never raises: it runs in a
    ``finally``, so a real test failure still surfaces from the body.
    """
    threads: list[threading.Thread] = []
    try:
        yield threads
    finally:
        for _ in range(len(threads) + 2):
            alive = sum(t.is_alive() for t in threads)
            if alive == 0:
                break
            clock["t"] += 1.0  # free one token → the current head deducts and exits
            stalled = time.monotonic() + 1.0
            while sum(t.is_alive() for t in threads) >= alive and time.monotonic() < stalled:
                time.sleep(0.001)
        for t in threads:
            t.join(timeout=2.0)


def _install_controlled_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[asyncio.Future[None]], Callable[[], Awaitable[None]]]:
    """Replace ``asyncio.sleep`` with a stub that parks each call on a test-owned
    future, and return ``(gates, pump)``.

    Each refill-sleep appends a future to ``gates`` and blocks on it, so the test
    controls exactly when a parked waiter re-polls — and ``len(gates)`` reveals how
    many waiters are parked at once. Under the FIFO lock only the *head* ever
    reaches the sleep (the rest block on the lock), so ``gates`` holds exactly one
    entry per outstanding head; a barging loop would park every waiter at once.
    ``pump`` drains the ready queue via the *real* ``asyncio.sleep(0)``.
    """
    loop = asyncio.get_running_loop()
    real_sleep = asyncio.sleep
    gates: list[asyncio.Future[None]] = []

    async def controlled_sleep(_delay: float) -> None:
        fut: asyncio.Future[None] = loop.create_future()
        gates.append(fut)
        await fut

    monkeypatch.setattr(asyncio, "sleep", controlled_sleep)

    async def pump() -> None:
        for _ in range(30):
            await real_sleep(0)

    return gates, pump


async def test_rpm_bucket_serves_async_waiters_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async RPM waiters are admitted strictly in arrival order — only the head
    contends for tokens.

    Each refill-sleep parks on a test-owned future, so exactly one waiter (the
    FIFO head) is ever parked — the ``len(gates) == 1`` check is what a barging
    loop fails (it would park all three at once). Releasing the head one token at a
    time serves them in arrival order, with no over-admission.
    """
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    gates, pump = _install_controlled_sleep(monkeypatch)

    bucket = RateBucket(rate_per_sec=1.0, capacity=1.0)
    assert bucket._try_acquire(1.0) == 0.0  # drain the initial token → empty, frozen

    order: list[int] = []

    async def waiter(i: int) -> None:
        await bucket.acquire_async(1.0)
        order.append(i)

    tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
    await pump()
    assert order == []  # all three blocked on the empty bucket
    assert len(gates) == 1  # ONLY the head is parked; the rest queue on the per-loop lock

    for expected in range(3):
        clock["t"] += 1.0  # refill exactly one token (capacity 1 caps the refill)
        gates[expected].set_result(None)  # wake the current head; it deducts and exits
        await pump()
        assert order == list(range(expected + 1))  # served strictly in arrival order
        assert len(gates) == min(expected + 2, 3)  # the next head parks on a fresh gate

    _ = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert order == [0, 1, 2]
    assert bucket._level == 0.0  # exactly three tokens consumed for three waiters


async def test_async_newcomer_cannot_barge_parked_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newcomer cannot take a token while an older waiter is parked ahead of it.

    The crux of the fairness fix: with a token *available* but the head still
    parked, a freshly-arrived waiter must NOT grab it — it is queued behind the
    head on the per-loop lock. Under the old barging loop the newcomer would poll
    freely and serve out of order, so ``order == []`` here is what distinguishes
    the fix.
    """
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    gates, pump = _install_controlled_sleep(monkeypatch)

    bucket = RateBucket(rate_per_sec=1.0, capacity=1.0)
    assert bucket._try_acquire(1.0) == 0.0  # drain → empty

    order: list[str] = []

    async def waiter(name: str) -> None:
        await bucket.acquire_async(1.0)
        order.append(name)

    head = asyncio.create_task(waiter("head"))
    await pump()
    assert len(gates) == 1 and order == []  # head parked on the empty bucket

    clock["t"] += 1.0  # a token is now available — but the head is still parked on its gate
    newcomer = asyncio.create_task(waiter("newcomer"))
    await pump()
    assert order == []  # newcomer did NOT barge the token; it is queued behind the head
    assert len(gates) == 1  # newcomer is on the lock, not polling the bucket

    gates[0].set_result(None)  # release the head; it takes the available token first
    await pump()
    assert order == ["head"]  # head served before the newcomer despite the newcomer arriving second

    clock["t"] += 1.0
    gates[1].set_result(None)  # now the newcomer (which parked once it became head) is served
    _ = await asyncio.wait_for(asyncio.gather(head, newcomer), timeout=1.0)
    assert order == ["head", "newcomer"]


def test_rpm_bucket_serves_sync_waiters_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync RPM waiters are admitted strictly in arrival order (Event ticket-queue).

    The threaded mirror of the async FIFO test: four threads queue on the empty
    bucket, tokens are released one at a time via the clock, and each goes to the
    current head. A high rate keeps the head's real spin-sleep ~1ms; the capacity
    cap makes "one token per clock step" exact regardless of float fuzz.
    """
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    bucket = RateBucket(rate_per_sec=1_000.0, capacity=1.0)  # empty-wait ≈ 1ms
    assert bucket._try_acquire(1.0) == 0.0  # drain → empty, frozen

    order: list[int] = []
    order_lock = threading.Lock()

    def waiter(i: int) -> None:
        bucket.acquire_sync(1.0)
        with order_lock:
            order.append(i)

    with _drained_on_exit(clock) as threads:
        for i in range(4):
            thread = threading.Thread(target=waiter, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)  # registered started, so a mid-test failure still drains it
            _wait_until(lambda i=i: len(bucket._sync_waiters) == i + 1)  # queued in arrival order

        assert order == []  # all four blocked on the empty bucket

        for expected in range(4):
            clock["t"] += 1.0  # +1000 tokens at 1000/s, capped to capacity 1 → one token
            _wait_until(lambda e=expected: len(order) == e + 1)
            with order_lock:
                assert order == list(range(expected + 1))  # head served; the rest wait their turn

        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()
        assert order == [0, 1, 2, 3]
        assert bucket._level == 0.0  # exactly four tokens consumed for four waiters


def test_sync_acquire_interrupt_does_not_wedge_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sync waiter interrupted mid-wait drops only itself; the queue still drains.

    Reproduces the wedge the ticket-queue must not have: a *non-head* waiter that
    raises (a ``KeyboardInterrupt`` delivered while parked in its ticket wait) must
    remove only its own ticket by identity and never strand the waiters behind it —
    and must never evict the still-running head with a positional pop.
    """
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])

    # Make the 2nd ticket created (waiter B, queued behind the head) raise from its
    # wait(), simulating a Ctrl-C delivered while B is parked. Threads are built
    # *before* patching so their internal Events stay real — only the bucket's
    # three tickets become interrupting Events.
    counter = {"n": 0}

    class _InterruptingEvent(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self._idx: int = counter["n"]
            counter["n"] += 1

        @override
        def wait(self, timeout: float | None = None) -> bool:
            if self._idx == 1:
                raise KeyboardInterrupt("simulated Ctrl-C during ticket.wait()")
            return super().wait(timeout)

    bucket = RateBucket(rate_per_sec=1_000.0, capacity=1.0)
    assert bucket._try_acquire(1.0) == 0.0  # drain → empty

    order: list[str] = []
    order_lock = threading.Lock()

    def waiter(name: str) -> None:
        with contextlib.suppress(KeyboardInterrupt):
            bucket.acquire_sync(1.0)
            with order_lock:
                order.append(name)

    a = threading.Thread(target=waiter, args=("A",), daemon=True)
    b = threading.Thread(target=waiter, args=("B",), daemon=True)
    c = threading.Thread(target=waiter, args=("C",), daemon=True)
    # Patch the threading module's ``Event`` (what the bucket constructs tickets
    # from). Threads are built above, *before* this, so their own internal Events
    # stay real and only the bucket's three tickets become interrupting Events.
    monkeypatch.setattr(threading, "Event", _InterruptingEvent)

    with _drained_on_exit(clock) as threads:
        a.start()
        threads.append(a)
        _wait_until(lambda: len(bucket._sync_waiters) == 1)  # A is the head, spinning

        b.start()  # B enqueues behind A, then raises from wait() and removes itself
        threads.append(b)
        b.join(timeout=2.0)
        assert not b.is_alive()
        assert len(bucket._sync_waiters) == 1  # only A remains; B's exit did not wedge the queue

        c.start()
        threads.append(c)
        _wait_until(lambda: len(bucket._sync_waiters) == 2)  # C queued behind A

        clock["t"] += 1.0  # release a token → A served, hands the baton to C (not to dead B)
        _wait_until(lambda: len(order) == 1)
        clock["t"] += 1.0  # C served next — proof B's interrupt never stranded it
        _wait_until(lambda: len(order) == 2)

        a.join(timeout=2.0)
        c.join(timeout=2.0)
        assert order == ["A", "C"]  # B interrupted; A and C served in order, queue intact


def test_sync_acquire_enqueue_window_exception_leaves_no_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised right after enqueue removes the ticket (no wedge).

    Guards the fix that put the enqueue *inside* the try/finally: an exception in
    the append→body window (here the deque raises immediately after appending,
    standing in for a signal during the append or lock release) must still be
    cleaned up by the finally. Pre-fix the append sat outside the try, so the
    ticket leaked and the whole queue wedged.
    """
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    bucket = RateBucket(rate_per_sec=1_000.0, capacity=1.0)
    assert bucket._try_acquire(1.0) == 0.0  # drain → empty

    armed = {"v": True}

    class _AppendBoomDeque(collections.deque[threading.Event]):
        @override
        def append(self, x: threading.Event, /) -> None:
            super().append(x)
            if armed["v"]:  # boom exactly once, right after the ticket is enqueued
                armed["v"] = False
                raise RuntimeError("boom right after enqueue")

    bucket._sync_waiters = _AppendBoomDeque()  # the bucket's own (empty) queue, instrumented

    with pytest.raises(RuntimeError, match="boom"):
        bucket.acquire_sync(1.0)
    assert list(bucket._sync_waiters) == []  # the appended ticket was removed, not orphaned

    clock["t"] += 1.0  # a token is available
    bucket.acquire_sync(1.0)  # a subsequent waiter drains — the queue never wedged


def test_async_fifo_lock_pruned_for_closed_loop() -> None:
    """The per-loop FIFO lock registry prunes entries whose loop has closed.

    The async FIFO lock is keyed by event loop (a lock's waiter futures bind to
    one loop); like the concurrency gate's registry it must shed closed loops so a
    process spinning short-lived loops cannot leak locks. One bucket is shared
    across loops; each loop installs its own lock, and a later touch prunes the
    dead one.
    """
    bucket = RateBucket(rate_per_sec=1.0, capacity=10.0)

    async def touch() -> None:
        await bucket.acquire_async(1.0)

    asyncio.run(touch())  # first loop installs its lock, then closes
    assert len(bucket._async_locks) == 1
    [first_loop] = list(bucket._async_locks)
    assert first_loop.is_closed()

    asyncio.run(touch())  # a second loop's touch prunes the closed one, installs its own
    assert len(bucket._async_locks) == 1
    [second_loop] = list(bucket._async_locks)
    assert second_loop is not first_loop


async def test_rpm_burst_is_capped_at_concurrency_not_a_full_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold/idle RPM burst is the concurrency width, not a whole minute's quota.

    Worked example for the ~2x-per-minute regression. The old bucket started at
    ``rpm`` (600), so a fan-out could emit 600 instantly *plus* 600 more refilled
    across the next 60s — ~2x the published per-minute limit inside one
    provider-side fixed window. Capacity is now ``min(max_concurrent, rpm)``, so
    the worst case over any 60s window is ``max_concurrent + rpm`` (8 + 600 =
    608, ~1.3% over), not ``2 * rpm``.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # freeze: no refill
    configure_rate_limit(max_concurrent=8, rpm=600)

    bucket = GlobalRateLimiter._get_rpm_bucket("openai")
    assert bucket is not None
    assert bucket._level == 8.0  # full at the concurrency-width capacity, not 600

    # Drain the burst: exactly ``max_concurrent`` requests go straight through...
    for _ in range(8):
        assert bucket._try_acquire(1.0) == 0.0
    # ...and the 9th must wait for refill — the bucket cannot bank a full minute.
    assert bucket._try_acquire(1.0) > 0.0

    # Even after an arbitrarily long idle stretch the level caps at capacity (8),
    # never back to 600 — so the post-idle burst stays bounded at 8, not rpm.
    monkeypatch.setattr(_tuning, "now", lambda: 100_000.0)
    bucket._refill_locked()
    assert bucket._level == 8.0


async def test_tpm_burst_reservoir_is_one_second_not_a_full_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idle TPM reservoir is ~1 second of tokens, not a whole minute.

    The token-dimension half of the same regression: the bucket no longer starts
    at ``tpm`` (a full minute, spendable in a burst before throttling) but at
    ``tpm * TPM_BURST_SECONDS / 60`` — a one-second reservoir — so an idle bucket
    can bank at most ~1.7% of ``tpm`` above the sustained rate.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(tpm=6_000)

    bucket = GlobalRateLimiter._get_tpm_bucket("openai")
    assert bucket is not None
    assert bucket._level == 100.0  # tpm/60, the 1s reservoir — not 6000

    # The cap also bounds refill: after a long idle the reservoir tops out at the
    # 1s capacity, never back at a full minute's tokens.
    monkeypatch.setattr(_tuning, "now", lambda: 100_000.0)
    bucket._refill_locked()
    assert bucket._level == 100.0


def test_rate_limit_slot_record_tokens_is_noop_without_bucket() -> None:
    """A slot with no TPM bucket (or no tokens) records nothing and never raises."""
    RateLimitSlot().record_tokens(1_234)
    RateLimitSlot().record_tokens(None)


def test_configure_rejects_nonpositive_rpm_tpm() -> None:
    """``rpm``/``tpm`` must be a positive int or None — zero/negative is a config error."""
    with pytest.raises(ValueError, match="rpm must be a positive integer"):
        configure_rate_limit(rpm=0)
    with pytest.raises(ValueError, match="tpm must be a positive integer"):
        configure_rate_limit(tpm=-5)


def test_configure_rejects_nonpositive_max_concurrent() -> None:
    """``max_concurrent`` must be positive — ``Semaphore(0)`` is a legal,
    permanently-locked semaphore, so accepting 0 would hang every LLM call."""
    with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
        configure_rate_limit(max_concurrent=0)
    with pytest.raises(ValueError, match="max_concurrent must be a positive integer"):
        configure_rate_limit(max_concurrent=-1)


async def test_rpm_off_by_default_creates_no_bucket() -> None:
    """With no rpm/tpm configured, acquiring creates no rate buckets (byte-identical)."""
    async with GlobalRateLimiter.acquire_async("openai"):
        pass
    assert GlobalRateLimiter._rpm_buckets == {}
    assert GlobalRateLimiter._tpm_buckets == {}


async def test_rpm_acquire_debits_the_request_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RPM-configured acquire deducts one request token from the provider bucket."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # freeze: no refill
    # Burst capacity is the concurrency width min(max_concurrent=8, rpm=120) = 8,
    # NOT a full minute's quota — so the bucket starts at 8, not 120.
    configure_rate_limit(rpm=120)  # 2 requests/sec sustained; burst 8

    async with GlobalRateLimiter.acquire_async("openai"):
        pass

    assert GlobalRateLimiter._rpm_buckets["openai"]._level == 7.0  # 8 - 1


async def test_tpm_slot_records_against_provider_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``slot.record_tokens`` debits the provider's TPM bucket by the usage."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # freeze
    # Burst capacity is a 1s reservoir = tpm/60 = 100 tokens (not a full minute),
    # so a 2000-token debit drives the level well negative — exactly the
    # smoothing signal that makes the next caller wait. The debit *amount* (2000)
    # is what this test pins; the start level is the small reservoir, not 6000.
    configure_rate_limit(tpm=6_000)

    async with GlobalRateLimiter.acquire_async("openai") as slot:
        slot.record_tokens(2_000)

    assert GlobalRateLimiter._tpm_buckets["openai"]._level == -1_900.0  # 100 - 2000


async def test_tpm_over_budget_blocks_then_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spending past the minute budget drives the bucket negative so the gate waits."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    configure_rate_limit(tpm=6_000)  # 100 tokens/sec sustained; burst 100 (1s)

    async with GlobalRateLimiter.acquire_async("openai") as slot:
        slot.record_tokens(9_000)  # spends far past the small reservoir

    bucket = GlobalRateLimiter._tpm_buckets["openai"]
    assert bucket._level == -8_900.0  # 100 reservoir - 9000 spent
    assert bucket._try_budget() > 0.0  # the next caller would block

    clock["t"] += 90.0  # +9000 tokens at 100/sec clears the 8900 deficit
    assert bucket._try_budget() == 0.0


async def test_tpm_budgets_are_independent_per_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draining one provider's TPM budget leaves another provider's untouched."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(tpm=6_000)  # burst 100 (1s reservoir) per provider

    async with GlobalRateLimiter.acquire_async("openai") as slot:
        slot.record_tokens(6_000)  # drain openai
    async with GlobalRateLimiter.acquire_async("google") as slot:
        slot.record_tokens(1_000)  # google has its own full reservoir

    # Each provider's debit lands only on its own bucket (independence is the
    # point); both start from the same 100-token reservoir.
    assert GlobalRateLimiter._tpm_buckets["openai"]._level == -5_900.0  # 100 - 6000
    assert GlobalRateLimiter._tpm_buckets["google"]._level == -900.0  # 100 - 1000


async def test_disabled_skips_rpm_and_tpm_dimensions() -> None:
    """When limiting is disabled, acquire creates no rate buckets and the slot is inert."""
    configure_rate_limit(enabled=False, rpm=1, tpm=1)
    async with GlobalRateLimiter.acquire_async("openai") as slot:
        slot.record_tokens(10_000)  # no bucket → no-op
    assert GlobalRateLimiter._rpm_buckets == {}
    assert GlobalRateLimiter._tpm_buckets == {}


async def test_call_layer_debits_tpm_from_response_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the real call layer debits TPM by the response's ``usage.total_tokens``.

    Fully offline — ``litellm.acompletion`` is faked to return a usage-bearing
    response, and the clock is frozen so the only level change is the debit.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(tpm=6_000)

    provider = MagicMock()
    provider.name = "openai"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.litellm_model = MagicMock(return_value="openai/fake")
    provider.reasoning_effort = None

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi"))],
            usage=MagicMock(total_tokens=2_000),
            _hidden_params={},
        )

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = await _litellm.acompletion_text(
            "hi",
            temperature=0.0,
            model=None,
            provider=provider,
        )

    assert text == "hi"
    # 1s reservoir (tpm/60 = 100) minus the response's 2000-token usage.
    assert GlobalRateLimiter._tpm_buckets["openai"]._level == -1_900.0


# ---------------------------------------------------------------------------
# Case-insensitive provider keys: llmkit's own call sites key by the display
# name ("OpenAI"), while a host following the public-helper docs may pass any
# casing. Keys are casefolded at the limiter boundary so every casing names
# ONE budget — a differently-cased host can never silently fork onto a
# separate budget it believes is shared.
# ---------------------------------------------------------------------------


async def test_rpm_budget_shared_across_key_casings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public-helper "openai" and internal-path "OpenAI" debit the SAME RPM bucket.

    With a frozen clock and rpm=2 (capacity 2), one lowercase acquire via the
    public helper plus one display-cased acquire via the internal path must
    drain the single shared bucket to zero — and the registry must hold exactly
    one (casefolded) entry, not one per casing.
    """
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # freeze: no refill
    configure_rate_limit(rpm=2)

    async with rate_limit_acquire_async("openai"):  # host casing, public helper
        pass
    async with GlobalRateLimiter.acquire_async("OpenAI"):  # llmkit's display name
        pass

    assert set(GlobalRateLimiter._rpm_buckets) == {"openai"}  # one bucket, casefolded key
    assert GlobalRateLimiter._rpm_buckets["openai"]._level == 0.0  # 2 - 1 - 1

    # The shared budget is now exhausted: a third acquire under ANY casing
    # would have to wait for refill.
    assert GlobalRateLimiter._rpm_buckets["openai"]._try_acquire(1.0) > 0.0


async def test_tpm_budget_shared_across_key_casings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tokens recorded under "OPENAI" are visible to a "openai" acquirer's bucket."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    # tpm chosen so the 1s reservoir (tpm/60 = 4000) comfortably holds both
    # debits: the first call must leave the *shared* bucket positive, or the
    # second casing's TPM gate would (correctly) block waiting for refill — which
    # under this frozen clock would never come. The sharing/casefold invariant,
    # not the gate's blocking, is what this test pins.
    configure_rate_limit(tpm=240_000)

    async with GlobalRateLimiter.acquire_async("OPENAI") as slot:
        slot.record_tokens(2_000)
    async with rate_limit_acquire_async("openai") as slot:
        slot.record_tokens(1_000)

    assert set(GlobalRateLimiter._tpm_buckets) == {"openai"}
    # Both debits land on the one shared (casefolded) bucket: 4000 reservoir
    # minus 2000 minus 1000.
    assert GlobalRateLimiter._tpm_buckets["openai"]._level == 1_000.0


async def test_concurrency_semaphore_shared_across_key_casings() -> None:
    """A slot held under "OpenAI" blocks a "openai" acquire: one semaphore, cap 1."""
    configure_rate_limit(max_concurrent=1)
    entered_second = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with GlobalRateLimiter.acquire_async("OpenAI"):  # internal casing
            _ = await release_first.wait()

    async def second() -> None:
        async with rate_limit_acquire_async("openai"):  # host casing, public helper
            entered_second.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())

    for _ in range(20):
        await asyncio.sleep(0)
    # If the casings forked into two semaphores, second() would be in already.
    assert not entered_second.is_set(), "differently-cased keys got separate semaphores"

    release_first.set()
    _ = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1.0)
    assert entered_second.is_set()


def test_sync_gate_shared_across_key_casings() -> None:
    """The sync path shares one concurrency gate across key casings too."""
    configure_rate_limit(max_concurrent=1)
    entered_second = threading.Event()
    release_first = threading.Event()
    first_holds = threading.Event()

    def first() -> None:
        with GlobalRateLimiter.acquire_sync("OpenAI"):  # internal casing
            first_holds.set()
            _ = release_first.wait(timeout=1.0)

    def second() -> None:
        with rate_limit_acquire_sync("openai"):  # host casing, public helper
            entered_second.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_holds.wait(timeout=1.0)

    second_thread.start()
    # One shared slot, held under "OpenAI": the "openai" acquire must block.
    assert not entered_second.wait(timeout=0.1)

    release_first.set()
    second_thread.join(timeout=1.0)
    first_thread.join(timeout=1.0)
    assert entered_second.is_set()
