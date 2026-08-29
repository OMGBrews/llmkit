"""The *concurrency* dimension: AIMD state and the two gates over it.

One :class:`AdaptiveState` per provider holds the live limit and the aggregate
in-flight count; the async and sync gates park on different primitives over that
same shared state, so a fan-out spread across both surfaces still obeys one
budget.

**Lock ordering is gate lock -> state lock, never the reverse.** Splitting these
three classes out of one file does not relax that: :meth:`SyncAdaptiveGate.
acquire` reads the live limit while holding its own lock, and
:class:`AdaptiveState`'s methods take the state lock as a leaf.
"""

import asyncio
import collections
import contextlib
import threading

import llmkit.rate_limiting._tuning as _tuning
from llmkit.rate_limiting._observability import BackpressureEvent
from llmkit.rate_limiting._tuning import (
    AIMD_DECREASE_COOLDOWN,
    AIMD_DECREASE_FACTOR,
    AIMD_RECOVERY_INTERVAL,
    SYNC_GATE_POLL_INTERVAL,
)


class AdaptiveState:
    """Per-provider AIMD limit, shared across every event loop in the process.

    One instance per provider *name* (like the RPM/TPM buckets — not per loop),
    holding the moving concurrency ``limit`` and the timestamps that pace it. The
    per-(provider, loop) :class:`AdaptiveGate`s read :meth:`limit` live on each
    admit and feed outcomes back via :meth:`on_throttle` / :meth:`on_success`.

    **Saturation is judged on the provider-wide aggregate.** Every gate — each
    per-loop :class:`AdaptiveGate` and the one per-provider
    :class:`SyncAdaptiveGate` — mirrors its in-flight count into
    ``_total_in_flight`` here via :meth:`note_acquire` / :meth:`note_release`, so
    :meth:`on_throttle` decides "were we at the shared limit?" against the sum
    across *all* populations, not one gate's local count. That is what lets a
    self-inflicted 429 in the multi-population regime (a host's own loop and the
    persistent sync loop both holding slots) trigger a decrease when no single
    gate is itself at the limit. Admission is unaffected — each gate still caps
    its own count independently (the per-population capacity model); only the
    saturation *judgment* aggregates.

    State is guarded by a plain :class:`threading.Lock` (shared safely across the
    async loops and sync threads, exactly like :class:`RateBucket`), held only
    for the O(1) arithmetic and **never across an await**. Because the aggregate
    counter lives under this same lock, read-aggregate + judge + cooldown + halve
    is one atomic critical section. Each mutator returns the
    :class:`BackpressureEvent` to emit (or ``None``) so the caller can fire the
    observability callback *outside* the lock.
    """

    def __init__(self, provider: str, ceiling: int) -> None:
        self._provider: str = provider
        self._ceiling: int = ceiling
        self._limit: int = ceiling
        # Provider-wide in-flight aggregate across every gate (all async gates +
        # the sync gate), mirrored by note_acquire/note_release. Read under the
        # lock to judge saturation on the shared limit, not one gate's local count.
        self._total_in_flight: int = 0
        self._lock: threading.Lock = threading.Lock()
        # Recovery is paced from this anchor; a decrease resets it (so a throttle
        # delays recovery) and each applied step advances it by one interval.
        self._recovery_anchor: float = _tuning.now()
        # Last decrease timestamp, for the refractory cooldown. ``-inf`` so the
        # first throttle is never suppressed.
        self._last_decrease: float = float("-inf")

    def limit(self) -> int:
        """The current admit ceiling (read live by the gate on every admit)."""
        with self._lock:
            return self._limit

    def note_acquire(self) -> None:
        """Count one slot grant into the provider-wide in-flight aggregate.

        Called by every gate (async and sync) the instant it increments its own
        ``_in_flight``, so :meth:`on_throttle` judges saturation on the sum across
        all populations. O(1) under the state lock.
        """
        with self._lock:
            self._total_in_flight += 1

    def note_release(self) -> None:
        """Count one slot release out of the provider-wide in-flight aggregate.

        The mirror of :meth:`note_acquire`; every gate calls it the instant it
        decrements its own ``_in_flight`` (including the async abandoned-grant
        rollback). A missed decrement would bias saturation permanently high and
        over-halve the limit.
        """
        with self._lock:
            self._total_in_flight -= 1

    def on_throttle(self) -> BackpressureEvent | None:
        """Account a provider overload signal; halve the limit if appropriate.

        Decreases only when the provider's aggregate in-flight — across every
        async gate and the sync gate — was at or above the shared limit when the
        throttle arrived; a throttle received while the aggregate is below the
        limit is provider-global noise, not evidence this client's concurrency is
        the problem, so halving then would be a pure self-inflicted regression.
        Judging on the aggregate (rather than one gate's local count) is what lets
        a self-inflicted 429 trigger a decrease even when the throttled call sat
        on a gate that was not itself at the limit. At most one decrease per
        ``AIMD_DECREASE_COOLDOWN``, so a fan-out's correlated burst of throttles
        collapses to a single halving instead of crashing to the floor. The
        aggregate read, the saturation judgment, the cooldown check, and the
        halving are one atomic critical section under the state lock. Returns the
        event to emit, or ``None`` when nothing changed.
        """
        with self._lock:
            if self._total_in_flight < self._limit:
                return None
            now = _tuning.now()
            if now - self._last_decrease < AIMD_DECREASE_COOLDOWN:
                return None
            self._last_decrease = now
            self._recovery_anchor = now  # a throttle restarts the recovery clock
            new = max(1, int(self._limit * AIMD_DECREASE_FACTOR))
            if new == self._limit:
                return None
            old, self._limit = self._limit, new
            return BackpressureEvent(self._provider, old, new, "throttle")

    def on_success(self) -> BackpressureEvent | None:
        """Account a successful call; recover the limit toward the ceiling.

        Wall-clock-paced: the limit climbs by one for each
        ``AIMD_RECOVERY_INTERVAL`` elapsed since the recovery anchor (a long idle
        gap recovers several steps at once). Triggered by a success but bounded by
        elapsed time, so recovery costs ``ceiling`` *intervals*, not
        O(ceiling**2) successes — the tail of a finite batch is not left
        depressed. Returns the event to emit, or ``None`` when nothing changed.
        """
        with self._lock:
            if self._limit >= self._ceiling:
                return None
            now = _tuning.now()
            steps = int((now - self._recovery_anchor) // AIMD_RECOVERY_INTERVAL)
            if steps < 1:
                return None
            self._recovery_anchor += steps * AIMD_RECOVERY_INTERVAL
            new = min(self._ceiling, self._limit + steps)
            if new == self._limit:
                return None
            old, self._limit = self._limit, new
            return BackpressureEvent(self._provider, old, new, "recover")


class AdaptiveGate:
    """A FIFO async concurrency gate whose capacity is a live ``AdaptiveState``.

    Replaces the fixed ``asyncio.Semaphore``: capacity is read from the shared
    per-provider :class:`AdaptiveState` on every admit, so it tracks the AIMD
    limit. One gate per (provider, loop) — the waiter futures bind to one loop,
    the same hard reality the semaphore had.

    **FIFO, no barging.** Waiters are served strictly in arrival order via a deque
    of futures (a newcomer queues behind anyone already waiting even if a slot is
    momentarily free), and ``in_flight`` is incremented *at grant time* so the
    count is always exact. A grown limit admits the next waiter(s) on the very
    next :meth:`release` (a recovery step happens on a success, immediately before
    that call releases), so no separate wake path is needed. Cancellation while
    queued — or in the narrow window after a grant but before resumption — never
    leaks a slot (the grant is handed on).

    All mutation runs on the gate's single event loop (asyncio is
    single-threaded), so no lock guards ``in_flight`` / ``_waiters``; the only lock
    is inside ``_state`` — for the cross-loop limit and the provider-wide in-flight
    aggregate this gate mirrors into (via ``_grant`` / ``_ungrant``) so saturation
    is judged across every population, not this gate alone.
    """

    def __init__(self, state: AdaptiveState) -> None:
        self._state: AdaptiveState = state
        self._in_flight: int = 0
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    def _grant(self) -> None:
        # The single place the async gate increments its in-flight count: bump the
        # local count and mirror it into the shared aggregate so saturation is
        # judged across all populations. Runs on the gate's loop (no gate lock);
        # ``note_acquire`` takes only the state lock (gate loop -> state lock,
        # never the reverse, so no lock inversion).
        self._in_flight += 1
        self._state.note_acquire()

    def _ungrant(self) -> None:
        # The mirror of ``_grant`` — the single place the async gate decrements.
        self._in_flight -= 1
        self._state.note_release()

    async def acquire(self) -> None:
        """Admit one call, blocking in FIFO order until a slot is free."""
        # Fast path: capacity available and nobody ahead of us — claim a slot.
        if not self._waiters and self._in_flight < self._state.limit():
            self._grant()
            return
        # Otherwise queue. ``_admit_next`` grants head-first, incrementing
        # ``in_flight`` *before* resolving our future, so on wake the slot is ours.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
        except BaseException:
            # Cancelled/errored while queued, or in the window just after a grant.
            if fut in self._waiters:
                self._waiters.remove(fut)
            if fut.done() and not fut.cancelled():
                # Granted a slot (``in_flight`` already bumped for us) but
                # abandoning it — hand it to the next waiter.
                self._ungrant()
                self._admit_next()
            raise

    def release(self) -> None:
        """Release a held slot and admit the next waiter(s) per the live limit."""
        self._ungrant()
        self._admit_next()

    def _admit_next(self) -> None:
        # Grant slots to head waiters in FIFO order while capacity allows. The
        # limit is read live, so a release after a recovery step admits more.
        while self._waiters and self._in_flight < self._state.limit():
            fut = self._waiters.popleft()
            if fut.cancelled():
                continue  # waiter already gone; don't spend a slot on it
            self._grant()
            fut.set_result(None)

    def on_throttle(self) -> BackpressureEvent | None:
        """Feed a throttle outcome to the shared state (saturation is judged
        there, on the provider-wide aggregate this gate mirrors into)."""
        return self._state.on_throttle()

    def on_success(self) -> BackpressureEvent | None:
        """Feed a success outcome to the shared state (drives recovery)."""
        return self._state.on_success()


class SyncAdaptiveGate:
    """A FIFO **sync** concurrency gate whose capacity is a live ``AdaptiveState``.

    The threading counterpart to :class:`AdaptiveGate`, and the replacement for
    the old fixed ``threading.Semaphore``. Both gates read their capacity from the
    *same* shared per-provider :class:`AdaptiveState` on every admit, so a
    throttle observed on **either** the async or the sync path lowers the limit
    the **other** path then honours (true shared backpressure, not a mirror). One
    gate per provider *name* — threads have no event-loop affinity, so unlike the
    async gate this needs no per-loop key.

    **FIFO, no barging.** Waiters queue on a deque of one-shot
    :class:`threading.Event` tickets; only the head contends for a slot and a
    newcomer always queues behind anyone already waiting, even if a slot is
    momentarily free. ``_in_flight`` is incremented at grant time so the count is
    always exact. A freed slot (:meth:`release`) wakes the head, which self-grants
    and wakes the next head in turn.

    **Never leaks a permit on interruption.** Every wait lives inside a
    ``try/finally``: a waiter interrupted (e.g. ``KeyboardInterrupt`` while parked)
    removes only itself and, if it was the head, hands the baton to the next
    ticket, so the queue can never wedge and no slot is lost. A caller that holds
    a slot releases it through :meth:`release` in the acquirer's ``finally``.

    **Bounded re-check (the cross-population subtlety).** The sync gate and the
    async gate park on different primitives over one shared
    :class:`AdaptiveState`, so an **async-side** success that *raises* the limit
    signals nothing on the sync side — no sync ``release`` runs. A blocked sync
    head must therefore not wait unboundedly on a condition only sync releases
    fire: it waits with a bounded :data:`SYNC_GATE_POLL_INTERVAL` timeout and
    re-reads the live limit each time, so a cross-population limit increase can
    never strand it (it is admitted within one poll interval). A same-side
    decrease/release is still honoured immediately.

    State (``_in_flight`` + the waiter deque) is guarded by a plain
    :class:`threading.Lock`, held only for O(1) bookkeeping, never across a wait;
    each grant/release also mirrors into the shared :class:`AdaptiveState`
    aggregate (gate lock -> state lock, never reversed — the same order the
    acquire path already uses reading ``self._state.limit()`` under the gate
    lock). Exposes the same :meth:`on_throttle` / :meth:`on_success` surface
    :class:`AdaptiveGate` does, so the shared outcome-classification helper
    (:func:`record_gate_outcome`) drives both paths identically.
    """

    def __init__(self, state: AdaptiveState) -> None:
        self._state: AdaptiveState = state
        self._in_flight: int = 0
        self._waiters: collections.deque[threading.Event] = collections.deque()
        self._lock: threading.Lock = threading.Lock()

    def _grant(self) -> None:
        # Caller holds ``self._lock``. Bump the local count and mirror into the
        # shared aggregate. ``note_acquire`` takes the state lock: ordering is
        # gate lock -> state lock (the same order the acquire fast path uses when
        # it reads ``self._state.limit()`` under ``self._lock``), never reversed,
        # and the state lock is a leaf held only for O(1) arithmetic.
        self._in_flight += 1
        self._state.note_acquire()

    def _ungrant(self) -> None:
        # Caller holds ``self._lock``. The mirror of ``_grant``.
        self._in_flight -= 1
        self._state.note_release()

    def acquire(self) -> None:
        """Admit one call, blocking in FIFO order until a slot is free.

        On success ``_in_flight`` is incremented for this caller; on *any*
        ``BaseException`` before the grant it holds no slot and leaves the queue
        contiguous (the ``finally`` drops this ticket and hands the head baton on).
        """
        ticket = threading.Event()
        slot_held = False
        try:
            with self._lock:
                # Fast path: nobody ahead of us and capacity free — take a slot.
                if not self._waiters and self._in_flight < self._state.limit():
                    self._grant()
                    slot_held = True
                    return
                self._waiters.append(ticket)
            while True:
                # Bounded wait: a same-side release sets our ticket immediately; the
                # timeout re-checks the live limit so a cross-population (async-side)
                # limit increase can never strand us. See the class docstring.
                _ = ticket.wait(SYNC_GATE_POLL_INTERVAL)
                with self._lock:
                    if (
                        self._waiters
                        and self._waiters[0] is ticket
                        and self._in_flight < self._state.limit()
                    ):
                        # Commit. Dequeue and wake the next head (idempotent calls)
                        # first; the adjacent ``_grant()``/``slot_held`` stores are
                        # the atomic grant (``_grant`` mirrors into the shared
                        # aggregate under the state leaf-lock — no await, no yield).
                        _ = self._waiters.popleft()
                        if self._waiters:
                            self._waiters[0].set()
                        self._grant()
                        slot_held = True
                        return
                    # Woken but cannot commit yet — a head roused with no capacity
                    # (the committing head wakes the next head unconditionally, and a
                    # release after an AIMD *decrease* may leave in_flight still >=
                    # the lowered limit). Clear the one-shot ticket before re-parking
                    # so ``wait`` genuinely blocks for the poll interval instead of
                    # returning instantly on a still-set Event and busy-spinning. Safe
                    # under ``self._lock`` because every ``set`` (commit above,
                    # ``release``, ``_drop_waiter``) also runs under it, so a
                    # concurrent set is serialized against this clear and any set that
                    # lands after the lock releases just makes the next ``wait`` return
                    # at once — no wakeup is lost, and the poll interval backstops it.
                    ticket.clear()
        finally:
            if not slot_held:
                self._drop_waiter(ticket)

    def _drop_waiter(self, ticket: threading.Event) -> None:
        """Remove an abandoned ticket, handing the baton on if it was the head."""
        with self._lock:
            was_head = bool(self._waiters) and self._waiters[0] is ticket
            with contextlib.suppress(ValueError):
                self._waiters.remove(ticket)
            # Only the head owes the baton. A non-head waiter that leaves keeps the
            # deque contiguous and the real head still wakes the next ticket.
            if was_head and self._waiters:
                self._waiters[0].set()

    def release(self) -> None:
        """Release a held slot and wake the head to claim the freed capacity."""
        with self._lock:
            self._ungrant()
            if self._waiters:
                self._waiters[0].set()

    def on_throttle(self) -> BackpressureEvent | None:
        """Feed a throttle outcome to the shared state (saturation is judged
        there, on the provider-wide aggregate this gate mirrors into)."""
        return self._state.on_throttle()

    def on_success(self) -> BackpressureEvent | None:
        """Feed a success outcome to the shared state (drives recovery)."""
        return self._state.on_success()
