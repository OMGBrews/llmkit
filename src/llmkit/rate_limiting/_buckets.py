"""The *rate* dimension: token buckets, and the slot handle they back.

:class:`RateBucket` implements both per-minute ceilings — requests (RPM, cost
known up front) and tokens (TPM, cost known only after the response) — over one
continuously-refilling reservoir. :class:`RateLimitSlot` is the public handle an
acquired slot yields; it exists only to carry the optional TPM bucket to the
caller's post-call :meth:`~RateLimitSlot.record_tokens`.

Kept separate from the *concurrency* dimension (``_adaptive`` / ``_breaker``):
these are the gates a caller passes before holding a slot, not the gate that
bounds how many slots exist.
"""

import asyncio
import collections
import contextlib
import threading
import time
from dataclasses import dataclass

import llmkit.rate_limiting._tuning as _tuning
from llmkit.rate_limiting._tuning import TPM_BURST_SECONDS as TPM_BURST_SECONDS


class RateBucket:
    """A per-provider token bucket for a *rate* limit (RPM or TPM).

    The bucket holds up to ``capacity`` tokens and refills continuously at
    ``rate_per_sec`` tokens/second. Two consumption shapes are supported,
    matching the two rate dimensions:

    * **Requests/minute (RPM)** — the cost is known *up front* (one request =
      one token). :meth:`acquire_async` / :meth:`acquire_sync` wait until at
      least ``cost`` tokens are available, then deduct them.
    * **Tokens/minute (TPM)** — the cost is known only *after* the call returns
      its usage. :meth:`wait_for_budget_async` / :meth:`wait_for_budget_sync`
      block only while the bucket is exhausted (level ``<= 0``); the caller
      then proceeds and, once the response is in hand, calls :meth:`record`
      with the measured token count. :meth:`record` may drive the level
      negative, which throttles the *next* caller until the bucket refills back
      above zero.

    State (the fill level and last-refill timestamp) is guarded by a plain
    :class:`threading.Lock`, held only for non-blocking O(1) bookkeeping — the
    refill-and-compare arithmetic and the per-running-loop FIFO-lock registry
    (below) — never across a sleep or ``await``, so a single bucket is safe to
    share between the async and sync acquire paths (one true per-provider budget).

    **FIFO admission (the cost-deducting RPM path).** A waiter that finds too few
    tokens must wait for a refill, but the deducted token is *scarce*: if every
    waiter just slept on its own computed delay and re-contended, a newcomer
    waking first could take the freshly refilled token an older waiter was about
    to claim, and under sustained saturation an individual waiter's latency would
    be unbounded. So :meth:`acquire_async` / :meth:`acquire_sync` admit in strict
    arrival order — only the queue *head* contends for tokens; everyone else waits
    behind it. Async waiters serialize on a **per-running-loop**
    :class:`asyncio.Lock` (CPython's lock is FIFO) held across the refill-sleep;
    it is keyed by loop — and closed loops pruned — because a lock's waiter
    futures bind to one loop and a bucket is awaited from several (the persistent
    sync loop, a host's own loop, the reentrant fallback's short-lived loops),
    exactly the per-loop reality :meth:`GlobalRateLimiter._get_async_gate` has.
    Sync waiters serialize on a deque of one-shot :class:`threading.Event`
    tickets (a plain :class:`threading.Lock` would not be FIFO); the head hands
    the baton to the next ticket when it exits — *including on an exception*, so a
    waiter interrupted (e.g. ``KeyboardInterrupt`` while parked in its wait)
    removes only itself and never wedges the queue. The token arithmetic still
    runs under :attr:`_lock` on every admit, so the **aggregate rate is exact
    regardless of queue**: serialization only orders *who* deducts next, never
    adds a token. FIFO is therefore per-loop (and the sync queue is independent of
    the async ones); cross-loop ordering is not strictly FIFO — the same bound the
    per-loop concurrency gate already documents — but the rate stays globally
    correct.

    **The TPM gate is deliberately not serialized.**
    :meth:`wait_for_budget_async` / :meth:`wait_for_budget_sync` deduct *nothing*
    at admission (the debit is post-call, in :meth:`record`); they only block
    while the bucket is exhausted. Once it refills above zero every blocked waiter
    passes and none can take anything from another, so the TPM gate is fair by
    construction — there is no scarce admission token to barge for. Adding a queue
    there would only force an ordering onto the hot entry path of every call for
    no fairness gain, so it keeps the plain refill-retry loop.
    """

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self._rate: float = rate_per_sec
        self._capacity: float = capacity
        self._level: float = capacity
        self._updated: float = _tuning.now()
        self._lock: threading.Lock = threading.Lock()
        # FIFO admission state for the cost-deducting acquire path (RPM); see the
        # class docstring. Async waiters queue on a per-running-loop asyncio.Lock
        # (keyed by loop and pruned, like _get_async_gate, since a Lock can't span
        # loops); sync waiters queue on a deque of one-shot Event tickets under its
        # own lock. wait_for_budget_* (TPM) is intentionally not serialized.
        self._async_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
        self._sync_waiters: collections.deque[threading.Event] = collections.deque()
        self._sync_lock: threading.Lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = _tuning.now()
        elapsed = now - self._updated
        if elapsed > 0:
            self._level = min(self._capacity, self._level + elapsed * self._rate)
            self._updated = now

    def _try_acquire(self, cost: float) -> float:
        """Deduct ``cost`` if available; else return the seconds to wait first.

        Returns ``0.0`` on success (tokens deducted), otherwise the time until
        enough tokens will have refilled for ``cost`` to succeed. ``cost`` is
        clamped to the bucket capacity, since the level can never exceed it.
        """
        # Clamp to capacity: the level caps at capacity on refill, so a cost
        # greater than capacity could otherwise never be satisfied (permanent wait).
        cost = min(cost, self._capacity)
        with self._lock:
            self._refill_locked()
            if self._level >= cost:
                self._level -= cost
                return 0.0
            return (cost - self._level) / self._rate

    def _try_budget(self) -> float:
        """Return ``0.0`` if any budget remains, else seconds until it refills
        above zero (for the TPM gate, which doesn't know the cost up front)."""
        with self._lock:
            self._refill_locked()
            if self._level > 0:
                return 0.0
            return (1.0 - self._level) / self._rate

    def record(self, amount: float) -> None:
        """Debit ``amount`` unconditionally (TPM accounting after a call).

        May drive the level negative; the deficit refills over the next
        window, so the next caller waits exactly long enough to bring the
        sustained rate back under the configured ceiling.
        """
        with self._lock:
            self._refill_locked()
            self._level -= amount

    def refund(self, amount: float) -> None:
        """Credit ``amount`` back, capped at capacity (never banks beyond full).

        The inverse of the up-front RPM deduction in :meth:`acquire_async`: when
        an acquirer is cancelled *after* its request token is deducted but
        *before* it holds a slot, the token is refunded here so a systematically
        cancelled workload doesn't silently shrink its own effective RPM. Capping
        at capacity keeps a refund from over-crediting a bucket that refilled in
        the meantime.
        """
        with self._lock:
            self._refill_locked()
            self._level = min(self._capacity, self._level + amount)

    def _fifo_lock_for_loop(self) -> asyncio.Lock:
        """The per-running-loop FIFO lock for the cost-deducting acquire path.

        Get-or-create the :class:`asyncio.Lock` for the running loop, first
        pruning entries whose loop has closed. Guarded by :attr:`_lock` (the
        token-state lock) because the dict is touched from every loop/thread,
        mirroring :meth:`GlobalRateLimiter._get_async_gate`. Keyed by loop because
        a lock's waiter futures bind to the loop they are created on and a single
        bucket is awaited from several loops; constructing the lock here is
        loop-inert on Python >= 3.12 (it binds the loop lazily on first
        ``acquire``), so it is safe to build under the threading lock. The running
        loop is by definition not closed, so the returned lock can never be pruned
        out from under the ``async with`` that follows.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [k for k in self._async_locks if k.is_closed()]:
                del self._async_locks[closed]
            lock = self._async_locks.get(loop)
            if lock is None:
                lock = asyncio.Lock()
                self._async_locks[loop] = lock
            return lock

    async def acquire_async(self, cost: float) -> None:
        """Deduct ``cost`` tokens, waiting in FIFO order behind earlier waiters.

        Holds this loop's FIFO lock across the refill-sleep so only the queue
        head contends for tokens and newcomers queue behind it (no barging). A
        cancellation mid-sleep releases the held lock and the FIFO lock hands the
        next waiter on; a cancellation while still *queued* just drops out of the
        lock's wait queue (it holds nothing). Either way no token leaks, because
        the deduct-and-return in :meth:`_try_acquire` is atomic with respect to
        the ``await`` — a token is deducted only on the iteration that returns
        ``0.0`` and exits, with no ``await`` between that deduction and return.
        """
        async with self._fifo_lock_for_loop():
            while (wait := self._try_acquire(cost)) > 0:
                await asyncio.sleep(wait)

    def acquire_sync(self, cost: float) -> None:
        """Deduct ``cost`` tokens, waiting in FIFO order behind earlier waiters.

        The sync mirror of :meth:`acquire_async`, with no event loop: each caller
        enqueues a one-shot :class:`threading.Event` ticket and the head alone
        contends for tokens, handing the baton to the next ticket on exit. The
        enqueue is *inside* the ``try``, so the ``finally`` covers a ticket from
        the instant it is appended: it removes *this* ticket by identity, so an
        exception at any point after enqueue (e.g. a ``KeyboardInterrupt``
        delivered during the append, the lock release, or the ``wait``) frees only
        this waiter — a non-head waiter that leaves keeps the deque contiguous and
        the real head still wakes the next ticket — so the queue can never wedge.
        """
        ticket = threading.Event()
        try:
            with self._sync_lock:
                is_head = not self._sync_waiters
                self._sync_waiters.append(ticket)
            if is_head:
                ticket.set()  # nobody ahead of us — proceed immediately
            _ = ticket.wait()
            while (wait := self._try_acquire(cost)) > 0:
                time.sleep(wait)
        finally:
            with self._sync_lock:
                was_head = bool(self._sync_waiters) and self._sync_waiters[0] is ticket
                with contextlib.suppress(ValueError):
                    self._sync_waiters.remove(ticket)
                # Only the head owes the baton. A non-head waiter that was
                # interrupted just drops itself; the still-running head will wake
                # the next ticket when it finishes.
                if was_head and self._sync_waiters:
                    self._sync_waiters[0].set()

    async def wait_for_budget_async(self) -> None:
        while (wait := self._try_budget()) > 0:
            await asyncio.sleep(wait)

    def wait_for_budget_sync(self) -> None:
        while (wait := self._try_budget()) > 0:
            time.sleep(wait)


@dataclass(frozen=True)
class RateLimitSlot:
    """Handle yielded by an acquired rate-limit slot.

    Carries the post-call hook for tokens-per-minute accounting: once a call
    returns its token usage, pass it to :meth:`record_tokens` so the provider's
    TPM budget is debited. A no-op when TPM limiting is not configured (the
    common case) or the usage is unknown, so call sites can always call it
    unconditionally.
    """

    _tpm_bucket: RateBucket | None = None

    def record_tokens(self, total_tokens: int | None) -> None:
        """Debit ``total_tokens`` from this provider's TPM budget (best-effort).

        Does nothing when TPM limiting is disabled or ``total_tokens`` is
        ``None`` / zero (e.g. a streamed call that reported no usage).
        """
        if self._tpm_bucket is not None and total_tokens:
            self._tpm_bucket.record(float(total_tokens))
