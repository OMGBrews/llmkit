"""Global, per-provider rate limiting for LLM API calls.

Bounds the LLM calls a process makes to each provider along three
independent, per-provider dimensions:

* **Concurrency** — the maximum number of *in-flight* calls per provider
  (a semaphore). On by default, cap 8.
* **Requests per minute (RPM)** — the sustained *request rate* per provider
  (a token bucket). Opt-in, off by default.
* **Tokens per minute (TPM)** — the sustained *token-consumption rate* per
  provider (a token bucket, debited by each call's measured usage). Opt-in,
  off by default.

The LiteLLM call layer (:mod:`llmkit._litellm`) wraps every provider call in
:meth:`GlobalRateLimiter.acquire_async`, keyed by the **provider name**. The
sync call wrappers drive that same async coroutine through
:func:`llmkit.sync.run_sync`, which routes every sync call onto **one
persistent event loop**. Because all sync calls share that loop, the
per-(provider, loop) async concurrency semaphore is genuinely shared among
them and bounds sync fan-out across threads directly — no separate
calling-thread semaphore is needed. The semaphore is keyed per (provider, loop)
only because an :class:`asyncio.Semaphore` binds to one loop and cannot be
awaited from another; the persistent sync loop, any loop a host runs its own
async calls on, and the short-lived loops the reentrant fallback spins up are
therefore each their own key.

One honest caveat follows from asyncio semaphores being per-loop: a process
that drives calls on more than one loop — e.g. a host running llmkit's *async*
call functions on its own event loop *and* llmkit's sync wrappers (on the
persistent loop) — caps each loop's population independently, so it can
momentarily hold up to ``loops x max_concurrent`` in-flight calls per provider.
Each population is capped; the caps do not share slots, because an asyncio
primitive cannot span loops. The hand-rolled sync acquire path
(:meth:`acquire_sync` / :func:`rate_limit_acquire_sync`) is bounded by a
loop-agnostic per-provider ``threading.Semaphore`` instead — for host code that
issues sync provider calls *outside* llmkit's own loop. RPM/TPM are unaffected
either way: those buckets are keyed by name and shared across every loop and
thread.

On by default, scoped per provider
----------------------------------
Rate limiting is **enabled out of the box** (``_enabled`` defaults to
``True``) — zero configuration needed. Every dimension is accounted **per
provider**, where the key is the provider *name* string (``provider.name`` /
:pyattr:`BaseProvider._provider_name`, e.g. ``"OpenAI"``, ``"Ollama"``). That
is exactly the value logging records as the *effective* provider (see
:func:`llmkit.capture.resolve_model_and_provider`), so a held slot —
and any token debit — is always accounted to the provider that actually runs
the call.

Provider keys are matched **case-insensitively**: every key is casefolded at
the limiter boundary (see :func:`_normalize_key`), so ``"openai"``,
``"OpenAI"``, and ``"OPENAI"`` all name one budget. A host joining the limit
by hand via :func:`rate_limit_acquire_async` / :func:`rate_limit_acquire_sync`
therefore cannot fork itself onto a separate budget by spelling the provider
name differently from llmkit's own call sites.

**Concurrency** defaults to a cap of **8 concurrent calls per provider**: eight
favours the common access pattern — fan-out — out of the box, bounding a
self-inflicted burst (and the bill / 429s that follow) without quietly
serialising real multi-call workloads. A host on a tightly-metered plan can
lower it via :func:`configure_rate_limit`; a local Ollama server happy to fan
out harder can raise it.

**RPM and TPM are opt-in and off by default.** Unlike concurrency, there is no
universal sane requests-/tokens-per-minute number — the right value is the
metered limit of *your* account/plan — so leaving them unset sends a
byte-identical request to the pre-feature behaviour (no throttle on those
dimensions). Set them to your account's published RPM/TPM and llmkit smooths
calls to stay under them. The binding limit on a metered cloud account is
usually RPM/TPM, **not** concurrency, so a migrator moving off a
requests-per-minute knob should set ``rpm=`` here rather than expect the
concurrency cap to stand in for it (the two limit different things, and the
concurrency cap alone leaves an old RPM tuning inert).

A single ``max_concurrent`` / ``rpm`` / ``tpm`` value applies to *all*
providers; there is intentionally no per-provider cap map, to keep the public
surface small (three numbers, one switch).

Token bucket, not sliding window
--------------------------------
RPM and TPM use a **token bucket** (:class:`_RateBucket`), not a sliding
window. A bucket holds O(1) state per provider — a fill level and a timestamp,
no per-event history — refills continuously at the configured rate, and
tolerates a burst up to its capacity before smoothing to the sustained rate.
A sliding window would track timestamps of recent calls for exactness at the
cost of unbounded per-provider history; for a thin, local-first limiter whose
job is to *smooth* fan-out under a metered ceiling (not to bill), the bucket's
constant-memory approximation is the right trade.

The bucket's **capacity is the burst depth, deliberately decoupled from the
per-minute number.** A bucket admits up to ``capacity + rate * T`` events in any
window of length ``T``, so a capacity of a full minute's quota (``= rpm`` /
``= tpm``) would let a cold or idle bucket emit ~2x the published per-minute
limit inside one provider-side fixed-minute window — breaking the "set them to
your account's published RPM/TPM" guidance for a bursty-after-idle workload.
Instead the burst is sized small: RPM caps it at the concurrency width
(``min(max_concurrent, rpm)``, since the RPM gate is passed before the
concurrency semaphore) and TPM at a one-second reservoir
(``_TPM_BURST_SECONDS``), so the worst-case overshoot is a small constant above
the sustained rate, not 2x. Each bucket still starts *full* at that small
capacity, so a quiet process's first call (and a cold fan-out up to
``max_concurrent``) never waits — only the empty-start alternative would. A
strict fixed-window provider therefore still sees a small burst above sustained,
so a workload known to be bursty against such a provider wants a little
headroom; the residual is documented in ``docs/planning/opinions.md`` §6.4.

Single-tenant by design; deliberate non-goals
----------------------------------------------
llmkit is designed for **single-tenant** applications: one credential per
provider. Two deliberate non-goals follow:

* **No per-credential scoping (deferred, not rejected).** Two configs of the
  same provider family that differ only by ``api_key`` / ``base_url`` share one
  budget, because the key is the provider *name*, not the credential. For a
  single-tenant app that is exact — there is only one credential per provider,
  so name-keying and credential-keying are identical. A genuine *multi-tenant*
  host (many accounts of one provider family in one process) would see them
  collapse into a shared budget; isolating those would require keying the slot
  by a credential hash, which would diverge it from how logging resolves the
  provider (by name). That multi-tenant direction is a possible future-version
  optimisation, recorded in ``docs/planning/opinions.md`` §6.4 — not built
  today, and a maintainer who needs it must opt into it explicitly.
* **No extra global ceiling.** There is no aggregate cap on top of the
  per-provider limits; total concurrency/rate is just the sum of the
  per-provider budgets. Per-provider only.
"""

import asyncio
import collections
import contextlib
import enum
import logging
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, Protocol

from llmkit.exceptions import CircuitOpenError, underlying_provider_error

logger = logging.getLogger(__name__)


def _now() -> float:
    """Monotonic clock read, indirected so offline tests can advance time.

    The token buckets read the clock through this one function; a test can
    monkeypatch ``llmkit.rate_limiting._now`` to drive refill deterministically
    without real sleeping.
    """
    return time.monotonic()


#: Burst depth for a TPM bucket, expressed in seconds of the sustained rate:
#: ``capacity = tpm * _TPM_BURST_SECONDS / 60``. A one-second reservoir keeps the
#: worst-case per-window overshoot to ~1.7% of ``tpm`` instead of the 2x a
#: full-minute capacity (``= tpm``) would allow after an idle stretch. TPM debits
#: *after* the call and gates only while exhausted, so even this small reservoir
#: never makes a quiet process's first call wait — capacity only bounds how much
#: an idle bucket may bank. (RPM's burst is the concurrency width instead — see
#: :meth:`GlobalRateLimiter._get_rpm_bucket` — because requests have a
#: concurrency analog and tokens do not.) Decoupling capacity from the per-minute
#: number is the burst-semantics decision recorded in opinions.md §6.4.
_TPM_BURST_SECONDS: float = 1.0

#: HTTP status codes that count as a provider *overload* signal for adaptive
#: concurrency (AIMD). 429 (rate limit), 503 (service unavailable / shared
#: capacity), 529 (overloaded). Deliberately **excludes** 500 (a generic server
#: error, not necessarily overload), 408, and bare network timeouts — those are
#: ambiguous transport noise, and over-reacting to them by collapsing concurrency
#: is the failure mode the saturation gate (below) exists to avoid. Widen on
#: evidence, not on speculation.
_THROTTLE_STATUS_CODES: frozenset[int] = frozenset({429, 503, 529})

#: AIMD tuning. These are deliberately **internal** constants, not public knobs:
#: the host owns the *ceiling* (``max_concurrent``) and the RPM/TPM numbers; the
#: library owns *how the limit moves beneath them* — mechanism it should own, like
#: the temperature default and the burst-depth choice (see opinions.md §6.4/§8).
#:
#: * ``_AIMD_DECREASE_FACTOR`` — multiplicative decrease on a throttle: the limit
#:   halves (floored at 1).
#: * ``_AIMD_DECREASE_COOLDOWN`` — refractory period (seconds): at most one
#:   decrease per window, so a fan-out's correlated burst of throttles collapses
#:   to a *single* halving instead of crashing to the floor on one bad instant.
#: * ``_AIMD_RECOVERY_INTERVAL`` — additive increase is **wall-clock-paced**: the
#:   limit climbs by one for each interval that has elapsed since the last change
#:   with no intervening throttle. Recovery from the floor to a ceiling of 8 is
#:   ~``7 * interval`` seconds regardless of offered load — bounded in wall-clock,
#:   unlike a per-success ("generation") rule that needs O(ceiling**2) successful
#:   calls and starves the tail of a finite batch.
_AIMD_DECREASE_FACTOR: float = 0.5
_AIMD_DECREASE_COOLDOWN: float = 1.0
_AIMD_RECOVERY_INTERVAL: float = 3.0

#: Circuit-breaker tuning. Like the AIMD constants these are deliberately
#: **internal**: the host owns the *switch* (``breaker``), the library owns the
#: mechanism. The breaker is the "limit is effectively 0 for a cooldown" case
#: AIMD's floor-of-1 cannot express — it stops doomed work outright when a
#: provider is *down* (see opinions.md §6.4). Defaults are conservative because
#: it is opt-in and flips "eventually succeeds" → "fails fast".
#:
#: * ``_BREAKER_WINDOW`` — size of the count-based ring of recent *real* outcomes
#:   (a throttle or a success; a fast ``CircuitOpenError`` and an ambiguous
#:   neutral error record nothing). O(N) state, no timestamps — one wrong OPEN is
#:   bounded to a single cooldown by the HALF_OPEN probe, so a count ring beats a
#:   time-bucketed window's complexity here.
#: * ``_BREAKER_MIN_SAMPLES`` — never trip before the ring holds this many
#:   outcomes, so one early throttle in a near-empty ring can't open the breaker.
#:   Equal to the window, so the ring must be *full* before it can trip.
#: * ``_BREAKER_THRESHOLD`` — open once the throttled fraction of a full ring
#:   reaches this (0.5 ⇒ half of the recent real outcomes were throttles).
#: * ``_BREAKER_COOLDOWN`` — seconds the breaker fast-fails while OPEN before it
#:   admits a single HALF_OPEN probe to test whether the provider has recovered.
_BREAKER_WINDOW: int = 20
_BREAKER_MIN_SAMPLES: int = 20
_BREAKER_THRESHOLD: float = 0.5
_BREAKER_COOLDOWN: float = 30.0


def _is_throttle_signal(exc: BaseException) -> bool:
    """Whether *exc* is a provider overload signal worth backing off on (AIMD).

    Unwraps first (:func:`~llmkit.exceptions.underlying_provider_error`) — a
    structured call surfaces the provider error wrapped in
    ``InstructorRetryException``, so without the unwrap a 429/503 from the
    *structured* path (the incident workload) would be invisible here. Classifies
    by HTTP ``status_code`` against :data:`_THROTTLE_STATUS_CODES`; LiteLLM/OpenAI
    status errors (including litellm's 503 ``ServiceUnavailableError``) carry the
    code, so a name check is unnecessary. Anything without a throttle status —
    a ``ValidationError``, a bare network timeout, a 500 — is neutral.
    """
    root = underlying_provider_error(exc)
    status = getattr(root, "status_code", None)
    return isinstance(status, int) and status in _THROTTLE_STATUS_CODES


def _normalize_key(provider_key: str) -> str:
    """Casefold a provider key so limiter budgets are shared case-insensitively.

    The single choke point where a provider key enters the limiter:
    :meth:`GlobalRateLimiter.acquire_async` / :meth:`acquire_sync` normalize
    through this before touching any registry, so ``"openai"``, ``"OpenAI"``,
    and ``"OPENAI"`` resolve to one semaphore and one RPM/TPM bucket. The
    normalized form is an internal registry key only — display/logging keeps
    the provider's original ``name``.
    """
    return provider_key.casefold()


class _RateBucket:
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
        self._updated: float = _now()
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
        now = _now()
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
        loop-inert on Python >= 3.13 (it binds the loop lazily on first
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

    _tpm_bucket: _RateBucket | None = None

    def record_tokens(self, total_tokens: int | None) -> None:
        """Debit ``total_tokens`` from this provider's TPM budget (best-effort).

        Does nothing when TPM limiting is disabled or ``total_tokens`` is
        ``None`` / zero (e.g. a streamed call that reported no usage).
        """
        if self._tpm_bucket is not None and total_tokens:
            self._tpm_bucket.record(float(total_tokens))


@dataclass(frozen=True)
class BackpressureEvent:
    """A per-provider backpressure transition, for observability.

    Emitted to the callback installed by :func:`backpressure_callback` whenever
    the AIMD limit actually moves *or* the opt-in circuit breaker changes state,
    so a host can *see* backpressure in real time (and wire its own metrics)
    instead of reconstructing it from call logs.

    Attributes:
        provider: The normalized (casefolded) provider key the budget is keyed by
            — the same identity the limiter accounts every dimension under.
        old_limit: The effective per-provider concurrency limit before the change.
        new_limit: The effective limit after the change.
        reason: which transition this is —

            * ``"throttle"`` — a provider overload signal lowered the AIMD limit;
            * ``"recover"`` — sustained no-throttle time raised the AIMD limit
              back toward the configured ceiling;
            * ``"breaker_open"`` — the breaker tripped and is now fast-failing;
              ``new_limit`` is ``0`` (it admits nothing) and ``old_limit`` is the
              ceiling it collapsed from;
            * ``"breaker_half_open"`` — the cooldown elapsed and the breaker
              admitted a single probe (``old_limit`` ``0`` → ``new_limit`` ``1``);
            * ``"breaker_closed"`` — the probe succeeded; the breaker is back to
              full capacity (``old_limit`` ``1`` → ``new_limit`` the ceiling).

            The breaker reasons report the breaker's *own* effective ceiling
            (ceiling → 0 → 1 → ceiling), which is independent of — and not to be
            confused with — the AIMD limit the ``"throttle"`` / ``"recover"``
            reasons carry.
    """

    provider: str
    old_limit: int
    new_limit: int
    reason: Literal[
        "throttle",
        "recover",
        "breaker_open",
        "breaker_half_open",
        "breaker_closed",
    ]


class BackpressureCallback(Protocol):
    """Receives :class:`BackpressureEvent`s from the adaptive limiter.

    Installed via :func:`backpressure_callback`. Invoked on the loop that observed
    the limit change; implementations must not raise (a raise is caught and
    logged, never allowed to break a call) and should not block. The event is
    passed positionally, so a plain ``def f(event): ...``, a lambda, or even
    ``list.append`` all satisfy this protocol.
    """

    def __call__(self, event: BackpressureEvent, /) -> None: ...


_backpressure_callback: ContextVar[BackpressureCallback | None] = ContextVar(
    "_backpressure_callback", default=None
)


def _emit_backpressure(event: BackpressureEvent | None) -> None:
    """Fire the installed backpressure callback for *event* (a no-op if ``None``).

    Read from a context variable so it propagates across the ``run_sync`` /
    ``to_thread`` boundaries (the persistent-loop bridge copies the caller's
    context) without threading a parameter through every call. A callback that
    raises is swallowed and logged — observability must never break a call.
    """
    if event is None:
        return
    callback = _backpressure_callback.get()
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        logger.exception("backpressure callback raised for %s", event.provider)


@contextlib.contextmanager
def backpressure_callback(callback: BackpressureCallback | None) -> Generator[None]:
    """Install a backpressure callback for the current dynamic scope.

    The callback receives a :class:`BackpressureEvent` each time the adaptive
    limiter changes a provider's concurrency limit (a throttle-driven decrease or
    a time-driven recovery) or the opt-in circuit breaker changes state
    (``breaker_open`` / ``breaker_half_open`` / ``breaker_closed``). Read from a
    context variable, so it crosses the ``run_sync`` thread boundary like the
    retry progress callback. Pass ``None`` to disable callbacks within an inner
    scope; the previous value is restored on exit.

    Usage::

        def on_event(event: BackpressureEvent) -> None:
            metrics.gauge(f"llm.concurrency.{event.provider}", event.new_limit)


        with backpressure_callback(on_event):
            ...  # calls here report limit changes to ``on_event``
    """
    token = _backpressure_callback.set(callback)
    try:
        yield
    finally:
        _backpressure_callback.reset(token)


class _AdaptiveState:
    """Per-provider AIMD limit, shared across every event loop in the process.

    One instance per provider *name* (like the RPM/TPM buckets — not per loop),
    holding the moving concurrency ``limit`` and the timestamps that pace it. The
    per-(provider, loop) :class:`_AdaptiveGate`s read :meth:`limit` live on each
    admit and feed outcomes back via :meth:`on_throttle` / :meth:`on_success`.

    State is guarded by a plain :class:`threading.Lock` (shared safely across the
    async loops and sync threads, exactly like :class:`_RateBucket`), held only
    for the O(1) arithmetic and **never across an await**. Each mutator returns
    the :class:`BackpressureEvent` to emit (or ``None``) so the caller can fire the
    observability callback *outside* the lock.
    """

    def __init__(self, provider: str, ceiling: int) -> None:
        self._provider: str = provider
        self._ceiling: int = ceiling
        self._limit: int = ceiling
        self._lock: threading.Lock = threading.Lock()
        # Recovery is paced from this anchor; a decrease resets it (so a throttle
        # delays recovery) and each applied step advances it by one interval.
        self._recovery_anchor: float = _now()
        # Last decrease timestamp, for the refractory cooldown. ``-inf`` so the
        # first throttle is never suppressed.
        self._last_decrease: float = float("-inf")

    def limit(self) -> int:
        """The current admit ceiling (read live by the gate on every admit)."""
        with self._lock:
            return self._limit

    def on_throttle(self, *, saturated: bool) -> BackpressureEvent | None:
        """Account a provider overload signal; halve the limit if appropriate.

        Decreases only when *saturated* (the gate was at its limit when the
        throttle arrived) — a throttle received while running below the limit is
        provider-global noise, not evidence this client's concurrency is the
        problem, so halving then would be a pure self-inflicted regression. At
        most one decrease per ``_AIMD_DECREASE_COOLDOWN``, so a fan-out's
        correlated burst of throttles collapses to a single halving instead of
        crashing to the floor. Returns the event to emit, or ``None`` when nothing
        changed.
        """
        if not saturated:
            return None
        with self._lock:
            now = _now()
            if now - self._last_decrease < _AIMD_DECREASE_COOLDOWN:
                return None
            self._last_decrease = now
            self._recovery_anchor = now  # a throttle restarts the recovery clock
            new = max(1, int(self._limit * _AIMD_DECREASE_FACTOR))
            if new == self._limit:
                return None
            old, self._limit = self._limit, new
            return BackpressureEvent(self._provider, old, new, "throttle")

    def on_success(self) -> BackpressureEvent | None:
        """Account a successful call; recover the limit toward the ceiling.

        Wall-clock-paced: the limit climbs by one for each
        ``_AIMD_RECOVERY_INTERVAL`` elapsed since the recovery anchor (a long idle
        gap recovers several steps at once). Triggered by a success but bounded by
        elapsed time, so recovery costs ``ceiling`` *intervals*, not
        O(ceiling**2) successes — the tail of a finite batch is not left
        depressed. Returns the event to emit, or ``None`` when nothing changed.
        """
        with self._lock:
            if self._limit >= self._ceiling:
                return None
            now = _now()
            steps = int((now - self._recovery_anchor) // _AIMD_RECOVERY_INTERVAL)
            if steps < 1:
                return None
            self._recovery_anchor += steps * _AIMD_RECOVERY_INTERVAL
            new = min(self._ceiling, self._limit + steps)
            if new == self._limit:
                return None
            old, self._limit = self._limit, new
            return BackpressureEvent(self._provider, old, new, "recover")


class _AdaptiveGate:
    """A FIFO async concurrency gate whose capacity is a live ``_AdaptiveState``.

    Replaces the fixed ``asyncio.Semaphore``: capacity is read from the shared
    per-provider :class:`_AdaptiveState` on every admit, so it tracks the AIMD
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
    is inside ``_state`` for the cross-loop limit.
    """

    def __init__(self, state: _AdaptiveState) -> None:
        self._state: _AdaptiveState = state
        self._in_flight: int = 0
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    async def acquire(self) -> None:
        """Admit one call, blocking in FIFO order until a slot is free."""
        # Fast path: capacity available and nobody ahead of us — claim a slot.
        if not self._waiters and self._in_flight < self._state.limit():
            self._in_flight += 1
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
                self._in_flight -= 1
                self._admit_next()
            raise

    def release(self) -> None:
        """Release a held slot and admit the next waiter(s) per the live limit."""
        self._in_flight -= 1
        self._admit_next()

    def _admit_next(self) -> None:
        # Grant slots to head waiters in FIFO order while capacity allows. The
        # limit is read live, so a release after a recovery step admits more.
        while self._waiters and self._in_flight < self._state.limit():
            fut = self._waiters.popleft()
            if fut.cancelled():
                continue  # waiter already gone; don't spend a slot on it
            self._in_flight += 1
            fut.set_result(None)

    def is_saturated(self) -> bool:
        """Whether the gate is at its limit right now (this holder included)."""
        return self._in_flight >= self._state.limit()

    def on_throttle(self) -> BackpressureEvent | None:
        """Feed a throttle outcome to the shared state (gated on saturation)."""
        return self._state.on_throttle(saturated=self.is_saturated())

    def on_success(self) -> BackpressureEvent | None:
        """Feed a success outcome to the shared state (drives recovery)."""
        return self._state.on_success()


class _Admit(enum.Enum):
    """A circuit breaker's verdict on one admission request.

    * ``NORMAL`` — admit and account the outcome into the rolling window (CLOSED).
    * ``PROBE`` — admit as the single HALF_OPEN probe; its outcome alone decides
      whether the breaker closes or re-opens.
    * ``REJECT`` — fast-fail with :class:`~llmkit.exceptions.CircuitOpenError`
      (OPEN within its cooldown, or HALF_OPEN with a probe already in flight).
    """

    NORMAL = "normal"
    PROBE = "probe"
    REJECT = "reject"


class _CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Per-provider circuit breaker over the shared throttle/success stream.

    The aggregate guard that complements AIMD: AIMD drives the per-provider limit
    *toward* 1 under sustained pushback; the breaker is the "limit is effectively
    0 for a cooldown" case — once a provider's throttle rate over a rolling window
    crosses the trip threshold it **opens**, and while open the limiter fast-fails
    every call with :class:`~llmkit.exceptions.CircuitOpenError` instead of letting
    each one burn its retry budget into the storm.

    One instance per provider *name* (like :class:`_AdaptiveState` and the
    RPM/TPM buckets — not per loop), shared across every event loop and sync
    thread. State is a ``CLOSED / OPEN / HALF_OPEN`` machine plus a count-based
    ring of the last :data:`_BREAKER_WINDOW` real outcomes:

    * **CLOSED** — admit normally; each real outcome (a throttle or a success;
      an ambiguous neutral error, like AIMD, records nothing) is appended to the
      ring. Once the ring is full and its throttled fraction reaches
      :data:`_BREAKER_THRESHOLD`, **open** (clearing the ring) and arm the
      cooldown.
    * **OPEN** — reject every admission for :data:`_BREAKER_COOLDOWN` seconds.
      The fast rejections feed nothing back (no real call ran).
    * **HALF_OPEN** — the first admission after the cooldown becomes a single,
      process-wide, lock-guarded **probe**; everyone else is rejected until it
      resolves. The probe's outcome alone decides: a clean success **closes** the
      breaker (and clears the ring); any failure — a throttle, a neutral error,
      or cancellation before/while it ran — **re-opens** it and re-arms the
      cooldown, so a probe can never wedge the breaker HALF_OPEN.

    State is guarded by a plain :class:`threading.Lock` (shared safely across
    loops and threads, exactly like :class:`_RateBucket` / :class:`_AdaptiveState`),
    held only for O(1) work and **never across an await**. Each mutator returns
    the :class:`BackpressureEvent` to emit (or ``None``) so the caller fires the
    observability callback *outside* the lock. The clock is read through
    :func:`_now`, so cooldown is deterministic under a monkeypatched clock.
    """

    def __init__(self, provider: str, ceiling: int) -> None:
        self._provider: str = provider
        # The breaker's own effective ceiling, reported on transitions so an
        # event reads ceiling → 0 (open) → 1 (one probe) → ceiling (closed).
        self._ceiling: int = ceiling
        self._state: _CircuitState = _CircuitState.CLOSED
        self._window: collections.deque[bool] = collections.deque(maxlen=_BREAKER_WINDOW)
        # When the breaker last opened, for the cooldown comparison. ``-inf`` is
        # unused while CLOSED (the state guard gates every read of it).
        self._opened_at: float = float("-inf")
        # Whether the single HALF_OPEN probe is currently out — the lock-guarded
        # flag that keeps a second concurrent call (on any loop) from probing too.
        self._probe_in_flight: bool = False
        self._lock: threading.Lock = threading.Lock()

    def admit(self) -> tuple[_Admit, BackpressureEvent | None]:
        """Decide whether to admit one call, transitioning OPEN → HALF_OPEN if the
        cooldown has elapsed. Returns the verdict and any transition event to emit.
        """
        with self._lock:
            if self._state is _CircuitState.CLOSED:
                return _Admit.NORMAL, None
            if self._state is _CircuitState.OPEN:
                if _now() - self._opened_at < _BREAKER_COOLDOWN:
                    return _Admit.REJECT, None
                # Cooldown elapsed: admit exactly one probe and announce HALF_OPEN.
                self._state = _CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return _Admit.PROBE, BackpressureEvent(self._provider, 0, 1, "breaker_half_open")
            # HALF_OPEN: the probe is in flight (or, defensively, just resolved on
            # another loop). Admit a probe only if none is out; otherwise reject.
            if self._probe_in_flight:
                return _Admit.REJECT, None
            self._probe_in_flight = True
            return _Admit.PROBE, None

    def on_record(self, *, throttled: bool) -> BackpressureEvent | None:
        """Account one *normal* (non-probe) call outcome; open if the ring trips.

        Only the CLOSED path accumulates the ring: a normal call that was admitted
        CLOSED but completes after the breaker has since opened (or a probe is
        underway) does not disturb the decision — the probe owns the transition.
        """
        with self._lock:
            if self._state is not _CircuitState.CLOSED:
                return None
            self._window.append(throttled)
            if len(self._window) < _BREAKER_MIN_SAMPLES:
                return None
            throttled_fraction = sum(self._window) / len(self._window)
            if throttled_fraction < _BREAKER_THRESHOLD:
                return None
            return self._open_locked()

    def on_probe_success(self) -> BackpressureEvent | None:
        """Resolve the HALF_OPEN probe with a clean success: close and clear."""
        with self._lock:
            self._probe_in_flight = False
            if self._state is not _CircuitState.HALF_OPEN:
                return None
            self._state = _CircuitState.CLOSED
            self._window.clear()
            return BackpressureEvent(self._provider, 1, self._ceiling, "breaker_closed")

    def on_probe_failure(self) -> BackpressureEvent | None:
        """Resolve the HALF_OPEN probe with *any* failure (throttle, neutral error,
        or cancellation before it ran): re-open and re-arm the cooldown.

        Always releases the probe claim, so the breaker can never wedge HALF_OPEN.
        """
        with self._lock:
            self._probe_in_flight = False
            if self._state is not _CircuitState.HALF_OPEN:
                return None
            return self._open_locked()

    def _open_locked(self) -> BackpressureEvent:
        # Caller holds ``self._lock``. Transition to OPEN, arm the cooldown, and
        # clear the ring so the next CLOSED period (after a probe closes it) starts
        # fresh. ``old_limit`` is the ceiling for every open — the breaker collapses
        # effective capacity to 0 regardless of whether it tripped from CLOSED or a
        # failed probe re-opened it.
        self._state = _CircuitState.OPEN
        self._opened_at = _now()
        self._window.clear()
        return BackpressureEvent(self._provider, self._ceiling, 0, "breaker_open")


class GlobalRateLimiter:
    """Process-global, per-provider rate limiter for LLM API calls.

    All state is class-level — this is a typed namespace, not a class you
    instantiate. Use the :meth:`acquire_async` / :meth:`acquire_sync` context
    managers, keyed by provider name (matched case-insensitively), to bound
    concurrency *and* (when configured) the request and token rate to each
    provider across the whole process.

    The async path (:meth:`acquire_async`) is what the LiteLLM call layer
    uses; the sync path (:meth:`acquire_sync`) is retained for synchronous
    LangChain-style chat-model wrappers whose ``_generate``/``_stream``
    methods cannot drive the async acquire.

    Each provider gets its own concurrency semaphore plus (when RPM/TPM are
    configured) its own request/token buckets, created on first touch and held
    in per-dimension registries keyed by provider name. Lazy creation is
    guarded by a ``threading.Lock`` so first-touch races can't construct
    competing limiters for the same key. Each acquirer snapshots its semaphore
    (and TPM bucket) locally, so a later :meth:`configure` swap (which clears
    the registries) does not strand in-flight callers — they release back onto,
    and debit, their own snapshot.

    The *async* concurrency semaphore is additionally keyed by the running event
    loop, because an ``asyncio.Semaphore`` binds to the loop it first blocks on
    and cannot be awaited from another. A process can have several loops — the
    persistent loop llmkit's sync bridge runs on (:func:`llmkit.sync.run_sync`),
    any loop a host drives its own async calls on, and the short-lived loops the
    sync bridge's reentrant fallback spins up — so a per-loop key keeps a
    saturated provider on one loop from raising "bound to a different event loop"
    on another; closed loops are pruned so they can't accumulate. A consequence
    is that the concurrency cap is enforced per (provider, loop): truly
    concurrent loops do not share one async semaphore (unavoidable — an asyncio
    primitive can't span loops). llmkit's own sync wrappers all share the *one*
    persistent loop, so its async semaphore bounds their cross-thread fan-out
    directly. Host code issuing sync provider calls *outside* that loop instead
    shares the loop-agnostic ``threading.Semaphore`` via :meth:`acquire_sync`.
    The multi-loop caveat (up to ``loops x max_concurrent`` momentarily, one cap
    per population) is documented in the module docstring.

    Enabled by default with a per-provider concurrency cap of 8; RPM and TPM
    are opt-in and off by default. See the module docstring for the rationale,
    the token-bucket choice, and the deliberate non-goals (single-tenant by
    design: no per-credential scoping, no extra global ceiling).
    """

    _lock: threading.Lock = threading.Lock()
    # Async concurrency gates are additionally keyed by the running event loop:
    # the gate's waiter futures bind to one loop and cannot be awaited from
    # another (the sync bridge runs one persistent loop, plus the reentrant
    # fallback's short-lived loops). The *adaptive limit* a gate enforces is
    # shared per provider (``_adaptive_states``, keyed by name like the buckets);
    # only the gate's in-flight count and waiter queue are per-loop. See
    # ``_get_async_gate``.
    _async_gates: dict[tuple[str, asyncio.AbstractEventLoop], _AdaptiveGate] = {}
    _adaptive_states: dict[str, _AdaptiveState] = {}
    # Circuit breakers are keyed by provider *name* alone (like the adaptive
    # states and buckets, not per-loop): the breaker's only state is the lock-
    # guarded outcome window, which has no loop affinity, so one breaker per
    # provider is shared across every loop and thread.
    _breaker_states: dict[str, _CircuitBreaker] = {}
    _sync_semaphores: dict[str, threading.Semaphore] = {}
    _rpm_buckets: dict[str, _RateBucket] = {}
    _tpm_buckets: dict[str, _RateBucket] = {}
    _max_concurrent: int = 8
    _rpm: int | None = None
    _tpm: int | None = None
    _enabled: bool = True
    _adaptive: bool = True
    _breaker: bool = False

    @classmethod
    def configure(
        cls,
        max_concurrent: int = 8,
        enabled: bool = True,
        rpm: int | None = None,
        tpm: int | None = None,
        adaptive: bool = True,
        breaker: bool = False,
    ) -> None:
        """Configure the global per-provider rate limit.

        Intended to be called once at startup before any LLM calls run.
        Calling again resets every dimension and clears all limiter registries
        (gates, adaptive states, circuit breakers, RPM/TPM buckets), so the new
        limits apply to *subsequent* acquires; in-flight callers continue to
        release on the gate object (and debit the TPM bucket) they snapshotted at
        acquire time, so reconfiguration is always safe.

        Args:
            max_concurrent: Maximum concurrent LLM API calls **per provider**.
                With ``adaptive`` on (the default) this is the *ceiling* the
                adaptive limit starts at and recovers toward — never exceeded.
            enabled: Whether rate limiting is active. When ``False``,
                :meth:`acquire_async`/:meth:`acquire_sync` are no-ops on every
                dimension.
            rpm: Sustained requests-per-minute **per provider**, or ``None``
                (the default) to leave the request-rate dimension off.
            tpm: Sustained tokens-per-minute **per provider**, or ``None``
                (the default) to leave the token-rate dimension off.
            adaptive: Whether the per-provider concurrency limit adapts (AIMD) to
                provider overload — on by default. When ``True``, a throttle
                signal (429/503/529) received while saturated halves the limit
                (floored at 1) and it recovers toward ``max_concurrent`` over time
                once throttling stops; with no throttles it sits at the ceiling,
                identical to a fixed cap. When ``False``, the limit is pinned at
                ``max_concurrent`` (the pre-feature behaviour).
            breaker: Whether the per-provider **circuit breaker** is armed — off
                by default. When ``True``, once a provider's throttle rate over a
                rolling window crosses the trip threshold the limiter fast-fails
                that provider's calls with
                :class:`~llmkit.exceptions.CircuitOpenError` for a cooldown (one
                HALF_OPEN probe then tests recovery), reading the *same* unwrapped
                outcome stream AIMD uses. Off by default because it flips
                "eventually succeeds" → "fails fast"; the host opts in.

        Raises:
            ValueError: if ``max_concurrent`` is non-positive, or ``rpm`` /
                ``tpm`` is set to a non-positive value. A gate with capacity 0
                would admit nothing, so without this check a ``max_concurrent=0``
                would silently hang every subsequent LLM call.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be a positive integer, got {max_concurrent!r}")
        if rpm is not None and rpm <= 0:
            raise ValueError(f"rpm must be a positive integer or None, got {rpm!r}")
        if tpm is not None and tpm <= 0:
            raise ValueError(f"tpm must be a positive integer or None, got {tpm!r}")
        with cls._lock:
            cls._max_concurrent = max_concurrent
            cls._enabled = enabled
            cls._rpm = rpm
            cls._tpm = tpm
            cls._adaptive = adaptive
            cls._breaker = breaker
            cls._async_gates = {}
            cls._adaptive_states = {}
            cls._breaker_states = {}
            cls._sync_semaphores = {}
            cls._rpm_buckets = {}
            cls._tpm_buckets = {}
        logger.info(
            "Configured LLM rate limit: max_concurrent=%d, rpm=%s, tpm=%s, enabled=%s, "
            + "adaptive=%s, breaker=%s",
            max_concurrent,
            rpm,
            tpm,
            enabled,
            adaptive,
            breaker,
        )

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether rate limiting is currently enabled."""
        return cls._enabled

    @classmethod
    def max_concurrent(cls) -> int:
        """The current per-provider concurrency cap (the symmetric read of the
        value ``configure`` sets)."""
        return cls._max_concurrent

    @classmethod
    def rpm(cls) -> int | None:
        """The current per-provider requests-per-minute limit, or ``None`` when
        the request-rate dimension is off."""
        return cls._rpm

    @classmethod
    def tpm(cls) -> int | None:
        """The current per-provider tokens-per-minute limit, or ``None`` when
        the token-rate dimension is off."""
        return cls._tpm

    @classmethod
    def adaptive(cls) -> bool:
        """Whether the per-provider concurrency limit adapts (AIMD) to overload."""
        return cls._adaptive

    @classmethod
    def breaker(cls) -> bool:
        """Whether the per-provider circuit breaker is armed (opt-in)."""
        return cls._breaker

    @classmethod
    def _get_breaker(cls, key: str) -> _CircuitBreaker:
        # One breaker per provider name, shared across loops/threads (the breaker
        # has no loop affinity — only a lock-guarded outcome window). Created at
        # the current ceiling; a ``configure`` swap clears this registry so a new
        # breaker picks up the new ceiling, exactly like ``_get_adaptive_state``.
        with cls._lock:
            breaker = cls._breaker_states.get(key)
            if breaker is None:
                breaker = _CircuitBreaker(provider=key, ceiling=cls._max_concurrent)
                cls._breaker_states[key] = breaker
            return breaker

    @classmethod
    def _get_adaptive_state_locked(cls, key: str) -> _AdaptiveState:
        # Caller holds ``cls._lock``. The adaptive state is shared per provider
        # *name* across loops (like the RPM/TPM buckets) — only the gate is
        # per-loop — so a throttle observed on any loop lowers the limit every
        # loop then sees, which is correct for a provider's shared server-side
        # capacity. Created at the current ceiling; a ``configure`` swap clears
        # this registry so a new state picks up a new ceiling.
        state = cls._adaptive_states.get(key)
        if state is None:
            state = _AdaptiveState(provider=key, ceiling=cls._max_concurrent)
            cls._adaptive_states[key] = state
        return state

    @classmethod
    def _get_async_gate(cls, key: str) -> _AdaptiveGate:
        # The gate's waiter futures bind to the loop they are created on, so a
        # process-global registry keyed by provider name alone would hand a
        # loop-A gate to loop B and crash under contention. A process can run
        # several loops — llmkit's persistent sync loop, a host's own async loop,
        # and the short-lived loops the sync bridge's reentrant fallback spins up
        # — so key it by (provider, running loop) and prune entries whose loop has
        # closed. The gate's *capacity* is the shared per-provider
        # ``_AdaptiveState``, so every loop enforces one adaptive limit even
        # though each gate's in-flight count is its own (the per-loop reality the
        # semaphore had).
        loop = asyncio.get_running_loop()
        with cls._lock:
            registry = cls._async_gates
            for stale in [k for k in registry if k[1].is_closed()]:
                del registry[stale]
            gate = registry.get((key, loop))
            if gate is None:
                gate = _AdaptiveGate(cls._get_adaptive_state_locked(key))
                registry[(key, loop)] = gate
            return gate

    @classmethod
    def _get_sync_semaphore(cls, key: str) -> threading.Semaphore:
        with cls._lock:
            sem = cls._sync_semaphores.get(key)
            if sem is None:
                sem = threading.Semaphore(cls._max_concurrent)
                cls._sync_semaphores[key] = sem
            return sem

    @classmethod
    def _get_rpm_bucket(cls, key: str) -> _RateBucket | None:
        """The provider's request-rate bucket, or ``None`` when RPM is off."""
        with cls._lock:
            rpm = cls._rpm
            if rpm is None:
                return None
            bucket = cls._rpm_buckets.get(key)
            if bucket is None:
                # Burst = the concurrency width, not a full minute's quota. The
                # RPM gate is passed *before* the concurrency semaphore, so
                # capacity bounds how many requests race to the provider before
                # the bucket throttles to the sustained rate; there is no point
                # letting that exceed ``max_concurrent`` (the burst llmkit
                # already tolerates). ``min`` with ``rpm`` keeps a mis-sized
                # ``max_concurrent`` > ``rpm`` from reintroducing a >2x overshoot.
                # Starting full at this smaller capacity lets a cold fan-out of up
                # to ``max_concurrent`` go immediately while an idle bucket can
                # never bank a whole minute (the ~2x-after-idle bug). See §6.4.
                capacity = float(min(cls._max_concurrent, rpm))
                bucket = _RateBucket(rate_per_sec=rpm / 60.0, capacity=capacity)
                cls._rpm_buckets[key] = bucket
            return bucket

    @classmethod
    def _get_tpm_bucket(cls, key: str) -> _RateBucket | None:
        """The provider's token-rate bucket, or ``None`` when TPM is off."""
        with cls._lock:
            tpm = cls._tpm
            if tpm is None:
                return None
            bucket = cls._tpm_buckets.get(key)
            if bucket is None:
                # Burst = a one-second reservoir, not a full minute of tokens.
                # TPM debits *after* the call and gates only while exhausted, so
                # even this small reservoir lets the first call (and a concurrent
                # first wave, bounded by ``max_concurrent``) through and then
                # smooths to the sustained rate; capacity only caps how much an
                # idle bucket banks, so a small one holds the per-window overshoot
                # to ~1.7% of ``tpm`` instead of the 2x a full-minute capacity
                # allowed after idle. See ``_TPM_BURST_SECONDS`` and §6.4.
                capacity = tpm * _TPM_BURST_SECONDS / 60.0
                bucket = _RateBucket(rate_per_sec=tpm / 60.0, capacity=capacity)
                cls._tpm_buckets[key] = bucket
            return bucket

    @classmethod
    @contextlib.asynccontextmanager
    async def acquire_async(cls, provider_key: str) -> AsyncGenerator[RateLimitSlot]:
        """Hold an async slot for ``provider_key`` for the ``async with`` block.

        ``provider_key`` is the provider name (``provider.name``, e.g.
        ``"OpenAI"``), matched case-insensitively (casefolded via
        :func:`_normalize_key` before touching any registry); each provider
        has an independent budget on every dimension. When the **circuit breaker**
        is armed it is consulted *first* — before any gate — so a known-open
        provider raises :class:`~llmkit.exceptions.CircuitOpenError` immediately,
        holding no slot and deducting no RPM token. Otherwise the call passes the
        request-rate (RPM) and token-rate (TPM) gates first — so a caller doesn't
        hold a scarce concurrency slot while waiting on a rate gate — then
        acquires the adaptive concurrency gate. Yields a :class:`RateLimitSlot`;
        call its :meth:`~RateLimitSlot.record_tokens` once the response's usage is
        known to debit the TPM budget. A no-op (yields an inert slot) when disabled.

        When ``adaptive`` is on, the call's outcome is observed at the ``yield``:
        a provider overload signal (429/503/529, **unwrapped** so a structured
        call's wrapped error is seen) received while saturated lowers the
        provider's limit; a success recovers it over time toward the ceiling. The
        breaker, when armed, reads the *same* unwrapped outcome: a throttle or
        success feeds its rolling window, and a HALF_OPEN probe's outcome decides
        whether it closes or re-opens. Cancellation while acquiring the
        rate/concurrency gates — before the slot is held — refunds the deducted
        RPM token (and releases a HALF_OPEN probe claim, re-opening the breaker so
        it never wedges).
        """
        # Unsynchronized read of _enabled is intentional: a concurrent
        # configure() flip only changes whether *this* call is bounded, and an
        # entered context always pairs its acquire with a release on the locally
        # snapshotted gate below, so no slot can leak across the swap.
        if not cls._enabled:
            yield RateLimitSlot()
            return
        provider_key = _normalize_key(provider_key)
        rpm_bucket = cls._get_rpm_bucket(provider_key)
        tpm_bucket = cls._get_tpm_bucket(provider_key)
        gate = cls._get_async_gate(provider_key)
        adaptive = cls._adaptive
        # Circuit breaker (opt-in) is consulted before any gate: a known-open
        # provider fast-fails here, before the RPM/TPM/concurrency phase below
        # deducts a token or holds a slot, so it occupies nothing while the
        # provider is down. ``admit`` may transition OPEN -> HALF_OPEN and hand
        # this call the single probe (``is_probe``); the probe's outcome alone
        # closes or re-opens the breaker.
        breaker = cls._get_breaker(provider_key) if cls._breaker else None
        is_probe = False
        if breaker is not None:
            decision, transition = breaker.admit()
            _emit_backpressure(transition)
            if decision is _Admit.REJECT:
                raise CircuitOpenError(provider_key)
            is_probe = decision is _Admit.PROBE
        # Acquire phase — RPM gate, then TPM gate, then the concurrency gate. If
        # cancelled here (after the RPM token is deducted but before a slot is
        # held), refund the token so a systematically-cancelled workload doesn't
        # silently shrink its own effective RPM — compounding the backpressure
        # that caused the cancellation. A probe cancelled here never ran, so it
        # is a probe failure: release the claim and re-open (no wedged breaker).
        rpm_debited = False
        try:
            if rpm_bucket is not None:
                await rpm_bucket.acquire_async(1.0)
                rpm_debited = True
            if tpm_bucket is not None:
                await tpm_bucket.wait_for_budget_async()
            await gate.acquire()
        except BaseException:
            if rpm_debited and rpm_bucket is not None:
                rpm_bucket.refund(1.0)
            if is_probe and breaker is not None:
                _emit_backpressure(breaker.on_probe_failure())
            raise
        # Slot held: classify the outcome for AIMD and the breaker (the exception,
        # if any, propagates back into this context manager at the ``yield``), then
        # always release the slot. Classification fires the backpressure callback
        # outside the gate/breaker locks. A non-throttle ("neutral") error is
        # ambiguous overload-wise, so — exactly as AIMD ignores it — it feeds
        # neither the window nor recovery; a probe, however, must resolve on *any*
        # outcome so it can never wedge HALF_OPEN.
        try:
            try:
                yield RateLimitSlot(tpm_bucket)
            except BaseException as exc:
                throttled = _is_throttle_signal(exc) if (adaptive or breaker is not None) else False
                if adaptive and throttled:
                    _emit_backpressure(gate.on_throttle())
                if breaker is not None:
                    if is_probe:
                        _emit_backpressure(breaker.on_probe_failure())
                    elif throttled:
                        _emit_backpressure(breaker.on_record(throttled=True))
                raise
            else:
                if adaptive:
                    _emit_backpressure(gate.on_success())
                if breaker is not None:
                    if is_probe:
                        _emit_backpressure(breaker.on_probe_success())
                    else:
                        _emit_backpressure(breaker.on_record(throttled=False))
        finally:
            gate.release()

    @classmethod
    @contextlib.contextmanager
    def acquire_sync(cls, provider_key: str) -> Generator[RateLimitSlot]:
        """Hold a sync slot for ``provider_key`` for the ``with`` block.

        The synchronous counterpart to :meth:`acquire_async`, with identical
        per-provider semantics across all three dimensions. ``provider_key`` is
        the provider name (``provider.name``, e.g. ``"OpenAI"``), matched
        case-insensitively. A no-op when disabled.
        """
        if not cls._enabled:
            yield RateLimitSlot()
            return
        provider_key = _normalize_key(provider_key)
        rpm_bucket = cls._get_rpm_bucket(provider_key)
        if rpm_bucket is not None:
            rpm_bucket.acquire_sync(1.0)
        tpm_bucket = cls._get_tpm_bucket(provider_key)
        if tpm_bucket is not None:
            tpm_bucket.wait_for_budget_sync()
        sem = cls._get_sync_semaphore(provider_key)
        _ = sem.acquire()
        try:
            yield RateLimitSlot(tpm_bucket)
        finally:
            sem.release()


@dataclass(frozen=True)
class RateLimitConfig:
    """A read-only snapshot of the effective rate-limit configuration.

    Returned by :func:`get_rate_limit_config` so a host can log or assert its
    effective limits at startup without reaching into limiter internals.
    ``rpm`` / ``tpm`` are ``None`` when those opt-in dimensions are off;
    ``adaptive`` is whether the concurrency limit adapts to provider overload;
    ``breaker`` is whether the per-provider circuit breaker is armed (opt-in).
    """

    enabled: bool
    max_concurrent: int
    rpm: int | None
    tpm: int | None
    adaptive: bool
    breaker: bool


def configure_rate_limit(
    max_concurrent: int = 8,
    enabled: bool = True,
    rpm: int | None = None,
    tpm: int | None = None,
    adaptive: bool = True,
    breaker: bool = False,
) -> None:
    """Configure the global, per-provider LLM rate limit.

    Rate limiting is on by default (concurrency cap 8 per provider, adaptive; RPM
    and TPM off; circuit breaker off). Call this to change the concurrency cap,
    turn limiting off, opt into a per-provider requests-/tokens-per-minute
    ceiling, turn off adaptive concurrency, or arm the circuit breaker. Call once
    at startup before any LLM calls run. When enabled, every provider call routed
    through the LiteLLM call layer passes through that provider's limiters.

    The binding limit on a metered cloud account is usually RPM/TPM rather than
    concurrency: set ``rpm=`` / ``tpm=`` to your account's published per-minute
    limits and llmkit smooths calls to stay under them (a token bucket per
    provider). Leaving them unset (the default) sends a byte-identical request
    to the pre-feature behaviour — the concurrency cap alone does not stand in
    for an RPM limit.

    Adaptive concurrency (``adaptive``, on by default) is the library-side
    generalization of a hand-tuned RPM ceiling: it *discovers* a safe concurrency
    by halving the per-provider limit on a provider overload signal (429/503/529)
    and recovering it toward ``max_concurrent`` once the provider stops pushing
    back — zero per-account tuning. It can only ever lower the limit *below*
    ``max_concurrent``, never above, so with no throttles it is identical to a
    fixed cap.

    The circuit breaker (``breaker``, **off** by default) is the aggregate guard
    for a provider that is *down*: armed, it fast-fails a provider's calls with
    :class:`~llmkit.exceptions.CircuitOpenError` once that provider's throttle
    rate over a rolling window crosses the trip threshold, for a cooldown, then
    lets a single probe test recovery. It is opt-in — unlike adaptive concurrency
    (which only ever *reduces* load and so is a safe default), the breaker flips
    "eventually succeeds" → "fails fast", so the host decides. ``CircuitOpenError``
    is in :data:`~llmkit.exceptions.LLM_RECOVERABLE_ERRORS` (a host's existing
    degrade-on-503 ``except`` keeps catching it) but never retried by the library.

    Args:
        max_concurrent: Maximum concurrent API calls **per provider** (the
            ceiling the adaptive limit starts at and recovers toward).
        enabled: Whether rate limiting is active.
        rpm: Sustained requests-per-minute **per provider**, or ``None`` to
            leave the request-rate dimension off (the default).
        tpm: Sustained tokens-per-minute **per provider**, or ``None`` to leave
            the token-rate dimension off (the default).
        adaptive: Whether the concurrency limit adapts to provider overload
            (on by default). ``False`` pins it at ``max_concurrent``.
        breaker: Whether the per-provider circuit breaker is armed (**off** by
            default). ``True`` makes a known-down provider fail fast for a
            cooldown instead of letting every call burn its retry budget.

    Raises:
        ValueError: if ``max_concurrent`` is non-positive, or ``rpm`` / ``tpm``
            is set to a non-positive value.
    """
    GlobalRateLimiter.configure(max_concurrent, enabled, rpm, tpm, adaptive, breaker)


def get_rate_limit_config() -> RateLimitConfig:
    """Read the effective rate-limit configuration.

    The symmetric read for :func:`configure_rate_limit`: lets a host log or
    assert its effective limits at startup without touching limiter internals
    (e.g. ``GlobalRateLimiter._max_concurrent``). The ``enabled`` flag on the
    returned snapshot is the public way to check whether limiting is active.

    Returns:
        A :class:`RateLimitConfig` snapshot of the current ``enabled`` flag,
        per-provider ``max_concurrent`` cap, per-provider ``rpm`` / ``tpm``
        limits (``None`` when those dimensions are off), the ``adaptive`` flag,
        and the ``breaker`` flag.
    """
    return RateLimitConfig(
        enabled=GlobalRateLimiter.is_enabled(),
        max_concurrent=GlobalRateLimiter.max_concurrent(),
        rpm=GlobalRateLimiter.rpm(),
        tpm=GlobalRateLimiter.tpm(),
        adaptive=GlobalRateLimiter.adaptive(),
        breaker=GlobalRateLimiter.breaker(),
    )


@contextlib.asynccontextmanager
async def rate_limit_acquire_async(provider_key: str) -> AsyncGenerator[RateLimitSlot]:
    """Hold an async slot on the global per-provider limit for the block.

    The public way to join the process-global, per-provider limit by hand — for
    a host that issues provider calls outside llmkit's own call functions (e.g.
    a LangChain chat-model wrapper) and still wants them bounded by the same
    concurrency / RPM / TPM budgets. ``provider_key`` is the provider name
    (``provider.name``, e.g. ``"OpenAI"``, ``"AWS Bedrock"``), matched
    **case-insensitively** — ``"openai"`` and ``"OpenAI"`` share one budget, so
    any casing joins the same budget llmkit's own calls debit. Each provider
    has an independent budget. Yields a :class:`RateLimitSlot`; call its
    :meth:`~RateLimitSlot.record_tokens` once you know the call's token usage to
    debit the TPM budget (a no-op when TPM is off). A no-op when limiting is
    disabled.

    Usage::

        async with rate_limit_acquire_async("OpenAI") as slot:
            response = ...  # one slot held against OpenAI's budget
            slot.record_tokens(response.usage.total_tokens)

    Behaviour is identical to the throttle llmkit's own async call path uses.
    """
    async with GlobalRateLimiter.acquire_async(provider_key) as slot:
        yield slot


@contextlib.contextmanager
def rate_limit_acquire_sync(provider_key: str) -> Generator[RateLimitSlot]:
    """Hold a sync slot on the global per-provider limit for the block.

    The synchronous counterpart to :func:`rate_limit_acquire_async`, for a host
    joining the global per-provider limit from a sync code path (e.g. a
    synchronous LangChain ``_generate``/``_stream`` wrapper). ``provider_key``
    is the provider name (``provider.name``, e.g. ``"OpenAI"``), matched
    **case-insensitively** — any casing joins the same budget llmkit's own
    calls debit. Each provider has an independent budget across all three
    dimensions. Yields a :class:`RateLimitSlot` whose
    :meth:`~RateLimitSlot.record_tokens` debits the TPM budget. A no-op when
    limiting is disabled.

    Usage::

        with rate_limit_acquire_sync("OpenAI") as slot:
            response = ...  # one slot held against OpenAI's budget
            slot.record_tokens(response.usage.total_tokens)
    """
    with GlobalRateLimiter.acquire_sync(provider_key) as slot:
        yield slot
