"""The process-global limiter namespace.

:class:`GlobalRateLimiter` owns every registry (per-provider RPM/TPM buckets,
per-(provider, loop) async gates, per-provider sync gates, AIMD states, circuit
breakers) and the two acquire context managers that walk them in order: breaker,
then RPM, then TPM, then concurrency.

**The class stays whole on purpose.** ``_get_adaptive_state_locked`` is called
with ``cls._lock`` already held, and that lock is a plain non-reentrant
``threading.Lock`` — relocating one of its callers and re-taking the lock would
deadlock, and contention-dependently at that.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncGenerator, Generator

import llmkit.rate_limiting._tuning as _tuning
from llmkit.exceptions import CircuitOpenError
from llmkit.rate_limiting._adaptive import AdaptiveGate, AdaptiveState, SyncAdaptiveGate
from llmkit.rate_limiting._breaker import Admit, CircuitBreaker
from llmkit.rate_limiting._buckets import RateBucket, RateLimitSlot
from llmkit.rate_limiting._observability import emit_backpressure, stamp_queue_wait
from llmkit.rate_limiting._outcomes import record_gate_outcome
from llmkit.rate_limiting._tuning import TPM_BURST_SECONDS

# Named explicitly rather than via ``__name__`` so every module in this package
# keeps emitting under the one logger name operators already filter on.
logger = logging.getLogger("llmkit.rate_limiting")


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
    methods cannot drive the async acquire. The two paths share their
    **backpressure state** — one per-provider :class:`AdaptiveState` (the AIMD
    limit) and one per-provider :class:`CircuitBreaker` — so a throttle observed
    on either lowers the limit the other honours and an open breaker fast-fails on
    both. They do **not** share the in-flight *count*: the async gate binds to a
    loop and threads are a third population, so each population caps
    independently (same-semantics-per-population, not one merged cap).

    Each provider gets its own concurrency gate plus (when RPM/TPM are
    configured) its own request/token buckets, created on first touch and held
    in per-dimension registries keyed by provider name. Lazy creation is
    guarded by a ``threading.Lock`` so first-touch races can't construct
    competing limiters for the same key. Each acquirer snapshots its gate
    (and TPM bucket) locally, so a later :meth:`configure` swap (which clears
    the registries) does not strand in-flight callers — they release back onto,
    and debit, their own snapshot.

    The *async* concurrency gate (:class:`AdaptiveGate`) is additionally keyed by
    the running event loop, because its waiter futures bind to the loop they are
    created on and cannot be awaited from another. A process can have several loops
    — the persistent loop llmkit's sync bridge runs on
    (:func:`llmkit.sync.run_sync`), any loop a host drives its own async calls on,
    and the short-lived loops the sync bridge's reentrant fallback spins up — so a
    per-loop key keeps a saturated provider on one loop from raising "bound to a
    different event loop" on another; closed loops are pruned so they can't
    accumulate. A consequence is that the concurrency cap is enforced per
    (provider, loop): truly concurrent loops do not share one async gate
    (unavoidable — an asyncio primitive can't span loops). llmkit's own sync
    wrappers all share the *one* persistent loop, so its async gate bounds their
    cross-thread fan-out directly. Host code issuing sync provider calls *outside*
    that loop instead shares the loop-agnostic per-provider
    :class:`SyncAdaptiveGate` via :meth:`acquire_sync`. Both gates read one shared
    per-provider :class:`AdaptiveState`, so the AIMD limit (and the breaker) is
    shared across every population even though each gate's in-flight *count* is its
    own. The multi-population caveat (up to ``populations x max_concurrent``
    momentarily, one cap per population) is documented in the module docstring.

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
    _async_gates: dict[tuple[str, asyncio.AbstractEventLoop], AdaptiveGate] = {}
    _adaptive_states: dict[str, AdaptiveState] = {}
    # Circuit breakers are keyed by provider *name* alone (like the adaptive
    # states and buckets, not per-loop): the breaker's only state is the lock-
    # guarded outcome window, which has no loop affinity, so one breaker per
    # provider is shared across every loop and thread.
    _breaker_states: dict[str, CircuitBreaker] = {}
    # Sync concurrency gates are keyed by provider *name* alone — threads have no
    # event-loop affinity, so unlike the async gate this needs no per-loop key.
    # Its capacity is the SAME shared per-provider ``AdaptiveState`` the async
    # gate reads, so a throttle on either path lowers the limit the other honours.
    # See ``_get_sync_gate``.
    _sync_gates: dict[str, SyncAdaptiveGate] = {}
    _rpm_buckets: dict[str, RateBucket] = {}
    _tpm_buckets: dict[str, RateBucket] = {}
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
            cls._sync_gates = {}
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
    def _get_breaker(cls, key: str) -> CircuitBreaker:
        # One breaker per provider name, shared across loops/threads (the breaker
        # has no loop affinity — only a lock-guarded outcome window). Created at
        # the current ceiling; a ``configure`` swap clears this registry so a new
        # breaker picks up the new ceiling, exactly like ``_get_adaptive_state``.
        with cls._lock:
            breaker = cls._breaker_states.get(key)
            if breaker is None:
                breaker = CircuitBreaker(provider=key, ceiling=cls._max_concurrent)
                cls._breaker_states[key] = breaker
            return breaker

    @classmethod
    def _get_adaptive_state_locked(cls, key: str) -> AdaptiveState:
        # Caller holds ``cls._lock``. The adaptive state is shared per provider
        # *name* across loops (like the RPM/TPM buckets) — only the gate is
        # per-loop — so a throttle observed on any loop lowers the limit every
        # loop then sees, which is correct for a provider's shared server-side
        # capacity. Created at the current ceiling; a ``configure`` swap clears
        # this registry so a new state picks up a new ceiling.
        state = cls._adaptive_states.get(key)
        if state is None:
            state = AdaptiveState(provider=key, ceiling=cls._max_concurrent)
            cls._adaptive_states[key] = state
        return state

    @classmethod
    def _get_async_gate(cls, key: str) -> AdaptiveGate:
        # The gate's waiter futures bind to the loop they are created on, so a
        # process-global registry keyed by provider name alone would hand a
        # loop-A gate to loop B and crash under contention. A process can run
        # several loops — llmkit's persistent sync loop, a host's own async loop,
        # and the short-lived loops the sync bridge's reentrant fallback spins up
        # — so key it by (provider, running loop) and prune entries whose loop has
        # closed. The gate's *capacity* is the shared per-provider
        # ``AdaptiveState``, so every loop enforces one adaptive limit even
        # though each gate's in-flight count is its own (the per-loop reality the
        # semaphore had).
        loop = asyncio.get_running_loop()
        with cls._lock:
            registry = cls._async_gates
            for stale in [k for k in registry if k[1].is_closed()]:
                del registry[stale]
            gate = registry.get((key, loop))
            if gate is None:
                gate = AdaptiveGate(cls._get_adaptive_state_locked(key))
                registry[(key, loop)] = gate
            return gate

    @classmethod
    def _get_sync_gate(cls, key: str) -> SyncAdaptiveGate:
        # One sync gate per provider name, shared across threads (threads have no
        # loop affinity, so unlike the async gate this needs no per-loop key). Its
        # capacity is the SAME shared per-provider ``AdaptiveState`` the async
        # gate reads, so a throttle observed on either path lowers the limit the
        # other honours. A ``configure`` swap clears this registry (and the shared
        # states) so a new gate picks up the new ceiling, exactly like
        # ``_get_async_gate``.
        with cls._lock:
            gate = cls._sync_gates.get(key)
            if gate is None:
                gate = SyncAdaptiveGate(cls._get_adaptive_state_locked(key))
                cls._sync_gates[key] = gate
            return gate

    @classmethod
    def _get_rpm_bucket(cls, key: str) -> RateBucket | None:
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
                bucket = RateBucket(rate_per_sec=rpm / 60.0, capacity=capacity)
                cls._rpm_buckets[key] = bucket
            return bucket

    @classmethod
    def _get_tpm_bucket(cls, key: str) -> RateBucket | None:
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
                # allowed after idle. See ``TPM_BURST_SECONDS`` and §6.4.
                capacity = tpm * TPM_BURST_SECONDS / 60.0
                bucket = RateBucket(rate_per_sec=tpm / 60.0, capacity=capacity)
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
            stamp_queue_wait(0.0)
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
            emit_backpressure(transition)
            if decision is Admit.REJECT:
                raise CircuitOpenError(provider_key)
            is_probe = decision is Admit.PROBE
        # Acquire phase — RPM gate, then TPM gate, then the concurrency gate. If
        # cancelled here (after the RPM token is deducted but before a slot is
        # held), refund the token so a systematically-cancelled workload doesn't
        # silently shrink its own effective RPM — compounding the backpressure
        # that caused the cancellation. A probe cancelled here never ran, so it
        # is a probe failure: release the claim and re-open (no wedged breaker).
        rpm_debited = False
        wait_start = _tuning.now()
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
                emit_backpressure(breaker.on_probe_failure())
            raise
        # Slot acquired: stamp how long this caller queued behind the RPM /
        # TPM / concurrency gates, for the call layer's log record.
        stamp_queue_wait((_tuning.now() - wait_start) * 1000.0)
        # Slot held: classify the outcome for AIMD and the breaker (the exception,
        # if any, propagates back into this context manager at the ``yield``), then
        # always release the slot. ``record_gate_outcome`` is the single shared
        # classifier :meth:`acquire_sync` also calls, so the feedback is one
        # implementation rather than two parallel copies that can drift.
        try:
            try:
                yield RateLimitSlot(tpm_bucket)
            except BaseException as exc:
                record_gate_outcome(
                    gate, breaker, is_probe=is_probe, adaptive=adaptive, outcome=exc
                )
                raise
            else:
                record_gate_outcome(
                    gate, breaker, is_probe=is_probe, adaptive=adaptive, outcome=None
                )
        finally:
            gate.release()

    @classmethod
    @contextlib.contextmanager
    def acquire_sync(cls, provider_key: str) -> Generator[RateLimitSlot]:
        """Hold a sync slot for ``provider_key`` for the ``with`` block.

        The synchronous counterpart to :meth:`acquire_async`, with the *same*
        per-provider backpressure semantics. ``provider_key`` is the provider name
        (``provider.name``, e.g. ``"OpenAI"``), matched case-insensitively
        (casefolded via :func:`_normalize_key`). When the **circuit breaker** is
        armed it is consulted *first* — before any gate, exactly as on the async
        path — so a known-open provider raises
        :class:`~llmkit.exceptions.CircuitOpenError` immediately, holding no slot
        and deducting no RPM token (a HALF_OPEN probe is tracked for the single
        process-wide probe). Otherwise the call passes the request-rate (RPM) and
        token-rate (TPM) gates first — so a caller doesn't hold a scarce
        concurrency slot while waiting on a rate gate — then acquires the adaptive
        concurrency gate (:class:`SyncAdaptiveGate`). Yields a
        :class:`RateLimitSlot`; call its :meth:`~RateLimitSlot.record_tokens` once
        the response's usage is known to debit the TPM budget. A no-op (yields an
        inert slot) when disabled.

        When ``adaptive`` is on, the call's outcome is observed at the ``yield``
        through the *same* :func:`record_gate_outcome` helper the async path uses:
        a provider overload signal (429/503/529, **unwrapped**) received while
        saturated lowers the provider's limit; a success recovers it. Because the
        concurrency gate reads the **shared** per-provider :class:`AdaptiveState`,
        a throttle observed here lowers the limit the *async* path then honours,
        and vice versa — the AIMD limit and the breaker are one per-provider state
        across both paths. The sync in-flight *count* is still its own population
        (an asyncio primitive cannot span loops, and threads are a third
        population), so this is same-semantics-per-population, not one merged cap.
        On any :class:`BaseException` while acquiring the rate/concurrency gates —
        before the slot is held — the deducted RPM token is refunded and a
        HALF_OPEN probe claim is resolved (re-opening the breaker so it never
        wedges); the slot release lives in ``finally``. Catching
        :class:`BaseException` (``KeyboardInterrupt`` is not an ``Exception``) is
        the permit-leak fix over the old fixed-semaphore path.
        """
        if not cls._enabled:
            yield RateLimitSlot()
            return
        provider_key = _normalize_key(provider_key)
        rpm_bucket = cls._get_rpm_bucket(provider_key)
        tpm_bucket = cls._get_tpm_bucket(provider_key)
        gate = cls._get_sync_gate(provider_key)
        adaptive = cls._adaptive
        # Circuit breaker (opt-in) is consulted before any gate, mirroring
        # ``acquire_async``: a known-open provider fast-fails here, before the
        # RPM/TPM/concurrency phase deducts a token or holds a slot. ``admit`` may
        # transition OPEN -> HALF_OPEN and hand this call the single probe
        # (``is_probe``); the breaker is the SAME object the async path consults.
        breaker = cls._get_breaker(provider_key) if cls._breaker else None
        is_probe = False
        if breaker is not None:
            decision, transition = breaker.admit()
            emit_backpressure(transition)
            if decision is Admit.REJECT:
                raise CircuitOpenError(provider_key)
            is_probe = decision is Admit.PROBE
        # Acquire phase — RPM gate, then TPM gate, then the concurrency gate. On
        # ANY BaseException here (after the RPM token is deducted but before a slot
        # is held), refund the token so a systematically-interrupted workload
        # doesn't silently shrink its own effective RPM, and resolve a probe claim
        # (a probe interrupted here never ran). ``BaseException`` — not
        # ``Exception`` — because ``KeyboardInterrupt`` is the permit-leak case the
        # old ``sem.acquire()``-outside-``try`` path could not cover.
        rpm_debited = False
        try:
            if rpm_bucket is not None:
                rpm_bucket.acquire_sync(1.0)
                rpm_debited = True
            if tpm_bucket is not None:
                tpm_bucket.wait_for_budget_sync()
            gate.acquire()
        except BaseException:
            if rpm_debited and rpm_bucket is not None:
                rpm_bucket.refund(1.0)
            if is_probe and breaker is not None:
                emit_backpressure(breaker.on_probe_failure())
            raise
        # Slot held: classify the outcome for AIMD and the breaker through the
        # single shared helper (identical to ``acquire_async``), then always
        # release the slot.
        try:
            try:
                yield RateLimitSlot(tpm_bucket)
            except BaseException as exc:
                record_gate_outcome(
                    gate, breaker, is_probe=is_probe, adaptive=adaptive, outcome=exc
                )
                raise
            else:
                record_gate_outcome(
                    gate, breaker, is_probe=is_probe, adaptive=adaptive, outcome=None
                )
        finally:
            gate.release()
