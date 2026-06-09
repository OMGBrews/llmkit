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
:meth:`GlobalRateLimiter.acquire_async`, keyed by the **provider name**; the
sync call path drives the same async coroutine, so it inherits the throttle.

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
import contextlib
import logging
import threading
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

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
    :class:`threading.Lock`, held only for the non-blocking refill-and-compare
    arithmetic — never across a sleep — so a single bucket is safe to share
    between the async and sync acquire paths (one true per-provider budget).
    """

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self._rate: float = rate_per_sec
        self._capacity: float = capacity
        self._level: float = capacity
        self._updated: float = _now()
        self._lock: threading.Lock = threading.Lock()

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

    async def acquire_async(self, cost: float) -> None:
        while (wait := self._try_acquire(cost)) > 0:
            await asyncio.sleep(wait)

    def acquire_sync(self, cost: float) -> None:
        while (wait := self._try_acquire(cost)) > 0:
            time.sleep(wait)

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
    and cannot be awaited from another. The sync bridge runs a fresh loop per
    call, so a per-loop key is what keeps a saturated provider from raising
    "bound to a different event loop" when the next sync call (or a mixed
    async/sync app) reuses the registry; closed loops are pruned so they can't
    accumulate. A consequence is that the concurrency cap is enforced per
    (provider, loop): truly concurrent loops in different threads do not share
    one async semaphore — which is unavoidable, since an asyncio primitive can't
    span loops. Cross-thread *sync* callers share the loop-agnostic
    ``threading.Semaphore`` and so do share one cap.

    Enabled by default with a per-provider concurrency cap of 8; RPM and TPM
    are opt-in and off by default. See the module docstring for the rationale,
    the token-bucket choice, and the deliberate non-goals (single-tenant by
    design: no per-credential scoping, no extra global ceiling).
    """

    _lock: threading.Lock = threading.Lock()
    # Async semaphores are additionally keyed by the running event loop: an
    # ``asyncio.Semaphore`` binds to the loop it first blocks on and cannot be
    # awaited from another (the sync bridge runs a fresh loop per call). See
    # ``_get_async_semaphore``.
    _async_semaphores: dict[tuple[str, asyncio.AbstractEventLoop], asyncio.Semaphore] = {}
    _sync_semaphores: dict[str, threading.Semaphore] = {}
    _rpm_buckets: dict[str, _RateBucket] = {}
    _tpm_buckets: dict[str, _RateBucket] = {}
    _max_concurrent: int = 8
    _rpm: int | None = None
    _tpm: int | None = None
    _enabled: bool = True

    @classmethod
    def configure(
        cls,
        max_concurrent: int = 8,
        enabled: bool = True,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> None:
        """Configure the global per-provider rate limit.

        Intended to be called once at startup before any LLM calls run.
        Calling again resets every dimension and clears all limiter registries,
        so the new limits apply to *subsequent* acquires; in-flight callers
        continue to release on the semaphore objects (and debit the TPM bucket)
        they snapshotted at acquire time, so reconfiguration is always safe.

        Args:
            max_concurrent: Maximum concurrent LLM API calls **per provider**.
            enabled: Whether rate limiting is active. When ``False``,
                :meth:`acquire_async`/:meth:`acquire_sync` are no-ops on every
                dimension.
            rpm: Sustained requests-per-minute **per provider**, or ``None``
                (the default) to leave the request-rate dimension off.
            tpm: Sustained tokens-per-minute **per provider**, or ``None``
                (the default) to leave the token-rate dimension off.

        Raises:
            ValueError: if ``max_concurrent`` is non-positive, or ``rpm`` /
                ``tpm`` is set to a non-positive value. ``Semaphore(0)`` is a
                legal, permanently-locked semaphore, so without this check a
                ``max_concurrent=0`` would be accepted here and silently hang
                every subsequent LLM call.
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
            cls._async_semaphores = {}
            cls._sync_semaphores = {}
            cls._rpm_buckets = {}
            cls._tpm_buckets = {}
        logger.info(
            "Configured LLM rate limit: max_concurrent=%d/provider, rpm=%s, tpm=%s, enabled=%s",
            max_concurrent,
            rpm,
            tpm,
            enabled,
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
    def _get_async_semaphore(cls, key: str) -> asyncio.Semaphore:
        # An ``asyncio.Semaphore`` lazily binds to the event loop it first
        # *blocks* on and thereafter raises ``RuntimeError`` ("bound to a
        # different event loop") if awaited from another. The sync bridge spins
        # up a fresh loop for every ``run_sync`` call, and an app may mix async
        # and sync entry points, so a process-global registry keyed by provider
        # name alone would hand a loop-A semaphore to loop B and crash under
        # contention. Key it by (provider, running loop) instead, and prune
        # entries whose loop has closed so the short-lived loops the sync bridge
        # retires can't accumulate without bound.
        loop = asyncio.get_running_loop()
        with cls._lock:
            registry = cls._async_semaphores
            for stale in [k for k in registry if k[1].is_closed()]:
                del registry[stale]
            sem = registry.get((key, loop))
            if sem is None:
                sem = asyncio.Semaphore(cls._max_concurrent)
                registry[(key, loop)] = sem
            return sem

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
        has an independent budget on every dimension. Passes the request-rate
        (RPM) and token-rate (TPM) gates first — so a caller doesn't hold a
        scarce concurrency slot while waiting on a rate gate — then acquires the
        concurrency semaphore. Yields a :class:`RateLimitSlot`; call its
        :meth:`~RateLimitSlot.record_tokens` once the response's usage is known
        to debit the TPM budget. A no-op (yields an inert slot) when disabled.
        """
        # Unsynchronized read of _enabled is intentional: a concurrent
        # configure() flip only changes whether *this* call is bounded, and an
        # entered context always pairs its acquire with a release on the locally
        # snapshotted semaphore below, so no slot can leak across the swap.
        if not cls._enabled:
            yield RateLimitSlot()
            return
        provider_key = _normalize_key(provider_key)
        rpm_bucket = cls._get_rpm_bucket(provider_key)
        if rpm_bucket is not None:
            await rpm_bucket.acquire_async(1.0)
        tpm_bucket = cls._get_tpm_bucket(provider_key)
        if tpm_bucket is not None:
            await tpm_bucket.wait_for_budget_async()
        sem = cls._get_async_semaphore(provider_key)
        async with sem:
            yield RateLimitSlot(tpm_bucket)

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
    ``rpm`` / ``tpm`` are ``None`` when those opt-in dimensions are off.
    """

    enabled: bool
    max_concurrent: int
    rpm: int | None
    tpm: int | None


def configure_rate_limit(
    max_concurrent: int = 8,
    enabled: bool = True,
    rpm: int | None = None,
    tpm: int | None = None,
) -> None:
    """Configure the global, per-provider LLM rate limit.

    Rate limiting is on by default (concurrency cap 8 per provider; RPM and TPM
    off). Call this to change the concurrency cap, turn limiting off, or opt
    into a per-provider requests-/tokens-per-minute ceiling. Call once at
    startup before any LLM calls run. When enabled, every provider call routed
    through the LiteLLM call layer passes through that provider's limiters.

    The binding limit on a metered cloud account is usually RPM/TPM rather than
    concurrency: set ``rpm=`` / ``tpm=`` to your account's published per-minute
    limits and llmkit smooths calls to stay under them (a token bucket per
    provider). Leaving them unset (the default) sends a byte-identical request
    to the pre-feature behaviour — the concurrency cap alone does not stand in
    for an RPM limit.

    Args:
        max_concurrent: Maximum concurrent API calls **per provider**.
        enabled: Whether rate limiting is active.
        rpm: Sustained requests-per-minute **per provider**, or ``None`` to
            leave the request-rate dimension off (the default).
        tpm: Sustained tokens-per-minute **per provider**, or ``None`` to leave
            the token-rate dimension off (the default).

    Raises:
        ValueError: if ``max_concurrent`` is non-positive, or ``rpm`` / ``tpm``
            is set to a non-positive value.
    """
    GlobalRateLimiter.configure(max_concurrent, enabled, rpm, tpm)


def get_rate_limit_config() -> RateLimitConfig:
    """Read the effective rate-limit configuration.

    The symmetric read for :func:`configure_rate_limit`: lets a host log or
    assert its effective limits at startup without touching limiter internals
    (e.g. ``GlobalRateLimiter._max_concurrent``). The ``enabled`` flag on the
    returned snapshot is the public way to check whether limiting is active.

    Returns:
        A :class:`RateLimitConfig` snapshot of the current ``enabled`` flag,
        per-provider ``max_concurrent`` cap, and per-provider ``rpm`` / ``tpm``
        limits (``None`` when those dimensions are off).
    """
    return RateLimitConfig(
        enabled=GlobalRateLimiter.is_enabled(),
        max_concurrent=GlobalRateLimiter.max_concurrent(),
        rpm=GlobalRateLimiter.rpm(),
        tpm=GlobalRateLimiter.tpm(),
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
