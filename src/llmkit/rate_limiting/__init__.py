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

One honest caveat follows from the concurrency gate being per-population: a
process that drives calls on more than one population — e.g. a host running
llmkit's *async* call functions on its own event loop, llmkit's sync wrappers
(on the persistent loop), and the hand-rolled sync path from its own threads —
caps each population independently, so it can momentarily hold up to
``populations x max_concurrent`` in-flight calls per provider. Each population
is capped; the caps do not share *slots*, because an asyncio primitive cannot
span loops and threads are a third population with no loop at all. The
hand-rolled sync acquire path (:meth:`acquire_sync` /
:func:`rate_limit_acquire_sync`) is bounded by a loop-agnostic per-provider
:class:`SyncAdaptiveGate` — for host code that issues sync provider calls
*outside* llmkit's own loop.

What the sync path *does* now share with the async path — the parity this
module deliberately provides — is the **backpressure state**, not the in-flight
count. Both paths read one per-provider :class:`AdaptiveState` (the AIMD limit)
and one per-provider :class:`CircuitBreaker`, so a throttle observed on either
path lowers the limit the *other* then honours, and an open breaker fast-fails
on both. That is same-semantics-per-population, not one merged cap: the sync
in-flight count is its own population (threads), but the *limit* it enforces and
the breaker it consults are shared. And AIMD *saturation* — the judgment that
decides whether a throttle should halve the limit — is measured on the
provider-wide **aggregate** in-flight (the sum across every async gate and the
sync gate, tracked on the shared :class:`AdaptiveState`), not any one gate's
local count: a self-inflicted 429 while the populations *together* sit at the
shared limit therefore triggers a decrease even when no single gate is itself at
the cap. Admission stays per-population (the capacity caveat above); only the
saturation judgment aggregates. RPM/TPM are likewise unaffected by
population: those buckets are keyed by name and shared across every loop and
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
surface small (three numbers, three switches: ``enabled`` / ``adaptive`` /
``breaker``).

Token bucket, not sliding window
--------------------------------
RPM and TPM use a **token bucket** (:class:`RateBucket`), not a sliding
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
(``TPM_BURST_SECONDS``), so the worst-case overshoot is a small constant above
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

Module layout
-------------

This module is the package facade: the design record above, the public
configuration and acquire surface, and nothing else. The mechanisms live in
private siblings, one per concern, so the layout itself says what is here —
the circuit breaker in particular was invisible in the single-file form:

* ``_tuning`` — the internal constants and the shared monotonic clock (a test
  seam; see that module before moving a clock read);
* ``_observability`` — backpressure events and the per-attempt queue-wait
  stamp, both context-scoped and both identity-critical;
* ``_buckets`` — the *rate* dimension (RPM/TPM token buckets) and
  :class:`RateLimitSlot`;
* ``_adaptive`` — the *concurrency* dimension: AIMD state and its async and
  sync gates;
* ``_breaker`` — the opt-in circuit breaker;
* ``_outcomes`` — the one throttle classifier and outcome-feedback path both
  acquire paths share, so their reactions cannot drift;
* ``_limiter`` — :class:`GlobalRateLimiter`, the process-global registries and
  the two acquire context managers.

Symbols inside those private modules carry public names because they cross a
module boundary; the package's public surface is exactly ``__all__`` below.
"""

import contextlib
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

from llmkit.rate_limiting._buckets import RateLimitSlot
from llmkit.rate_limiting._limiter import GlobalRateLimiter
from llmkit.rate_limiting._observability import (
    BackpressureCallback,
    BackpressureEvent,
    backpressure_callback,
    begin_queue_wait,
    current_queue_wait_ms,
)

__all__ = [
    "BackpressureCallback",
    "BackpressureEvent",
    "GlobalRateLimiter",
    "RateLimitConfig",
    "RateLimitSlot",
    "backpressure_callback",
    "begin_queue_wait",
    "configure_rate_limit",
    "current_queue_wait_ms",
    "get_rate_limit_config",
    "rate_limit_acquire_async",
    "rate_limit_acquire_sync",
]


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
