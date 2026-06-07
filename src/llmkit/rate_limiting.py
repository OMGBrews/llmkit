"""Global, per-provider concurrency limiting for LLM API calls.

Bounds the number of *concurrent* LLM calls the process makes to each
provider. The LiteLLM call layer (:mod:`llmkit._litellm`) wraps every
provider call in :meth:`GlobalRateLimiter.acquire_async`, keyed by the
**provider name**; the sync call path drives the same async coroutine, so
it inherits the throttle.

On by default, scoped per provider
----------------------------------
Rate limiting is **enabled out of the box** (``_enabled`` defaults to
``True``) — zero configuration needed. Slots are accounted **per
provider**, where the key is the provider *name* string
(``provider.name`` / :pyattr:`BaseProvider._provider_name`, e.g.
``"openai"``, ``"ollama"``). That is exactly the value logging records as
the *effective* provider (see
:func:`llmkit.structured_output._resolve_model_and_provider`), so a held
slot is always accounted to the provider that actually runs the call.

The default cap is **8 concurrent calls per provider**. Eight favours the
common access pattern — fan-out — out of the box: it still bounds a
self-inflicted burst (and the bill / 429s that follow) but doesn't quietly
serialise the multi-call workloads consumers actually run. A host on a
tightly-metered plan can lower it once via :func:`configure_rate_limit`; a
local Ollama server happy to fan out harder can raise it. A single cap value
applies to *all* providers; there is intentionally no per-provider cap map, to
keep the public surface small (one number, one switch).

Deliberate non-goals
--------------------
* **No per-credential scoping.** Two configs of the same provider family
  that differ only by ``api_key`` / ``base_url`` share one budget, because
  the key is the provider *name*, not the credential. This is a deliberate
  choice (the slot should match how logging resolves the provider), not an
  oversight — a future maintainer who needs credential-level isolation must
  opt into it explicitly.
* **No extra global ceiling.** There is no aggregate cap on top of the
  per-provider caps; total concurrency is just the sum of the per-provider
  budgets. Per-provider only.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """Process-global, per-provider rate limiter for LLM API calls.

    All state is class-level — this is a typed namespace, not a class you
    instantiate. Use the :meth:`acquire_async` / :meth:`acquire_sync`
    context managers, keyed by provider name, to bound concurrent calls to
    each provider across the whole process.

    The async path (:meth:`acquire_async`) is what the LiteLLM call layer
    uses; the sync path (:meth:`acquire_sync`) is retained for synchronous
    LangChain-style chat-model wrappers whose ``_generate``/``_stream``
    methods cannot drive the async acquire.

    Each provider gets its own semaphore, created on first touch and held in
    the ``_async_semaphores`` / ``_sync_semaphores`` registries (keyed by
    provider name). Lazy creation is guarded by a ``threading.Lock`` so
    first-touch races can't construct competing semaphores for the same key.
    Each acquirer snapshots its semaphore object locally, so a later
    :meth:`configure` swap (which clears the registries) does not strand
    in-flight callers on a different semaphore than the one they acquired
    from — they release back onto their own snapshot.

    Enabled by default with a per-provider cap of 8; see the module
    docstring for the rationale and the deliberate non-goals (no
    per-credential scoping, no extra global ceiling).
    """

    _lock: threading.Lock = threading.Lock()
    _async_semaphores: dict[str, asyncio.Semaphore] = {}
    _sync_semaphores: dict[str, threading.Semaphore] = {}
    _max_concurrent: int = 8
    _enabled: bool = True

    @classmethod
    def configure(cls, max_concurrent: int = 8, enabled: bool = True) -> None:
        """Configure the global per-provider rate limit.

        Intended to be called once at startup before any LLM calls run.
        Calling again resets the cap/enabled flag and clears both semaphore
        registries, so the new cap applies to *subsequent* acquires; in-flight
        callers continue to release on the semaphore objects they snapshotted
        at acquire time, so reconfiguration is always safe.

        Args:
            max_concurrent: Maximum concurrent LLM API calls **per provider**.
            enabled: Whether rate limiting is active. When ``False``,
                :meth:`acquire_async`/:meth:`acquire_sync` are no-ops.
        """
        with cls._lock:
            cls._max_concurrent = max_concurrent
            cls._enabled = enabled
            cls._async_semaphores = {}
            cls._sync_semaphores = {}
        logger.info(
            "Configured global LLM rate limit: max_concurrent=%d (per provider), enabled=%s",
            max_concurrent,
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
    def _get_async_semaphore(cls, key: str) -> asyncio.Semaphore:
        with cls._lock:
            sem = cls._async_semaphores.get(key)
            if sem is None:
                sem = asyncio.Semaphore(cls._max_concurrent)
                cls._async_semaphores[key] = sem
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
    @contextlib.asynccontextmanager
    async def acquire_async(cls, provider_key: str) -> AsyncIterator[None]:
        """Hold an async slot for ``provider_key`` for the ``async with`` block.

        ``provider_key`` is the provider name (``provider.name``); each
        provider has an independent budget. A no-op when disabled.
        """
        # Unsynchronized read of _enabled is intentional: a concurrent
        # configure() flip only changes whether *this* call is bounded, and an
        # entered context always pairs its acquire with a release on the locally
        # snapshotted semaphore below, so no slot can leak across the swap.
        if not cls._enabled:
            yield
            return
        sem = cls._get_async_semaphore(provider_key)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    @classmethod
    @contextlib.contextmanager
    def acquire_sync(cls, provider_key: str) -> Iterator[None]:
        """Hold a sync slot for ``provider_key`` for the ``with`` block.

        ``provider_key`` is the provider name (``provider.name``); each
        provider has an independent budget. A no-op when disabled.
        """
        if not cls._enabled:
            yield
            return
        sem = cls._get_sync_semaphore(provider_key)
        sem.acquire()
        try:
            yield
        finally:
            sem.release()


@dataclass(frozen=True)
class RateLimitConfig:
    """A read-only snapshot of the effective rate-limit configuration.

    Returned by :func:`get_rate_limit_config` so a host can log or assert its
    effective limits at startup without reaching into limiter internals.
    """

    enabled: bool
    max_concurrent: int


def configure_rate_limit(max_concurrent: int = 8, enabled: bool = True) -> None:
    """Configure the global, per-provider LLM rate limit.

    Rate limiting is on by default (cap 8 per provider); call this only to
    change the cap or turn it off. Call once at startup before any LLM calls
    run. When enabled, every provider call routed through the LiteLLM call
    layer passes through that provider's semaphore.

    Args:
        max_concurrent: Maximum concurrent API calls **per provider**.
        enabled: Whether rate limiting is active.
    """
    GlobalRateLimiter.configure(max_concurrent, enabled)


def get_rate_limit_config() -> RateLimitConfig:
    """Read the effective rate-limit configuration.

    The symmetric read for :func:`configure_rate_limit`: lets a host log or
    assert its effective limits at startup without touching limiter internals
    (e.g. ``GlobalRateLimiter._max_concurrent``). The ``enabled`` flag on the
    returned snapshot is the public way to check whether limiting is active.

    Returns:
        A :class:`RateLimitConfig` snapshot of the current ``enabled`` flag and
        per-provider ``max_concurrent`` cap.
    """
    return RateLimitConfig(
        enabled=GlobalRateLimiter.is_enabled(),
        max_concurrent=GlobalRateLimiter.max_concurrent(),
    )


@contextlib.asynccontextmanager
async def rate_limit_acquire_async(provider_key: str) -> AsyncIterator[None]:
    """Hold an async slot on the global per-provider limit for the block.

    The public way to join the process-global, per-provider concurrency limit
    by hand — for a host that issues provider calls outside llmkit's own call
    functions (e.g. a LangChain chat-model wrapper) and still wants them
    bounded by the same budget. ``provider_key`` is the provider name
    (``provider.name``, e.g. ``"openai"``); each provider has an independent
    budget capped at :func:`get_rate_limit_config`'s ``max_concurrent`` (8 by
    default). A no-op when limiting is disabled.

    Usage::

        async with rate_limit_acquire_async("openai"):
            ...  # one slot held against openai's budget

    Behaviour is identical to the throttle llmkit's own async call path uses.
    """
    async with GlobalRateLimiter.acquire_async(provider_key):
        yield


@contextlib.contextmanager
def rate_limit_acquire_sync(provider_key: str) -> Iterator[None]:
    """Hold a sync slot on the global per-provider limit for the block.

    The synchronous counterpart to :func:`rate_limit_acquire_async`, for a host
    joining the global per-provider limit from a sync code path (e.g. a
    synchronous LangChain ``_generate``/``_stream`` wrapper). ``provider_key``
    is the provider name (``provider.name``); each provider has an independent
    budget capped at :func:`get_rate_limit_config`'s ``max_concurrent`` (8 by
    default). A no-op when limiting is disabled.

    Usage::

        with rate_limit_acquire_sync("openai"):
            ...  # one slot held against openai's budget
    """
    with GlobalRateLimiter.acquire_sync(provider_key):
        yield
