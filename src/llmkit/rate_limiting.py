"""Global rate limiting for LLM API calls.

Provides a process-global async semaphore that bounds the number of
concurrent LLM calls across the whole process. The LiteLLM call layer
(:mod:`llmkit._litellm`) wraps every provider call in
:meth:`GlobalRateLimiter.acquire_async`; the sync call path drives the
same async coroutine, so it inherits the throttle.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """Process-global rate limiter for LLM API calls.

    All state is class-level — this is a typed namespace, not a class
    you instantiate. Use the :meth:`acquire_async` / :meth:`acquire_sync`
    context managers to serialize concurrent calls across the whole
    process.

    The async path (:meth:`acquire_async`) is what the LiteLLM call layer
    uses; the sync path (:meth:`acquire_sync`) is retained for the
    eval-owned LangChain chat-model wrappers (``evals/shared``), whose
    synchronous ``_generate``/``_stream`` cannot drive the async acquire.

    Lazy initialization is guarded by a ``threading.Lock`` so first-touch
    races can't construct competing semaphores. Each acquirer snapshots
    the semaphore object locally, so a later :meth:`configure` swap does
    not strand in-flight callers on a different semaphore than the one
    they acquired from.
    """

    _lock: threading.Lock = threading.Lock()
    _async_semaphore: asyncio.Semaphore | None = None
    _sync_semaphore: threading.Semaphore | None = None
    _max_concurrent: int = 2
    _enabled: bool = False

    @classmethod
    def configure(cls, max_concurrent: int = 2, enabled: bool = True) -> None:
        """Configure the global rate limit.

        Intended to be called once at startup before any LLM calls run.
        Calling again resets the underlying semaphores; in-flight callers
        continue to release on the prior semaphore objects (snapshotted at
        acquire time), so reconfiguration is safe but the new limit only
        applies to acquires that follow.

        Args:
            max_concurrent: Maximum number of concurrent LLM API calls.
            enabled: Whether rate limiting is active. When ``False``,
                :meth:`acquire_async`/:meth:`acquire_sync` are no-ops.
        """
        with cls._lock:
            cls._max_concurrent = max_concurrent
            cls._enabled = enabled
            cls._async_semaphore = None
            cls._sync_semaphore = None
        logger.info(
            "Configured global LLM rate limit: max_concurrent=%d, enabled=%s",
            max_concurrent,
            enabled,
        )

    @classmethod
    def is_enabled(cls) -> bool:
        """Whether rate limiting is currently enabled."""
        return cls._enabled

    @classmethod
    def _get_async_semaphore(cls) -> asyncio.Semaphore:
        with cls._lock:
            if cls._async_semaphore is None:
                cls._async_semaphore = asyncio.Semaphore(cls._max_concurrent)
            return cls._async_semaphore

    @classmethod
    def _get_sync_semaphore(cls) -> threading.Semaphore:
        with cls._lock:
            if cls._sync_semaphore is None:
                cls._sync_semaphore = threading.Semaphore(cls._max_concurrent)
            return cls._sync_semaphore

    @classmethod
    @contextlib.asynccontextmanager
    async def acquire_async(cls) -> AsyncIterator[None]:
        """Hold an async slot for the duration of the ``async with`` block."""
        if not cls._enabled:
            yield
            return
        sem = cls._get_async_semaphore()
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    @classmethod
    @contextlib.contextmanager
    def acquire_sync(cls) -> Iterator[None]:
        """Hold a sync slot for the duration of the ``with`` block."""
        if not cls._enabled:
            yield
            return
        sem = cls._get_sync_semaphore()
        sem.acquire()
        try:
            yield
        finally:
            sem.release()


def configure_rate_limit(max_concurrent: int = 2, enabled: bool = True) -> None:
    """Configure the global LLM rate limit.

    Call once at startup before any LLM calls run. When enabled, every
    provider call routed through the LiteLLM call layer passes through a
    shared semaphore.

    Args:
        max_concurrent: Maximum concurrent API calls.
        enabled: Whether rate limiting is active.
    """
    GlobalRateLimiter.configure(max_concurrent, enabled)
