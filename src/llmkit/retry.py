"""Composable async retry utility.

Audit logging, timing, and error recovery are the caller's concern.
"""

import asyncio
import contextvars
import logging
import random
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

logger = logging.getLogger(__name__)


class RetryProgressCallback(Protocol):
    """Receives per-attempt failure events from :func:`with_retries`.

    Invoked once per non-final failed attempt. The final-failure case
    does not call back — callers learn about exhaustion by catching the
    re-raised exception. Implementations must not raise.
    """

    def __call__(
        self,
        *,
        label: str,
        attempt: int,
        max_retries: int,
        error: BaseException,
    ) -> None: ...


_progress_callback: contextvars.ContextVar[RetryProgressCallback | None] = contextvars.ContextVar(
    "_retry_progress_callback", default=None
)


@contextmanager
def retry_progress_callback(callback: RetryProgressCallback | None) -> Iterator[None]:
    """Install a progress callback for retries within this dynamic scope.

    The callback is read by :func:`with_retries` from a context variable,
    so it propagates across ``asyncio.to_thread`` and ``asyncio.run``
    boundaries without threading a parameter through every caller. Set
    ``callback=None`` to explicitly disable callbacks within an inner
    scope.
    """
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 1,
    label: str | None = None,
    backoff_base_seconds: float = 0.0,
) -> T:
    """Retry an async callable up to *max_retries* times.

    Args:
        fn: Zero-argument async callable to execute.
        max_retries: Total number of attempts (1 = no retry).
        label: Optional identifier for log messages (e.g. an op_id).
        backoff_base_seconds: When > 0, sleep before each retry using
            exponential "full jitter" backoff — a random delay in
            ``[0, backoff_base_seconds * 2**(attempt-1)]``. Defaults to
            0.0 (retry immediately), which preserves prior behaviour for
            callers that don't opt in. Jitter spreads concurrent retries
            so a transient provider-saturation window (the dominant
            failure mode for the eval fan-out) isn't re-hit by every
            caller at once.

    Returns:
        The result of the first successful call.

    Raises:
        The exception from the last failed attempt if all retries
        are exhausted.
    """
    tag = label or "retry"
    last_error: Exception = RuntimeError(f"{tag}: no attempts made (max_retries={max_retries})")

    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning("%s: attempt %d/%d failed: %s", tag, attempt, max_retries, e)
                callback = _progress_callback.get()
                if callback is not None:
                    try:
                        callback(label=tag, attempt=attempt, max_retries=max_retries, error=e)
                    except Exception:
                        logger.exception("%s: retry progress callback raised", tag)
                if backoff_base_seconds > 0:
                    ceiling = backoff_base_seconds * (2 ** (attempt - 1))
                    await asyncio.sleep(random.uniform(0, ceiling))
            else:
                logger.error("%s: all %d attempts failed: %s", tag, max_retries, e)

    raise last_error
