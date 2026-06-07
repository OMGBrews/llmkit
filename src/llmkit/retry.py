"""Async transient-error retry layer.

The call functions in :mod:`llmkit.structured_output` retry *transient*
provider errors on their own by default (see :class:`RetryPolicy`); this
module holds the loop they share. :func:`with_retries` is also exported as
the explicit, composable advanced path a caller can wrap any awaitable in.

Audit logging and timing remain the caller's concern: each attempt is its
own LLM call (and its own log record), because the retry loop wraps the
logging call functions rather than living inside them.
"""

import asyncio
import contextvars
import logging
import random
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from llmkit.exceptions import LLM_RECOVERABLE_ERRORS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """The transient-error retry budget a call function applies by default.

    Realizes the "transient retries are on by default" opinion: the call
    functions (:func:`~llmkit.structured_llm_call`,
    :func:`~llmkit.text_llm_call`, :func:`~llmkit.stream_text_with_log`,
    and the sync wrapper) retry the ``retry_on`` errors with bounded
    full-jitter backoff, without the caller wrapping every call.

    This layer is kept deliberately separate from instructor's in-call
    schema-repair budget (``validation_retries``, default 1) — the two are
    never conflated, so attempts are not double-counted *within* one call.
    Note the layering at the seam: when instructor exhausts its own repair
    budget it raises ``InstructorRetryException``, which is itself in
    ``LLM_RECOVERABLE_ERRORS`` — so a persistent schema failure is treated as
    transient and triggers a fresh outer attempt (each attempt runs its own
    low schema-repair budget). That is layering, not summation: the inner
    budget stays 1 per attempt.

    Attributes:
        max_attempts: Total attempts, including the first (``1`` = no
            retry). The default permits two retries after the first try.
        backoff_base_seconds: Full-jitter backoff base; the sleep before
            retry *n* is a random delay in ``[0, base * 2**(n-1)]``.
        retry_on: The exception types treated as transient and worth
            retrying. Anything outside this set (e.g. a programming error)
            propagates immediately. Defaults to
            :data:`~llmkit.LLM_RECOVERABLE_ERRORS`.
    """

    max_attempts: int = 3
    backoff_base_seconds: float = 0.5
    retry_on: tuple[type[BaseException], ...] = field(default=LLM_RECOVERABLE_ERRORS)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")


#: The budget applied when a call function's ``retry`` argument is left at
#: its default — three attempts with full-jitter backoff over the curated
#: transient-error set.
DEFAULT_RETRY_POLICY = RetryPolicy()

#: A policy that disables retries (a single attempt). Pass ``retry=NO_RETRY``
#: to opt a latency-sensitive call out of automatic transient recovery.
NO_RETRY = RetryPolicy(max_attempts=1)


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


async def handle_retry_failure(
    *,
    tag: str,
    attempt: int,
    max_retries: int,
    error: BaseException,
    backoff_base_seconds: float,
) -> None:
    """Run the shared book-keeping for one *non-final* failed attempt.

    Logs a warning, fires the installed progress callback (swallowing any
    error it raises), then sleeps a full-jitter backoff when configured —
    in that order. Shared by :func:`with_retries` and the streaming retry
    loop in :mod:`llmkit.structured_output` so both surfaces back off,
    warn, and report identically. The final failure is *not* routed here:
    callers learn about exhaustion from the re-raised exception.
    """
    logger.warning("%s: attempt %d/%d failed: %s", tag, attempt, max_retries, error)
    callback = _progress_callback.get()
    if callback is not None:
        try:
            callback(label=tag, attempt=attempt, max_retries=max_retries, error=error)
        except Exception:
            logger.exception("%s: retry progress callback raised", tag)
    if backoff_base_seconds > 0:
        ceiling = backoff_base_seconds * (2 ** (attempt - 1))
        await asyncio.sleep(random.uniform(0, ceiling))


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 1,
    label: str | None = None,
    backoff_base_seconds: float = 0.0,
    retry_on: tuple[type[BaseException], ...] | None = None,
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
        retry_on: When set, only exceptions matching this tuple are
            retried; anything else propagates immediately on the first
            raise (so programming errors are never retried). ``None`` (the
            default) retries on any :class:`Exception`, preserving prior
            behaviour for direct callers.

    Returns:
        The result of the first successful call.

    Raises:
        The exception from the last failed attempt if all retries
        are exhausted, or any non-matching exception immediately when
        ``retry_on`` is set.
    """
    tag = label or "retry"
    last_error: Exception = RuntimeError(f"{tag}: no attempts made (max_retries={max_retries})")

    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            if retry_on is not None and not isinstance(e, retry_on):
                raise
            last_error = e
            if attempt < max_retries:
                await handle_retry_failure(
                    tag=tag,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=e,
                    backoff_base_seconds=backoff_base_seconds,
                )
            else:
                logger.error("%s: all %d attempts failed: %s", tag, max_retries, e)

    raise last_error
