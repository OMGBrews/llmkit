"""Async transient-error retry layer.

The call functions in :mod:`llmkit.structured_output` retry *transient*
provider errors on their own by default (see :class:`RetryPolicy`); this
module holds the loop they share. :func:`with_retries` is also exported as
the explicit, composable advanced path a caller can wrap any awaitable in.

Audit logging and timing remain the caller's concern: each attempt is its
own LLM call (and its own log record), because the retry loop wraps the
logging call functions rather than living inside them.

Two budgets, kept separate: *transport* failures (rate limits, transient
5xx, network/timeout) get the full :attr:`RetryPolicy.max_attempts` budget;
*schema-validation* failures get the lower
:attr:`RetryPolicy.validation_max_attempts` budget, so a deterministic schema
failure can't burn the full transport budget on doomed re-asks.

Composing :func:`with_retries` around a call function that *already* retries
(every call function does, by default) would otherwise multiply the budgets
(the ``3 x 3 = 9`` trap). A context variable marks an active llmkit retry
loop, so the outer :func:`with_retries` detects the inner policy and runs a
single pass instead of multiplying — see :func:`with_retries`.
"""

import asyncio
import contextvars
import logging
import random
import warnings
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Protocol, cast

from pydantic import ValidationError

from llmkit.exceptions import (
    LLM_SCHEMA_ERRORS,
    LLM_TRANSPORT_ERRORS,
    underlying_provider_error,
)

logger = logging.getLogger(__name__)

# The unwrapped causes that are *genuinely* schema-shaped when dug out of an
# ``InstructorRetryException``: pydantic rejected the parse, or the response
# wasn't valid JSON at all. A wrapped cause outside this set (and outside the
# transport set) is a *permanent* error — e.g. an exhausted 401/400/403 that
# instructor re-raised — which must fail fast instead of burning the
# validation budget on doomed re-asks.
_SCHEMA_SHAPED_CAUSES: tuple[type[Exception], ...] = (ValidationError, JSONDecodeError)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """The transient-error retry budget a call function applies by default.

    Realizes the "transient retries are on by default" opinion: the call
    functions (:func:`~llmkit.structured_llm_call`,
    :func:`~llmkit.text_llm_call`, :func:`~llmkit.stream_text_with_log`,
    and the sync wrapper) retry the ``retry_on`` / ``validation_retry_on``
    errors with bounded full-jitter backoff, without the caller wrapping
    every call.

    **Two separate budgets.** Transport failures (rate limits, transient
    5xx, network/timeout — :data:`~llmkit.exceptions.LLM_TRANSPORT_ERRORS`)
    get ``max_attempts`` tries, since a retry on a fresh connection
    routinely succeeds. Schema-validation failures (pydantic
    ``ValidationError`` / instructor ``InstructorRetryException`` —
    :data:`~llmkit.exceptions.LLM_SCHEMA_ERRORS`) get the lower
    ``validation_max_attempts``, so a deterministically-wrong schema can't
    burn the full transport budget on doomed re-asks while a
    transiently-malformed JSON response still earns one cross-call retry.
    The budgets are counted independently: each failure is charged to
    whichever budget matches its class, and the call stops when *that*
    budget is spent.

    This layer is kept deliberately separate from instructor's in-call
    schema-repair budget (``max_retries=2``: two attempts total, i.e. one
    schema-repair re-ask per call) — the two are never conflated, so attempts
    are not double-counted *within* one call.
    Note the layering at the seam: when instructor exhausts its own repair
    budget it raises ``InstructorRetryException``, which is in
    :data:`~llmkit.exceptions.LLM_SCHEMA_ERRORS` — so a persistent schema
    failure triggers a *fresh outer attempt* on the lower validation budget
    (each attempt runs its own low in-call repair budget). That is layering,
    not summation: the inner budget stays one re-ask per attempt. Because instructor
    wraps *transport* failures in the same exception, the loop first unwraps it
    (:func:`~llmkit.exceptions.underlying_provider_error`) and routes a wrapped
    transport cause to ``max_attempts`` — so a 429/5xx/network blip inside a
    structured call gets the full transport budget, exactly like the text path.
    A wrapped cause that is neither transport- nor schema-shaped — a *permanent*
    error such as an exhausted 401/400/403 — is not retried on either budget:
    it propagates immediately, matching the fail-fast contract the bare-4xx
    text path already honours.

    Attributes:
        max_attempts: Total *transport* attempts, including the first
            (``1`` = no retry). The default permits two retries after the
            first try.
        validation_max_attempts: Total *schema-validation* attempts,
            including the first (``1`` = no retry). Defaults to ``2`` (one
            retry) — lower than ``max_attempts`` on purpose.
        backoff_base_seconds: Full-jitter backoff base; the sleep before
            retry *n* is a random delay in
            ``[0, min(base * 2**(n-1), max_backoff_seconds)]``.
        max_backoff_seconds: Ceiling on any single backoff sleep. Caps the
            exponential term so a large ``max_attempts`` can't grow the
            worst-case sleep unboundedly (at the default base, attempt 15
            would otherwise permit a multi-hour sleep). Defaults to 30
            seconds — generous spacing for transient-error recovery while
            bounding the worst case. Must be > 0; to disable backoff
            entirely, set ``backoff_base_seconds=0`` instead.
        retry_on: The exception types treated as transient *transport*
            errors, retried against ``max_attempts``. Anything outside both
            sets (e.g. a programming error) propagates immediately. Defaults
            to :data:`~llmkit.exceptions.LLM_TRANSPORT_ERRORS`.
        validation_retry_on: The exception types treated as schema-validation
            errors, retried against ``validation_max_attempts``. Defaults to
            :data:`~llmkit.exceptions.LLM_SCHEMA_ERRORS`.
    """

    max_attempts: int = 3
    validation_max_attempts: int = 2
    backoff_base_seconds: float = 0.5
    max_backoff_seconds: float = 30.0
    retry_on: tuple[type[BaseException], ...] = field(default=LLM_TRANSPORT_ERRORS)
    validation_retry_on: tuple[type[BaseException], ...] = field(default=LLM_SCHEMA_ERRORS)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.validation_max_attempts < 1:
            raise ValueError(
                f"validation_max_attempts must be >= 1, got {self.validation_max_attempts}"
            )
        if self.max_backoff_seconds <= 0:
            raise ValueError(f"max_backoff_seconds must be > 0, got {self.max_backoff_seconds}")


#: The budget applied when a call function's ``retry`` argument is left at
#: its default — three transport attempts (and two validation attempts) with
#: full-jitter backoff over the curated transient-error sets.
DEFAULT_RETRY_POLICY = RetryPolicy()

#: A policy that disables retries (a single attempt) for *both* budgets. Pass
#: ``retry=NO_RETRY`` to opt a latency-sensitive call out of automatic
#: transient recovery, or to opt the inner layer out when wrapping a call
#: function in :func:`with_retries`.
NO_RETRY = RetryPolicy(max_attempts=1, validation_max_attempts=1)


#: Marks "an llmkit retry loop is active in this dynamic scope." Set by
#: :func:`with_retries` while it runs, so a *nested* :func:`with_retries`
#: (the classic case: a host wrapping a call function that already retries)
#: detects the active inner policy and runs a single pass instead of
#: multiplying the budgets (the ``3 x 3 = 9`` trap).
_retry_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_llmkit_retry_active", default=False
)


class RetryProgressCallback(Protocol):
    """Receives per-attempt failure events from :func:`with_retries`.

    Invoked once per non-final failed attempt. The final-failure case
    does not call back — callers learn about exhaustion by catching the
    re-raised exception. Implementations must not raise.

    The ``max_attempts`` keyword carries the *total attempts* for the budget
    the failing error is charged against (transport or validation), matching
    the name used across the rest of the retry surface.
    """

    def __call__(
        self,
        *,
        label: str,
        attempt: int,
        max_attempts: int,
        error: BaseException,
    ) -> None: ...


_progress_callback: contextvars.ContextVar[RetryProgressCallback | None] = contextvars.ContextVar(
    "_retry_progress_callback", default=None
)


@contextmanager
def retry_progress_callback(callback: RetryProgressCallback | None) -> Generator[None]:
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
    max_attempts: int,
    error: BaseException,
    backoff_base_seconds: float,
    max_backoff_seconds: float = 30.0,
) -> None:
    """Run the shared book-keeping for one *non-final* failed attempt.

    Logs a warning, fires the installed progress callback (swallowing any
    error it raises), then sleeps a full-jitter backoff when configured —
    in that order. Shared by :func:`with_retries` and the streaming retry
    loop in :mod:`llmkit.structured_output` so both surfaces back off,
    warn, and report identically. The final failure is *not* routed here:
    callers learn about exhaustion from the re-raised exception.

    ``max_attempts`` is the budget the failing class is charged against
    (transport or validation), so the logged ``attempt/max_attempts`` pair
    reflects the actual ceiling for this error.

    The full-jitter ceiling ``backoff_base_seconds * 2**(attempt-1)`` is
    capped at ``max_backoff_seconds`` (default 30s), so a large attempt
    budget can't grow the worst-case sleep unboundedly.
    """
    logger.warning("%s: attempt %d/%d failed: %s", tag, attempt, max_attempts, error)
    callback = _progress_callback.get()
    if callback is not None:
        try:
            callback(label=tag, attempt=attempt, max_attempts=max_attempts, error=error)
        except Exception:
            logger.exception("%s: retry progress callback raised", tag)
    if backoff_base_seconds > 0:
        # int ** int widens to Any in the stubs (negative exponents yield
        # float); attempt >= 1 keeps the exponent non-negative, so cast the
        # power back to the int it really is before scaling to a float ceiling.
        # The exponential ceiling is capped at max_backoff_seconds so late
        # attempts under a large budget can't sleep for hours.
        ceiling = min(
            backoff_base_seconds * cast("int", 2 ** (attempt - 1)),
            max_backoff_seconds,
        )
        await asyncio.sleep(random.uniform(0, ceiling))


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int | None = None,
    label: str | None = None,
    backoff_base_seconds: float = 0.0,
    max_backoff_seconds: float = 30.0,
    retry_on: tuple[type[BaseException], ...] | None = None,
    validation_max_attempts: int | None = None,
    validation_retry_on: tuple[type[BaseException], ...] | None = None,
) -> T:
    """Retry an async callable up to *max_attempts* times.

    .. warning::

        The call functions (:func:`~llmkit.structured_llm_call`,
        :func:`~llmkit.text_llm_call`, :func:`~llmkit.stream_text_with_log`,
        and the sync wrapper) **already retry internally** by default. Wrapping
        one of them in :func:`with_retries` would otherwise multiply the two
        budgets (the ``3 x 3 = 9`` trap). To prevent that, :func:`with_retries`
        detects (via a context variable) when it is running *inside* an
        already-active llmkit retry loop and collapses that **inner** layer to
        a **single pass** — the outer loop owns the retries — so the net effect
        is one shared budget, not the product. An *accidental* double-wrap (an
        inner layer that would itself have retried) also emits a
        ``RuntimeWarning`` (de-duplicated by Python's default warning filter).
        This guard only fires when there *is* an active llmkit retry loop in
        scope: wrapping a plain (non-llmkit) awaitable retries normally. To drive
        retries entirely from your wrapper, opt the inner call out with
        ``retry=NO_RETRY`` — the clean, warning-free path.

    Args:
        fn: Zero-argument async callable to execute.
        max_attempts: Total number of *transport* attempts (1 = no retry).
            Defaults to ``1`` when not given.
        label: Optional identifier for log messages (e.g. an op_id).
        backoff_base_seconds: When > 0, sleep before each retry using
            exponential "full jitter" backoff — a random delay in
            ``[0, min(backoff_base_seconds * 2**(attempt-1), max_backoff_seconds)]``.
            Defaults to 0.0 (retry immediately), which preserves prior
            behaviour for callers that don't opt in. Jitter spreads
            concurrent retries so a transient provider-saturation window
            (the dominant failure mode for the eval fan-out) isn't re-hit
            by every caller at once.
        max_backoff_seconds: Ceiling on any single backoff sleep (default
            30s), capping the exponential term so a large ``max_attempts``
            can't grow the worst-case sleep unboundedly.
        retry_on: When set, only exceptions matching this tuple are retried
            against ``max_attempts``; anything else propagates immediately
            on the first raise (so programming errors are never retried).
            ``None`` (the default) retries on any :class:`Exception`,
            preserving prior behaviour for direct callers. Budget *routing*
            still distinguishes the two classes when a validation budget is
            configured: with ``retry_on=None`` the unwrapped cause is
            classified against
            :data:`~llmkit.exceptions.LLM_TRANSPORT_ERRORS`, so an
            ``InstructorRetryException`` wrapping a 429/5xx/network blip is
            charged the full ``max_attempts``, and only genuine validation
            failures are charged the lower validation budget. A wrapped cause
            that is neither transport- nor schema-shaped (a permanent
            401/400/403) propagates immediately under every configuration,
            unless ``retry_on`` explicitly lists the wrapper type.
        validation_max_attempts: Total number of *schema-validation*
            attempts (1 = no retry). When set together with
            ``validation_retry_on``, failures matching that tuple are charged
            against this separate, typically-lower budget instead of
            ``max_attempts``. ``None`` (the default) means no separate
            validation budget — every retryable error shares ``max_attempts``.
        validation_retry_on: The exception types charged against
            ``validation_max_attempts``. Only meaningful when
            ``validation_max_attempts`` is set.

    Returns:
        The result of the first successful call.

    Raises:
        The exception from the last failed attempt if all retries
        are exhausted, or any non-matching exception immediately when
        ``retry_on`` is set.
    """
    if max_attempts is None:
        max_attempts = 1

    tag = label or "retry"

    # Nested-retry guard: if an llmkit retry loop is already active in this
    # dynamic scope (the classic case: a host wrapped a call function that
    # already retries internally), this inner layer collapses to a single pass
    # so the two budgets don't multiply (the 3 x 3 = 9 trap). The outer,
    # already-running loop owns the retries. The accidental double-wrap warns
    # (deduped by the default warning filter); an explicit NO_RETRY inner does not.
    if _retry_active.get():
        # Only the *accidental* double-wrap is worth warning about: an inner
        # layer that would itself have retried (budget > 1). When the inner is
        # already a single pass — e.g. retry=NO_RETRY, the documented way to
        # drive retries from an outer wrapper — there is nothing to multiply, so
        # stay quiet. Either way the inner runs exactly one pass; the outer loop
        # owns the retries.
        inner_would_retry = max_attempts > 1 or (validation_max_attempts or 1) > 1
        if inner_would_retry:
            warnings.warn(
                f"with_retries({tag!r}) is nested inside an already-retrying llmkit "
                + "retry loop; this inner layer will run a single pass to avoid "
                + "multiplying retry budgets (the outer loop owns the retries). To "
                + "drive retries from an outer wrapper around a call function, opt "
                + "the inner call out with retry=NO_RETRY.",
                RuntimeWarning,
                stacklevel=2,
            )
        return await fn()

    # Separate budgets: validation failures (when a validation budget is
    # configured) are charged against ``validation_max_attempts``; everything
    # else retryable against ``max_attempts``. Each failure increments only
    # its own counter, and the call stops when that counter hits its ceiling.
    use_validation_budget = validation_max_attempts is not None and validation_retry_on is not None
    transport_attempt = 0
    validation_attempt = 0

    token = _retry_active.set(True)
    try:
        while True:
            try:
                return await fn()
            except Exception as e:
                # instructor re-raises *any* exhausted attempt — transport
                # failures included — as InstructorRetryException, which is in
                # the schema set. Unwrap to the underlying provider error so a
                # wrapped transport failure (429/5xx/network) claims the
                # transport budget first, matching the plain-text path; a
                # wrapped schema failure (or an inner error in neither set)
                # still falls through to the validation budget below.
                cause = underlying_provider_error(e)
                # With retry_on=None ("retry on any Exception") the routing
                # still needs a transport set to classify the unwrapped cause
                # against — otherwise a wrapped 429/network blip would match
                # ``validation_retry_on`` below and be charged the *lower*
                # validation budget, the exact misclassification the unwrap
                # exists to prevent. Fall back to the curated transport set.
                transport_types = retry_on if retry_on is not None else LLM_TRANSPORT_ERRORS
                is_transport_cause = isinstance(cause, transport_types)
                # Wrapped *permanent* errors fail fast: instructor wraps an
                # exhausted 401/400/403 (any non-transient provider error) in
                # the same InstructorRetryException as genuine schema failures,
                # so without this guard the wrapper would match
                # ``validation_retry_on`` below and burn a validation retry on
                # a request that can never succeed — violating the documented
                # fail-fast contract the bare-4xx text path already honours.
                # When the unwrapped cause differs from the wrapper and is
                # neither transport- nor schema-shaped, propagate immediately.
                # An explicit ``retry_on`` that lists the wrapper type still
                # wins: the caller asked for exactly that retry.
                wraps_permanent_cause = (
                    cause is not e
                    and not is_transport_cause
                    and not isinstance(cause, _SCHEMA_SHAPED_CAUSES)
                )
                if wraps_permanent_cause and (retry_on is None or not isinstance(e, retry_on)):
                    raise
                is_validation = (
                    use_validation_budget
                    and not is_transport_cause
                    and isinstance(
                        e,
                        validation_retry_on,  # pyright: ignore[reportArgumentType]  # None excluded by use_validation_budget
                    )
                )
                if is_validation:
                    validation_attempt += 1
                    attempt = validation_attempt
                    budget = validation_max_attempts
                    assert budget is not None  # guaranteed by use_validation_budget
                elif retry_on is None or isinstance(e, retry_on) or is_transport_cause:
                    transport_attempt += 1
                    attempt = transport_attempt
                    budget = max_attempts
                else:
                    # Not retryable on either budget — propagate immediately.
                    raise
                if attempt < budget:
                    await handle_retry_failure(
                        tag=tag,
                        attempt=attempt,
                        max_attempts=budget,
                        error=e,
                        backoff_base_seconds=backoff_base_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                else:
                    logger.error("%s: all %d attempts failed: %s", tag, budget, e)
                    raise
    finally:
        _retry_active.reset(token)
