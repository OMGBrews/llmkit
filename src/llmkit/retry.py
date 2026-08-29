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
failure can't burn the full transport budget on doomed re-asks. A third class
gets **zero budget** under every configuration — including ``retry_on=None``:
an open circuit (:class:`~llmkit.exceptions.CircuitOpenError`) must never be
re-asked, because the breaker already knows the provider is down, and an
output-limit truncation (:class:`~llmkit.exceptions.OutputLimitError`) re-asked
with an identical token budget can only truncate again; only a ``retry_on``
that explicitly lists the type opts back in.

Composing :func:`with_retries` around a call function that *already* retries
(every call function does, by default) would otherwise multiply the budgets
(the ``3 x 3 = 9`` trap). A context variable marks the active llmkit retry loop
by the task that owns it, so an inner :func:`with_retries` reached in the
*same* task detects the outer policy and runs a single pass instead of
multiplying — while a *distinct* call in its own task (spawned via
``create_task`` or driven across the sync bridge) keeps its own budget. See
:func:`with_retries`.
"""

import asyncio
import contextvars
import logging
import random
import warnings
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, cast, final

import httpx

from llmkit.exceptions import (
    LLM_BACKPRESSURE_ERRORS,
    LLM_OUTPUT_LIMIT_ERRORS,
    LLM_SCHEMA_ERRORS,
    LLM_TRANSPORT_ERRORS,
    REPAIRABLE_PARSE_ERRORS,
    underlying_provider_error,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Wall-clock read, indirected so offline tests can pin ``Retry-After`` dates.

    The only consumer is :func:`_retry_after_seconds` when a provider sends a
    ``Retry-After`` as an HTTP-date (rather than a delta-seconds count): the
    delay is ``parsed_date - _utcnow()``. A test can monkeypatch
    ``llmkit.retry._utcnow`` to make that arithmetic deterministic without real
    sleeping — the same seam ``llmkit.rate_limiting._tuning.now`` provides for
    the token buckets.
    """
    return datetime.now(UTC)


#: Default ceiling on a *server-supplied* ``Retry-After`` wait, in seconds. A
#: separate concern from :attr:`RetryPolicy.max_backoff_seconds` (which caps the
#: library's *computed* exponential): this bounds a value the provider chose, so
#: a hostile or mistaken ``Retry-After: 99999`` cannot wedge a call, while a
#: legitimate larger-than-``max_backoff_seconds`` directive (e.g. 45 s) is still
#: honoured up to this cap.
_DEFAULT_RETRY_AFTER_CAP: float = 60.0


def _as_float(value: object) -> float | None:
    """Best-effort parse of a ``Retry-After``-style value to non-negative seconds.

    Accepts an ``int``/``float`` (an attribute) or a numeric ``str`` (a header);
    returns ``None`` for anything else (a non-numeric string, ``None``, an odd
    type). ``bool`` is rejected explicitly so a stray ``True`` is not read as
    ``1.0``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _http_date_delay(raw: str) -> float | None:
    """Seconds until an HTTP-date ``Retry-After``, or ``None`` if unparseable.

    Clamps at zero (a past date means "retry now"). Reads the present through
    :func:`_utcnow` so tests can pin it.
    """
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - _utcnow()).total_seconds())


def _header(headers: httpx.Headers, name: str) -> str | None:
    """Read a single header value (case-insensitive), or ``None`` when absent.

    Uses ``__getitem__`` (typed ``-> str``) rather than ``httpx.Headers.get``
    (typed ``-> Any``) so the value stays precisely typed at the boundary.
    """
    try:
        return headers[name]
    except KeyError:
        return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The provider's requested wait from a retried error, or ``None`` if absent.

    Honours an explicit ``Retry-After`` so a retry comes back *when the provider
    said to* instead of on a blind exponential. The exception is **unwrapped
    first** (:func:`~llmkit.exceptions.underlying_provider_error`) because a
    structured call surfaces the provider error wrapped in
    ``InstructorRetryException`` — without the unwrap the header would be invisible
    on exactly the structured path. Read order, first hit wins:

    1. a numeric ``retry_after`` attribute (the OpenAI/LiteLLM SDK surface);
    2. a ``retry-after-ms`` header (milliseconds), divided to seconds;
    3. a ``retry-after`` header, as delta-seconds or an HTTP-date.

    Returns non-negative seconds, or ``None`` when no parseable value is present
    (a ``ValidationError``/timeout/network error carries none) — the caller then
    falls back to the computed exponential, byte-identically to before.
    """
    root = underlying_provider_error(exc)
    direct = _as_float(getattr(root, "retry_after", None))
    if direct is not None:
        return max(0.0, direct)
    headers = getattr(getattr(root, "response", None), "headers", None)
    if not isinstance(headers, httpx.Headers):
        return None
    ms = _as_float(_header(headers, "retry-after-ms"))
    if ms is not None:
        return max(0.0, ms / 1000.0)
    raw = _header(headers, "retry-after")
    if raw is None:
        return None
    seconds = _as_float(raw)
    if seconds is not None:
        return max(0.0, seconds)
    return _http_date_delay(raw)


# The unwrapped causes that are *genuinely* schema-shaped when dug out of an
# ``InstructorRetryException`` — exactly the parse failures instructor re-asks
# in-call. Bound to the single source of truth in :mod:`llmkit.exceptions` so
# the two can never drift: any type instructor re-asks in-call earns the
# validation budget here rather than being mis-classified. A wrapped cause
# *outside* this set (and outside the transport set) is a *permanent* error —
# e.g. an exhausted 401/400/403 that instructor re-raised — which must fail fast
# instead of burning the validation budget on doomed re-asks.
_SCHEMA_SHAPED_CAUSES: tuple[type[Exception], ...] = REPAIRABLE_PARSE_ERRORS


# Zero-budget fail-fast signals: retried on neither budget under every
# configuration, including retry_on=None. Backpressure (an open circuit must
# never be re-asked) and output-limit truncation (an identical token budget
# can only truncate again). An explicit retry_on listing the type opts back in.
_ZERO_BUDGET_ERRORS: tuple[type[Exception], ...] = (
    *LLM_OUTPUT_LIMIT_ERRORS,
    *LLM_BACKPRESSURE_ERRORS,
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """The transient-error retry budget a call function applies by default.

    Realizes the "transient retries are on by default" opinion: the call
    functions (:func:`~llmkit.structured_llm_call`,
    :func:`~llmkit.text_llm_call`, :func:`~llmkit.text_llm_call_stream`,
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

    **Two classes get zero budget.** An output-limit truncation
    (:class:`~llmkit.exceptions.OutputLimitError`, ``finish_reason='length'``)
    and an open-circuit backpressure signal
    (:class:`~llmkit.exceptions.CircuitOpenError`) are retried on *neither*
    budget and propagate immediately, under every configuration including
    ``retry_on=None``: a re-ask with an identical token budget can only
    truncate again, and the breaker already knows the provider is down, so
    neither a fresh connection nor a schema repair can help. Listing the type
    explicitly in ``retry_on`` opts back in.

    This layer is kept deliberately separate from instructor's in-call
    schema-repair budget (two attempts total, i.e. one schema-repair re-ask
    per call — fired only for a genuine parse failure, and refused for a length
    truncation; a transport or permanent error is not re-asked in-call) — the
    two are never conflated, so attempts are not double-counted *within* one
    call.
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
            ``[0, min(base * 2**(n-1), max_backoff_seconds)]``. Must be >= 0
            (``0`` disables the computed backoff); a negative base would jitter
            a server ``Retry-After`` *backwards* into an immediate retry.
        max_backoff_seconds: Ceiling on any single backoff sleep. Caps the
            exponential term so a large ``max_attempts`` can't grow the
            worst-case sleep unboundedly (at the default base, attempt 15
            would otherwise permit a multi-hour sleep). Defaults to 30
            seconds — generous spacing for transient-error recovery while
            bounding the worst case. Must be > 0; to disable backoff
            entirely, set ``backoff_base_seconds=0`` instead.
        retry_after_cap: Ceiling on a *server-supplied* ``Retry-After`` wait
            (seconds). When a retried provider error carries ``Retry-After``,
            the backoff honours it (capped here, plus a little jitter) instead
            of the computed exponential — a server directive, honoured even
            when ``backoff_base_seconds`` is 0. Distinct from
            ``max_backoff_seconds`` (which caps the library's *own* exponential):
            this bounds a value the provider chose, so a hostile
            ``Retry-After: 99999`` can't wedge a call while a legitimate 45 s
            directive is still respected. Defaults to 60 s; must be > 0.
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
    retry_after_cap: float = _DEFAULT_RETRY_AFTER_CAP
    retry_on: tuple[type[BaseException], ...] = field(default=LLM_TRANSPORT_ERRORS)
    validation_retry_on: tuple[type[BaseException], ...] = field(default=LLM_SCHEMA_ERRORS)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.validation_max_attempts < 1:
            raise ValueError(
                f"validation_max_attempts must be >= 1, got {self.validation_max_attempts}"
            )
        if self.backoff_base_seconds < 0:
            raise ValueError(f"backoff_base_seconds must be >= 0, got {self.backoff_base_seconds}")
        if self.max_backoff_seconds <= 0:
            raise ValueError(f"max_backoff_seconds must be > 0, got {self.max_backoff_seconds}")
        if self.retry_after_cap <= 0:
            raise ValueError(f"retry_after_cap must be > 0, got {self.retry_after_cap}")


#: The budget applied when a call function's ``retry`` argument is left at
#: its default — three transport attempts (and two validation attempts) with
#: full-jitter backoff over the curated transient-error sets.
DEFAULT_RETRY_POLICY = RetryPolicy()

#: A policy that disables retries (a single attempt) for *both* budgets. Pass
#: ``retry=NO_RETRY`` to opt a latency-sensitive call out of automatic
#: transient recovery, or to opt the inner layer out when wrapping a call
#: function in :func:`with_retries`.
NO_RETRY = RetryPolicy(max_attempts=1, validation_max_attempts=1)


@final
class _RetryScope:
    """Identity of one running llmkit retry loop: the task that owns it.

    :func:`with_retries` (and the streaming loop in
    :mod:`llmkit.structured_output`) installs one of these while it runs, so a
    *nested* retry loop reached **in the same task** — the classic ``3 x 3 = 9``
    trap of a host wrapping a call function that already retries — detects the
    active policy and runs a single pass. Binding the scope to
    :func:`asyncio.current_task` is what keeps a *distinct* call that merely
    inherited the context across a task boundary (``create_task`` copies it; the
    sync bridge captures ``copy_context()`` onto a new task) from wrongly
    collapsing: it runs in its own task, so it keeps its own retry budget.
    """

    __slots__ = ("task",)

    def __init__(self) -> None:
        self.task: asyncio.Task[object] | None = asyncio.current_task()


#: Identifies the llmkit retry loop active in this dynamic scope by the task
#: that owns it (``None`` when no loop is active). Set by :func:`with_retries`
#: while it runs; read through :func:`_in_active_retry_scope` so the collapse
#: only fires for a nested loop in the *owning* task, not an unrelated call that
#: inherited the context across a task boundary (the ``3 x 3 = 9`` trap guard).
_retry_scope: contextvars.ContextVar[_RetryScope | None] = contextvars.ContextVar(
    "_llmkit_retry_scope", default=None
)


def _in_active_retry_scope() -> bool:
    """True when an llmkit retry loop is active in this scope AND owned by the
    current task.

    A scope inherited across a task boundary (``create_task`` / the sync bridge
    copy the context into a NEW task) does not count: a distinct call in its own
    task keeps its own retries. Only meaningful when called from a running loop.
    """
    scope = _retry_scope.get()
    return scope is not None and scope.task is asyncio.current_task()


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
    retry_after_cap: float = _DEFAULT_RETRY_AFTER_CAP,
) -> None:
    """Run the shared book-keeping for one *non-final* failed attempt.

    Logs a warning, fires the installed progress callback (swallowing any
    error it raises), then sleeps — in that order. Shared by
    :func:`with_retries` and the streaming retry loop in
    :mod:`llmkit.structured_output` so both surfaces back off, warn, and report
    identically. The final failure is *not* routed here: callers learn about
    exhaustion from the re-raised exception.

    ``max_attempts`` is the budget the failing class is charged against
    (transport or validation), so the logged ``attempt/max_attempts`` pair
    reflects the actual ceiling for this error.

    **The sleep honours ``Retry-After`` first.** If ``error`` carries a
    provider ``Retry-After`` (:func:`_retry_after_seconds`), the sleep is that
    duration capped at ``retry_after_cap`` plus a little jitter — a server
    *directive*, honoured even when ``backoff_base_seconds`` is 0. Otherwise the
    sleep is the usual full-jitter exponential whose ceiling
    ``backoff_base_seconds * 2**(attempt-1)`` is capped at ``max_backoff_seconds``
    (default 30s), so a large attempt budget can't grow the worst-case sleep
    unboundedly. With no ``Retry-After`` and ``backoff_base_seconds == 0`` there
    is no sleep — byte-identical to the pre-feature behaviour.
    """
    logger.warning("%s: attempt %d/%d failed: %s", tag, attempt, max_attempts, error)
    callback = _progress_callback.get()
    if callback is not None:
        try:
            callback(label=tag, attempt=attempt, max_attempts=max_attempts, error=error)
        except Exception:
            logger.exception("%s: retry progress callback raised", tag)
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        # The provider told us when to come back: honour it (capped so a hostile
        # value can't wedge the call), plus jitter from the same band as the
        # exponential so a fan-out doesn't synchronize its retries. Independent
        # of ``backoff_base_seconds`` — a Retry-After is a directive, not opt-in
        # backoff; when the base is 0 the jitter collapses to 0 and the sleep is
        # exactly the capped server value.
        delay = min(retry_after, retry_after_cap) + random.uniform(0, backoff_base_seconds)
        await asyncio.sleep(delay)
    elif backoff_base_seconds > 0:
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
    retry_after_cap: float = _DEFAULT_RETRY_AFTER_CAP,
    retry_on: tuple[type[BaseException], ...] | None = None,
    validation_max_attempts: int | None = None,
    validation_retry_on: tuple[type[BaseException], ...] | None = None,
) -> T:
    """Retry an async callable up to *max_attempts* times.

    .. warning::

        The call functions (:func:`~llmkit.structured_llm_call`,
        :func:`~llmkit.text_llm_call`, :func:`~llmkit.text_llm_call_stream`,
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
        retry_after_cap: Ceiling on a *server-supplied* ``Retry-After`` wait
            (default 60s). When a retried error carries ``Retry-After``, the
            backoff honours it (capped here, plus jitter) instead of the
            computed exponential — see :func:`handle_retry_failure`.
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
            unless ``retry_on`` explicitly lists the wrapper type. An
            output-limit truncation
            (:class:`~llmkit.exceptions.OutputLimitError`) likewise propagates
            immediately even with ``retry_on=None`` — an identical token
            budget can only truncate again — unless ``retry_on`` explicitly
            lists the type. A bare
            :class:`~llmkit.exceptions.CircuitOpenError` fails fast the same
            way under every configuration — the breaker already knows the
            provider is down, so re-asking an open circuit is pointless —
            unless ``retry_on`` explicitly lists the type.
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
    # task (the classic case: a host wrapped a call function that already
    # retries internally, awaited inline in the same task), this inner layer
    # collapses to a single pass so the two budgets don't multiply (the
    # 3 x 3 = 9 trap). The outer, already-running loop owns the retries. The
    # scope is keyed on the owning task, so a *distinct* call that only
    # inherited the context across a task boundary (create_task / the sync
    # bridge, each copying the context into a NEW task) is NOT collapsed — it
    # keeps its own retry budget. Accepted, documented edges of the
    # task-identity check: a distinct call *directly awaited in the same task*
    # inside ``fn`` still collapses (its exception genuinely propagates to this
    # outer loop, so un-collapsing would recreate real multiplication — the
    # RuntimeWarning signals it and NO_RETRY is the escape); an
    # ``asyncio.current_task()`` of None in an exotic harness reproduces today's
    # collapse; and a same-task ``copy_context().run(...)`` still collapses. The
    # accidental double-wrap warns (deduped by the default warning filter); an
    # explicit NO_RETRY inner does not.
    if _in_active_retry_scope():
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

    token = _retry_scope.set(_RetryScope())
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
                # Zero-budget fail-fast signals are permanent under every
                # configuration, including retry_on=None ("retry on any
                # Exception"): an open circuit
                # (:class:`~llmkit.exceptions.CircuitOpenError`) must never be
                # re-asked — the breaker already knows the provider is down —
                # and an output-limit truncation retried with an identical
                # token budget can only truncate again. Without this guard the
                # bare CircuitOpenError / OutputLimitError would fall into the
                # retry-anything branch below and burn the full transport
                # budget on doomed re-asks. Checked on the unwrapped cause for
                # defense in depth (the transport raises them bare today; a
                # wrapped form would also fail fast via wraps_permanent_cause
                # below — the wrapped CircuitOpenError is already caught there).
                # A retry_on that explicitly lists the type still wins, matching
                # the wraps_permanent_cause escape.
                if isinstance(cause, _ZERO_BUDGET_ERRORS) and (
                    retry_on is None or not isinstance(e, retry_on)
                ):
                    raise
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
                        retry_after_cap=retry_after_cap,
                    )
                else:
                    logger.error("%s: all %d attempts failed: %s", tag, budget, e)
                    raise
    finally:
        _retry_scope.reset(token)
