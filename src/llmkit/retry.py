"""Async transient-error retry layer.

The call functions in :mod:`llmkit.calls` retry *transient* provider errors on
their own by default (see :class:`RetryPolicy`); this module holds the loops
they share. There are two, because a stream is not an awaitable:

* :func:`with_retries` retries an awaitable, and is also exported as the
  explicit, composable advanced path a caller can wrap any awaitable in;
* :func:`with_retries_stream` retries an *async generator*, and only up to its
  first yielded chunk — a partially-consumed stream cannot be transparently
  restarted.

Both install the same task-keyed scope, so they compose with each other exactly
as either composes with itself.

Audit logging and timing remain the caller's concern: each attempt is its
own LLM call (and its own log record), because the retry loop wraps the
logging call functions rather than living inside them. The loop does own one
piece of logging identity: a call function re-run by either loop joins the
loop's logical call, so its records share one ``call_id`` and number their
``attempt`` across passes (see :class:`_CallLedger`).

One overall outer-attempt ceiling, with a smaller validation-retry allowance
inside it: every pass counts against :attr:`RetryPolicy.max_attempts`, and a
validation failure can cause at most ``validation_max_attempts - 1`` retries.
Switching error categories never resets either count. Output-limit truncation
and an open circuit fail immediately unless explicitly included in ``retry_on``.

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
import math
import random
import uuid
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import aclosing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, final

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
    ``1.0``. A value that parses to NaN or infinity (``"inf"``, ``"nan"``,
    ``"1e400"``, a nonfinite attribute) is unparseable too: it is no wait a
    server could mean, and passing it on would let it wedge or skip the sleep.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if math.isfinite(parsed) else None


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

    A nonfinite value (NaN or infinity) from any source is skipped exactly like
    an unparseable one, so the next source is read; nothing is clamped.

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


# Zero-budget fail-fast signals: never retried under the default policy.
# Backpressure (an open circuit must
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
    :func:`~llmkit.tool_llm_call`, :func:`~llmkit.tool_llm_call_stream`,
    and the sync wrappers) retry the ``retry_on`` / ``validation_retry_on``
    errors with bounded full-jitter backoff, without the caller wrapping
    every call.

    **One ceiling with a validation sublimit.** ``max_attempts`` bounds every
    outer attempt, including the first, regardless of failure category.
    ``validation_max_attempts - 1`` limits how many retries validation failures
    can cause within that ceiling. The default is three outer attempts with
    at most one validation-caused retry. No count resets after another error
    class intervenes. A pass may issue more than one provider request because
    schema repair and temperature reissues happen inside it; this is not a
    provider-request, spending, or elapsed-time cap.

    **Two classes get zero budget.** An output-limit truncation
    (:class:`~llmkit.exceptions.OutputLimitError`, ``finish_reason='length'``)
    and an open-circuit backpressure signal
    (:class:`~llmkit.exceptions.CircuitOpenError`) are retried on *neither*
    budget and propagate immediately under the default policy: a re-ask with
    an identical token budget can only truncate again, and the breaker already knows the provider is down, so
    neither a fresh connection nor a schema repair can help. Listing the type
    explicitly in ``retry_on`` opts back in.

    This layer is kept deliberately separate from instructor's in-call
    schema-repair budget (two attempts total, i.e. one schema-repair re-ask
    per call — fired only for a genuine parse failure, and refused for a length
    truncation; a transport or permanent error is not re-asked in-call) — the
    two are never conflated, so attempts are not double-counted *within* one
    call.
    Note the layering at the seam: when instructor exhausts its own repair
    budget on a parse failure, the structured lane re-raises its
    ``InstructorRetryException`` as
    :class:`~llmkit.exceptions.StructuredOutputError`, which is in
    :data:`~llmkit.exceptions.LLM_SCHEMA_ERRORS` — so a persistent schema
    failure triggers a *fresh outer attempt* on the lower validation budget
    (each attempt runs its own low in-call repair budget). That is layering,
    not summation: the inner budget stays one re-ask per attempt. Because instructor
    wraps *transport* failures in the same exception, the loop first unwraps it
    (:func:`~llmkit.exceptions.underlying_provider_error`) and routes a wrapped
    transport cause to ``max_attempts`` — so a 429/5xx/network blip inside a
    structured call can use the overall ceiling, exactly like the text path.
    A wrapped cause that is neither transport- nor schema-shaped — a *permanent*
    error such as an exhausted 401/400/403 — is not retried on either budget:
    it propagates immediately, matching the fail-fast contract the bare-4xx
    text path already honours.

    Attributes:
        max_attempts: Total outer attempts across all error classes, including
            the first (``1`` = no retry). Defaults to ``3``. Must be an
            ``int`` >= 1; a ``bool`` or a float such as ``3.0`` is rejected.
        validation_max_attempts: A value of N permits at most N-1 retries caused
            by validation failures, within ``max_attempts``. ``1`` permits none;
            the default ``2`` permits one. A larger value cannot raise the total
            ceiling. Must be an ``int`` >= 1, like ``max_attempts``.
        backoff_base_seconds: Full-jitter backoff base; the sleep before
            retry *n* is a random delay in
            ``[0, min(base * 2**(n-1), max_backoff_seconds)]``. Must be finite
            and >= 0 (``0`` disables the computed backoff); a negative base would jitter
            a server ``Retry-After`` *backwards* into an immediate retry.
        max_backoff_seconds: Ceiling on any single backoff sleep. Caps the
            exponential term so a large ``max_attempts`` can't grow the
            worst-case sleep unboundedly (at the default base, attempt 15
            would otherwise permit a multi-hour sleep). Defaults to 30
            seconds — generous spacing for transient-error recovery while
            bounding the worst case. Must be finite and > 0; to disable backoff
            entirely, set ``backoff_base_seconds=0`` instead.
        retry_after_cap: Ceiling on a *server-supplied* ``Retry-After`` wait
            (seconds). When a retried provider error carries ``Retry-After``,
            the backoff honours it (capped here, plus a little jitter) instead
            of the computed exponential — a server directive, honoured even
            when ``backoff_base_seconds`` is 0. Distinct from
            ``max_backoff_seconds`` (which caps the library's *own* exponential):
            this bounds a value the provider chose, so a hostile
            ``Retry-After: 99999`` can't wedge a call while a legitimate 45 s
            directive is still respected. Defaults to 60 s; must be finite and > 0.
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
        _check_count("max_attempts", self.max_attempts)
        _check_count("validation_max_attempts", self.validation_max_attempts)
        _check_duration("backoff_base_seconds", self.backoff_base_seconds, allow_zero=True)
        _check_duration("max_backoff_seconds", self.max_backoff_seconds, allow_zero=False)
        _check_duration("retry_after_cap", self.retry_after_cap, allow_zero=False)


def _check_count(name: str, value: int) -> None:
    """Reject an attempt count that is not a positive ``int``.

    ``type(...) is int`` (not ``isinstance``) refuses ``bool``, which would
    otherwise read as ``1``, as well as floats such as ``2.5`` or ``3.0`` and
    integer-like foreign types. The same rule as
    :class:`~llmkit.logging.LogDeliveryPolicy`'s ``queue_capacity``.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be >= 1 (an int, not a bool), got {value!r}")


def _check_duration(name: str, value: float, *, allow_zero: bool) -> None:
    """Reject a duration that is a ``bool``, nonfinite, or below its floor.

    NaN slips past every comparison and infinity defeats every cap, so either
    would silently disable the guard the field exists for. An ``int`` is fine.
    """
    bound = ">= 0" if allow_zero else "> 0"
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(f"{name} must be {bound} (finite, not a bool), got {value!r}")


#: The budget applied when a call function's ``retry`` argument is left at
#: its default — three outer attempts, at most one validation-caused retry, with
#: full-jitter backoff over the curated transient-error sets.
DEFAULT_RETRY_POLICY = RetryPolicy()

#: A policy that disables retries (a single outer attempt). Pass
#: ``retry=NO_RETRY`` to opt a latency-sensitive call out of automatic
#: transient recovery, or to opt the inner layer out when wrapping a call
#: function in :func:`with_retries`.
NO_RETRY = RetryPolicy(max_attempts=1, validation_max_attempts=1)


@final
class _CallLedger:
    """The logical calls one retry loop re-runs, across passes.

    ``LLMCallRecord`` promises one ``call_id`` per *logical* call and a 1-based
    ``attempt`` within it. A call function opted out of its own retries and
    re-run by an outer :func:`with_retries` — or a stream function a host's
    attempt factory calls afresh under :func:`with_retries_stream` — is invoked
    afresh on every pass, so left alone it would mint a new id each time; the
    ledger is what lets it rejoin the logical call it belongs to instead. A
    rejoined call also gets a nonzero attempt offset, which is how only the
    first attempt reads the response cache.

    A call is identified by its ``(feature, label)`` and the occurrence index of
    that pair within the current pass — not by raw position, so a conditional
    call that runs on only some passes cannot shift every later call onto the
    wrong id. ``attempt`` counts tries *of that logical call*: one first reached
    on pass 2 records attempt 1.

    The non-nested paths of both loops install one. The stream loop carries
    its one ledger across every scope it re-arms around a ``yield``, so every
    call an attempt prepares is counted by the same ledger, before its first
    chunk or after. llmkit's own stream functions
    prepare their call before entering the loop and pass the id to each
    attempt as a parameter, so for them the ledger changes nothing.
    """

    __slots__ = ("_entries", "_seen")

    def __init__(self) -> None:
        # (feature, label, occurrence) -> (call_id, attempts claimed so far).
        self._entries: dict[tuple[str, str | None, int], tuple[str, int]] = {}
        # (feature, label) -> occurrences claimed in the current pass.
        self._seen: dict[tuple[str, str | None], int] = {}

    def begin_pass(self) -> None:
        """Start a new pass: occurrence counting restarts from zero."""
        self._seen.clear()

    def claim(self, feature: str, label: str | None) -> tuple[str, int]:
        """Return the ``call_id`` and attempt offset for the next call with
        this ``(feature, label)`` in the current pass, recording the claim."""
        occurrence = self._seen.get((feature, label), 0)
        self._seen[(feature, label)] = occurrence + 1
        key = (feature, label, occurrence)
        entry = self._entries.get(key)
        call_id, claimed = entry if entry is not None else (uuid.uuid4().hex, 0)
        self._entries[key] = (call_id, claimed + 1)
        return call_id, claimed


@final
class _RetryScope:
    """Identity of one running llmkit retry loop: the task that owns it.

    :func:`with_retries` (and the streaming loop in
    :func:`with_retries_stream`) installs one of these while it runs, so a
    *nested* retry loop reached **in the same task** — the classic ``3 x 3 = 9``
    trap of a host wrapping a call function that already retries — detects the
    active policy and runs a single pass. Binding the scope to
    :func:`asyncio.current_task` is what keeps a *distinct* call that merely
    inherited the context across a task boundary (``create_task`` copies it; the
    sync bridge captures ``copy_context()`` onto a new task) from wrongly
    collapsing: it runs in its own task, so it keeps its own retry budget.

    The same ownership test decides which calls join the loop's logical calls:
    *ledger* is the :class:`_CallLedger` a non-nested :func:`with_retries` or
    :func:`with_retries_stream` installs, read through
    :func:`claim_logical_call`; ``None`` everywhere else.
    """

    __slots__ = ("ledger", "task")

    def __init__(self, ledger: _CallLedger | None = None) -> None:
        self.task: asyncio.Task[object] | None = asyncio.current_task()
        self.ledger: _CallLedger | None = ledger


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


def claim_logical_call(feature: str, label: str | None) -> tuple[str, int]:
    """The ``call_id`` and attempt offset for a call about to be prepared.

    Inside a :func:`with_retries` or :func:`with_retries_stream` loop owned by
    the current task, the call joins
    the loop's logical call through its :class:`_CallLedger`: the same id as the
    matching call on earlier passes, and an offset of the attempts already made
    under it. Anywhere else — no loop, or a loop owned by another task, which is
    the same ownership rule the nested-retry guard applies — it gets a fresh
    ``uuid4`` hex and offset 0.
    """
    scope = _retry_scope.get()
    if scope is None or scope.ledger is None or scope.task is not asyncio.current_task():
        return uuid.uuid4().hex, 0
    return scope.ledger.claim(feature, label)


class RetryProgressCallback(Protocol):
    """Receives per-attempt failure events from :func:`with_retries`.

    Invoked once per non-final failed attempt. The final-failure case
    does not call back — callers learn about exhaustion by catching the
    re-raised exception. Implementations must not raise.

    ``attempt`` is the monotonic outer-pass number and ``max_attempts`` is
    its total ceiling. The validation allowance may stop retries earlier.
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
    :func:`with_retries_stream` so both surfaces back off, warn, and report
    identically. The final failure is *not* routed here: callers learn about
    exhaustion from the re-raised exception.

    ``attempt`` and ``max_attempts`` report the monotonic outer-pass number
    and total ceiling. The validation allowance may stop retries earlier.

    **The sleep honours ``Retry-After`` first.** If ``error`` carries a
    provider ``Retry-After`` (:func:`_retry_after_seconds`), the sleep is that
    duration capped at ``retry_after_cap`` plus a little jitter — a server
    *directive*, honoured even when ``backoff_base_seconds`` is 0. Otherwise the
    sleep is the usual full-jitter exponential whose ceiling
    ``backoff_base_seconds * 2**(attempt-1)`` is capped at ``max_backoff_seconds``
    (default 30s), so a large attempt budget can't grow the worst-case sleep
    unboundedly. With no ``Retry-After`` and ``backoff_base_seconds == 0`` there
    is no sleep — byte-identical to the pre-feature behaviour.

    The keyword budgets are trusted as given: the library passes a validated
    :class:`RetryPolicy`'s fields, and a direct caller is responsible for
    passing values inside that policy's domains.
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
        # The exponential ceiling is capped at max_backoff_seconds so late
        # attempts under a large budget can't sleep for hours. The power is a
        # float on an exponent capped at 1023, the largest a float power of two
        # can hold: an int power would overflow converting to float from
        # attempt 1025, and an infinite product is harmless inside min().
        ceiling = min(
            backoff_base_seconds * 2.0 ** min(attempt - 1, 1023),
            max_backoff_seconds,
        )
        await asyncio.sleep(random.uniform(0, ceiling))


async def with_retries_stream[T](
    attempt_factory: Callable[[int], AsyncGenerator[T]],
    *,
    policy: RetryPolicy,
    label: str,
    surface: str,
    warn_stacklevel: int = 2,
) -> AsyncGenerator[T]:
    """:func:`with_retries` for a stream: retry only *before* the first chunk.

    The second of this module's two retry loops. It cannot be expressed as
    :func:`with_retries` because a stream is not a ``Callable[[],
    Awaitable[T]]`` that can simply be re-awaited: a partially-consumed stream
    cannot be transparently restarted, so once any chunk has reached the
    consumer a mid-stream error propagates unretried. Everything before the
    first chunk follows ``policy``'s total ceiling and validation sublimit.

    *attempt_factory* is called with the 1-based attempt number and returns one
    attempt's async generator; each attempt is a distinct call (and, for the
    call functions, its own log record, sharing the stream's ``call_id``). A
    factory that calls a stream function afresh on each attempt joins one
    logical call through the loop's :class:`_CallLedger`, as under
    :func:`with_retries`: one ``call_id``, a rising ``attempt``, and a response
    cache read on the first attempt only.
    *surface* names the caller in the double-wrap warning, and *label* is the
    tag used for the warning and the exhaustion log line — the same
    ``"%s: stopped after %d attempts: %s"`` message :func:`with_retries` emits,
    from the same logger, so an operator greps the streaming and non-streaming
    surfaces identically.

    Two things about async generators drive the shape of the body, and neither
    is optional:

    * **The scope is released around every ``yield``.** An async generator's
      body runs in its *consumer's* context, so holding the retry scope across
      a suspension would leak it into the consumer's own llmkit calls between
      chunks — and, on an early ``break``, leave it set in their context for
      good. Each re-arm installs a fresh :class:`_RetryScope`, because a
      generator may resume in a *different* task than the one it suspended in,
      carrying the loop's one :class:`_CallLedger`.
    * **Each attempt is wrapped in ``aclosing``.** An abandoning consumer's
      close must reach the attempt generator while this frame is still live, so
      the attempt's own ``finally`` runs deterministically rather than whenever
      the garbage collector finalizes a suspended generator — which is what
      makes an abandoned stream's log record honest about being truncated.

    *warn_stacklevel* exists because the double-wrap warning must name the
    **consumer's** ``async for``, and this is an async generator: its body runs
    in whatever frame is driving it, so the frame count depends on how many
    generators sit between here and the consumer. The default ``2`` is correct
    when the consumer iterates this generator directly; a wrapper that re-yields
    from it (``text_llm_call_stream`` and ``tool_llm_call_stream`` both do)
    adds one frame and passes ``3``. It is chosen per call site, not inherited:
    a second wrapper's correct value is not implied by the first one's.
    Getting this wrong does not fail loudly — it blames a line of llmkit for the
    caller's double-wrap, and collapses the default warning filter's
    per-call-site de-duplication onto that one line, so a host with two distinct
    double-wrap sites is told about only the first.

    A pre-yield validation failure consumes the same persistent validation
    allowance as a buffered call. Once anything has been yielded, no error is
    replayed, even if its type is explicitly configured as retryable.
    """
    tag = label
    # Nested-retry guard, mirroring :func:`with_retries`: when an outer llmkit
    # retry loop is already active in this task (the documented composable path
    # — a host wrapping stream consumption, since mid-stream errors propagate
    # unretried), collapse to a single pass so the budgets don't multiply (the
    # 3 x 3 = 9 trap). The accidental double-wrap warns; an explicit NO_RETRY
    # inner stays silent.
    if _in_active_retry_scope():
        yielded_any = False
        try:
            async with aclosing(attempt_factory(1)) as attempt_stream:
                async for chunk in attempt_stream:
                    yielded_any = True
                    yield chunk
            return
        except Exception as exc:
            # Stricter than ``with_retries``'s policy-shaped test on purpose:
            # the inner layer only "would have retried" if the failure was
            # actually retryable *and* arrived pre-first-chunk.
            retry_class = _retry_class(exc, policy)
            would_have_retried = (
                policy.max_attempts > 1
                and not yielded_any
                and retry_class is not None
                and (not retry_class or policy.validation_max_attempts > 1)
            )
            if would_have_retried:
                warnings.warn(
                    f"{surface}({tag!r}) is nested inside an already-retrying "
                    + "llmkit retry loop; this inner layer ran a single pass to avoid "
                    + "multiplying retry budgets (the outer loop owns the retries). To "
                    + "drive retries from an outer wrapper around a call function, opt "
                    + "the inner call out with retry=NO_RETRY.",
                    RuntimeWarning,
                    stacklevel=warn_stacklevel,
                )
            raise

    validation_failures = 0
    ledger = _CallLedger()
    token = _retry_scope.set(_RetryScope(ledger))
    try:
        for attempt in range(1, policy.max_attempts + 1):
            ledger.begin_pass()
            yielded_any = False
            try:
                async with aclosing(attempt_factory(attempt)) as attempt_stream:
                    async for chunk in attempt_stream:
                        yielded_any = True
                        _retry_scope.reset(token)
                        try:
                            yield chunk
                        finally:
                            # The loop's one ledger, so a call the attempt
                            # prepares after this chunk is counted by it too.
                            token = _retry_scope.set(_RetryScope(ledger))
                return
            except Exception as exc:
                is_validation = _retry_class(exc, policy)
                if yielded_any or is_validation is None:
                    raise
                if is_validation:
                    validation_failures += 1
                if attempt >= policy.max_attempts or (
                    is_validation and validation_failures >= policy.validation_max_attempts
                ):
                    logger.error("%s: stopped after %d attempts: %s", tag, attempt, exc)
                    raise
                await handle_retry_failure(
                    tag=tag,
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    error=exc,
                    backoff_base_seconds=policy.backoff_base_seconds,
                    max_backoff_seconds=policy.max_backoff_seconds,
                    retry_after_cap=policy.retry_after_cap,
                )
    finally:
        _retry_scope.reset(token)


def _retry_class(error: BaseException, policy: RetryPolicy) -> bool | None:
    """True for validation, False for transport, None for an ineligible error.

    Instructor wraps transport and permanent errors as well as parse failures.
    Classify its cause first so the wrapper cannot turn a permanent error into
    a validation retry. Explicit transport-type overrides retain precedence
    over fail-fast exclusions, as on the buffered call path.
    """
    cause = underlying_provider_error(error)
    explicit = isinstance(error, policy.retry_on)
    if isinstance(cause, _ZERO_BUDGET_ERRORS) and not explicit:
        return None
    is_transport = isinstance(cause, policy.retry_on)
    if (
        cause is not error
        and not is_transport
        and not isinstance(cause, _SCHEMA_SHAPED_CAUSES)
        and not explicit
    ):
        return None
    if not is_transport and isinstance(error, policy.validation_retry_on):
        return True
    if explicit or is_transport:
        return False
    return None


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    label: str | None = None,
) -> T:
    """Retry an async callable under the same policy as ordinary calls.

    ``policy.max_attempts`` is the total outer-pass ceiling, including the
    first pass. At most ``validation_max_attempts - 1`` retries may be caused
    by validation failures, within that ceiling. Switching failure categories
    never resets either count or the exponential backoff. Defaults permit
    three passes, with at most one validation-caused retry. Use ``NO_RETRY``
    for a single pass, or a custom ``RetryPolicy`` to set budgets and types.
    Permanent and programming errors fail immediately under the default policy.

    The ceiling counts executions of ``fn``, not provider requests, spending,
    or elapsed time: a call may perform an in-call schema repair or temperature
    reissue. Provider ``Retry-After`` still takes precedence over computed
    backoff, within the policy's cap.

    The call functions already retry internally. A retry loop nested in this
    loop's task collapses to one pass, so budgets do not multiply. An accidental
    double-wrap warns; set the inner call's ``retry=NO_RETRY`` to make the outer
    policy its sole owner. A separate task retains its own budget. Calls re-run
    by the outer loop share a logical ``call_id`` and number their attempts.

    Args:
        fn: Zero-argument async callable to execute.
        policy: Retry classification, total ceiling, validation allowance and
            backoff settings. Defaults to ``DEFAULT_RETRY_POLICY``.
        label: Optional identifier for log messages.

    Returns:
        The result of the first successful call.

    Raises:
        The last failure when either applicable limit is exhausted, or an
        ineligible error immediately.
    """
    tag = label or "retry"
    if _in_active_retry_scope():
        if policy.max_attempts > 1:
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

    attempt = 0
    validation_failures = 0
    ledger = _CallLedger()
    token = _retry_scope.set(_RetryScope(ledger))
    try:
        while True:
            attempt += 1
            ledger.begin_pass()
            try:
                return await fn()
            except Exception as error:
                is_validation = _retry_class(error, policy)
                if is_validation is None:
                    raise
                if is_validation:
                    validation_failures += 1
                if attempt >= policy.max_attempts or (
                    is_validation and validation_failures >= policy.validation_max_attempts
                ):
                    logger.error("%s: stopped after %d attempts: %s", tag, attempt, error)
                    raise
                await handle_retry_failure(
                    tag=tag,
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    error=error,
                    backoff_base_seconds=policy.backoff_base_seconds,
                    max_backoff_seconds=policy.max_backoff_seconds,
                    retry_after_cap=policy.retry_after_cap,
                )
    finally:
        _retry_scope.reset(token)
