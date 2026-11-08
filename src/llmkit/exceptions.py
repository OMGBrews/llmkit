"""Recoverable exception types for LLM service calls.

Use ``LLM_RECOVERABLE_ERRORS`` in ``except`` clauses to catch expected LLM
operational failures (network errors, rate limits, transient 5xx provider
errors, schema-validation/parsing failures, timeouts) while letting
programming errors (TypeError, AttributeError) — and *permanent* request
errors such as authentication (401) and bad-request (400) failures —
propagate.

The recoverable set is split into five subsets the retry layer classifies
within one overall attempt ceiling (see :mod:`llmkit.retry`):

* :data:`LLM_TRANSPORT_ERRORS` — transient *transport* failures (rate
  limits, transient 5xx, network/timeout). These may retry up to the overall
  attempt ceiling, since a retry on a fresh connection routinely succeeds.
* :data:`LLM_SCHEMA_ERRORS` — *schema-validation* failures (llmkit's
  :class:`StructuredOutputError`, raised when a structured answer fails
  validation after instructor's in-call repair, or when a compose lane's final
  answer fails validation; pydantic ``ValidationError``; instructor's
  ``InstructorRetryException`` for an exhausted attempt that was not a parse
  failure). These get a **lower** retry allowance inside that ceiling: a
  transiently-malformed JSON response still earns one cross-call retry, but a deterministically-wrong
  schema can't burn the full attempt ceiling on doomed re-asks.

  Caveat — ``InstructorRetryException`` is a *wrapper*: instructor re-raises
  **every** exhausted attempt this way, including transport failures (a 429
  or network blip caught inside its create call). llmkit re-owns the
  parse-shaped ones as :class:`StructuredOutputError`; the rest keep the
  wrapper, and the retry layer unwraps it via :func:`underlying_provider_error`
  and gives a wrapped transport cause no validation charge — so
  membership here is the default-but-not-final classification, not a promise
  the failure is genuinely schema-shaped.
* :data:`LLM_BACKPRESSURE_ERRORS` — llmkit's *own* fail-fast backpressure
  signal (:class:`CircuitOpenError`, raised by the opt-in per-provider circuit
  breaker while it is open). It is recoverable in the catch-set sense — a host
  that degrades on ``LLM_RECOVERABLE_ERRORS`` keeps catching it (a fast
  fallback) instead of crashing on a new uncaught type — but it is deliberately
  **not** retried by default: it lives outside the transport set,
  and ``with_retries`` additionally carves it out as a zero-budget fail-fast
  signal, so neither ``with_retries`` with its default policy
  nor the streaming loop (which keys off the transport/schema subsets, never
  the union) ever re-asks a circuit the breaker already knows is open — unless
  a caller explicitly lists the type in ``retry_on``.
* :data:`LLM_OUTPUT_LIMIT_ERRORS` — llmkit's *own* fail-fast truncation signal
  (:class:`OutputLimitError`, raised when a structured or text completion, or
  a compose call's final answer, is cut off by the output-token limit). Recoverable in the same catch-set sense as the
  breaker, but **never retried** on either budget by default:
  re-asking with an identical token budget can only truncate again — unless
  a caller explicitly lists the type in ``retry_on``.
* :data:`LLM_TOOL_ERRORS` — a tool round the model got wrong
  (:class:`ToolArgumentError`: an unknown tool, or arguments that are not
  valid JSON or fail the tool's model). The tool lanes charge it to the
  **validation** allowance, so a round whose every call is malformed is re-asked
  like a malformed schema response — on the streaming lane only while nothing
  has been yielded.

``LLM_RECOVERABLE_ERRORS`` is the **union** of the five, preserved as the
single documented ``except``-clause catch-set. At llmkit's provider boundaries,
uncovered OpenAI/LiteLLM API errors carrying mapped 500/502/529 become
:class:`TransientProviderError`, a plain member of the transport tuple.
Already recognized transport errors retain their types. Custom retry policies
match the normalized type, not its original ``__cause__``. The 503 case deserves a note:
litellm's own ``ServiceUnavailableError`` cannot be listed statically
(that would cost every ``import llmkit`` the multi-second litellm import),
so the transport boundary (:mod:`llmkit._litellm`) re-raises every
litellm-native 503 as llmkit's own :class:`ServiceUnavailableError` — a
plain class in the tuple, so the documented ``except`` pattern genuinely
catches it. A lazy stand-in (:class:`_LiteLLMServiceUnavailableError`)
additionally keeps *raw* litellm 503s classifying as transport in
``isinstance`` checks (e.g. a host's own litellm call wrapped in
:func:`~llmkit.retry.with_retries`).

``with_retries()`` (see :mod:`llmkit.retry`) is the transient-retry
layer; in-call schema repair is handled separately by instructor's retry
loop (two total attempts, pinned in :mod:`llmkit._litellm` — with the one
carve-out that a length truncation is never re-asked and surfaces as
:class:`OutputLimitError` instead).
"""

import functools
import sys
from dataclasses import dataclass
from json import JSONDecodeError
from typing import override

import httpx
import openai
from instructor.core.exceptions import (
    AsyncValidationError,
    InstructorRetryException,
    ResponseParsingError,
)
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from tenacity import RetryError

# Transient *transport* failures — the default transport retry budget retries
# this exact set. We name the specific transient ``openai`` subclasses rather
# than their shared ``openai.APIError`` base, because that base also covers
# permanent 4xx errors (``AuthenticationError``/401, ``BadRequestError``/400,
# ``PermissionDeniedError``/403, ...) which must fail fast instead of burning
# the retry budget. Transport set:
#   - ``RateLimitError``      — 429
#   - ``InternalServerError`` — 5xx
#   - ``APIConnectionError``  — network failures (``APITimeoutError`` subclasses it)
#   - ``litellm ServiceUnavailableError`` — 503 (see below)
#   - ``TransientProviderError`` — uncovered mapped 500/502/529 at our boundary
#   - ``httpx.RequestError``  — lower-level network failures
#   - ``TimeoutError``        — builtin timeout
#
# LiteLLM raises its *own* exception classes, which we catch via their
# ``openai`` bases: ``litellm.RateLimitError`` subclasses
# ``openai.RateLimitError``, ``litellm.InternalServerError`` subclasses
# ``openai.InternalServerError``, ``litellm.Timeout``/``litellm.APIConnectionError``
# subclass ``openai.APIConnectionError``. Native 503 does not follow that
# pattern: ``litellm.exceptions.ServiceUnavailableError``
# (LiteLLM's mapping for HTTP 503): its MRO goes straight to
# ``openai.APIStatusError``, skipping ``openai.InternalServerError`` — so it
# must be listed explicitly or real 503s would propagate unretried.
#
# The provider boundary additionally normalizes uncovered API errors carrying
# mapped status 500/502/529 to TransientProviderError. Generic APIError and
# APIStatusError remain outside the tuple: permanent errors and nonselected
# statuses must not gain retries. Already recognized transport types stay intact.


class _LazyServiceUnavailableMeta(type):
    """Resolves litellm's 503 class at *classification* time, not import time.

    ``litellm.exceptions.ServiceUnavailableError`` belongs in the transport
    set, but a module-level ``import litellm`` here would make every
    ``import llmkit`` pay the full multi-second litellm import (this module is
    imported eagerly by ``llmkit/__init__.py``). A litellm exception
    *instance* cannot exist unless ``litellm`` is already in ``sys.modules``,
    so the class is resolved lazily inside ``__instancecheck__``: before
    litellm is loaded nothing can match, and once it is, the resolved class
    is cached.
    """

    _resolved: type[Exception] | None = None

    @override
    def __instancecheck__(cls, instance: object) -> bool:
        if cls._resolved is None:
            if "litellm" not in sys.modules:
                # litellm not loaded — no litellm exception instance can exist.
                return False
            import litellm

            cls._resolved = litellm.exceptions.ServiceUnavailableError
        return isinstance(instance, cls._resolved)


class _LiteLLMServiceUnavailableError(Exception, metaclass=_LazyServiceUnavailableMeta):
    """Lazy stand-in for ``litellm.exceptions.ServiceUnavailableError``.

    Matches exactly what the real class matches in ``isinstance`` checks —
    the form the retry layer and rate limiter use to classify errors —
    without importing litellm at ``import llmkit`` time. Never raise this
    class directly.

    Known limit: ``except`` matching bypasses ``__instancecheck__``, so an
    ``except`` clause over the transport tuple does not match this entry.
    That limit no longer reaches hosts through llmkit's call functions: the
    transport boundary (:func:`normalize_transient_provider_error`, applied in
    :mod:`llmkit._litellm`) re-raises every litellm-native 503 as
    :class:`ServiceUnavailableError`, a plain tuple member ``except`` does
    match. This stand-in stays for ``isinstance`` classification of the raw
    litellm class — the shape a host's *own* litellm call wrapped in
    :func:`~llmkit.retry.with_retries` still raises.
    """


class LlmkitError(Exception):
    """The base of every exception class llmkit defines.

    ``except LlmkitError`` catches every failure llmkit itself raises: a
    provider outage it normalized (:class:`ServiceUnavailableError` or
    :class:`TransientProviderError`), its
    own fail-fast signals (:class:`CircuitOpenError`,
    :class:`OutputLimitError`), validation it ran or re-owned
    (:class:`StructuredOutputError`, :class:`ToolArgumentError`,
    :class:`ResultValidationError`), and its configuration errors
    (:class:`LLMClientNotConfiguredError`, :class:`ComposeUnsupportedError`,
    :class:`ToolChoiceUnsupportedError`,
    :class:`ReasoningEffortUnsupportedError`, :class:`SchemaEngineError`,
    :class:`UnsupportedSchemaError`). It
    does not catch the third-party errors a call can also raise (``openai``
    and ``httpx`` transport errors, a bare pydantic ``ValidationError``, and
    instructor's ``InstructorRetryException`` wrapping a transport or
    permanent provider error); :data:`LLM_RECOVERABLE_ERRORS` and its subsets name those, and
    remain the sets to catch for retry-shaped policy.

    Do not list this class in :attr:`~llmkit.RetryPolicy.retry_on`: retry
    matching is ``isinstance``, so it would re-enable retries for every
    subclass that is deliberately given no budget, such as
    :class:`CircuitOpenError` and :class:`OutputLimitError`.
    """


class LLMClientNotConfiguredError(LlmkitError, RuntimeError):
    """A call needed the registered client config and none is registered.

    Raised before the first attempt when a call names no ``provider=`` and
    :func:`~llmkit.configure_llm_client` was never called or the registration
    was cleared, and by any other read of the registration. A
    :class:`RuntimeError` subclass, so code that caught the earlier plain
    ``RuntimeError`` keeps working.
    """


class ServiceUnavailableError(LlmkitError):
    """A provider reported itself unavailable (HTTP 503).

    llmkit's **owned, catchable** form of litellm's ``ServiceUnavailableError``.
    litellm's class cannot appear in the transport tuple statically (the
    import-time cost), and the lazy stand-in above matches only in
    ``isinstance`` checks — CPython's ``except`` matching bypasses metaclass
    hooks — so a raw litellm 503 escaping the library would slip through the
    documented ``except LLM_RECOVERABLE_ERRORS`` net. The transport boundary
    (:mod:`llmkit._litellm`) therefore re-raises every litellm-native 503 as
    this class via :func:`normalize_transient_provider_error`, with the original on
    ``__cause__`` — the :class:`CircuitOpenError` / :class:`OutputLimitError`
    precedent of one plain llmkit-owned type per fail signal.

    Carries exactly the fields downstream consumers read (measured against
    litellm's class, 2026-07-18): ``status_code == 503`` keeps the AIMD/breaker
    throttle classification seeing the overload signal, and ``response`` keeps
    a server ``Retry-After`` header readable by the retry layer's backoff —
    so normalization changes the *type* a host catches, never the retry timing
    or backpressure behaviour.

    Attributes:
        provider: litellm's provider tag for the failing call (its
            ``llm_provider``), or ``None`` when the original carried none.
        model: The model string the call ran against, or ``None``.
        status_code: Always ``503``.
        response: The original error's ``httpx.Response`` (carrying any
            ``Retry-After`` header), or ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None,
        model: str | None,
        response: httpx.Response | None,
    ) -> None:
        super().__init__(message)
        self.provider: str | None = provider
        self.model: str | None = model
        self.status_code: int = 503
        self.response: httpx.Response | None = response


class TransientProviderError(LlmkitError):
    """An otherwise unclassified provider API error with mapped HTTP 500/502/529.

    Raised at llmkit's provider boundaries, with the original on ``__cause__``.
    Already recognized transport errors keep their types, including the
    existing :class:`ServiceUnavailableError` for native LiteLLM 503s. Custom
    ``RetryPolicy.retry_on`` tuples must name this type to retry these newly
    normalized failures; classification does not follow the original cause.

    Metadata reflects what the mapped error still carries, not necessarily
    the original wire response. ``provider`` and ``model`` are optional tags;
    ``status_code`` retains the mapped status, ``response`` retains the original
    httpx response, and numeric ``retry_after`` takes precedence over its headers.
    Missing metadata is ``None``. Normalization cannot reconstruct discarded
    wire status or headers, and does not cover independent raw LiteLLM calls.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        provider: str | None,
        model: str | None,
        response: httpx.Response | None,
        retry_after: int | float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int = status_code
        self.provider: str | None = provider
        self.model: str | None = model
        self.response: httpx.Response | None = response
        self.retry_after: int | float | None = retry_after


def normalize_transient_provider_error(exc: Exception) -> Exception:
    """Own native LiteLLM 503s and uncovered API errors with mapped 500/502/529.

    All other errors retain their identity, including already recognized
    transport errors and non-provider exceptions with a status-like attribute.
    The caller re-raises changed results ``from`` the original. Metadata reads
    are best-effort; the normalizer is internal and never imports LiteLLM eagerly.
    """
    native_503 = isinstance(exc, _LiteLLMServiceUnavailableError)
    if not native_503 and (
        isinstance(exc, LLM_TRANSPORT_ERRORS) or not isinstance(exc, openai.APIError)
    ):
        return exc
    provider = getattr(exc, "llm_provider", None)
    model = getattr(exc, "model", None)
    response = getattr(exc, "response", None)
    provider = provider if isinstance(provider, str) else None
    model = model if isinstance(model, str) else None
    response = response if isinstance(response, httpx.Response) else None
    if native_503:
        return ServiceUnavailableError(str(exc), provider=provider, model=model, response=response)
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or status not in {500, 502, 529}:
        return exc
    retry_after = getattr(exc, "retry_after", None)
    return TransientProviderError(
        str(exc),
        status_code=status,
        provider=provider,
        model=model,
        response=response,
        retry_after=retry_after if isinstance(retry_after, (int, float)) else None,
    )


LLM_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    # llmkit-owned 503 — what the transport boundary re-raises for every
    # litellm-native 503, so ``except`` clauses over this tuple catch it.
    ServiceUnavailableError,
    TransientProviderError,
    # Raw litellm-native 503 — matches in ``isinstance`` classification only
    # (see the stand-in's docstring); kept so a host's own litellm call
    # wrapped in ``with_retries`` still retries its 503s. The lazy stand-in
    # keeps ``import llmkit`` litellm-free.
    _LiteLLMServiceUnavailableError,
    openai.APIConnectionError,
    httpx.RequestError,
    TimeoutError,
)


class StructuredOutputError(LlmkitError):
    """A structured answer did not validate against the caller's schema.

    Raised by :func:`~llmkit.structured_llm_call` when instructor's in-call
    schema repair is exhausted on a genuine parse failure (malformed JSON, a
    field or model validator that rejected the answer, required fields
    missing) or the provider returned no choices, and by
    :func:`~llmkit.tool_llm_call` with ``output_schema=``
    when the final answer does not validate or the turn carried neither tool
    calls nor text. One llmkit-owned type for the one failure the contract
    names, so a caller can read what failed without knowing instructor's
    wrapper type. The cause is on ``__cause__``: instructor's exception on the
    structured lane, pydantic's ``ValidationError`` on the compose lane, and
    nothing when a compose turn carried no text. The message names each
    failing field and reason (:func:`validation_failure_summary`), never the
    answer itself.

    Only parse-shaped failures are re-owned: a transport or permanent provider
    error that escapes instructor keeps its ``InstructorRetryException`` wrap,
    so :func:`underlying_provider_error`, ``Retry-After`` handling and throttle
    detection read it exactly as before.

    A member of :data:`LLM_SCHEMA_ERRORS` and so of
    :data:`LLM_RECOVERABLE_ERRORS`: the call re-asks it on the validation
    budget (``RetryPolicy.validation_max_attempts``), and the last one
    propagates when that budget is spent.

    Attributes:
        schema_name: The ``__name__`` of the requested output schema.
        raw_text: The answer text that failed, unparsed, from the last
            in-call attempt; ``None`` when the completion carried no text
            (no choices, or a compose turn with neither tool calls nor text).
            Never put into the message.
        errors: pydantic's structured errors
            (``ValidationError.errors(include_url=False)``) when pydantic
            rejected the answer, each naming the failing field in ``loc``
            (malformed JSON reports one ``json_invalid`` error with an empty
            ``loc``); ``None`` when the failure was not a pydantic
            ``ValidationError``, such as instructor's ``ResponseParsingError``
            for a response with nothing to parse.
        attempt: The call-level attempt number this failure ended (the same
            value the attempt's log record carries), or ``None``.
        model: The routed LiteLLM model id the call ran against (such as
            ``openai/gpt-4.1-mini``), or ``None``.
        provider: The provider name the call ran against, or ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        schema_name: str,
        raw_text: str | None,
        errors: list[ErrorDetails] | None,
        attempt: int | None,
        model: str | None,
        provider: str | None,
    ) -> None:
        super().__init__(message)
        self.schema_name: str = schema_name
        self.raw_text: str | None = raw_text
        self.errors: list[ErrorDetails] | None = errors
        self.attempt: int | None = attempt
        self.model: str | None = model
        self.provider: str | None = provider

    @override
    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        # Keyword-only fields: rebuild through a partial so the error pickles
        # (a process pool or a task queue may carry it across a boundary).
        return (
            functools.partial(
                type(self),
                self.args[0] if self.args else "",
                schema_name=self.schema_name,
                raw_text=self.raw_text,
                errors=self.errors,
                attempt=self.attempt,
                model=self.model,
                provider=self.provider,
            ),
            (),
        )


def validation_failure_summary(exc: BaseException) -> str:
    """A one-line account of why an answer failed, without the answer itself.

    For :class:`StructuredOutputError`'s message, which lands in the call
    log's ``error``: pydantic's own message quotes each failing input, so a
    ``ValidationError`` is summarized from its errors with inputs left out
    (``"n: Field required [missing]"``). Any other cause keeps its own text.
    Not exported: an internal formatting helper.
    """
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(part) for part in error['loc']) or '(answer)'}: "
            + f"{error['msg']} [{error['type']}]"
            for error in exc.errors(include_url=False, include_input=False, include_context=False)
        ]
        return f"{exc.error_count()} validation error(s): " + "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"


# Schema-validation failures — a lower retry allowance within the overall
# attempt ceiling, distinct from the transport set, so a deterministic schema
# failure cannot consume the full ceiling on doomed re-asks while a
# transiently-malformed JSON response still earns one cross-call retry.
#   - ``StructuredOutputError``     — llmkit's own type for a structured answer
#     that failed validation, on either the structured or the compose lane.
#     The structured lane re-raises an exhausted parse-cause
#     ``InstructorRetryException`` as this type.
#   - ``ValidationError``           — pydantic could not parse the response
#   - ``InstructorRetryException``  — instructor's in-call loop gave up on a
#     failure that is *not* parse-shaped. instructor wraps *any* failure
#     escaping that loop this way — transport and permanent errors included,
#     which llmkit's parse-only predicate declines after a single attempt (no
#     re-ask). The retry layer unwraps it (``underlying_provider_error``) so a
#     wrapped transport cause does not consume this allowance, and a wrapped
#     permanent error fails fast. Every pass still counts toward max_attempts.
LLM_SCHEMA_ERRORS: tuple[type[Exception], ...] = (
    StructuredOutputError,
    ValidationError,
    InstructorRetryException,
)


class ToolArgumentError(LlmkitError):
    """A model requested an unknown tool or supplied invalid arguments.

    ``round_errors`` carries every rejection in an all-malformed round, for
    repair re-asks. Individual salvaged-round errors leave it empty. The
    streaming lane re-asks the round too, but only when it streamed no prose
    first; after a yield it raises with the round attached.
    """

    def __init__(
        self,
        tool_name: str | None,
        call_id: str | None,
        arguments_raw: str | None,
        reason: str,
        *,
        round_errors: tuple["ToolArgumentError", ...] = (),
    ):
        super().__init__(reason)
        self.tool_name: str | None = tool_name
        self.call_id: str | None = call_id
        self.arguments_raw: str | None = arguments_raw
        self.round_errors: tuple[ToolArgumentError, ...] = round_errors


LLM_TOOL_ERRORS: tuple[type[Exception], ...] = (ToolArgumentError,)

# The genuine *parse* failures — a malformed or schema-invalid response the model
# can plausibly fix if asked again. This is the **single source of truth** for two
# decisions that must never drift apart (a type instructor re-asks in-call must
# also earn the validation budget across calls, never be mis-routed to fail-fast):
#   * instructor's in-call re-ask predicate
#     (:func:`llmkit._litellm._schema_repair_retrying`) retries exactly this set,
#     mirroring instructor's own private ``_RETRYABLE_PARSE_ERRORS`` from public
#     sources rather than importing the private tuple; and
#   * the outer retry layer (:mod:`llmkit.retry`) treats an unwrapped cause in
#     this set as *schema-shaped* — charged the lower validation budget — while a
#     wrapped cause outside it (and outside the transport set) is a *permanent*
#     error (an exhausted 401/400/403) that must fail fast.
#   - ``ValidationError``       — pydantic rejected the parse
#   - ``JSONDecodeError``       — the response wasn't valid JSON
#   - ``AsyncValidationError``  — an async field validator rejected the parse.
#     Unreachable on the supported instructor range (1.17.x): instructor never
#     runs async validators, and llmkit refuses a schema that declares one with
#     :class:`UnsupportedSchemaError` before any request. Kept so this set keeps
#     mirroring instructor's own parse-error set.
#   - ``ResponseParsingError``  — instructor couldn't parse the response at all
#     (e.g. a blocked/empty Gemini ``Mode.JSON`` generation)
# Deliberately *not* in the exported ``LLM_*_ERRORS`` family: these raw parse
# types are an internal routing detail (a structured call surfaces an exhausted
# one as :class:`StructuredOutputError` ∈ ``LLM_SCHEMA_ERRORS``, which is what a
# host catches), not a user-facing catch-set.
REPAIRABLE_PARSE_ERRORS: tuple[type[Exception], ...] = (
    ValidationError,
    JSONDecodeError,
    AsyncValidationError,
    ResponseParsingError,
)


class CircuitOpenError(LlmkitError):
    """Raised by the per-provider circuit breaker when it declines to admit a call.

    The breaker (opt-in via ``configure_rate_limit(breaker=True)``) is the
    aggregate guard that stops doomed work when a provider is *down*: once its
    outage rate over a rolling window crosses the trip threshold it opens, and
    the limiter then raises this **immediately** — before any concurrency slot is
    held or RPM token deducted — instead of admitting a call that would only burn
    its retry budget into the storm. It is raised whenever the breaker won't admit
    the call: while **OPEN** within its cooldown, and while **HALF_OPEN** when the
    single recovery probe is already in flight (every other concurrent caller is
    turned away so exactly one probe tests the provider). Either way the right
    response is the same — fall back fast — and it carries :attr:`provider`, the
    normalized (casefolded) provider key the breaker is keyed under (the same
    identity :class:`~llmkit.BackpressureEvent` reports), so a host's fallback can
    branch on *which* provider tripped.

    Deliberately a **distinct, fail-fast** type, kept **out** of
    :data:`LLM_TRANSPORT_ERRORS` (and so out of the default ``retry_on``):
    retrying a circuit the breaker already knows is open would defeat its
    purpose. :func:`~llmkit.retry.with_retries` treats it as a zero-budget
    fail-fast signal: it is not re-asked unless a caller explicitly lists
    the type in ``RetryPolicy.retry_on``
    (an explicit opt-in wins, matching :class:`OutputLimitError`). It *is* a
    member of :data:`LLM_BACKPRESSURE_ERRORS` and therefore of
    :data:`LLM_RECOVERABLE_ERRORS`, so a host that already writes
    ``except LLM_RECOVERABLE_ERRORS`` to degrade on a 503 keeps catching this
    (and falls back fast) rather than crashing on a new uncaught type.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(f"circuit breaker open for provider {provider!r}")
        self.provider: str = provider


# llmkit's own fail-fast backpressure signal — kept in its own subset so the
# recoverable *union* can carry it (for host ``except`` clauses) while the retry
# layer, which budgets off the transport/schema subsets, never retries it.
LLM_BACKPRESSURE_ERRORS: tuple[type[Exception], ...] = (CircuitOpenError,)


@dataclass(frozen=True)
class OutputLimitDiagnostics:
    """What a truncated structured or text completion reported, best-effort.

    Carried by :attr:`OutputLimitError.diagnostics` and written to the call
    log: ``partial_text`` as the record's ``partial_text`` field, the rest as
    its ``output_limit`` mapping (:meth:`to_log_dict`). Every field is
    ``None`` when the completion did not report it; an empty ``partial_text``
    (``""``) means the model returned content with no text in it.

    Attributes:
        partial_text: The unfinished answer text, unparsed — not a validated
            response. For a text stream, the transcript already yielded.
        finish_reason: The provider's stop reason (``"length"`` for an
            OpenAI-shaped truncation).
        prompt_tokens: Input tokens, from the completion's usage.
        completion_tokens: Generated tokens, from the completion's usage. May
            include reasoning tokens that never appear in ``partial_text``.
        total_tokens: Prompt plus completion tokens, as reported.
        reasoning_tokens: The reasoning share of ``completion_tokens`` when
            the provider breaks it out. ``0`` is ambiguous: instructor's usage
            accumulator writes ``0`` when the provider reported no breakdown.
    """

    partial_text: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    def to_log_dict(self) -> dict[str, int | str | None]:
        """The non-text fields as the call record's ``output_limit`` mapping."""
        return {
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


class OutputLimitError(LlmkitError):
    """A structured, text or compose final-answer completion was truncated by the output limit.

    Raised by the structured path when the provider stopped generating at the
    token ceiling (``finish_reason='length'`` / Gemini ``MAX_TOKENS``) before a
    complete response could be parsed, and by the buffered and streamed text
    paths when the provider explicitly reports ``finish_reason='length'`` — a
    text stream raises it after its last chunk, so the text already yielded
    is the partial answer. A missing finish reason is never read as a
    truncation. A compose call's final answer (:func:`~llmkit.tool_llm_call`
    with ``output_schema=``) raises it too; every other tool call reports the
    reason as ``stop_reason`` instead.
    Deliberately **never retried**:
    re-asking with an identical token budget can only truncate again — the
    motivating production failure was a degenerate repetition loop that burned
    to the provider ceiling on the original ask *and* on every blind re-ask —
    so instructor's in-call repair loop refuses the re-ask (see
    :mod:`llmkit._litellm`) and :func:`~llmkit.retry.with_retries` propagates
    it immediately under every configuration, unless a caller explicitly lists
    the type in ``retry_on`` (an explicit opt-in wins — though a text stream
    that has already yielded text is never replayed, even then).

    Carries what the truncated completion reported, so the failure can be
    investigated without paying to reproduce it — but a truncation does not
    say *why* it happened. The answer may legitimately need more room, the
    model may have been repeating itself, reasoning may have consumed the
    budget, or the prompt may have lacked the evidence to finish; read
    :attr:`diagnostics` (the partial text above all) before deciding.
    ``completion_tokens`` may include reasoning tokens the provider does not
    surface as text, so a count at the cap is not proof the visible answer
    filled it. Fail-fast is the default; retrying is an explicit
    ``retry_on`` opt-in.

    Follows the :class:`CircuitOpenError` precedent — a distinct fail-fast
    type, kept **out** of :data:`LLM_TRANSPORT_ERRORS` (a fresh connection
    doesn't change the budget) and :data:`LLM_SCHEMA_ERRORS` (a repair re-ask
    doesn't either), but **in** :data:`LLM_OUTPUT_LIMIT_ERRORS` and therefore
    in :data:`LLM_RECOVERABLE_ERRORS`, so a host that degrades on
    ``except LLM_RECOVERABLE_ERRORS`` keeps catching it instead of crashing on
    a new uncaught type.

    Attributes:
        model: The resolved litellm model string the call ran against.
        max_tokens: The cap sent on this call; ``None`` means no cap was set
            (the provider's own ceiling applied).
        completion_tokens: Tokens actually generated before truncation,
            best-effort from the truncated completion's usage (``None`` when
            the provider reported none).
        diagnostics: The truncated completion's partial text, finish reason
            and token counts as an :class:`OutputLimitDiagnostics`, or
            ``None`` when the error was constructed without them. Any field
            inside may itself be ``None`` when the completion did not report
            it. The partial text is never put into the message.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int | None,
        completion_tokens: int | None,
        diagnostics: OutputLimitDiagnostics | None = None,
    ) -> None:
        cap = "provider ceiling" if max_tokens is None else f"max_tokens={max_tokens}"
        super().__init__(
            f"completion for model {model!r} truncated by the output-token limit "
            + f"({cap}, generated {completion_tokens} tokens); not retried — an identical "
            + "budget can only truncate again"
        )
        self.model: str = model
        self.max_tokens: int | None = max_tokens
        self.completion_tokens: int | None = completion_tokens
        self.diagnostics: OutputLimitDiagnostics | None = diagnostics


# llmkit's own fail-fast truncation signal — same pattern as the backpressure
# subset: the recoverable *union* carries it (so host ``except`` nets keep
# degrading gracefully) while the retry layer never retries it on any budget.
LLM_OUTPUT_LIMIT_ERRORS: tuple[type[Exception], ...] = (OutputLimitError,)

# The full recoverable set is the union of the five subsets — preserved as the
# documented single catch-set for ``except`` clauses, so callers keep catching
# exactly what they did before even though the retry layer now budgets the
# subsets within one ceiling (and, unless a caller opts in through
# ``retry_on``, never retries the backpressure/output-limit subsets).
LLM_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    *LLM_TRANSPORT_ERRORS,
    *LLM_SCHEMA_ERRORS,
    *LLM_BACKPRESSURE_ERRORS,
    *LLM_OUTPUT_LIMIT_ERRORS,
    *LLM_TOOL_ERRORS,
)


def underlying_provider_error(exc: BaseException) -> BaseException:
    """Dig the original provider error out of an ``InstructorRetryException``.

    instructor wraps *every* failure that escapes its in-call loop this way —
    transport failures (rate limits, 5xx, network) and permanent errors
    (401/400/403) just as much as schema-validation ones. The wrapper alone
    can't tell the retry layer which budget the failure belongs to, so this
    returns the wrapped root error for the caller to classify.

    The wrap shape on the supported instructor 1.17.x (unchanged since 1.15):
    ``args[0]`` is always the
    message ``str`` (never the exception), and the root error rides
    ``__cause__`` **raw**. instructor's blanket
    ``except Exception as e: raise InstructorRetryException(...) from e`` catches
    whatever escapes its tenacity loop, and because llmkit pins that loop with
    ``reraise=True`` (:func:`llmkit._litellm._schema_repair_retrying`), tenacity
    re-raises the original error bare on *both* paths — a *declined* predicate
    (every non-parse error, after the in-call re-ask was restricted to parse
    failures) and an *exhausted* parse re-ask alike. So ``__cause__`` is the raw
    provider/parse error in all current cases, handled by the second branch.

    The first branch — ``__cause__`` a tenacity ``RetryError`` whose
    ``last_attempt`` Future holds the root error — is **defensive**: it is the
    shape instructor produced *before* ``reraise=True`` (and the shape a future
    ``instructor<2`` could reintroduce), unwrapped via the public
    ``Future.exception()``. The ``args[0]``-is-an-exception branch is likewise
    retained only as harmless compatibility with instructor's pre-1.15 shape;
    both are dead on the supported range but cheap insurance against it moving.

    Any other exception (a bare ``ValidationError``, a transport error that
    never went through instructor) is returned unchanged, so callers can
    classify ``underlying_provider_error(exc)`` uniformly.
    """
    if isinstance(exc, InstructorRetryException):
        # Pre-1.15 compat only: older instructor stored the root exception as
        # the first positional arg. Since 1.15 ``args[0]`` is the message str,
        # so this branch never fires there — kept because it is harmless.
        inner = exc.args[0] if exc.args else None
        if isinstance(inner, BaseException):
            return inner
        cause = exc.__cause__
        if isinstance(cause, RetryError):
            # Defensive (pre-``reraise=True`` / future-instructor shape): the
            # last attempt's Future carries the root error. Public accessor,
            # not the private ``._exception``.
            recorded = cause.last_attempt.exception() if cause.last_attempt else None
            if isinstance(recorded, BaseException):
                return recorded
        elif isinstance(cause, BaseException):
            # The 1.15-and-later shape for every current case (``reraise=True``):
            # instructor wrapped the raw error directly via ``raise ... from e``
            # — whether the in-call predicate declined it (transport/permanent)
            # or an exhausted parse re-ask re-raised it bare.
            return cause
    return exc


class ResultValidationError(LlmkitError):
    """Raised by a call function's ``on_result`` hook to reject a result.

    The signal a host uses to fold an LLM-then-validate-then-re-roll loop into
    the call functions themselves. When a structured/text call's ``on_result``
    callback raises this (a result that *parsed* but is *semantically* wrong —
    an empty risk register, a citation that doesn't resolve, a total that
    doesn't reconcile), the call re-rolls **within the same validation budget**
    that governs schema-validation retries (at most
    ``RetryPolicy.validation_max_attempts - 1`` retries within the overall
    ``max_attempts`` ceiling), so a deterministically-bad result cannot spend
    the full ceiling on validation re-asks.

    It is in spirit a *schema-validation* failure (the content is wrong, not the
    transport), so it consumes the validation allowance as well as the
    overall attempt count. On budget exhaustion the last ``ResultValidationError``
    propagates to the caller, carrying the rejecting message (and, if the host
    raised it ``from`` another error, that ``__cause__``).
    """


class ComposeUnsupportedError(LlmkitError):
    """Raised before transport when tools and a response schema cannot compose.

    This is a permanent route/model configuration error, intentionally outside
    :data:`LLM_RECOVERABLE_ERRORS`: retrying cannot make a provider support the
    combined request.  Use the portable two-step pattern (tool loop followed by
    ``structured_llm_call``) instead.
    """


class ToolChoiceUnsupportedError(LlmkitError, ValueError):
    """Raised before transport when a route cannot honour the requested ``tool_choice``.

    A route without ``tool_choice`` support (Ollama) accepts only ``"auto"``,
    which is what it does anyway; ``"none"``, ``"required"`` and a
    :class:`~llmkit.ToolName` would be dropped on the way to the wire and the
    model could still call a tool or skip one, so llmkit refuses them before the
    first attempt. A permanent configuration error, outside
    :data:`LLM_RECOVERABLE_ERRORS`. It subclasses ``ValueError``, the type this
    refusal raised before, so an existing ``except ValueError`` keeps catching it.
    """


class ReasoningEffortUnsupportedError(LlmkitError):
    """Thinking cannot be disabled, or the model's off capability is unknown.

    A permanent configuration error, outside ``LLM_RECOVERABLE_ERRORS``:
    retrying cannot establish a missing capability or disable an always-on model.
    """


class SchemaEngineError(LlmkitError):
    """The JSON Schema validator could not evaluate the contract.

    A permanent engine or configuration fault, outside the repairable parse
    and recoverable error sets: another generated answer cannot repair it.
    The original engine exception is available through ``__cause__``.
    """


class UnsupportedSchemaError(LlmkitError, TypeError):
    """An ``output_schema`` llmkit cannot hold to the structured-output contract.

    Raised before any request (and before the response cache is read) by
    :func:`~llmkit.structured_llm_call` and by :func:`~llmkit.tool_llm_call`
    with ``output_schema=``, when the schema is not a pydantic ``BaseModel``
    subclass, is one of instructor's DSL classes (``IterableModel``,
    ``Partial``, a simple-type adapter), or declares an instructor async
    validator (``async_field_validator`` / ``async_model_validator``), which no
    lane runs. A permanent programming error, outside
    :data:`LLM_RECOVERABLE_ERRORS`, and a :class:`TypeError` subclass because
    the schema, not the provider, is wrong.
    """
