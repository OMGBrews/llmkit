"""Recoverable exception types for LLM service calls.

Use ``LLM_RECOVERABLE_ERRORS`` in ``except`` clauses to catch expected LLM
operational failures (network errors, rate limits, transient 5xx provider
errors, schema-validation/parsing failures, timeouts) while letting
programming errors (TypeError, AttributeError) — and *permanent* request
errors such as authentication (401) and bad-request (400) failures —
propagate.

The recoverable set is split into three subsets the retry layer budgets
**separately** (see :mod:`llmkit.retry`):

* :data:`LLM_TRANSPORT_ERRORS` — transient *transport* failures (rate
  limits, transient 5xx, network/timeout). These get the full transport
  retry budget, since a retry on a fresh connection routinely succeeds.
* :data:`LLM_SCHEMA_ERRORS` — *schema-validation* failures (pydantic
  ``ValidationError``; instructor's ``InstructorRetryException``, raised
  when its own in-call repair budget is exhausted). These get a **lower**
  budget: a transiently-malformed JSON response still earns one cross-call
  retry, but a deterministically-wrong schema can't burn the full transport
  budget on doomed re-asks.

  Caveat — ``InstructorRetryException`` is a *wrapper*: instructor re-raises
  **every** exhausted attempt this way, including transport failures (a 429
  or network blip caught inside its create call), not only schema ones. The
  retry layer therefore unwraps it via :func:`underlying_provider_error` and
  charges a wrapped transport cause to the transport budget — so membership
  here is the default-but-not-final classification, not a promise the failure
  is genuinely schema-shaped.
* :data:`LLM_BACKPRESSURE_ERRORS` — llmkit's *own* fail-fast backpressure
  signal (:class:`CircuitOpenError`, raised by the opt-in per-provider circuit
  breaker while it is open). It is recoverable in the catch-set sense — a host
  that degrades on ``LLM_RECOVERABLE_ERRORS`` keeps catching it (a fast
  fallback) instead of crashing on a new uncaught type — but it is deliberately
  **not** retried: it lives outside the transport set, so ``with_retries`` and
  the streaming loop (which key off the transport/schema subsets, never the
  union) never re-ask a circuit the breaker already knows is open.

``LLM_RECOVERABLE_ERRORS`` is the **union** of the three, preserved as the
single documented ``except``-clause catch-set so existing callers keep
catching exactly what they did before. One member is special: the
litellm-native 503 entry is a lazy stand-in
(:class:`_LiteLLMServiceUnavailableError`) that resolves litellm's class at
``isinstance`` time, keeping ``import llmkit`` free of the multi-second
litellm import — see its docstring for the one ``except``-clause limit.

``with_retries()`` (see :mod:`llmkit.retry`) is the transient-retry
layer; in-call schema repair is handled separately by instructor's retry
loop (``max_retries``, pinned to 2 in :mod:`llmkit._litellm`).
"""

import sys
from typing import override

import httpx
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

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
#   - ``httpx.RequestError``  — lower-level network failures
#   - ``TimeoutError``        — builtin timeout
#
# LiteLLM raises its *own* exception classes, which we catch via their
# ``openai`` bases: ``litellm.RateLimitError`` subclasses
# ``openai.RateLimitError``, ``litellm.InternalServerError`` subclasses
# ``openai.InternalServerError``, ``litellm.Timeout``/``litellm.APIConnectionError``
# subclass ``openai.APIConnectionError``. The one transient class that does
# NOT follow that pattern is ``litellm.exceptions.ServiceUnavailableError``
# (LiteLLM's mapping for HTTP 503): its MRO goes straight to
# ``openai.APIStatusError``, skipping ``openai.InternalServerError`` — so it
# must be listed explicitly or real 503s would propagate unretried.
#
# Deliberate scope: we do not retry generic ``openai.APIStatusError`` with
# status >= 500. LiteLLM maps the 5xx family to the named classes above
# (500 → ``InternalServerError``, 503 → ``ServiceUnavailableError``), and an
# isinstance-tuple can't express a status-code predicate; a raw
# ``APIStatusError`` outside those mappings is unexpected enough to surface
# rather than retry.


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
    the form every consumer of the transport set uses to classify errors —
    without importing litellm at ``import llmkit`` time. Never raise this
    class directly.

    Known limit: ``except`` matching bypasses ``__instancecheck__``, so a
    bare ``except LLM_TRANSPORT_ERRORS`` clause does not catch the
    litellm-native 503 specifically (every other member still matches; an
    openai-SDK 503 arrives as ``openai.InternalServerError`` and matches
    too). Use ``isinstance`` — as the retry layer does — where the
    litellm-native class matters.
    """


LLM_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    # litellm-native 503 — does not subclass ``openai.InternalServerError``
    # (only ``openai.APIStatusError``), so it needs an explicit entry; the
    # lazy stand-in keeps ``import llmkit`` litellm-free.
    _LiteLLMServiceUnavailableError,
    openai.APIConnectionError,
    httpx.RequestError,
    TimeoutError,
)

# Schema-validation failures — retried on their own *lower* budget, distinct
# from the transport set, so a deterministic schema failure can't consume the
# full transport budget on doomed re-asks while a transiently-malformed JSON
# response still earns one cross-call retry.
#   - ``ValidationError``           — pydantic could not parse the response
#   - ``InstructorRetryException``  — instructor exhausted its in-call repair
#     budget (``max_retries=2``: two attempts total, i.e. one schema-repair
#     re-ask) for this attempt. NOTE: instructor wraps *any*
#     exhausted attempt this way, transport failures included — the retry layer
#     unwraps it (``underlying_provider_error``) so a wrapped transport cause is
#     charged the transport budget rather than this lower one.
LLM_SCHEMA_ERRORS: tuple[type[Exception], ...] = (
    ValidationError,
    InstructorRetryException,
)


class CircuitOpenError(Exception):
    """Raised by the per-provider circuit breaker when it declines to admit a call.

    The breaker (opt-in via ``configure_rate_limit(breaker=True)``) is the
    aggregate guard that stops doomed work when a provider is *down*: once its
    throttle rate over a rolling window crosses the trip threshold it opens, and
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
    purpose. It *is* a member of :data:`LLM_BACKPRESSURE_ERRORS` and therefore of
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

# The full recoverable set is the union of the three subsets — preserved as the
# documented single catch-set for ``except`` clauses, so callers keep catching
# exactly what they did before even though the retry layer now budgets the
# subsets separately (and never retries the backpressure subset).
LLM_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    *LLM_TRANSPORT_ERRORS,
    *LLM_SCHEMA_ERRORS,
    *LLM_BACKPRESSURE_ERRORS,
)


def underlying_provider_error(exc: BaseException) -> BaseException:
    """Dig the original provider error out of an ``InstructorRetryException``.

    instructor re-raises *every* exhausted attempt — transport failures (rate
    limits, 5xx, network) just as much as schema-validation failures — wrapped
    in a single ``InstructorRetryException``. The wrapper alone can't tell the
    retry layer which budget the failure belongs to, so this returns the
    wrapped root error: instructor stores it as the first positional ``arg``,
    with the tenacity ``RetryError.__cause__``'s recorded last attempt as a
    fallback. Any other exception (a bare ``ValidationError``, a transport
    error that never went through instructor) is returned unchanged, so callers
    can classify ``underlying_provider_error(exc)`` uniformly.
    """
    if isinstance(exc, InstructorRetryException):
        inner = exc.args[0] if exc.args else None
        if isinstance(inner, BaseException):
            return inner
        # Fallback: instructor raises ``from`` a tenacity RetryError whose
        # ``last_attempt`` Future holds the root exception.
        last_attempt = getattr(exc.__cause__, "last_attempt", None)
        recorded = getattr(last_attempt, "_exception", None)
        if isinstance(recorded, BaseException):
            return recorded
    return exc


class ResultValidationError(Exception):
    """Raised by a call function's ``on_result`` hook to reject a result.

    The signal a host uses to fold an LLM-then-validate-then-re-roll loop into
    the call functions themselves. When a structured/text call's ``on_result``
    callback raises this (a result that *parsed* but is *semantically* wrong —
    an empty risk register, a citation that doesn't resolve, a total that
    doesn't reconcile), the call re-rolls **within the same validation budget**
    that governs schema-validation retries (``RetryPolicy.validation_max_attempts``),
    so a deterministically-bad result can't burn the full transport budget.

    It is in spirit a *schema-validation* failure (the content is wrong, not the
    transport), so it is charged against the validation budget rather than the
    transport one. On budget exhaustion the last ``ResultValidationError``
    propagates to the caller, carrying the rejecting message (and, if the host
    raised it ``from`` another error, that ``__cause__``).
    """
