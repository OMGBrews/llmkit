"""Recoverable exception types for LLM service calls.

Use ``LLM_RECOVERABLE_ERRORS`` in ``except`` clauses to catch expected LLM
operational failures (network errors, rate limits, transient 5xx provider
errors, schema-validation/parsing failures, timeouts) while letting
programming errors (TypeError, AttributeError) — and *permanent* request
errors such as authentication (401) and bad-request (400) failures —
propagate.

The recoverable set is split into two subsets the retry layer budgets
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

``LLM_RECOVERABLE_ERRORS`` is the **union** of the two, preserved as the
single documented ``except``-clause catch-set so existing callers keep
catching exactly what they did before.

``with_retries()`` (see :mod:`llmkit.retry`) is the transient-retry
layer; instructor's own ``validation_retries`` handles in-call schema-repair
separately.
"""

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
#   - ``httpx.RequestError``  — lower-level network failures
#   - ``TimeoutError``        — builtin timeout
LLM_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
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
#     budget (``max_retries=1``) for this attempt. NOTE: instructor wraps *any*
#     exhausted attempt this way, transport failures included — the retry layer
#     unwraps it (``underlying_provider_error``) so a wrapped transport cause is
#     charged the transport budget rather than this lower one.
LLM_SCHEMA_ERRORS: tuple[type[Exception], ...] = (
    ValidationError,
    InstructorRetryException,
)

# The full recoverable set is the union of the two subsets — preserved as the
# documented single catch-set for ``except`` clauses, so callers keep catching
# exactly what they did before even though the retry layer now budgets the two
# subsets separately.
LLM_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    *LLM_TRANSPORT_ERRORS,
    *LLM_SCHEMA_ERRORS,
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
