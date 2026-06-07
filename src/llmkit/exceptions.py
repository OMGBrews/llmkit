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
#   - ``InstructorRetryException``  — instructor exhausted its in-call
#     ``validation_retries`` repair budget (default 1) for this attempt
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
