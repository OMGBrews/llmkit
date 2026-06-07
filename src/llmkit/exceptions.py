"""Recoverable exception types for LLM service calls.

Use ``LLM_RECOVERABLE_ERRORS`` in ``except`` clauses to catch expected LLM
operational failures (network errors, rate limits, transient 5xx provider
errors, schema-validation/parsing failures, timeouts) while letting
programming errors (TypeError, AttributeError) — and *permanent* request
errors such as authentication (401) and bad-request (400) failures —
propagate.

``with_retries()`` (see :mod:`llmkit.retry`) is the transient-retry
layer; instructor's own ``max_retries`` handles schema-repair separately.
"""

import httpx
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

# Only *genuinely transient* failures belong here — the default retry policy
# retries this exact set. We name the specific transient ``openai`` subclasses
# rather than their shared ``openai.APIError`` base, because that base also
# covers permanent 4xx errors (``AuthenticationError``/401, ``BadRequestError``/400,
# ``PermissionDeniedError``/403, ...) which must fail fast instead of burning the
# retry budget. Transient set:
#   - ``RateLimitError``      — 429
#   - ``InternalServerError`` — 5xx
#   - ``APIConnectionError``  — network failures (``APITimeoutError`` subclasses it)
# ``InstructorRetryException`` is raised when instructor exhausts its in-call
# schema-validation retries.
LLM_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    InstructorRetryException,
    httpx.RequestError,
    ValidationError,
    TimeoutError,
)
