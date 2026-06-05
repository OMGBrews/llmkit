"""Recoverable exception types for LLM service calls.

Use ``LLM_RECOVERABLE_ERRORS`` in ``except`` clauses to catch expected LLM
operational failures (network errors, rate limits, transient provider
errors, schema-validation/parsing failures, timeouts) while letting
programming errors (TypeError, AttributeError) propagate.

``with_retries()`` (see :mod:`llmkit.retry`) is the transient-retry
layer; instructor's own ``max_retries`` handles schema-repair separately.
"""

import httpx
import openai
from instructor.core import InstructorRetryException
from pydantic import ValidationError

# LiteLLM's transient errors (RateLimitError, Timeout, APIConnectionError,
# ServiceUnavailableError, InternalServerError, ...) all subclass
# ``openai.APIError``, so it covers them in one entry. ``InstructorRetryException``
# is raised when instructor exhausts its in-call schema-validation retries.
LLM_RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.APIError,
    InstructorRetryException,
    httpx.RequestError,
    ValidationError,
    TimeoutError,
)
