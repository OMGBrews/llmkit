"""LLM client with multi-provider support.

Provides a thin, opinionated layer over **LiteLLM** (with ``instructor``
for structured output) that gives the application a unified, provider-
agnostic call surface across cloud providers (OpenRouter, Google,
Anthropic, OpenAI) and local Ollama.

This package provides:
- The structured / plain-text / streaming call functions
- Provider switching based on a host-supplied config
- A process-global async rate limiter shared across all calls
- Default-on transient-error retries with full-jitter backoff (RetryPolicy)
- Per-call invocation logging via a pluggable sink (with approximate cost)

The headline surface below is what a typical consumer needs. Lower-level
symbols stay importable from their submodules (``llmkit.providers``,
``llmkit.retry``, ``llmkit.rate_limiting``) without crowding the top level:
the seven concrete ``*Provider`` classes, ``describe_llm`` / ``LLMInfo``,
``with_retries``, and ``GlobalRateLimiter`` all live there.
"""

from llmkit.exceptions import (
    LLM_RECOVERABLE_ERRORS,
    LLM_SCHEMA_ERRORS,
    LLM_TRANSPORT_ERRORS,
)
from llmkit.json_schema import model_from_json_schema
from llmkit.logging import (
    LLMCallRecord,
    LocalYamlLogSink,
    LogSink,
    configure_llm_logging,
)
from llmkit.providers import (
    LLMClientConfig,
    LLMProviderInterface,
    Provider,
    build_provider,
    configure_llm_client,
    make_provider,
)
from llmkit.rate_limiting import (
    RateLimitConfig,
    configure_rate_limit,
    get_rate_limit_config,
    rate_limit_acquire_async,
    rate_limit_acquire_sync,
)
from llmkit.retry import (
    DEFAULT_RETRY_POLICY,
    NO_RETRY,
    RetryPolicy,
    RetryProgressCallback,
    retry_progress_callback,
)
from llmkit.structured_output import (
    LLMCallOptions,
    capture_llm_log_paths,
    capture_llm_records,
    stream_text_with_log,
    structured_data_call,
    structured_data_call_sync,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
)

__all__ = [
    # Providers + config
    "LLMProviderInterface",
    "Provider",
    "LLMClientConfig",
    "configure_llm_client",
    "build_provider",
    "make_provider",
    # Logging
    "LLMCallRecord",
    "LogSink",
    "LocalYamlLogSink",
    "configure_llm_logging",
    # Rate limiting
    "configure_rate_limit",
    "get_rate_limit_config",
    "RateLimitConfig",
    "rate_limit_acquire_async",
    "rate_limit_acquire_sync",
    # Retries (transient-error recovery, on by default via RetryPolicy)
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
    "NO_RETRY",
    "retry_progress_callback",
    "RetryProgressCallback",
    # Structured + plain-text call functions (the public call surface)
    "structured_llm_call",
    "structured_llm_call_sync",
    "structured_data_call",
    "structured_data_call_sync",
    "text_llm_call",
    "stream_text_with_log",
    "capture_llm_log_paths",
    "capture_llm_records",
    "LLMCallOptions",
    # JSON-schema-dict structured output (build-once-reuse helper)
    "model_from_json_schema",
    # Exception handling
    "LLM_RECOVERABLE_ERRORS",
    "LLM_TRANSPORT_ERRORS",
    "LLM_SCHEMA_ERRORS",
]
