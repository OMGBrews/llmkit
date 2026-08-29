"""LLM client with multi-provider support.

Provides a thin, opinionated layer over **LiteLLM** (with ``instructor``
for structured output) that gives the application a unified, provider-
agnostic call surface across cloud providers (OpenRouter, Google AI Studio,
Google Vertex AI, Anthropic, OpenAI, DeepSeek, AWS Bedrock) and local Ollama.

This package provides:
- The structured / plain-text / streaming call functions
- Provider switching based on a host-supplied config
- A process-global async rate limiter shared across all calls
- Default-on transient-error retries with full-jitter backoff (RetryPolicy)
- Per-call invocation logging via a pluggable sink (with approximate cost)
- Run scoping, so every record and index line is filterable by run id

The headline surface below is what a typical consumer needs. Lower-level
symbols stay importable from their submodules (``llmkit.providers``,
``llmkit.retry``, ``llmkit.rate_limiting``) without crowding the top level:
the eight concrete ``*Provider`` classes, ``describe_llm`` / ``LLMInfo``,
``with_retries``, and ``GlobalRateLimiter`` all live there.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from llmkit._types import (
    AssistantToolMessage,
    ChatMessage,
    Message,
    ReasoningEffort,
    ToolResultMessage,
)
from llmkit.capture import (
    capture_llm_log_paths,
    capture_llm_records,
)
from llmkit.exceptions import (
    LLM_BACKPRESSURE_ERRORS,
    LLM_OUTPUT_LIMIT_ERRORS,
    LLM_RECOVERABLE_ERRORS,
    LLM_SCHEMA_ERRORS,
    LLM_TOOL_ERRORS,
    LLM_TRANSPORT_ERRORS,
    CircuitOpenError,
    ComposeUnsupportedError,
    OutputLimitError,
    ResultValidationError,
    ServiceUnavailableError,
    ToolArgumentError,
)
from llmkit.json_schema import model_from_json_schema
from llmkit.logging import (
    LLMCallRecord,
    LocalYamlLogSink,
    LogSink,
    configure_llm_logging,
    default_log_dir,
)
from llmkit.options import (
    DEFAULT_TEMPERATURE,
    UNSET,
    LLMCallOptions,
    Unset,
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
    BackpressureCallback,
    BackpressureEvent,
    RateLimitConfig,
    RateLimitSlot,
    backpressure_callback,
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
from llmkit.run_scope import (
    RUN_ID_ENV_VAR,
    get_run_id,
    run_scope,
    set_run_id,
)
from llmkit.structured_output import (
    STREAM_ABANDONED_ERROR,
    stream_text_with_log,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_sync,
    tool_llm_call,
    tool_llm_call_sync,
)
from llmkit.tools import (
    TokenUsage,
    ToolCall,
    ToolCallResult,
    ToolChoice,
    ToolComposeResult,
    ToolDefinition,
    ToolName,
    tool_result_message,
)

# Distribution version, resolved from installed metadata rather than hardcoded
# so it can never drift from what pip installed. The distribution is
# ``omg-llmkit`` (the import name is ``llmkit``); a source tree with no dist
# metadata falls back to a sentinel instead of raising at import.
try:
    __version__ = _version("omg-llmkit")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

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
    "default_log_dir",
    # Run scoping (tag every record + index line with a run id)
    "RUN_ID_ENV_VAR",
    "get_run_id",
    "run_scope",
    "set_run_id",
    # Rate limiting
    "configure_rate_limit",
    "get_rate_limit_config",
    "RateLimitConfig",
    "RateLimitSlot",
    "rate_limit_acquire_async",
    "rate_limit_acquire_sync",
    # Backpressure observability (adaptive concurrency limit changes)
    "backpressure_callback",
    "BackpressureCallback",
    "BackpressureEvent",
    # Retries (transient-error recovery, on by default via RetryPolicy)
    "RetryPolicy",
    "DEFAULT_RETRY_POLICY",
    "NO_RETRY",
    "retry_progress_callback",
    "RetryProgressCallback",
    # Structured + plain-text call functions (the public call surface)
    "structured_llm_call",
    "structured_llm_call_sync",
    "text_llm_call",
    "text_llm_call_sync",
    "text_llm_call_stream",
    "tool_llm_call",
    "tool_llm_call_sync",
    # Deprecated pre-1.0 alias for ``text_llm_call_stream`` (removed at 1.0).
    "stream_text_with_log",
    "STREAM_ABANDONED_ERROR",
    "capture_llm_log_paths",
    "capture_llm_records",
    "LLMCallOptions",
    "Unset",
    "UNSET",
    "DEFAULT_TEMPERATURE",
    "Message",
    "ChatMessage",
    "ReasoningEffort",
    "AssistantToolMessage",
    "ToolResultMessage",
    "ToolDefinition",
    "ToolName",
    "ToolChoice",
    "ToolCall",
    "ToolCallResult",
    "ToolComposeResult",
    "TokenUsage",
    "tool_result_message",
    # JSON-schema-dict structured output (build-once-reuse helper)
    "model_from_json_schema",
    # Exception handling
    "LLM_RECOVERABLE_ERRORS",
    "LLM_TRANSPORT_ERRORS",
    "LLM_SCHEMA_ERRORS",
    "LLM_TOOL_ERRORS",
    "LLM_BACKPRESSURE_ERRORS",
    "LLM_OUTPUT_LIMIT_ERRORS",
    "CircuitOpenError",
    "OutputLimitError",
    "ResultValidationError",
    "ComposeUnsupportedError",
    "ServiceUnavailableError",
    "ToolArgumentError",
    # Package metadata
    "__version__",
]
