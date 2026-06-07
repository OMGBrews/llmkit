"""LLM client with multi-provider support.

Provides a thin, opinionated layer over **LiteLLM** (with ``instructor``
for structured output) that gives the application a unified, provider-
agnostic call surface across cloud providers (OpenRouter, Google,
Anthropic, OpenAI) and local Ollama.

This package provides:
- The structured / plain-text / streaming call functions
- Provider switching based on a host-supplied config
- A process-global async rate limiter shared across all calls
- Per-call invocation logging via a pluggable sink (with approximate cost)
"""

from llmkit.exceptions import LLM_RECOVERABLE_ERRORS
from llmkit.logging import (
    LLMCallRecord,
    LocalYamlLogSink,
    LogSink,
    configure_llm_logging,
)
from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
    DeepSeekProvider,
    GoogleProvider,
    LLMClientConfig,
    LLMInfo,
    LLMProviderInterface,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
    configure_llm_client,
    get_llm_config,
    get_provider,
)
from llmkit.rate_limiting import (
    GlobalRateLimiter,
    configure_rate_limit,
)
from llmkit.retry import (
    RetryProgressCallback,
    retry_progress_callback,
    with_retries,
)
from llmkit.structured_output import (
    capture_llm_log_paths,
    stream_text_with_log,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
)

__all__ = [
    # Providers + config
    "LLMProviderInterface",
    "OpenRouterProvider",
    "OllamaProvider",
    "GoogleProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "BedrockProvider",
    "Provider",
    "LLMClientConfig",
    "LLMInfo",
    "configure_llm_client",
    "get_provider",
    "get_llm_config",
    # Logging
    "LLMCallRecord",
    "LogSink",
    "LocalYamlLogSink",
    "configure_llm_logging",
    # Rate limiting
    "GlobalRateLimiter",
    "configure_rate_limit",
    # Retries (composable helper you wrap calls in — not automatic)
    "with_retries",
    "retry_progress_callback",
    "RetryProgressCallback",
    # Structured + plain-text call functions (the public call surface)
    "structured_llm_call",
    "structured_llm_call_sync",
    "text_llm_call",
    "stream_text_with_log",
    "capture_llm_log_paths",
    # Exception handling
    "LLM_RECOVERABLE_ERRORS",
]
