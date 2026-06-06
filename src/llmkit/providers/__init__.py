"""LLM provider package: shared contract, concrete providers, and dispatch.

Public facade for the provider layer. The provider-agnostic core lives in
:mod:`llmkit.providers.base`; each concrete provider lives in its own
sibling module (``openrouter``, ``ollama``, ``google``, ``anthropic``,
``openai``). This package is the **single place** that imports every
provider module and maps a :class:`Provider` to its implementation.

Adding a provider: create a ``providers/<name>.py`` with a
:class:`BaseProvider` subclass and a ``build(config)`` classmethod, then
wire it here — one import and one ``case`` in :func:`get_provider`. The
``match`` has no catch-all, so a new :class:`Provider` enum member left
unwired is reported by the type checker (non-exhaustive match), raises a
clear error at runtime, and fails the exhaustiveness unit test — it can
never silently fall back to another provider.
"""

from __future__ import annotations

from typing import assert_never

from llmkit.providers.anthropic import AnthropicProvider
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    LLMInfo,
    LLMProviderInterface,
    Provider,
    _active_config,
    configure_llm_client,
)
from llmkit.providers.deepseek import DeepSeekProvider
from llmkit.providers.google import GoogleProvider
from llmkit.providers.ollama import OllamaProvider
from llmkit.providers.openai import OpenAIProvider
from llmkit.providers.openrouter import OpenRouterProvider


def get_provider(config: LLMClientConfig | None = None) -> BaseProvider:
    """Get an LLM provider instance for the given (or configured) config.

    Args:
        config: Explicit config to build from. When ``None``, the source
            registered via :func:`configure_llm_client` supplies it.

    Returns:
        A provider constructed from the config's credentials.

    Raises:
        AssertionError: If ``config.provider`` is not wired to a provider
            here (an unwired enum member). See :func:`typing.assert_never`.
    """
    if config is None:
        config = _active_config()

    match config.provider:
        case Provider.OPENROUTER:
            return OpenRouterProvider.build(config)
        case Provider.OLLAMA:
            return OllamaProvider.build(config)
        case Provider.GOOGLE:
            return GoogleProvider.build(config)
        case Provider.ANTHROPIC:
            return AnthropicProvider.build(config)
        case Provider.OPENAI:
            return OpenAIProvider.build(config)
        case Provider.DEEPSEEK:
            return DeepSeekProvider.build(config)
    # Every Provider member is handled above, so the subject narrows to Never
    # here. assert_never makes that a *static* guarantee: basedpyright reports
    # a type error if a newly-added Provider member is missing its case, and it
    # raises loudly at runtime for any unwired value — never a silent fallback.
    assert_never(config.provider)


def get_llm_config(config: LLMClientConfig | None = None) -> LLMInfo:
    """Get descriptive metadata for the active (or given) configuration.

    Returns:
        :class:`LLMInfo` snapshot of the resolved provider + model.
    """
    if config is None:
        config = _active_config()
    provider = get_provider(config)

    return LLMInfo(
        provider=config.provider,
        provider_name=provider.name,
        model=provider.model,
        is_local=config.provider == Provider.OLLAMA,
    )


__all__ = [
    "LLMProviderInterface",
    "BaseProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "GoogleProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "Provider",
    "LLMClientConfig",
    "LLMInfo",
    "configure_llm_client",
    "get_provider",
    "get_llm_config",
]
