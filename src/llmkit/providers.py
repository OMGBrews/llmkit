"""LLM provider implementations.

Defines the provider protocol and concrete implementations for
OpenRouter, Ollama, Google AI Studio, Anthropic, and OpenAI.

Each provider supplies what the LiteLLM call layer needs: a LiteLLM
**model string** (provider-prefixed, e.g. ``openrouter/<model>``,
``gemini/<model>``), the completion kwargs that carry credentials
(``api_key`` / ``api_base``), and the :class:`instructor.Mode` that pins
structured output to the provider's *native* JSON-schema mode. The mode
map is explicit on purpose — instructor's auto-mode falls back to
``Mode.TOOLS``, which silently regresses Gemini structured output
(measured 0% valid / silent-empty shapes), so every provider names its
mode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import instructor

logger = logging.getLogger(__name__)


class Provider(StrEnum):
    """LLM providers the library can route to.

    Library-owned enum (no dependency on the host's configuration).
    """

    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass(frozen=True)
class LLMClientConfig:
    """Provider selection + credentials for the active LLM provider.

    The library is config-source-agnostic: the host application builds
    this from whatever settings system it uses and registers a source
    via :func:`configure_llm_client`, so the library never imports the
    host's config module.

    Only the *active* provider's fields need be populated. ``api_key`` is
    unused by Ollama (which talks to a local endpoint) and ``base_url`` is
    unused by Google/Anthropic (whose endpoints are fixed). Per-call
    ``model`` overrides (e.g. the strong/small roles the host resolves)
    are passed at call time and are not part of this config — this carries
    only the provider's *default* model.

    ``reasoning_effort`` controls provider "thinking"/reasoning tokens,
    mirroring LiteLLM's standard param (``"disable" | "low" | "medium" |
    "high"``). ``None`` (the default) sends no reasoning kwarg, leaving the
    provider's own default in place — byte-identical to prior behaviour. Set
    it once here (e.g. ``"disable"``) to have **every** call inherit the
    setting; Gemini's default-on thinking otherwise spends reasoning tokens
    against ``max_tokens`` and can truncate small-capped structured output.
    """

    provider: Provider
    model: str
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None


# Module-level config source, registered once by the host application at a
# universally-imported wiring edge (see ``app/__init__.py``). Mirrors the
# ``configure_rate_limit`` module-level pattern so call sites that only pass
# ``feature``/``label``/``model`` keep working with zero per-call wiring.
_config_source: Callable[[], LLMClientConfig] | None = None


def configure_llm_client(config_source: Callable[[], LLMClientConfig]) -> None:
    """Register the source that supplies the active :class:`LLMClientConfig`.

    The source is a zero-arg callable invoked on each :func:`get_provider`
    call, mirroring the previous behaviour of reading the host's
    (cached) settings on every provider construction — so a settings
    change is picked up without re-registration.
    """
    global _config_source
    _config_source = config_source


def _active_config() -> LLMClientConfig:
    """Return the configured :class:`LLMClientConfig`, or raise if unset."""
    if _config_source is None:
        raise RuntimeError(
            "LLM client not configured: call configure_llm_client(...) at "
            "application startup before constructing a provider."
        )
    return _config_source()


@runtime_checkable
class LLMProviderInterface(Protocol):
    """Protocol for LLM provider implementations."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def model(self) -> str:
        """Current default model name (unprefixed)."""
        ...

    @property
    def instructor_mode(self) -> instructor.Mode:
        """The instructor mode pinning structured output for this provider."""
        ...

    @property
    def reasoning_effort(self) -> str | None:
        """Configured reasoning/thinking effort, or ``None`` for provider default."""
        ...

    def litellm_model(self, model: str | None = None) -> str:
        """The provider-prefixed LiteLLM model string for ``model`` (or the default)."""
        ...

    def completion_kwargs(self) -> dict[str, str]:
        """Credential kwargs (api_key / api_base) forwarded to ``litellm`` calls."""
        ...


class BaseProvider(ABC):
    """Base class for LLM providers.

    Implements the shared public interface; subclasses supply the
    LiteLLM-specific routing via the ``_model_prefix`` / ``instructor_mode``
    hooks and ``completion_kwargs``.
    """

    _provider_name: str = ""
    _model_prefix: str = ""
    _mode: instructor.Mode = instructor.Mode.JSON_SCHEMA

    def __init__(self, model: str, reasoning_effort: str | None = None) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort

    @property
    def name(self) -> str:
        """Provider name."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Current default model name (unprefixed)."""
        return self._model

    @property
    def instructor_mode(self) -> instructor.Mode:
        """The instructor mode pinning structured output for this provider."""
        return self._mode

    @property
    def reasoning_effort(self) -> str | None:
        """Configured reasoning/thinking effort, or ``None`` for provider default.

        Forwarded to LiteLLM as ``reasoning_effort`` when set (see
        :mod:`llmkit._litellm`); ``None`` sends nothing, so the provider's
        own thinking default stands.
        """
        return self._reasoning_effort

    def litellm_model(self, model: str | None = None) -> str:
        """Return the provider-prefixed LiteLLM model string.

        ``model`` overrides the provider's configured default when set
        (e.g. the host's strong/small role resolved to a concrete model).
        """
        return f"{self._model_prefix}{model or self._model}"

    @abstractmethod
    def completion_kwargs(self) -> dict[str, str]:
        """Return the credential kwargs forwarded to every ``litellm`` call."""
        ...


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider.

    Routes through OpenRouter's unified endpoint (``openrouter/<model>``),
    using its native structured-outputs mode for instructor.
    """

    _provider_name = "OpenRouter"
    _model_prefix = "openrouter/"
    _mode = instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.0-flash-001",
        base_url: str = "https://openrouter.ai/api/v1",
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key = api_key
        self._base_url = base_url

    def completion_kwargs(self) -> dict[str, str]:
        return {"api_key": self._api_key, "api_base": self._base_url}


class OllamaProvider(BaseProvider):
    """Ollama LLM provider.

    Talks to a locally-running Ollama instance (``ollama_chat/<model>``)
    for fully local/private LLM access. No data leaves the host.
    """

    _provider_name = "Ollama"
    _model_prefix = "ollama_chat/"
    _mode = instructor.Mode.JSON_SCHEMA

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._base_url = base_url

    def completion_kwargs(self) -> dict[str, str]:
        return {"api_base": self._base_url}


class GoogleProvider(BaseProvider):
    """Google AI Studio LLM provider.

    Uses Gemini directly (``gemini/<model>``) for high rate limits and low
    latency, pinned to Gemini's native JSON-schema mode for instructor.
    """

    _provider_name = "Google AI Studio"
    _model_prefix = "gemini/"
    _mode = instructor.Mode.JSON_SCHEMA

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key = api_key

    def completion_kwargs(self) -> dict[str, str]:
        return {"api_key": self._api_key}


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider.

    Routes Claude through the Anthropic API (``anthropic/<model>``), pinned
    to Anthropic's native JSON mode for instructor.
    """

    _provider_name = "Anthropic"
    _model_prefix = "anthropic/"
    _mode = instructor.Mode.ANTHROPIC_JSON

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key = api_key

    def completion_kwargs(self) -> dict[str, str]:
        return {"api_key": self._api_key}


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider.

    Calls OpenAI (GPT / o-series / GPT-5) directly (``openai/<model>``),
    pinned to OpenAI's native structured-outputs mode (``Mode.JSON_SCHEMA``
    — a ``strict: true`` ``response_format`` JSON schema). This is the
    direct alternative to the indirect ``openrouter/openai/...`` hop, which
    adds a markup and a different structured-output mode.

    ``Mode.TOOLS_STRICT`` (forced strict function call) was the original
    pick but the live smoke test caught it silently returning empty fields
    on ``gpt-4.1-mini`` (0/4 valid; ``gpt-4o-mini`` / GPT-5 affected too) —
    the exact silent-regression failure mode this map exists to prevent.
    ``Mode.JSON_SCHEMA`` round-trips cleanly across GPT-4.1 / 4o / o-series
    / GPT-5, and matches the JSON-schema mode the other providers pin.

    An optional ``base_url`` points the same provider at an OpenAI-compatible
    gateway; left unset, LiteLLM uses OpenAI's default endpoint (so
    ``api_base`` is only forwarded when a ``base_url`` is given).
    """

    _provider_name = "OpenAI"
    _model_prefix = "openai/"
    _mode = instructor.Mode.JSON_SCHEMA

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key = api_key
        self._base_url = base_url

    def completion_kwargs(self) -> dict[str, str]:
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["api_base"] = self._base_url
        return kwargs


@dataclass
class LLMInfo:
    """Descriptive metadata about the active LLM configuration.

    A read-only snapshot for display/telemetry (provider name, model,
    locality) — distinct from :class:`LLMClientConfig`, which carries the
    credentials used to *construct* a provider.
    """

    provider: Provider
    provider_name: str
    model: str
    is_local: bool


def get_provider(config: LLMClientConfig | None = None) -> BaseProvider:
    """Get an LLM provider instance for the given (or configured) config.

    Args:
        config: Explicit config to build from. When ``None``, the source
            registered via :func:`configure_llm_client` supplies it.

    Returns:
        A provider constructed from the config's credentials.
    """
    if config is None:
        config = _active_config()

    if config.provider == Provider.OPENROUTER:
        return OpenRouterProvider(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            reasoning_effort=config.reasoning_effort,
        )
    elif config.provider == Provider.GOOGLE:
        return GoogleProvider(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.provider == Provider.ANTHROPIC:
        return AnthropicProvider(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.provider == Provider.OPENAI:
        return OpenAIProvider(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )
    else:
        return OllamaProvider(
            base_url=config.base_url or "http://localhost:11434",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )


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
