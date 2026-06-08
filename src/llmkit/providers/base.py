"""Provider-agnostic core for the provider package.

Defines the shared contract every LLM provider builds on: the
library-owned :class:`Provider` enum, the :class:`LLMClientConfig`
credentials carrier, the module-level config-source registration
(:func:`configure_llm_client`), the :class:`LLMProviderInterface`
protocol, the :class:`BaseProvider` ABC, and the :class:`LLMInfo`
display snapshot.

This module knows nothing about any concrete provider. Each provider
lives in its own sibling module (``openrouter.py``, ``ollama.py``, …)
and supplies what the LiteLLM call layer needs: a LiteLLM **model
string** (provider-prefixed, e.g. ``openrouter/<model>``,
``gemini/<model>``), the completion kwargs that carry credentials
(``api_key`` / ``api_base``), and the :class:`instructor.Mode` that pins
structured output to the provider's *native* JSON-schema mode. The mode
map is explicit on purpose — instructor's auto-mode falls back to
``Mode.TOOLS``, which silently regresses Gemini structured output
(measured 0% valid / silent-empty shapes), so every provider names its
mode.

Dispatch (config → provider) and the public facade live in this
package's ``__init__.py``.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import instructor

logger = logging.getLogger(__name__)


def require_anthropic_sdk(provider_name: str) -> None:
    """Raise a clear, actionable error if the Anthropic SDK is not installed.

    The Anthropic and Bedrock providers both pin instructor's
    ``ANTHROPIC_JSON`` mode. instructor reaches the Anthropic SDK only at
    *call time*, on its ANTHROPIC_* usage-accounting path (``from
    anthropic.types import Usage`` inside ``instructor/core/retry.py``), so
    plain ``import llmkit`` and a Google-only flow never touch it. The SDK
    therefore ships in the opt-in ``omg-llmkit[anthropic]`` extra rather than
    the core install. Call this at provider construction so the failure
    surfaces *eagerly* with a fix, instead of as a cryptic ``ModuleNotFound``
    deep on the first completion.
    """
    if importlib.util.find_spec("anthropic") is None:
        raise ModuleNotFoundError(
            f"The {provider_name} provider requires the Anthropic SDK, which is "
            + "not installed. It ships in an opt-in extra so non-Anthropic hosts "
            + "take on no Anthropic dependency. Install it with:\n\n"
            + "    pip install 'omg-llmkit[anthropic]'\n\n"
            + "(Bedrock routes Claude and needs it too: 'omg-llmkit[bedrock]' "
            + "pulls it in.)"
        )


class Provider(StrEnum):
    """LLM providers the library can route to.

    Library-owned enum (no dependency on the host's configuration).
    """

    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    BEDROCK = "bedrock"


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

    ``model`` is optional: leave it ``None`` (or empty) to inherit the
    selected provider's own default model rather than naming one here. A
    falsy ``model`` resolves to the concrete provider's built-in default
    (e.g. ``claude-sonnet-4-6`` for Anthropic), so the assembled LiteLLM
    id is always well-formed — never a dangling ``"anthropic/"``.

    ``reasoning_effort`` controls provider "thinking"/reasoning tokens,
    mirroring LiteLLM's standard param (``"disable" | "low" | "medium" |
    "high"``). ``None`` (the default) sends no reasoning kwarg, leaving the
    provider's own default in place — byte-identical to prior behaviour. Set
    it once here (e.g. ``"disable"``) to have **every** call inherit the
    setting; Gemini's default-on thinking otherwise spends reasoning tokens
    against ``max_tokens`` and can truncate small-capped structured output.

    ``aws_region_name`` carries the AWS region for the Bedrock provider and
    is unused by every other provider (which authenticate with ``api_key`` /
    ``base_url``). It is the *only* AWS-shaped field on this config on
    purpose: Bedrock's secrets (access key / secret / session token, or
    instance-role credentials) resolve from the **ambient AWS credential
    chain** — environment, shared config, or instance/role — so they never
    pass through ``LLMClientConfig``. Leave it ``None`` to let the region
    resolve from the chain too (``AWS_REGION_NAME`` / ``AWS_REGION``).
    """

    provider: Provider
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    aws_region_name: str | None = None


# Module-level config source, registered once by the host application at a
# universally-imported wiring edge (see ``app/__init__.py``). Mirrors the
# ``configure_rate_limit`` module-level pattern so call sites that only pass
# ``feature``/``label``/``model`` keep working with zero per-call wiring.
_config_source: Callable[[], LLMClientConfig] | None = None


def configure_llm_client(config_source: Callable[[], LLMClientConfig]) -> None:
    """Register the source that supplies the active :class:`LLMClientConfig`.

    The source is a zero-arg callable invoked on each :func:`build_provider`
    call, mirroring the previous behaviour of reading the host's
    (cached) settings on every provider construction — so a settings
    change is picked up without re-registration.
    """
    global _config_source
    _config_source = config_source


def active_config() -> LLMClientConfig:
    """Return the configured :class:`LLMClientConfig`, or raise if unset."""
    if _config_source is None:
        raise RuntimeError(
            "LLM client not configured: call configure_llm_client(...) at "
            + "application startup before constructing a provider."
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

    Each concrete subclass also provides a ``build(config)`` classmethod
    that maps an :class:`LLMClientConfig` to a constructed instance
    (including the provider's own credential/endpoint defaults). ``build``
    is a package-internal construction convention used by
    :func:`llmkit.providers.build_provider`; it is deliberately **not** part
    of :class:`LLMProviderInterface`, which is the surface consumers type
    against.
    """

    _provider_name: str = ""
    _model_prefix: str = ""
    _mode: instructor.Mode = instructor.Mode.JSON_SCHEMA
    _default_model: str = ""

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None) -> None:
        # A falsy model (None or "") falls back to the provider's own default
        # so the LiteLLM id assembled by ``litellm_model`` is always well-formed
        # (``<prefix><model>``) and never a dangling ``"<prefix>/"``. Each
        # concrete provider sets ``_default_model`` to the same value its ctor
        # documents as the default, so config-built and directly-built providers
        # agree on the fallback model.
        self._model: str = model or self._default_model
        self._reasoning_effort: str | None = reasoning_effort

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
