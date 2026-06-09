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
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol, runtime_checkable

import instructor

logger = logging.getLogger(__name__)


def _require_sdk(
    import_name: str, *, provider_name: str, extra: str, label: str | None = None, note: str = ""
) -> None:
    """Raise a clear, actionable error if ``import_name`` is not importable.

    Generic backing for the per-dependency ``require_*_sdk`` wrappers below.
    Optional provider dependencies ship in opt-in pip extras (so hosts that
    never touch a given provider take on no extra dependency) and are checked
    *eagerly* at provider construction, so a missing dep fails with a fix
    instead of as a cryptic ``ModuleNotFound`` deep on the first completion.

    ``label`` is the human-readable dependency name used in the message (e.g.
    ``"the Anthropic SDK"``); it defaults to ``import_name``. ``note`` is an
    optional trailing sentence appended to the message (e.g. to cross-reference
    a related extra).
    """
    if importlib.util.find_spec(import_name) is None:
        message = (
            f"The {provider_name} provider requires {label or import_name}, which is "
            "not installed. It ships in an opt-in extra so hosts that don't use "
            "this provider take on no extra dependency. Install it with:\n\n"
            f"    pip install 'omg-llmkit[{extra}]'"
        )
        if note:
            message += f"\n\n{note}"
        raise ModuleNotFoundError(message)


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
    _require_sdk(
        "anthropic",
        provider_name=provider_name,
        extra="anthropic",
        label="the Anthropic SDK",
        note=("(Bedrock routes Claude and needs it too: 'omg-llmkit[bedrock]' pulls it in.)"),
    )


def require_boto3_sdk(provider_name: str) -> None:
    """Raise a clear, actionable error if ``boto3`` is not installed.

    LiteLLM's Bedrock path uses ``boto3`` to sign requests (AWS SigV4), so the
    Bedrock provider needs it at call time. It ships in the opt-in
    ``omg-llmkit[bedrock]`` extra rather than the core install. Call this at
    provider construction so a missing ``boto3`` fails *eagerly* with a fix,
    instead of as a cryptic ``ModuleNotFound`` deep on the first completion.
    """
    _require_sdk(
        "boto3",
        provider_name=provider_name,
        extra="bedrock",
        note="(boto3 signs Bedrock requests with AWS SigV4.)",
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
    unused by Google/Anthropic/DeepSeek/Bedrock (whose endpoints are
    fixed). Per-call
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

    def completion_kwargs(self) -> dict[str, object]:
        """Credential + routing kwargs forwarded to ``litellm`` calls.

        Carries the credential kwargs (``api_key`` / ``api_base`` /
        ``aws_region_name``) and any provider-specific routing preference (e.g.
        OpenRouter's ``extra_body`` ``require_parameters``). The value type is
        ``object`` rather than ``str`` because a routing preference is a nested
        mapping, not a string.
        """
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

    # Routing-contract hooks every concrete provider MUST set. They are
    # declared without class-level default *values* on purpose: a silent
    # default here is a footgun (a forgotten ``_mode`` would inherit a
    # JSON-schema mode that silently regresses Gemini/DeepSeek structured
    # output; a forgotten ``_default_model`` would assemble a dangling
    # ``"<prefix>/"`` id; a forgotten ``_provider_name`` would degrade logging,
    # rate-limit keying, and ``describe_llm``). ``__init_subclass__`` below
    # turns the "every provider names these" convention into an enforced
    # contract — an incomplete provider fails loudly at import, not at routing
    # time with a provider-shaped misbehavior.
    _provider_name: ClassVar[str]
    _mode: ClassVar[instructor.Mode]
    _default_model: ClassVar[str]
    # ``_model_prefix`` keeps a ``""`` default deliberately: an empty prefix is
    # legitimate (it still assembles a well-formed bare-model id), so unlike the
    # three hooks above a missing value is not a footgun and is left unguarded.
    _model_prefix: ClassVar[str] = ""
    #: Whether this provider runs against a local endpoint (no data leaves the
    #: host). A provider trait rather than a dispatch special-case, so locality
    #: stays defined alongside the provider it describes (see
    #: :func:`~llmkit.providers.describe_llm`). Cloud providers leave this False.
    is_local: ClassVar[bool] = False

    #: The routing-contract hooks validated for every concrete subclass.
    _required_hooks: ClassVar[tuple[str, ...]] = ("_provider_name", "_mode", "_default_model")

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject a concrete provider that omits a routing-contract hook.

        Fires at class definition (import time), so an incomplete provider
        fails the moment its module is imported — earlier and louder than a
        construction-time check, and before any call can route through it.

        Only *concrete* (instantiable) subclasses must satisfy the contract:
        an intermediate ABC that leaves an abstract method (e.g.
        ``completion_kwargs``) unimplemented is exempt until a concrete
        subclass fills it in. ``inspect.isabstract`` is the structural signal,
        so no provider needs to opt in or out by hand.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        missing = [hook for hook in cls._required_hooks if not getattr(cls, hook, None)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is an incomplete provider: it must set "
                + f"{', '.join(missing)}. Every concrete BaseProvider subclass names its own "
                + "provider, instructor mode, and default model so structured-output routing, "
                + "rate-limit keying, and the assembled LiteLLM id are explicit — never "
                + "inherited from a silent class-level default."
            )

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None) -> None:
        # A falsy model (None or "") falls back to the provider's own default
        # so the LiteLLM id assembled by ``litellm_model`` is always well-formed
        # (``<prefix><model>``) and never a dangling ``"<prefix>/"``. The
        # ``__init_subclass__`` contract guarantees ``_default_model`` is a
        # non-empty value every concrete provider set, so this fallback is
        # well-formed by construction, not merely by convention.
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
    def completion_kwargs(self) -> dict[str, object]:
        """Return the credential + routing kwargs forwarded to every ``litellm`` call.

        Values are typed ``object`` (not ``str``) because a provider may add a
        nested routing preference (e.g. OpenRouter's ``extra_body``) alongside
        the string credential kwargs.
        """
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
