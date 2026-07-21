"""LLM provider package: shared contract, concrete providers, and dispatch.

Public facade for the provider layer. The provider-agnostic core lives in
:mod:`llmkit.providers.base`; each concrete provider lives in its own
sibling module (``openrouter``, ``ollama``, ``google``, ``anthropic``,
``openai``, ``deepseek``, ``bedrock``, ``vertex``). This package is the
**single place** that imports every provider module and maps a
:class:`Provider` to its implementation.

Adding a provider: create a ``providers/<name>.py`` with a
:class:`BaseProvider` subclass and a ``build(config)`` classmethod, then
wire it here — one import and one ``case`` in :func:`build_provider`. The
``match`` has no catch-all, so a new :class:`Provider` enum member left
unwired is reported by the type checker (non-exhaustive match), raises a
clear error at runtime, and fails the exhaustiveness unit test — it can
never silently fall back to another provider.
"""

from __future__ import annotations

from typing import Literal, assert_never

from llmkit._types import ReasoningEffort
from llmkit.providers.anthropic import AnthropicProvider
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    LLMInfo,
    LLMProviderInterface,
    Provider,
    active_config,
    configure_llm_client,
)
from llmkit.providers.bedrock import BedrockProvider
from llmkit.providers.deepseek import DeepSeekProvider
from llmkit.providers.google import GoogleProvider
from llmkit.providers.ollama import OllamaProvider
from llmkit.providers.openai import OpenAIProvider
from llmkit.providers.openrouter import OpenRouterProvider
from llmkit.providers.vertex import VertexProvider


def _provider_class(provider: Provider) -> type[BaseProvider]:
    """Map a :class:`Provider` to its concrete provider class.

    The single dispatch switch — resolving the *class* (rather than a built
    instance) lets :func:`build_provider` validate the config against the
    provider's contract before constructing it. The ``match`` has no catch-all,
    so a new enum member left unwired is caught three ways (see the module
    docstring): a basedpyright non-exhaustive-match error, an ``assert_never``
    at runtime, and the exhaustiveness unit test.
    """
    match provider:
        case Provider.OPENROUTER:
            return OpenRouterProvider
        case Provider.OLLAMA:
            return OllamaProvider
        case Provider.GOOGLE:
            return GoogleProvider
        case Provider.ANTHROPIC:
            return AnthropicProvider
        case Provider.OPENAI:
            return OpenAIProvider
        case Provider.DEEPSEEK:
            return DeepSeekProvider
        case Provider.BEDROCK:
            return BedrockProvider
        case Provider.VERTEX:
            return VertexProvider
    # Every Provider member is handled above, so the subject narrows to Never
    # here. assert_never makes that a *static* guarantee: basedpyright reports
    # a type error if a newly-added Provider member is missing its case, and it
    # raises loudly at runtime for any unwired value — never a silent fallback.
    assert_never(provider)


def build_provider(config: LLMClientConfig | None = None) -> BaseProvider:
    """Construct an LLM provider instance from a config.

    The canonical *construct-from-config* entry point. ``build_*`` /
    ``make_*`` name construction; ``get_*`` is reserved for *reading*
    effective state (see :func:`describe_llm`,
    :func:`llmkit.get_rate_limit_config`).

    This is the single seam every construction path flows through
    (``make_provider``, the :func:`configure_llm_client` source, and a direct
    ``build_provider(config)``), so it is where the config is validated against
    the selected provider's contract: a populated knob the provider will not read
    is rejected here (:func:`~llmkit.providers.base.reject_unread_config_fields`),
    not silently ignored.

    Args:
        config: Explicit config to build from. When ``None``, the source
            registered via :func:`configure_llm_client` supplies it.

    Returns:
        A provider constructed from the config's credentials. A falsy
        ``config.model`` resolves to the provider's own default model.

    Raises:
        ValueError: If the config populates a provider-shaped knob the selected
            provider does not read (e.g. an ``api_key`` for Bedrock/Vertex, a
            ``base_url`` for a fixed-endpoint provider), or if a bearer provider
            has no key configured and no matching environment variable.
        AssertionError: If ``config.provider`` is not wired to a provider
            here (an unwired enum member). See :func:`typing.assert_never`.
    """
    if config is None:
        config = active_config()

    cls = _provider_class(config.provider)
    cls.reject_unread_config_fields(config)
    return cls.build(config)


def make_provider(
    provider: Provider,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    aws_region_name: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    gemini_structured_output: Literal["schema", "json"] = "schema",
) -> BaseProvider:
    """Construct a provider directly from raw credentials.

    The one-liner for the per-call ``provider=`` override: build a provider
    from a raw key (and optional model) without standing up an
    :class:`LLMClientConfig` or touching the module-level config source. Reach
    for it to route a single call through a different provider family, model, or
    credential than the globally-configured one::

        provider = make_provider(Provider.ANTHROPIC, api_key=key)
        result = structured_llm_call_sync(..., provider=provider)

    Note this does not isolate rate-limit budgets: the limiter is keyed by
    provider *name*, so calls sharing a provider share its budget (llmkit is
    single-tenant by design — see :mod:`llmkit.rate_limiting`).

    This function's keywords are the fields of :class:`LLMClientConfig`, but a
    knob the selected provider does not read is **rejected**, not silently
    ignored: passing ``api_key`` to Ollama/Bedrock/Vertex, ``base_url`` to
    Bedrock/Vertex, ``aws_region_name`` to anything but Bedrock, ``vertex_*`` to
    anything but Vertex, or a non-default ``gemini_structured_output`` to a
    non-Gemini provider raises ``ValueError``. Populate only the fields the
    selected provider uses. A falsy ``model`` resolves to the provider's own
    default, so the assembled LiteLLM id is always well-formed.

    Args:
        provider: Which provider to construct.
        api_key: Bearer credential for the five key-authenticated providers;
            resolves from this value, else the provider's environment variable,
            else raises. Not accepted by Ollama (local) or Bedrock/Vertex (which
            sign via their ambient AWS/Google credential chains).
        model: Default model for this provider; ``None`` uses the provider's
            built-in default.
        base_url: Endpoint (gateway, proxy, compatible endpoint) for every
            key-authenticated provider and Ollama, forwarded to LiteLLM as
            ``api_base``: it resolves from this value, else the provider's own
            documented endpoint variable(s) (``OPENAI_BASE_URL`` and friends,
            still honoured), else that provider's measured default — resolved
            per call, so it reflects the environment the call goes out in. That
            keeps a source llmkit does not read from choosing the endpoint, with
            two documented exceptions: Google AI Studio owns no default (its base
            carries a model-derived API version) and so sends nothing when
            neither is set, and Ollama's LiteLLM route ranks ``litellm.api_base``
            above the value sent. Not accepted by Bedrock/Vertex, whose endpoints
            derive from region/location.
        reasoning_effort: Provider "thinking" effort forwarded to LiteLLM;
            ``None`` leaves the provider default in place.
        aws_region_name: AWS region for Bedrock; not accepted by any other
            provider.
        vertex_project: GCP project id for Vertex AI; not accepted by any other
            provider. ``None`` resolves from the environment / ADC.
        vertex_location: Vertex AI region — the data-residency control; not
            accepted by any other provider. ``None`` resolves from the
            environment, else Google's default region.
        gemini_structured_output: Structured-output strategy for the Gemini
            providers (Vertex, Google AI Studio); a non-default value is not
            accepted by any other provider. ``"schema"`` (default) keeps Gemini's
            native JSON-schema constrained decoding; ``"json"`` switches to
            JSON-mime-type output
            with client-side validation to escape the constrained-decoding
            repetition-loop trap (see the Gemini provider docstrings).

    Returns:
        A constructed provider, ready to pass as a per-call ``provider=``.

    Raises:
        ValueError: If a passed knob is not read by ``provider`` (see above), or
            if a bearer provider has no ``api_key`` and no matching environment
            variable.
        AssertionError: If ``provider`` is not wired here (an unwired enum
            member). See :func:`typing.assert_never`.
    """
    # This function's keyword params *are* the fields of LLMClientConfig, so
    # synthesize a config and delegate to the single seam in build_provider —
    # which resolves the provider class, rejects any knob the provider does not
    # read, and builds. There is no second, parallel match to keep in sync, and
    # the endpoint-default literals live only in each provider.
    config = LLMClientConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        aws_region_name=aws_region_name,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        gemini_structured_output=gemini_structured_output,
    )
    return build_provider(config)


def describe_llm(config: LLMClientConfig | None = None) -> LLMInfo:
    """Read descriptive metadata for the active (or given) configuration.

    A *read* accessor (``get_*``/``describe_*`` semantics), distinct from the
    ``build_*`` / ``make_*`` constructors: it builds a provider only to read
    back the resolved provider + model for display/telemetry.

    Returns:
        :class:`LLMInfo` snapshot of the resolved provider + model.
    """
    if config is None:
        config = active_config()
    provider = build_provider(config)

    return LLMInfo(
        provider=config.provider,
        provider_name=provider.name,
        model=provider.model,
        is_local=provider.is_local,
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
    "BedrockProvider",
    "VertexProvider",
    "Provider",
    "LLMClientConfig",
    "LLMInfo",
    "configure_llm_client",
    "build_provider",
    "make_provider",
    "describe_llm",
]
