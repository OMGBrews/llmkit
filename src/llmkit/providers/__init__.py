"""LLM provider package: shared contract, concrete providers, and dispatch.

Public facade for the provider layer. The provider-agnostic core lives in
:mod:`llmkit.providers.base`; each concrete provider lives in its own
sibling module (``openrouter``, ``ollama``, ``google``, ``anthropic``,
``openai``, ``deepseek``, ``bedrock``, ``vertex``, ``cerebras``, ``moonshot``,
``alibaba``). This package is the
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

import importlib
from typing import Literal, assert_never

from llmkit._types import ReasoningEffort
from llmkit.providers.alibaba import AlibabaProvider
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
from llmkit.providers.cerebras import CerebrasProvider
from llmkit.providers.deepseek import DeepSeekProvider
from llmkit.providers.google import GoogleProvider
from llmkit.providers.moonshot import MoonshotProvider
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
        case Provider.CEREBRAS:
            return CerebrasProvider
        case Provider.MOONSHOT:
            return MoonshotProvider
        case Provider.ALIBABA:
            return AlibabaProvider
    # Every Provider member is handled above, so the subject narrows to Never
    # here. assert_never makes that a *static* guarantee: basedpyright reports
    # a type error if a newly-added Provider member is missing its case, and it
    # raises loudly at runtime for any unwired value — never a silent fallback.
    assert_never(provider)


# The inverse of :func:`_provider_class`, for :func:`describe_llm`: a built
# instance does not record its member, so the member is recovered from the
# instance's exact class. The exhaustiveness test guarantees the classes, one
# per member, are distinct, so the inverse is unambiguous.
_MEMBER_BY_CLASS: dict[type[BaseProvider], Provider] = {_provider_class(m): m for m in Provider}


def provider_display_name(provider: Provider) -> str:
    """Return the name llmkit's calls on ``provider``'s own endpoint report.

    A member names the provider's default endpoint. An instance built with an
    explicit ``base_url`` elsewhere reports its own route name instead (see
    :attr:`~llmkit.providers.base.BaseProvider.name`).

    The name differs from the member's value for some members
    (``Provider.VERTEX`` is ``"vertex"`` but reports ``"Google Agent
    Platform"``), so code that keys state by the provider name, such as the rate
    limiter, translates a member through this rather than using its value.
    """
    return _provider_class(provider).display_name()


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
            provider does not read (e.g. an ``api_key`` for Bedrock or Gemini Enterprise Agent
            Platform (formerly Vertex AI), a ``base_url`` for a fixed-endpoint provider), or if a
            bearer provider has no key configured and no matching environment variable.
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

    For a per-call ``provider=`` that needs its own key, model or other knob:
    build a provider from a raw key (and optional model) without standing up an
    :class:`LLMClientConfig` or touching the module-level config source. A call
    that needs only the provider's environment credentials and default model
    passes the :class:`Provider` member itself (``provider=Provider.ANTHROPIC``),
    which the call builds with this function. Passing the result as
    ``provider=`` switches that call's provider, model, or credential the way
    ``model=`` switches its model, so one program can send calls to several
    providers, for example to compare them on the same prompt, with or without
    a registered config::

        provider = make_provider(Provider.ANTHROPIC, api_key=key)
        result = structured_llm_call_sync(..., provider=provider)

    Note this does not isolate one credential from another: two instances of
    the same provider, for example two accounts, share its rate-limit budget
    and circuit breaker (both keyed by provider *name*) and its cached answers
    (the response-cache key leaves credentials out). llmkit is single-tenant by
    design; see :mod:`llmkit.rate_limiting`.

    This function's keywords are the fields of :class:`LLMClientConfig`, but a knob the selected
    provider does not read is **rejected**, not silently ignored: passing ``api_key`` to Ollama,
    Bedrock, or Google's Agent Platform, ``base_url`` to Bedrock or Google's Agent Platform,
    ``aws_region_name`` to anything but Bedrock, ``vertex_*`` to anything but Google's Agent
    Platform, or a non-default ``gemini_structured_output`` to a non-Gemini provider raises
    ``ValueError``. Populate only the fields the selected provider uses. A falsy ``model`` resolves
    to the provider's own default, so the assembled LiteLLM id is always well-formed.

    Args:
        provider: Which provider to construct.
        api_key: Bearer credential for the key-authenticated providers
            (OpenRouter, Google AI Studio, Anthropic, OpenAI, DeepSeek, Cerebras, Moonshot,
            Alibaba); resolves from
            this value, else the provider's environment variable as ``os.environ`` holds it when
            this function runs (a key kept only in a ``.env`` file is not loaded until the first
            call imports LiteLLM, so load the file first), else raises. Not accepted by
            Ollama (local) or Bedrock or Google's Agent Platform (which sign via their ambient
            AWS/Google credential chains).
        model: Default model for this provider; ``None`` uses the provider's
            built-in default.
        base_url: Endpoint (gateway, proxy, compatible endpoint) for every
            key-authenticated provider and Ollama, forwarded to LiteLLM as ``api_base``.
            Anthropic, OpenAI, DeepSeek, Cerebras, Moonshot, Alibaba and Google AI Studio resolve
            it per call from this value, else their endpoint variable(s) (``OPENAI_BASE_URL``,
            ``GEMINI_API_BASE`` and friends), else a measured default. Google AI Studio has no default and sends
            nothing when neither this value nor ``GEMINI_API_BASE`` is set (its base carries a
            model-derived API version). OpenRouter and Ollama read no variable: this value, else
            llmkit's default, fixed when the provider is built. Ollama's LiteLLM route ranks
            ``litellm.api_base`` above the value sent. Not accepted by Bedrock or Google's Agent
            Platform, whose endpoints LiteLLM derives from ``aws_region_name`` /
            ``vertex_location`` — for those two an ambient endpoint value can override the pinned
            region, which matters because that region is the residency control (see each provider's
            class docstring).
        reasoning_effort: Provider "thinking" effort translated to its
            request control; ``None`` leaves the provider default in place.
        aws_region_name: AWS region for Bedrock; not accepted by any other
            provider.
        vertex_project: GCP project id for Google's Agent Platform; not accepted by any other
            provider. ``None`` resolves from the environment / ADC.
        vertex_location: Google's Agent Platform region — the data-residency control; not
            accepted by any other provider. ``None`` resolves from the
            environment, else LiteLLM's default ``us-central1``, which does
            not serve the default model.
        gemini_structured_output: Structured-output strategy for the Gemini
            providers (Google's Agent Platform, Google AI Studio); a non-default value is not
            accepted by any other provider. ``"schema"`` (default) keeps Gemini's native
            JSON-schema constrained decoding; ``"json"`` switches to JSON-mime-type output with
            client-side validation to escape the constrained-decoding repetition-loop trap (see the
            Gemini provider docstrings).

    Returns:
        A constructed provider, ready to pass as a per-call ``provider=``.

    Raises:
        TypeError: If ``provider`` is not a :class:`Provider` member, a plain
            string included (``make_provider("ollama")``); parse a name with
            ``Provider(value)``, which ignores case.
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


def coerce_call_provider(
    value: LLMProviderInterface | Provider | None,
) -> LLMProviderInterface | None:
    """Turn a call's ``provider=`` into the instance it runs on, or refuse it.

    Shared by every call and by :func:`describe_llm`, so ``describe_llm(provider=x)``
    describes exactly the provider a call passing ``provider=x`` runs on.

    A :class:`~llmkit.Provider` member is built with
    :func:`~llmkit.make_provider` from environment credentials, after importing
    LiteLLM: that import loads a ``.env`` file, so the first call of a process
    sees a key kept there as every later call does. The build's own error (a
    missing key, a missing extra) propagates, so a member that cannot be built
    fails before any attempt rather than degrading to the registered provider
    the way a call's global path does (``llmkit.calls._shared.build_call_provider``).

    A plain string raises :class:`TypeError`: a call site is code that can
    import the enum, so the string is refused rather than parsed. The member
    test comes first because a ``StrEnum`` member is also a ``str``. Anything
    else (an instance, or ``None``) is returned unchanged.
    """
    if isinstance(value, Provider):
        # Importing the transport imports LiteLLM, which loads ``.env``; by
        # name rather than ``import llmkit._litellm``, which would bind the
        # package and close an import cycle through ``llmkit/__init__``.
        _ = importlib.import_module("llmkit._litellm")
        return make_provider(value)
    if isinstance(value, str):
        raise TypeError(_bare_provider_string_message(value))
    return value


def _bare_provider_string_message(value: str) -> str:
    """The fix-naming message for a provider name passed as a plain string."""
    head = f"provider= takes a Provider member or a built provider, not the string {value!r}: "
    try:
        member = Provider(value)
    except ValueError:
        expected = ", ".join(f"Provider.{m.name}" for m in Provider)
        return (
            head
            + "pass a Provider member (one of "
            + expected
            + "), or make_provider(Provider.X, api_key=...) for a call that needs its own key."
        )
    return (
        head
        + f"pass Provider.{member.name}, or make_provider(Provider.{member.name}, api_key=...) "
        + "for a call that needs its own key."
    )


def _member_for(instance: LLMProviderInterface | None) -> tuple[Provider, BaseProvider]:
    """Recover the :class:`Provider` member of a shipped provider instance.

    The match is on the exact class: a host's own provider, including a
    subclass of a shipped class, is a route llmkit has not measured, so it has
    no member and raises :class:`TypeError` naming the accepted inputs.
    """
    if isinstance(instance, BaseProvider):
        member = _MEMBER_BY_CLASS.get(type(instance))
        if member is not None:
            return member, instance
    shipped = ", ".join(cls.__name__ for cls in _MEMBER_BY_CLASS)
    raise TypeError(
        "describe_llm describes an LLMClientConfig passed as config, or a Provider "
        + "member or an instance of a shipped provider class passed as provider= "
        + f"(one of {shipped}); got an instance of {type(instance).__qualname__}, "
        + "which llmkit has not measured. Read its decisions from the instance's own "
        + "instructor_mode, strict_json_schema, single_tool_call_per_schema and "
        + "supports_tool_choice properties."
    )


def _snapshot(member: Provider, instance: BaseProvider) -> LLMInfo:
    """Fill every :class:`LLMInfo` field from a built shipped provider."""
    return LLMInfo(
        provider=member,
        provider_name=instance.name,
        model=instance.model,
        is_local=instance.is_local,
        compose_tools_schema=instance.compose_tools_schema,
        instructor_mode=instance.instructor_mode,
        strict_json_schema=instance.strict_json_schema,
        single_tool_call_per_schema=instance.single_tool_call_per_schema,
        supports_tool_choice=instance.supports_tool_choice,
    )


def describe_llm(
    config: LLMClientConfig | None = None,
    *,
    provider: Provider | LLMProviderInterface | None = None,
) -> LLMInfo:
    """Describe the provider a call would use, and the decisions it makes.

    A *read* accessor (``get_*``/``describe_*`` semantics), distinct from the
    ``build_*`` / ``make_*`` constructors: it reads back the resolved provider
    and model, plus the route's correctness decisions (how a structured answer
    is requested and checked), without making a call.

    Three inputs, matching what a call can run on:

    - No argument: the config registered with :func:`configure_llm_client`.
    - ``config``: that config, built with :func:`build_provider`, so a missing
      key, a missing extra or a rejected field raises here, before any call.
    - ``provider=``: the same values a call's ``provider=`` takes. A
      :class:`Provider` member is built the way a call builds it, with
      :func:`make_provider` from environment credentials after importing
      LiteLLM (which loads ``.env``), so it reports that provider's default
      model, not the registered config's. An instance of a shipped provider
      class is described as built. A plain string raises the
      :class:`TypeError` a call raises.

    The snapshot does not resolve the routed model id, and it does not report
    decisions made per model rather than per route, such as Anthropic's
    compose guard or how ``reasoning_effort="disable"`` and ``"lowest"`` are
    translated.

    Args:
        config: Explicit config to describe. When ``None`` and ``provider`` is
            not given, the registered config is described.
        provider: A :class:`Provider` member or a built instance of a shipped
            provider class to describe instead of a config.

    Returns:
        :class:`LLMInfo` with every field filled.

    Raises:
        TypeError: If both ``config`` and ``provider`` are given, if
            ``provider`` is a plain string, or if it is an instance of a class
            llmkit does not ship (a host subclass of a shipped class included).
        ValueError: If the provider cannot be built, such as a bearer provider
            with no key configured.
    """
    if provider is not None:
        if config is not None:
            raise TypeError(
                "describe_llm takes a config or a provider=, not both: pass the "
                + "config to describe the program-wide setting, or provider= to "
                + "describe what a call with that provider= uses."
            )
        member, shipped = _member_for(coerce_call_provider(provider))
        return _snapshot(member, shipped)

    if config is None:
        config = active_config()
    return _snapshot(config.provider, build_provider(config))


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
    "CerebrasProvider",
    "MoonshotProvider",
    "AlibabaProvider",
    "Provider",
    "LLMClientConfig",
    "LLMInfo",
    "configure_llm_client",
    "build_provider",
    "make_provider",
    "describe_llm",
]
