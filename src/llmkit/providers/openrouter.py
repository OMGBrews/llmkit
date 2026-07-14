"""OpenRouter LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig

#: OpenRouter's public API endpoint — the fallback when no ``base_url`` is
#: configured. Defined once so the ctor default and :meth:`build` agree.
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider.

    Routes through OpenRouter's unified endpoint (``openrouter/<model>``),
    pinned to ``Mode.JSON_SCHEMA`` for instructor: the request carries the
    OpenAI-style strict ``response_format={"type": "json_schema", ...}``,
    which is exactly OpenRouter's structured-outputs surface.
    ``Mode.OPENROUTER_STRUCTURED_OUTPUTS`` (the pin through 0.6.x) sent the
    same request shape but was removed from the mode registry
    ``from_litellm`` validates against at client construction in instructor
    1.15.3, so it now raises ``RegistryError`` before any request is sent.

    **Schema-honoring routing (the sharp edge).** OpenRouter advertises
    ``structured_outputs`` as a *model-level* capability, but the strict
    ``response_format`` is actually enforced by whichever *serving* provider
    OpenRouter routes the request to. A model can list the capability while one
    of its serving endpoints quietly ignores the schema and returns free-form
    JSON — which then surfaces only as a confusing downstream validation
    failure, with nothing pointing at routing as the cause. To close that gap
    this provider sets OpenRouter's ``provider.require_parameters`` routing
    preference (via ``extra_body``) **by default**, so a request only lands on
    an endpoint that honors every parameter sent — including the structured
    ``response_format``. It restricts routing to capable endpoints, which can in
    principle reduce availability or shift cost; pass ``require_parameters=False``
    to opt out (you then accept the silent-free-form-JSON risk above). That opt-out
    is a **direct-construction-only escape hatch**: it is deliberately not exposed
    through the :func:`~llmkit.providers.build_provider` /
    :func:`~llmkit.providers.make_provider` factories, since
    :class:`~llmkit.LLMClientConfig` is provider-agnostic and does not carry an
    OpenRouter-specific flag.
    """

    _provider_name: ClassVar[str] = "OpenRouter"
    _model_prefix: ClassVar[str] = "openrouter/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "google/gemini-2.0-flash-001"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        reasoning_effort: str | None = None,
        require_parameters: bool = True,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key
        self._base_url: str = base_url
        self._require_parameters: bool = require_parameters

    @override
    def completion_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"api_key": self._api_key, "api_base": self._base_url}
        if self._require_parameters:
            # OpenRouter routing preference: only route to serving endpoints that
            # support every parameter in the request — so a strict
            # ``response_format`` is never silently dropped by an endpoint that
            # ignores it. LiteLLM forwards ``extra_body`` to OpenRouter verbatim.
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
        return kwargs

    @classmethod
    def build(cls, config: LLMClientConfig) -> OpenRouterProvider:
        """Construct from an :class:`LLMClientConfig` (OpenRouter endpoint default).

        ``require_parameters`` stays on (schema-honoring routing); construct the
        provider directly to opt out.
        """
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url or _DEFAULT_BASE_URL,
            reasoning_effort=config.reasoning_effort,
        )
