"""OpenRouter LLM provider."""

from __future__ import annotations

from typing import override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider.

    Routes through OpenRouter's unified endpoint (``openrouter/<model>``),
    using its native structured-outputs mode for instructor.

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
    to opt out (you then accept the silent-free-form-JSON risk above).
    """

    _provider_name: str = "OpenRouter"
    _model_prefix: str = "openrouter/"
    _mode: instructor.Mode = instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
    _default_model: str = "google/gemini-2.0-flash-001"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
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
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            reasoning_effort=config.reasoning_effort,
        )
