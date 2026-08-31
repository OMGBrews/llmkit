"""OpenRouter LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import BaseProvider, LLMClientConfig, resolve_api_key

#: OpenRouter's public API endpoint — the fallback when no ``base_url`` is
#: configured. Defined once so the ctor default and :meth:`build` agree.
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider.

    Routes through OpenRouter's unified endpoint (``openrouter/<model>``),
    pinned to ``Mode.JSON_SCHEMA`` for instructor: the request carries the
    OpenAI-style ``response_format={"type": "json_schema", ...}``, which is
    exactly OpenRouter's structured-outputs surface.
    ``Mode.OPENROUTER_STRUCTURED_OUTPUTS`` (the pin through 0.6.x) sent the
    same request shape but was removed from the mode registry
    ``from_litellm`` validates against at client construction in instructor
    1.15.3, so it now raises ``RegistryError`` before any request is sent.

    **Strict enforcement is restored at the call seam.** The removed mode's
    handler additionally sent ``"strict": true`` and ``additionalProperties:
    false``; instructor's core ``JSON_SCHEMA`` handler sends neither, and
    OpenRouter treats a non-strict json_schema as *advisory* — measured
    2026-07-14 on ``mistralai/mistral-nemo``: the model stochastically echoed
    the schema itself with values nested inside ``properties``. This provider
    therefore sets ``strict_json_schema``, which upgrades the
    ``response_format`` at the LiteLLM call seam to the exact wire shape the
    original pin was measured on.

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
    #: Gemini 2.5 Flash-Lite — cheap, fast, and honors the strict json_schema
    #: ``response_format`` this provider sends (measured 2026-07-14: 10/10
    #: valid structured round-trips through the public surface). Succeeds
    #: ``google/gemini-2.0-flash-001``, which OpenRouter retired — it left the
    #: catalog entirely, so through 0.7.0 *every* default-model call 404'd with
    #: "No endpoints found". The live smoke suite now drives this default with no
    #: ``model=`` override (``test_openrouter_live_default_model``), so the next
    #: retirement fails the release gate instead of shipping.
    _default_model: ClassVar[str] = "google/gemini-3.1-flash-lite"
    _strict_json_schema: ClassVar[bool] = True
    _api_key_env_var: ClassVar[str] = "OPENROUTER_API_KEY"
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url"})

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        reasoning_effort: ReasoningEffort | None = None,
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

    @override
    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Translate portable effort into OpenRouter's native ``reasoning`` body.

        Gemini 3.x requires thinking, so its portable ``disable`` intent maps
        to OpenRouter's lowest supported effort, ``minimal``. Other OpenRouter
        models can receive the explicit ``none`` control. Values outside the
        portable aliases pass through: OpenRouter accepts additional native
        levels and should reject an unsupported value itself.
        """
        native_effort = (
            "minimal"
            if effort == "disable" and model.startswith("google/gemini-3")
            else "none"
            if effort == "disable"
            else effort
        )
        return {"extra_body": {"reasoning": {"effort": native_effort}}}

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> OpenRouterProvider:
        """Construct from an :class:`LLMClientConfig` (OpenRouter endpoint default).

        ``api_key`` resolves from the config, else the ``OPENROUTER_API_KEY``
        environment variable, else raises (see :func:`resolve_api_key`).
        ``require_parameters`` stays on (schema-honoring routing); construct the
        provider directly to opt out.
        """
        return cls(
            api_key=resolve_api_key(
                config.api_key, env_var=cls._api_key_env_var, provider_name=cls._provider_name
            ),
            model=config.model,
            base_url=config.base_url or _DEFAULT_BASE_URL,
            reasoning_effort=config.reasoning_effort,
        )
