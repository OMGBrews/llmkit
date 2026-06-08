"""OpenRouter LLM provider."""

from __future__ import annotations

from typing import override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider.

    Routes through OpenRouter's unified endpoint (``openrouter/<model>``),
    using its native structured-outputs mode for instructor.
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
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key
        self._base_url: str = base_url

    @override
    def completion_kwargs(self) -> dict[str, str]:
        return {"api_key": self._api_key, "api_base": self._base_url}

    @classmethod
    def build(cls, config: LLMClientConfig) -> OpenRouterProvider:
        """Construct from an :class:`LLMClientConfig` (OpenRouter endpoint default)."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
            reasoning_effort=config.reasoning_effort,
        )
