"""Google AI Studio LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class GoogleProvider(BaseProvider):
    """Google AI Studio LLM provider.

    Uses Gemini directly (``gemini/<model>``) for high rate limits and low
    latency, pinned to Gemini's native JSON-schema mode for instructor.
    """

    _provider_name: ClassVar[str] = "Google AI Studio"
    _model_prefix: str = "gemini/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_key": self._api_key}

    @classmethod
    def build(cls, config: LLMClientConfig) -> GoogleProvider:
        """Construct from an :class:`LLMClientConfig`."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
