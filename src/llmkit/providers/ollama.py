"""Ollama LLM provider."""

from __future__ import annotations

from typing import override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class OllamaProvider(BaseProvider):
    """Ollama LLM provider.

    Talks to a locally-running Ollama instance (``ollama_chat/<model>``)
    for fully local/private LLM access. No data leaves the host.
    """

    _provider_name: str = "Ollama"
    _model_prefix: str = "ollama_chat/"
    _mode: instructor.Mode = instructor.Mode.JSON_SCHEMA
    _default_model: str = "llama3.2"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._base_url: str = base_url

    @override
    def completion_kwargs(self) -> dict[str, str]:
        return {"api_base": self._base_url}

    @classmethod
    def build(cls, config: LLMClientConfig) -> OllamaProvider:
        """Construct from an :class:`LLMClientConfig` (local endpoint default)."""
        return cls(
            base_url=config.base_url or "http://localhost:11434",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
