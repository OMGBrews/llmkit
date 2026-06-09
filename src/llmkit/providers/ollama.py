"""Ollama LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig

#: Default local Ollama endpoint — the fallback when no ``base_url`` is
#: configured. Defined once so the ctor default and :meth:`build` agree.
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Ollama LLM provider.

    Talks to a locally-running Ollama instance (``ollama_chat/<model>``)
    for fully local/private LLM access. No data leaves the host.
    """

    _provider_name: ClassVar[str] = "Ollama"
    _model_prefix: str = "ollama_chat/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "llama3.2"
    is_local: bool = True

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._base_url: str = base_url

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_base": self._base_url}

    @classmethod
    def build(cls, config: LLMClientConfig) -> OllamaProvider:
        """Construct from an :class:`LLMClientConfig` (local endpoint default)."""
        return cls(
            base_url=config.base_url or _DEFAULT_BASE_URL,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
