"""OpenAI LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider.

    Calls OpenAI (GPT / o-series / GPT-5) directly (``openai/<model>``),
    pinned to OpenAI's native structured-outputs mode (``Mode.JSON_SCHEMA``
    — a ``strict: true`` ``response_format`` JSON schema). This is the
    direct alternative to the indirect ``openrouter/openai/...`` hop, which
    adds a markup and a different structured-output mode.

    ``Mode.TOOLS_STRICT`` (forced strict function call) was the original
    pick but the live smoke test caught it silently returning empty fields
    on ``gpt-4.1-mini`` (0/4 valid; ``gpt-4o-mini`` / GPT-5 affected too) —
    the exact silent-regression failure mode this map exists to prevent.
    ``Mode.JSON_SCHEMA`` round-trips cleanly across GPT-4.1 / 4o / o-series
    / GPT-5, and matches the JSON-schema mode the other providers pin.

    An optional ``base_url`` points the same provider at an OpenAI-compatible
    gateway; left unset, LiteLLM uses OpenAI's default endpoint (so
    ``api_base`` is only forwarded when a ``base_url`` is given).
    """

    _provider_name: ClassVar[str] = "OpenAI"
    _model_prefix: str = "openai/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gpt-4.1-mini"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key
        self._base_url: str | None = base_url

    @override
    def completion_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["api_base"] = self._base_url
        return kwargs

    @classmethod
    def build(cls, config: LLMClientConfig) -> OpenAIProvider:
        """Construct from an :class:`LLMClientConfig` (``base_url`` passed through)."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )
