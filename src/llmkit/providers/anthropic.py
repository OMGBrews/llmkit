"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider.

    Routes Claude through the Anthropic API (``anthropic/<model>``), pinned
    to ``Mode.JSON`` for instructor: the schema travels in a system message,
    ``response_format={"type": "json_object"}`` rides along (LiteLLM
    translates it for the Anthropic API), and the response text is parsed
    and validated against the schema.

    ``Mode.ANTHROPIC_JSON`` (the pin through 0.6.x) is gone: instructor
    1.15.3 removed it from the mode registry ``from_litellm`` validates
    against at client construction, so it now raises ``RegistryError``
    before any request is sent. ``Mode.JSON`` is the surviving core
    equivalent — the same schema-in-system-prompt wire shape the original
    pin was chosen for.

    No Anthropic SDK is required: LiteLLM speaks the Anthropic HTTP API
    directly, and instructor (>=1.15.4) reaches the SDK only on an optional
    usage-accounting branch guarded by ``try/except ImportError``. This
    provider therefore has no extra — a core install routes Claude.
    """

    _provider_name: ClassVar[str] = "Anthropic"
    _model_prefix: ClassVar[str] = "anthropic/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "claude-sonnet-4-6"

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
    def build(cls, config: LLMClientConfig) -> AnthropicProvider:
        """Construct from an :class:`LLMClientConfig`."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
