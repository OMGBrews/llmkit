"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

from typing import override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig, require_anthropic_sdk


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider.

    Routes Claude through the Anthropic API (``anthropic/<model>``), pinned
    to Anthropic's native JSON mode for instructor.

    instructor reaches the Anthropic SDK only at call time on its
    ``ANTHROPIC_JSON`` usage-accounting path, so the SDK ships in the opt-in
    ``omg-llmkit[anthropic]`` extra. Constructing this provider without it
    raises a clear "install omg-llmkit[anthropic]" error (see
    :func:`~llmkit.providers.base.require_anthropic_sdk`).
    """

    _provider_name: str = "Anthropic"
    _model_prefix: str = "anthropic/"
    _mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON
    _default_model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        require_anthropic_sdk(self._provider_name)
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
