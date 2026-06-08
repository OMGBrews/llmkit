"""DeepSeek LLM provider."""

from __future__ import annotations

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


class DeepSeekProvider(BaseProvider):
    """DeepSeek LLM provider.

    Calls DeepSeek directly (``deepseek/<model>``) on a first-party key —
    the direct alternative to the indirect ``openrouter/deepseek/...`` hop
    (which adds a gateway markup). Covers ``deepseek-chat`` (V3) and
    ``deepseek-reasoner`` (R1).

    Structured output is pinned to DeepSeek's native JSON mode
    (``instructor.Mode.JSON`` → ``response_format={"type": "json_object"}``).
    ``Mode.JSON_SCHEMA`` (the strict ``response_format`` json-schema the
    OpenAI/Google/Ollama providers pin) is **not** accepted by DeepSeek's
    API — it returns a ``BadRequestError`` — so this provider pins
    ``Mode.JSON``, measured to validate on both ``deepseek-chat`` and
    ``deepseek-reasoner`` (see ``tests/integration/test_live_providers.py``).

    ``reasoning_effort`` is forwarded for ``deepseek-reasoner`` (R1) and is
    harmless on ``deepseek-chat`` (litellm tolerates it). DeepSeek keeps the
    reasoner's thinking tokens in a budget separate from ``max_tokens``, and
    because output is validated through instructor, a too-small ``max_tokens``
    cap surfaces as a loud retry error rather than the silently-empty shape
    seen with other providers' reasoning modes.
    """

    _provider_name: str = "DeepSeek"
    _model_prefix: str = "deepseek/"
    _mode: instructor.Mode = instructor.Mode.JSON
    _default_model: str = "deepseek-chat"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key

    def completion_kwargs(self) -> dict[str, str]:
        return {"api_key": self._api_key}

    @classmethod
    def build(cls, config: LLMClientConfig) -> DeepSeekProvider:
        """Construct from an :class:`LLMClientConfig`."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
