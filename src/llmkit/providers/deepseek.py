"""DeepSeek LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import BaseProvider, LLMClientConfig, resolve_api_key


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

    An optional ``base_url`` points the provider at a DeepSeek-compatible
    gateway (LiteLLM accepts ``api_base`` on the ``deepseek/`` route); left
    unset, LiteLLM uses DeepSeek's default endpoint (so ``api_base`` is only
    forwarded when a ``base_url`` is given).
    """

    _provider_name: ClassVar[str] = "DeepSeek"
    _model_prefix: ClassVar[str] = "deepseek/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "deepseek-chat"
    _api_key_env_var: ClassVar[str] = "DEEPSEEK_API_KEY"
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url"})

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
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

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> DeepSeekProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``DEEPSEEK_API_KEY``
        environment variable, else raises (see :func:`resolve_api_key`);
        ``base_url`` is passed through.
        """
        return cls(
            api_key=resolve_api_key(
                config.api_key, env_var=cls._api_key_env_var, provider_name=cls._provider_name
            ),
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )
