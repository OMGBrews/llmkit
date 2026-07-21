"""DeepSeek LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_api_key,
    resolve_base_url,
)

#: DeepSeek's public API endpoint — the fallback when neither ``base_url`` nor an
#: endpoint environment variable is set. **Measured, not copied from LiteLLM's
#: source**: this exact string reproduces the pre-fix request URL byte for byte
#: (``…/beta/chat/completions``), verified against litellm 1.92.0 on 2026-07-21.
#: The ``/beta`` segment is LiteLLM's own choice of DeepSeek base, not a typo.
_DEFAULT_BASE_URL = "https://api.deepseek.com/beta"


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
    gateway (LiteLLM accepts ``api_base`` on the ``deepseek/`` route).
    ``api_base`` is **always** forwarded: it resolves from ``base_url`` if set,
    else ``DEEPSEEK_API_BASE``, else :data:`_DEFAULT_BASE_URL`. Sending it
    unconditionally is what keeps a source llmkit does **not** read — the
    ``litellm.api_base`` module global, a LiteLLM key-management backend, an alias
    a future release adds — from steering a caller's explicitly-pinned key to an
    endpoint they never named. The variable named above still steers,
    deliberately: llmkit reads it itself now, so a host already using it is
    unaffected (see :func:`resolve_base_url`). The wire shape with nothing
    configured is unchanged, because the default is the endpoint LiteLLM itself
    resolved.
    """

    _provider_name: ClassVar[str] = "DeepSeek"
    _model_prefix: ClassVar[str] = "deepseek/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "deepseek-chat"
    _api_key_env_var: ClassVar[str] = "DEEPSEEK_API_KEY"
    #: The endpoint variable LiteLLM consulted for this route (measured, not read
    #: off the source). Read here so a host that relies on it keeps working —
    #: explicitly, and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("DEEPSEEK_API_BASE",)
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
        # The *configured* endpoint, not the resolved one: resolution reads the
        # environment and so must happen at the read point, never here (see
        # ``completion_kwargs``).
        self._base_url: str | None = base_url

    @override
    def completion_kwargs(self) -> dict[str, object]:
        # Resolved per call, deliberately — see the note in ``OpenAIProvider``
        # and :func:`resolve_base_url`: constructing a provider before LiteLLM's
        # import-time ``load_dotenv()`` would otherwise freeze the wrong endpoint.
        return {
            "api_key": self._api_key,
            "api_base": resolve_base_url(
                self._base_url, env_vars=self._base_url_env_vars, default=_DEFAULT_BASE_URL
            ),
        }

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> DeepSeekProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``DEEPSEEK_API_KEY``
        environment variable, else raises (see :func:`resolve_api_key`).
        ``base_url`` is passed through as configured; the endpoint is resolved
        later, in ``completion_kwargs()``, so it observes the environment the
        call actually goes out in (see :func:`resolve_base_url`).
        """
        return cls(
            api_key=resolve_api_key(
                config.api_key, env_var=cls._api_key_env_var, provider_name=cls._provider_name
            ),
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )
