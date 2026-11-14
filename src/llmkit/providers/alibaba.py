"""Alibaba (QwenCloud) LLM provider."""

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

#: QwenCloud's Singapore pay-as-you-go endpoint, the fallback when neither
#: ``base_url`` nor ``QWENCLOUD_API_BASE`` is set. QwenCloud keys are regional, so
#: an account in another region sets one of those. This is **not** the endpoint
#: LiteLLM's ``qwencloud/`` route picks with nothing configured
#: (``dashscope-intl.aliyuncs.com``); llmkit always sends its own.
_DEFAULT_BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"


class AlibabaProvider(BaseProvider):
    """Alibaba (QwenCloud) LLM provider for the Qwen models.

    Calls QwenCloud's OpenAI-compatible API directly through LiteLLM's native
    ``qwencloud/`` route, on a first-party key. The default model is
    ``qwen3.7-plus``. It reasons by default.

    Measured live on 2026-11-14 against litellm 1.104.0 (the floor), 1.106.6 and
    1.109.1, with instructor 1.17.0, on the default model unless stated:

    * **Structured output** is pinned to ``instructor.Mode.JSON_SCHEMA`` without
      the strict upgrade. A fully specified extraction validated with every
      field populated, with thinking on and off, as did a ``pattern``/``const``
      schema. ``Mode.JSON`` also validated; the schema mode was chosen because
      it is the constrained one.
    * **Tools**: an unforced tool turn, a named ``tool_choice``, ``"required"``,
      ``"none"`` and a nested, unnormalised tool schema all work, with thinking
      on.
    * **Tools plus a response schema** in one request were not measured, so
      the compose trait stays off and callers get the portable two-step steer.
    * **Streaming**: both streams report usage. With thinking on, an unforced
      streamed tool turn sent no prose before its call (twice); with thinking
      off, prose and the call interleaved.
    * **Reasoning**: QwenCloud accepts ``"none"``, ``"minimal"``, ``"low"``,
      ``"medium"``, ``"high"`` and ``"xhigh"``, and rejects anything else with a
      400, ``"disable"`` included. ``"none"`` turned thinking off, so the
      portable ``"disable"`` and ``"lowest"`` are both sent as ``"none"``.
      Other values pass through for QwenCloud to accept or reject.

    Pricing comes from LiteLLM's cost map, which prices the default by token
    tier; the measured streams recorded positive estimates.

    An optional ``base_url`` points the provider at another QwenCloud region or
    a compatible endpoint. ``api_base`` is **always** forwarded: it resolves from
    ``base_url`` if set, else ``QWENCLOUD_API_BASE``, else
    :data:`_DEFAULT_BASE_URL`, so a source llmkit does not read (the
    ``litellm.api_base`` module global, a LiteLLM key-management backend) cannot
    steer the caller's key elsewhere (see :func:`resolve_base_url`). The key
    resolves only from ``api_key`` or ``QWENCLOUD_API_KEY``; LiteLLM's own
    fallback to ``DASHSCOPE_API_KEY`` is never reached, because llmkit always
    sends a key.
    """

    _provider_name: ClassVar[str] = "Alibaba"
    _default_base_url: ClassVar[str | None] = _DEFAULT_BASE_URL
    _model_prefix: ClassVar[str] = "qwencloud/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "qwen3.7-plus"
    _api_key_env_var: ClassVar[str] = "QWENCLOUD_API_KEY"
    #: The endpoint variable LiteLLM's ``qwencloud/`` route consults. Read here so
    #: a host that relies on it keeps working, explicitly and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("QWENCLOUD_API_BASE",)
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url"})
    # Stated rather than inherited: tool_choice was measured; compose was not.
    _supports_tool_choice: ClassVar[bool] = True
    _compose_tools_schema: ClassVar[bool] = False

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
    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Translate ``"disable"`` and ``"lowest"`` to ``"none"``, QwenCloud's thinking-off value.

        QwenCloud rejects ``"disable"``, and thinking can be turned off, so the
        least reasoning the route supports is off. Other values pass through.
        """
        del model
        return {"reasoning_effort": "none" if effort in ("disable", "lowest") else effort}

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> AlibabaProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``QWENCLOUD_API_KEY``
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
