"""Moonshot AI (Kimi) LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_api_key,
    resolve_base_url,
)

#: Moonshot's international API endpoint (the ``platform.kimi.ai`` account's), the
#: fallback when neither ``base_url`` nor ``MOONSHOT_API_BASE`` is set. It is also
#: the endpoint LiteLLM's ``moonshot/`` route resolves with nothing configured, at
#: litellm 1.104.0 and 1.106.6, so the wire shape with nothing configured is LiteLLM's.
_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


class MoonshotProvider(BaseProvider):
    """Moonshot AI (Kimi) LLM provider.

    Calls Moonshot's OpenAI-compatible API directly through LiteLLM's native
    ``moonshot/`` route, on a first-party key. The default model is
    ``kimi-k3.1-pro``. It reasons by default.

    Measured live on 2026-11-14 against litellm 1.104.0 (the floor), 1.106.6 and
    1.109.1, with instructor 1.17.0, on the default model unless stated:

    * **Structured output** is pinned to ``instructor.Mode.JSON`` (a
      ``json_object`` response format, with the schema in the prompt and
      validated locally), not ``Mode.JSON_SCHEMA``. Moonshot does not reliably
      enforce a ``json_schema`` response format: with thinking on,
      ``kimi-k2.6`` answered three of three raw requests carrying one with
      fenced JSON under keys the schema does not name (litellm 1.104.0 and
      1.109.1; the request carried the schema), and through instructor
      ``JSON_SCHEMA`` failed or needed a second attempt on it. ``Mode.JSON``
      validated a fully specified extraction on the first attempt nine times of
      nine across ``kimi-k2.6``, ``kimi-k3`` and ``kimi-k3.1-pro`` with thinking
      on (litellm 1.109.1), and on the default with thinking on and off, plus a
      ``pattern``/``const`` schema (1.106.6). ``JSON_SCHEMA`` did validate on
      ``kimi-k3`` and the default. DeepSeek is the precedent for this pin.
    * **Tools**: an unforced tool turn, ``tool_choice="none"`` and a nested,
      unnormalised tool schema all work. A **named** ``tool_choice`` works only
      with thinking off: with thinking on Moonshot answers a 400 (``tool_choice
      'specified' is incompatible with thinking enabled``), and at ``"low"``
      too. llmkit does not turn thinking off for you, which would change what
      the call does; pass ``reasoning_effort="disable"`` with a named tool.
      ``"required"`` is not forwarded at all: LiteLLM removes it and appends a
      user message asking the model to pick a tool, so it is a request, not a
      guarantee.
    * **Tools plus a response schema** in one request were not measured, so
      the compose trait stays off and callers get the portable two-step steer.
    * **Streaming**: a streamed tool turn interleaves prose and the tool call,
      and both streams report usage.
    * **Reasoning**: Moonshot turns thinking off for ``"none"``. Forwarded
      unchanged, ``"disable"`` was accepted and the model kept thinking, so on
      a reasoning model the portable ``"disable"`` and ``"lowest"`` are both
      sent as ``"none"``. LiteLLM's route accepts ``reasoning_effort`` only for
      models its metadata marks as reasoning, and raises
      ``UnsupportedParamsError`` before sending for any other model (checked
      offline at litellm 1.109.1 on ``moonshot-v1-8k``). So on a model the
      metadata lists as non-reasoning (the ``moonshot-v1`` family) the portable
      values send nothing, since it already does not think, and on a model the
      metadata does not list they raise
      :class:`~llmkit.exceptions.ReasoningEffortUnsupportedError` before
      transport. Other values pass through unchanged, for LiteLLM and Moonshot
      to accept or reject. On reasoning models LiteLLM also drops any
      ``temperature``.
    * **Thinking-on tool loops** were measured only as single turns. A later
      turn that resends an assistant tool call relies on LiteLLM inserting a
      placeholder ``reasoning_content``, which Moonshot requires on those
      messages, because llmkit's tool messages carry none; that path is not
      measured. The measured multi-turn loop ran with thinking off.

    An optional ``base_url`` points the provider at a Moonshot-compatible
    endpoint, such as Moonshot's China endpoint. ``api_base`` is **always**
    forwarded: it resolves from ``base_url`` if set, else ``MOONSHOT_API_BASE``,
    else :data:`_DEFAULT_BASE_URL`, so a source llmkit does not read (the
    ``litellm.api_base`` module global, a LiteLLM key-management backend) cannot
    steer the caller's key elsewhere (see :func:`resolve_base_url`).
    """

    _provider_name: ClassVar[str] = "Moonshot"
    _default_base_url: ClassVar[str | None] = _DEFAULT_BASE_URL
    _model_prefix: ClassVar[str] = "moonshot/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "kimi-k3.1-pro"
    _api_key_env_var: ClassVar[str] = "MOONSHOT_API_KEY"
    #: The endpoint variable LiteLLM's ``moonshot/`` route consults. Read here so
    #: a host that relies on it keeps working, explicitly and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("MOONSHOT_API_BASE",)
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset({"api_key", "base_url"})
    # Stated rather than inherited: tool_choice works, within the thinking
    # constraint the class docstring describes; compose was not measured.
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
        """Turn thinking off for ``"disable"`` and ``"lowest"`` by what LiteLLM knows of ``model``.

        Moonshot accepts ``"disable"`` but keeps thinking, so neither portable
        value may reach it unchanged. A model LiteLLM marks as reasoning gets
        ``"none"``, Moonshot's thinking-off value; one it lists without
        reasoning gets nothing, because LiteLLM rejects the parameter there and
        the model does not think; an unlisted model raises, because LiteLLM
        would reject the parameter and nothing establishes how to turn thinking
        off. Other values pass through unchanged.
        """
        if effort not in ("disable", "lowest"):
            return {"reasoning_effort": effort}

        import litellm

        bare = model.removeprefix(self._model_prefix)
        try:
            info = litellm.get_model_info(bare, custom_llm_provider="moonshot")
        except Exception:
            # Do not leak dependency errors (ModelNotMappedError) or send blind.
            raise ReasoningEffortUnsupportedError(
                f"{bare!r} is not in LiteLLM's model metadata, so llmkit cannot establish how "
                + f'to turn thinking off on it; reasoning_effort="{effort}" is not available. '
                + "Choose a supported native reasoning_effort explicitly, or leave it unset."
            ) from None
        if info.get("supports_reasoning") is True:
            return {"reasoning_effort": "none"}
        return {}

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> MoonshotProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``MOONSHOT_API_KEY``
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
