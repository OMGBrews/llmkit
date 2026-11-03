"""Ollama LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError
from llmkit.providers.base import BaseProvider, LLMClientConfig

#: Default local Ollama endpoint — the fallback when no ``base_url`` is
#: configured. Defined once so the ctor default and :meth:`build` agree.
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Ollama LLM provider.

    Talks to a locally-running Ollama instance (``ollama_chat/<model>``)
    for fully local/private LLM access. No data leaves the host.

    One accepted edge, unique to this route: both Ollama dispatch arms order the
    endpoint chain ``litellm.api_base or api_base or
    get_secret("OLLAMA_API_BASE") or "http://localhost:11434"``
    (``litellm/main.py`` 4072, 4112 and 6465, measured against litellm 1.92.0 on
    2026-07-21) — **inverted** relative to every other route, where an explicit
    ``api_base`` comes first. So for Ollama alone the ``litellm.api_base`` module
    global outranks even the value this provider sends. That is accepted, not
    fixed: it is a global a host must deliberately set inside its own process
    (not ambient environment, and not a stray ``.env``), and llmkit cannot close
    it without reaching into a dependency's globals.

    Reasoning: LiteLLM sends ``think: false`` for ``"disable"`` on every model
    except gpt-oss, whose ``think`` takes an effort string and cannot be
    turned off, so ``"disable"`` raises
    :class:`~llmkit.exceptions.ReasoningEffortUnsupportedError` before transport
    on gpt-oss (matched by name, as LiteLLM does) and ``"lowest"`` sends
    ``"low"`` there. On every other model the server is the authority: llmkit
    sends its explicit off control, and a model that cannot stop thinking may
    ignore it. Measured 2026-11-02 against Ollama 0.40.11: ``deepseek-r1:8b``
    returned reasoning with ``think: false``, through llmkit and in a direct
    ``/api/chat`` request. No list of always-thinking local models is kept.
    ``"lowest"`` therefore sends the off control too: off is the least
    reasoning this route can ask for.
    """

    _provider_name: ClassVar[str] = "Ollama"
    _model_prefix: ClassVar[str] = "ollama_chat/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "llama3.2"
    is_local: ClassVar[bool] = True
    _supports_tool_choice: ClassVar[bool] = False
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset({"base_url"})

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        super().__init__(model, reasoning_effort)
        self._base_url: str = base_url

    @override
    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Refuse ``"disable"`` on gpt-oss; otherwise send Ollama's explicit off control.

        ``"lowest"`` becomes ``"low"`` on gpt-oss and ``"disable"`` (LiteLLM:
        ``think: false``) elsewhere. Other values pass through.
        """
        gpt_oss = model.startswith("gpt-oss")
        if effort == "disable" and gpt_oss:
            raise ReasoningEffortUnsupportedError(
                f"{model!r} cannot turn thinking off (gpt-oss always reasons); "
                + 'reasoning_effort="disable" is not available on it. Send "lowest" '
                + "for the least reasoning it supports."
            )
        if effort == "lowest":
            return {"reasoning_effort": "low" if gpt_oss else "disable"}
        return {"reasoning_effort": effort}

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_base": self._base_url}

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> OllamaProvider:
        """Construct from an :class:`LLMClientConfig` (local endpoint default)."""
        return cls(
            base_url=config.base_url or _DEFAULT_BASE_URL,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
