"""Cerebras LLM provider."""

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

#: Cerebras's public API endpoint — the fallback when neither ``base_url`` nor an
#: endpoint environment variable is set. **Measured, not copied from LiteLLM's
#: source**: this exact string reproduces the unconfigured request URL byte for
#: byte (``…/v1/chat/completions``), verified against litellm 1.106.2 on 2026-10-22.
_DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"


def _is_gpt_oss(model: str) -> bool:
    """Whether ``model`` is a gpt-oss model, whose reasoning cannot be disabled."""
    return model.removeprefix("cerebras/").startswith("gpt-oss")


class CerebrasProvider(BaseProvider):
    """Cerebras LLM provider.

    Calls Cerebras's OpenAI-compatible inference API directly
    (``cerebras/<model>``) on a first-party key — the direct alternative to an
    OpenRouter hop. Shared inference serves ``gpt-oss-120b`` (the default) and
    ``qwen-3.8-27b``; other families are dedicated-inference only.

    Structured output is pinned to ``instructor.Mode.JSON_SCHEMA`` (a
    ``response_format`` JSON schema) without the strict upgrade. Measured live on
    2026-10-22 (litellm 1.106.2, instructor 1.17.0): five consecutive structured
    round-trips on ``gpt-oss-120b`` with strict off all validated (as did
    ``qwen-3.8-27b``), so strict on and ``Mode.JSON`` were never tried: the
    measurement plan escalated only on drift. Cerebras's strict mode would also reject
    ``pattern``, ``format``, ``minItems``/``maxItems``, ``oneOf``/``allOf`` and
    recursive ``$ref``, and require ``additionalProperties: false`` on every
    nested object, which the library's top-level strict upgrade does not add.

    Tools plus a response schema in one request are **not** supported:
    ``gpt-oss-120b`` answers a request carrying both with a 400 (``"tools" is
    incompatible with "response_format"``, measured 2026-10-22), so the compose
    trait stays off and callers get the portable two-step steer. A plain tool
    round-trip validates on the default model (measured live, 2026-10-22).

    LiteLLM's Cerebras route drops ``stream_options`` silently (it is not in the
    route's supported parameters and the route does not consult
    ``drop_params``), so a streamed tool turn does not request a usage block.
    Measured live on 2026-10-22: Cerebras reports usage on the stream anyway, on
    both served models, so streamed tool-turn usage is real on this route. On
    ``gpt-oss-120b`` an unforced streamed tool turn delivered no text deltas
    before the call (its prose goes to reasoning); ``qwen-3.8-27b`` streamed
    both.

    Reasoning: both served models reason by default, and reasoning tokens count
    against ``max_tokens`` — a tight cap surfaces as a validation retry, not a
    silent empty answer. The portable ``disable`` maps to Cerebras's ``none``,
    except on ``gpt-oss`` models, which reject ``none`` ("Supported values are
    'low', 'medium', and 'high'", measured 2026-10-22) and so receive their lowest
    supported effort, ``low``. LiteLLM raises ``UnsupportedParamsError`` for an
    effort sent to a model it does not mark as reasoning-capable.

    An optional ``base_url`` points the provider at a Cerebras-compatible
    gateway. ``api_base`` is **always** forwarded: it resolves from ``base_url``
    if set, else ``CEREBRAS_API_BASE``, else :data:`_DEFAULT_BASE_URL`, so a
    source llmkit does not read (the ``litellm.api_base`` module global, a
    LiteLLM key-management backend) cannot steer the caller's key elsewhere (see
    :func:`resolve_base_url`).
    """

    _provider_name: ClassVar[str] = "Cerebras"
    _model_prefix: ClassVar[str] = "cerebras/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gpt-oss-120b"
    _api_key_env_var: ClassVar[str] = "CEREBRAS_API_KEY"
    #: The endpoint variable LiteLLM consulted for this route (measured, not read
    #: off the source). Read here so a host that relies on it keeps working —
    #: explicitly, and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("CEREBRAS_API_BASE",)
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
    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Translate portable effort into Cerebras's ``reasoning_effort`` values.

        ``disable`` becomes ``none``, or ``low`` on ``gpt-oss`` models, which
        cannot disable reasoning (see the class docstring). Other values pass
        through for Cerebras to accept or reject.
        """
        if effort == "disable":
            return {"reasoning_effort": "low" if _is_gpt_oss(model) else "none"}
        return {"reasoning_effort": effort}

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> CerebrasProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``CEREBRAS_API_KEY``
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
