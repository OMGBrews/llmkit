"""OpenAI LLM provider."""

from __future__ import annotations

from typing import ClassVar, cast, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_api_key,
    resolve_base_url,
)

#: OpenAI's public API endpoint — the fallback when neither ``base_url`` nor an
#: endpoint environment variable is set. **Measured, not copied from LiteLLM's
#: source**: this exact string reproduces the pre-fix request URL byte for byte
#: (``…/v1/chat/completions``), verified against litellm 1.92.0 on 2026-07-21.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
    gateway. ``api_base`` is **always** forwarded: it resolves from ``base_url``
    if set, else ``OPENAI_BASE_URL``, else ``OPENAI_API_BASE``, else
    :data:`_DEFAULT_BASE_URL`. Sending it unconditionally is what keeps a source
    llmkit does **not** read — the ``litellm.api_base`` module global, a LiteLLM
    key-management backend, an alias a future release adds — from steering a
    caller's explicitly-pinned key to an endpoint they never named. The two
    variables named above still steer, deliberately: llmkit reads them itself now,
    so a host already using them is unaffected (see :func:`resolve_base_url`). The
    wire shape with nothing configured is unchanged, because the default is the
    endpoint LiteLLM itself resolved.
    """

    _provider_name: ClassVar[str] = "OpenAI"
    _model_prefix: ClassVar[str] = "openai/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gpt-4.1-mini"
    #: Measured on the native OpenAI route; both parameters share the standard
    #: chat-completions request shape.
    _compose_tools_schema: ClassVar[bool] = True
    _api_key_env_var: ClassVar[str] = "OPENAI_API_KEY"
    #: The endpoint variables LiteLLM consulted for this route, in its own
    #: precedence order (measured, not read off the source). Read here so a host
    #: that relies on one keeps working — explicitly, and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
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
    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Disable thinking only when the effective model's capabilities establish how.

        Missing metadata is not evidence of non-reasoning. Native effort values
        remain an explicit escape hatch and pass through without lookup.
        """
        if effort != "disable":
            return {"reasoning_effort": effort}

        import litellm

        unknown = ReasoningEffortUnsupportedError(
            f"{model!r}: cannot establish how to turn thinking off; model capabilities "
            + "are unknown or inconsistent. Choose a supported native reasoning_effort "
            + 'explicitly, or remove the "disable" requirement.'
        )
        try:
            info = litellm.get_model_info(model, custom_llm_provider="openai")
            reasoning = info.get("supports_reasoning")
            none = info.get("supports_none_reasoning_effort")
            minimal = info.get("supports_minimal_reasoning_effort")
            low = info.get("supports_low_reasoning_effort")
            always_on = info.get("thinking_always_on")
            params = info.get("supported_openai_params")

            # Contradictory affirmative capabilities must not select a success path.
            if (reasoning is False and (none is True or minimal is True or low is True)) or (
                always_on is True and (none is True or reasoning is False)
            ):
                raise unknown
            if none is True:
                if params is not None and "reasoning_effort" not in params:
                    raise unknown
                return {"reasoning_effort": "none"}
            if reasoning is False:
                return {}
            if reasoning is None and none is None and always_on is not True:
                # Older non-reasoning catalogue entries omit the reasoning flag.
                # Require both catalogue identity and a config excluding effort;
                # generic/unknown OpenAI-compatible aliases prove neither.
                catalog = cast(
                    "set[str]", litellm.open_ai_chat_completion_models
                )  # raw-llm: unparameterized catalogue set
                if (
                    model in catalog
                    and info.get("key") in (model, f"openai/{model}")
                    and params is not None
                    and "reasoning_effort" not in params
                    and minimal is not True
                    and low is not True
                ):
                    return {}
        except ReasoningEffortUnsupportedError:
            raise
        except Exception:
            # Do not leak dependency errors or retry transport on missing metadata.
            raise unknown from None

        if none is False or always_on is True:
            alternative = ' Choose a different model or remove the "disable" requirement.'
            if minimal is True or low is True:
                native = "minimal" if minimal is True else "low"
                alternative = f' Send "{native}" explicitly for lower effort; it still reasons.'
            raise ReasoningEffortUnsupportedError(
                f'{model!r} cannot turn thinking off; reasoning_effort="disable" is not '
                + "available on it."
                + alternative
            )
        raise unknown

    @override
    def completion_kwargs(self) -> dict[str, object]:
        # Resolved per call, deliberately. LiteLLM read these variables when it
        # built the request, and its own import runs ``load_dotenv()`` — so a
        # provider constructed before the first call (the documented
        # ``make_provider`` pattern) would otherwise freeze a default and miss the
        # host's ``.env`` endpoint. See :func:`resolve_base_url`.
        return {
            "api_key": self._api_key,
            "api_base": resolve_base_url(
                self._base_url, env_vars=self._base_url_env_vars, default=_DEFAULT_BASE_URL
            ),
        }

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> OpenAIProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``OPENAI_API_KEY``
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
