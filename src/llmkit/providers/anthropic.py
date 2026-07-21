"""Anthropic (Claude) LLM provider."""

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

#: Anthropic's public API endpoint — the fallback when neither ``base_url`` nor
#: an endpoint environment variable is set. **Measured, not copied from
#: LiteLLM's source**: this exact string reproduces the pre-fix request URL byte
#: for byte (``…/v1/messages``), verified against litellm 1.92.0 on 2026-07-21.
#: Note the deliberate absence of a trailing ``/v1`` — LiteLLM appends its own,
#: so pinning ``https://api.anthropic.com/v1`` yields ``/v1/v1/messages``.
_DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider.

    Routes Claude through the Anthropic API (``anthropic/<model>``), pinned
    to ``Mode.JSON_SCHEMA`` for instructor: the request carries the strict
    ``response_format={"type": "json_schema", ...}``, which LiteLLM
    translates for the Anthropic API, and the response text validates
    directly against the schema.

    ``Mode.ANTHROPIC_JSON`` (the pin through 0.6.x) is gone: instructor
    1.15.3 removed it from the mode registry ``from_litellm`` validates
    against at client construction, so it now raises ``RegistryError``
    before any request is sent. Of the surviving core modes, measured live
    against Haiku 4.5 (2026-07-14): ``JSON_SCHEMA`` and ``MD_JSON`` validate
    every field; ``JSON`` fails — Claude wraps the (correct) JSON in a
    markdown fence and instructor's parse path does not strip fences; and
    ``TOOLS`` cannot run at all on a lean install — LiteLLM's tools branch
    imports its proxy machinery, which needs ``fastapi``. ``JSON_SCHEMA``
    wins as the strictest surviving mode, and matches most other providers'
    pins.

    No Anthropic SDK is required: LiteLLM speaks the Anthropic HTTP API
    directly, and instructor (>=1.15.4) reaches the SDK only on an optional
    usage-accounting branch guarded by ``try/except ImportError``. This
    provider therefore has no extra — a core install routes Claude.

    An optional ``base_url`` points the provider at an Anthropic-compatible
    gateway (LiteLLM accepts ``api_base`` on the ``anthropic/`` route).
    ``api_base`` is **always** forwarded: it resolves from ``base_url`` if set,
    else ``ANTHROPIC_API_BASE``, else ``ANTHROPIC_BASE_URL``, else
    :data:`_DEFAULT_BASE_URL`. Sending it unconditionally is what keeps a source
    llmkit does **not** read — the ``litellm.api_base`` module global, a LiteLLM
    key-management backend, an alias a future release adds — from steering a
    caller's explicitly-pinned key to an endpoint they never named. The two
    variables named above still steer, deliberately: llmkit reads them itself now,
    so a host already using them is unaffected (see :func:`resolve_base_url`). The
    wire shape with nothing configured is unchanged, because the default is the
    endpoint LiteLLM itself resolved.
    """

    _provider_name: ClassVar[str] = "Anthropic"
    _model_prefix: ClassVar[str] = "anthropic/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "claude-sonnet-4-6"
    _api_key_env_var: ClassVar[str] = "ANTHROPIC_API_KEY"
    #: The endpoint variables LiteLLM consulted for this route, in its own
    #: precedence order (measured, not read off the source). Read here so a host
    #: that relies on one keeps working — explicitly, and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL")
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
    def build(cls, config: LLMClientConfig) -> AnthropicProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``ANTHROPIC_API_KEY``
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
