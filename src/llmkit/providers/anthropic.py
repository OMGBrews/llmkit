"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig


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
    gateway (LiteLLM accepts ``api_base`` on the ``anthropic/`` route); left
    unset, LiteLLM uses Anthropic's default endpoint (so ``api_base`` is only
    forwarded when a ``base_url`` is given).
    """

    _provider_name: ClassVar[str] = "Anthropic"
    _model_prefix: ClassVar[str] = "anthropic/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
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

    @classmethod
    def build(cls, config: LLMClientConfig) -> AnthropicProvider:
        """Construct from an :class:`LLMClientConfig` (``base_url`` passed through)."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )
