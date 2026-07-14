"""AWS Bedrock LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    require_boto3_sdk,
)


class BedrockProvider(BaseProvider):
    """AWS Bedrock LLM provider.

    Routes Claude (and other Bedrock-hosted models) through AWS Bedrock
    (``bedrock/<model>``). Unlike every other provider, Bedrock does **not**
    authenticate with a bearer ``api_key``: LiteLLM signs requests with the
    standard **AWS credential chain** (environment variables, shared config,
    or instance/role credentials). This provider therefore carries only an
    optional ``aws_region_name`` — the secrets stay in the ambient chain and
    never pass through :class:`~llmkit.LLMClientConfig`. When the region is
    also left unset, it resolves from the chain (``AWS_REGION_NAME`` /
    ``AWS_REGION``) like the credentials.

    Structured output is pinned to ``instructor.Mode.JSON`` — the same mode
    the direct :class:`AnthropicProvider` pins, for the same reason: the
    underlying model is Claude. ``Mode.ANTHROPIC_JSON`` (the pin through
    0.6.x) was measured to validate against Haiku 4.5 over LiteLLM — as were
    ``JSON`` / ``JSON_SCHEMA`` / ``TOOLS``, while ``ANTHROPIC_TOOLS`` did not
    — but instructor 1.15.3 removed it from the mode registry
    ``from_litellm`` validates against at client construction, so it now
    raises ``RegistryError`` before any request is sent; ``JSON`` is the
    surviving measured mode. Note ``Mode.BEDROCK_JSON`` is deliberately
    **not** used: it targets instructor's native ``from_bedrock`` (boto3)
    client, whose request param is ``modelId``, so when driven through
    ``from_litellm`` (this library's call seam) it drops ``model`` and
    LiteLLM's ``acompletion`` fails with a missing-``model`` error. Non-Claude
    Bedrock families (Llama, Titan) may need a different mode and are out of
    scope for this first cut.

    The default model is Haiku 4.5 — the model the mode measurements above
    were actually run against — via its **cross-region inference profile** id
    (``us.anthropic....``). Newer Claude models on Bedrock are generally
    reachable only through such a profile, not a bare on-demand id, so the
    default carries the ``us.`` prefix; callers in other AWS partitions
    should pass their region group's profile id (e.g. ``eu.anthropic....``)
    as the ``model``. Inference profiles are otherwise out of scope here.

    ``reasoning_effort`` is forwarded to LiteLLM for Bedrock models that
    support thinking (e.g. Claude) and is harmless on those that don't.

    Bedrock pulls in ``boto3`` for request signing (AWS SigV4); it ships via
    the ``omg-llmkit[bedrock]`` extra rather than the core install, so
    non-Bedrock users take on no extra dependency. It is checked *eagerly* at
    construction: constructing this provider without ``boto3`` raises a clear
    "install omg-llmkit[bedrock]" error instead of a cryptic
    ``ModuleNotFound`` deep on the first completion. (The Anthropic SDK is
    **not** needed — see :class:`AnthropicProvider`; instructor >=1.15.4
    never requires it on the LiteLLM path.)
    """

    _provider_name: ClassVar[str] = "AWS Bedrock"
    _model_prefix: ClassVar[str] = "bedrock/"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(
        self,
        model: str | None = None,
        aws_region_name: str | None = None,
        reasoning_effort: str | None = None,
    ):
        require_boto3_sdk(self._provider_name)
        super().__init__(model, reasoning_effort)
        self._aws_region_name: str | None = aws_region_name

    @override
    def completion_kwargs(self) -> dict[str, object]:
        """Return Bedrock credential kwargs forwarded to ``litellm``.

        Only ``aws_region_name`` is sent, and only when set; the access
        key / secret / session token resolve from the ambient AWS credential
        chain. An empty dict leaves region resolution to the chain as well.
        """
        return {"aws_region_name": self._aws_region_name} if self._aws_region_name else {}

    @classmethod
    def build(cls, config: LLMClientConfig) -> BedrockProvider:
        """Construct from an :class:`LLMClientConfig` (region only; no secrets)."""
        return cls(
            model=config.model,
            aws_region_name=config.aws_region_name,
            reasoning_effort=config.reasoning_effort,
        )
