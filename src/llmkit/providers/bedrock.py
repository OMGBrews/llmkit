"""AWS Bedrock LLM provider."""

from __future__ import annotations

import instructor

from llmkit.providers.base import BaseProvider, LLMClientConfig, require_anthropic_sdk


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

    Structured output is pinned to ``instructor.Mode.ANTHROPIC_JSON`` — the
    same Claude-native JSON mode the direct :class:`AnthropicProvider` pins,
    for the same reason: the underlying model is Claude. Note ``Mode.BEDROCK_JSON``
    is deliberately **not** used: it targets instructor's native ``from_bedrock``
    (boto3) client, whose request param is ``modelId``, so when driven through
    ``from_litellm`` (this library's call seam) it drops ``model`` and LiteLLM's
    ``acompletion`` fails with a missing-``model`` error. Over LiteLLM the
    working Claude mode is ``ANTHROPIC_JSON`` (measured against Haiku 4.5;
    ``JSON`` / ``JSON_SCHEMA`` / ``TOOLS`` also validate, ``ANTHROPIC_TOOLS``
    does not). Non-Claude Bedrock families (Llama, Titan) may need a different
    mode and are out of scope for this first cut.

    The default model is a plain **on-demand** Claude-on-Bedrock id. Newer
    Claude 4.x models on Bedrock are typically reachable only through a
    cross-region inference profile — pass the profile-prefixed id as the
    ``model`` (e.g. ``us.anthropic.claude-sonnet-4-...``); inference profiles
    are otherwise out of scope here.

    ``reasoning_effort`` is forwarded to LiteLLM for Bedrock models that
    support thinking (e.g. Claude) and is harmless on those that don't.

    Bedrock pulls in ``boto3`` for request signing; it ships via the
    ``omg-llmkit[bedrock]`` extra rather than the core install, so non-Bedrock
    users take on no extra dependency. Because it routes Claude under
    ``ANTHROPIC_JSON``, it also needs the Anthropic SDK at call time — the
    ``[bedrock]`` extra therefore pulls in the ``[anthropic]`` extra, and
    constructing this provider without the Anthropic SDK raises a clear
    "install omg-llmkit[anthropic]" error.
    """

    _provider_name = "AWS Bedrock"
    _model_prefix = "bedrock/"
    _mode = instructor.Mode.ANTHROPIC_JSON

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region_name: str | None = None,
        reasoning_effort: str | None = None,
    ):
        require_anthropic_sdk(self._provider_name)
        super().__init__(model, reasoning_effort)
        self._aws_region_name = aws_region_name

    def completion_kwargs(self) -> dict[str, str]:
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
