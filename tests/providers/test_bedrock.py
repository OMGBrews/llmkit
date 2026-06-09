"""Unit tests for the AWS Bedrock provider.

Parity with the other curated providers, plus the one thing Bedrock does
differently: it authenticates through the **AWS credential chain**, not a
bearer ``api_key``. The provider must expose the right LiteLLM model-prefix
string (``bedrock/<model>``), pin Claude's native JSON mode
(``Mode.ANTHROPIC_JSON`` — the same mode the direct ``AnthropicProvider`` uses,
since the model is Claude; ``Mode.BEDROCK_JSON`` is for instructor's
``from_bedrock`` client and drops ``model`` over ``from_litellm``), and carry
only ``aws_region_name`` — and only when set, so an
unset region (and the secrets, always) resolve from the ambient AWS chain.
``build_provider`` must map ``Provider.BEDROCK`` explicitly and thread model /
region / reasoning_effort.

These are mocked-shape assertions; a real round-trip lives in
``tests/integration/test_live_providers.py::test_bedrock_live`` (gated on the
``[bedrock]`` extra + an AWS account).
"""

from __future__ import annotations

import instructor

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import BedrockProvider


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``bedrock/`` prefix and Claude-on-Bedrock's native JSON mode."""
    provider = BedrockProvider(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
    assert provider.name == "AWS Bedrock"
    assert provider.litellm_model() == "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    assert provider.litellm_model("anthropic.claude-3-5-haiku-20241022-v1:0") == (
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    assert provider.instructor_mode is instructor.Mode.ANTHROPIC_JSON


def test_completion_kwargs_with_region() -> None:
    """An explicit region is forwarded as ``aws_region_name``."""
    provider = BedrockProvider(aws_region_name="us-east-1")
    assert provider.completion_kwargs() == {"aws_region_name": "us-east-1"}


def test_completion_kwargs_without_region() -> None:
    """No region → empty kwargs; region + secrets resolve from the AWS chain."""
    provider = BedrockProvider()
    assert provider.completion_kwargs() == {}


def test_default_model() -> None:
    """A sane, structured-output-capable on-demand default when none is given."""
    assert BedrockProvider().model == "anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_build_provider_maps_bedrock() -> None:
    """``build_provider`` builds a ``BedrockProvider`` and threads every field."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.BEDROCK,
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region_name="eu-west-1",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, BedrockProvider)
    assert provider.litellm_model() == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {"aws_region_name": "eu-west-1"}


def test_build_provider_bedrock_defaults() -> None:
    """Without region/effort, kwargs are empty (ambient chain) and effort is None."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.BEDROCK,
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
    )
    assert isinstance(provider, BedrockProvider)
    assert provider.reasoning_effort is None
    assert provider.completion_kwargs() == {}
