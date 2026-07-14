"""The Anthropic SDK is not a dependency of any llmkit path — not even Claude's.

Through 0.6.x the Anthropic and Bedrock providers pinned ``ANTHROPIC_JSON``
and eagerly required the SDK (the ``omg-llmkit[anthropic]`` extra) for
instructor's usage-accounting path. With the ``Mode.JSON_SCHEMA`` repin that
coupling is gone: LiteLLM speaks the Anthropic HTTP API directly, and
instructor (>=1.15.4) reaches the SDK only on an optional usage branch
guarded by ``try/except ImportError``. These tests pin the new contract:
every Claude-routing provider constructs cleanly with the SDK absent, so the
extra (and its eager construction gate) can never quietly come back without
a deliberate decision.

The SDK's absence is simulated by stubbing ``importlib.util.find_spec``,
mirroring how the removed ``require_anthropic_sdk`` gate used to observe it.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec

import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
)


@pytest.fixture
def anthropic_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any eager SDK probe observe the Anthropic SDK as not installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "anthropic":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.mark.usefixtures("anthropic_absent")
def test_anthropic_provider_constructs_without_sdk() -> None:
    """Direct Claude routing needs no Anthropic SDK — the core install suffices."""
    provider = AnthropicProvider(api_key="x")
    assert provider.name == "Anthropic"
    assert provider.litellm_model() == "anthropic/claude-sonnet-4-6"


@pytest.mark.usefixtures("anthropic_absent")
def test_bedrock_provider_constructs_without_sdk() -> None:
    """Claude-on-Bedrock needs boto3 (present in dev), never the Anthropic SDK."""
    provider = BedrockProvider()
    assert provider.name == "AWS Bedrock"


@pytest.mark.usefixtures("anthropic_absent")
def test_build_provider_anthropic_constructs_without_sdk() -> None:
    """The dispatch entrypoint constructs Claude routing SDK-free too."""
    provider = build_provider(
        LLMClientConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-4-6", api_key="x")
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.litellm_model() == "anthropic/claude-sonnet-4-6"
