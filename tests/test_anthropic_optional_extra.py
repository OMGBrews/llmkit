"""The Anthropic SDK is an opt-in extra, not a core dependency.

instructor reaches the Anthropic SDK only at *call time*, on its ANTHROPIC_*
usage-accounting path (``from anthropic.types import Usage`` inside
``instructor/core/retry.py``, guarded by the mode). Plain ``import llmkit`` and
a non-Anthropic flow never touch it, so it ships in the ``omg-llmkit[anthropic]``
extra. The Anthropic and Bedrock providers (both pin ``ANTHROPIC_JSON``) must
therefore fail *eagerly and clearly* at construction when the SDK is absent —
instead of with a cryptic ``ModuleNotFound`` deep on the first completion — and a
non-Anthropic provider must construct fine without it.

These tests simulate the SDK's absence by stubbing ``importlib.util.find_spec``;
the dev/CI environment installs the SDK (via the ``dev`` extra) so the rest of
the suite runs against a real, importable ``anthropic``.
"""

from __future__ import annotations

import importlib.util

import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
    GoogleProvider,
    base,
)


@pytest.fixture
def anthropic_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``require_anthropic_sdk`` observe the Anthropic SDK as not installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            return None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(base.importlib.util, "find_spec", fake_find_spec)


def test_anthropic_provider_raises_clearly_without_sdk(anthropic_absent: None) -> None:
    """Constructing the Anthropic provider without the SDK names the install fix."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[anthropic\]"):
        AnthropicProvider(api_key="x")


def test_bedrock_provider_raises_clearly_without_sdk(anthropic_absent: None) -> None:
    """Bedrock pins ANTHROPIC_JSON, so it needs the SDK too and errors the same way."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[anthropic\]"):
        BedrockProvider()


def test_build_provider_anthropic_raises_without_sdk(anthropic_absent: None) -> None:
    """The error surfaces through the dispatch entrypoint, not just direct construction."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[anthropic\]"):
        build_provider(
            LLMClientConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-4-6", api_key="x")
        )


def test_non_anthropic_provider_builds_without_sdk(anthropic_absent: None) -> None:
    """A Google-only flow constructs cleanly with the Anthropic SDK absent."""
    provider = build_provider(
        LLMClientConfig(provider=Provider.GOOGLE, model="gemini-2.5-flash", api_key="x")
    )
    assert isinstance(provider, GoogleProvider)
    assert provider.litellm_model() == "gemini/gemini-2.5-flash"


def test_anthropic_provider_builds_when_sdk_present() -> None:
    """With the SDK installed (dev/CI), construction succeeds as before."""
    provider = AnthropicProvider(api_key="x")
    assert provider.name == "Anthropic"
    assert provider.litellm_model() == "anthropic/claude-sonnet-4-6"
