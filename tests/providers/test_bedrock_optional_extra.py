"""boto3 is an opt-in extra for the Bedrock provider, not a core dependency.

LiteLLM's Bedrock path signs requests with ``boto3`` (AWS SigV4), so the
Bedrock provider needs it at call time — but a host that never routes to
Bedrock should take on no ``boto3`` dependency. It therefore ships in the
``omg-llmkit[bedrock]`` extra and is checked *eagerly* at construction, so a
missing ``boto3`` fails with a fix instead of as a cryptic ``ModuleNotFound``
deep on the first completion.

``boto3`` is the provider's only eager construction check — the Anthropic
SDK is not needed at all since the ``Mode.JSON`` repin (see
``test_anthropic_sdk_not_required.py``).

These tests simulate ``boto3``'s absence by stubbing
``importlib.util.find_spec``; the dev/CI environment installs ``boto3`` (via
the ``dev`` extra) so the rest of the suite runs against a real, importable
``boto3``.
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
from llmkit.providers import BedrockProvider


@pytest.fixture
def boto3_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``require_boto3_sdk`` observe ``boto3`` as not installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "boto3":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.mark.usefixtures("boto3_absent")
def test_bedrock_provider_raises_clearly_without_boto3() -> None:
    """Constructing Bedrock without boto3 names the install fix."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[bedrock\]") as excinfo:
        _ = BedrockProvider()
    message = str(excinfo.value)
    # The message identifies the missing dep and why it's needed.
    assert "boto3" in message
    assert "SigV4" in message


@pytest.mark.usefixtures("boto3_absent")
def test_build_provider_bedrock_raises_without_boto3() -> None:
    """The boto3 error surfaces through the dispatch entrypoint, not just direct construction."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[bedrock\]"):
        _ = build_provider(LLMClientConfig(provider=Provider.BEDROCK))


@pytest.mark.skipif(
    importlib.util.find_spec("boto3") is None,
    reason=(
        "boto3 is opt-in (omg-llmkit[bedrock]); skip the happy-path sanity "
        "check when it isn't installed (the dev extra normally pulls it in)."
    ),
)
def test_bedrock_provider_builds_when_boto3_present() -> None:
    """With boto3 installed, construction succeeds (sanity)."""
    provider = BedrockProvider()
    assert provider.name == "AWS Bedrock"
    assert provider.litellm_model() == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
