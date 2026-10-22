"""Unit tests for the direct Cerebras provider.

The Cerebras provider must expose the ``cerebras/<model>`` LiteLLM prefix, pin
``Mode.JSON_SCHEMA`` (measured live, see the provider docstring), and forward
``api_key`` and ``api_base`` always — the configured ``base_url`` first, else
``CEREBRAS_API_BASE``, else the measured default (``https://api.cerebras.ai/v1``).
``build_provider`` must map ``Provider.CEREBRAS`` explicitly and thread model /
key / base_url / reasoning_effort through, and the provider owns the translation
of the portable ``disable`` effort.

Resolution order is pinned in ``test_base_url_resolution.py`` and the resulting
URL in ``test_endpoint_routing.py``. These are mocked-shape assertions; real
round-trips live in ``tests/integration/test_live_providers.py``.
"""

from __future__ import annotations

import instructor
import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import CerebrasProvider


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``cerebras/`` prefix and the measured ``Mode.JSON_SCHEMA`` pin."""
    provider = CerebrasProvider(api_key="csk-test", model="gpt-oss-120b")
    assert provider.name == "Cerebras"
    assert provider.litellm_model() == "cerebras/gpt-oss-120b"
    assert provider.litellm_model("qwen-3.8-27b") == "cerebras/qwen-3.8-27b"
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA


def test_measured_traits() -> None:
    """Strict stays off (five non-strict runs validated); tools+schema is rejected."""
    provider = CerebrasProvider(api_key="csk-test")
    assert provider.strict_json_schema is False
    assert provider.compose_tools_schema is False
    assert provider.single_tool_call_per_schema is False


def test_default_model() -> None:
    """The default is the faster of the two shared-inference models."""
    assert CerebrasProvider(api_key="csk-test").model == "gpt-oss-120b"


def test_completion_kwargs_uses_the_library_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``base_url`` and no endpoint variable → the measured default as ``api_base``."""
    monkeypatch.delenv("CEREBRAS_API_BASE", raising=False)
    provider = CerebrasProvider(api_key="csk-test")
    assert provider.completion_kwargs() == {
        "api_key": "csk-test",
        "api_base": "https://api.cerebras.ai/v1",
    }


def test_completion_kwargs_with_base_url() -> None:
    """A ``base_url`` (Cerebras-compatible gateway) is forwarded as ``api_base``."""
    provider = CerebrasProvider(api_key="csk-test", base_url="https://gateway.example/v1")
    assert provider.completion_kwargs() == {
        "api_key": "csk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_build_provider_maps_cerebras() -> None:
    """``build_provider`` constructs a ``CerebrasProvider`` and threads every field."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.CEREBRAS,
            model="qwen-3.8-27b",
            api_key="csk-test",
            base_url="https://gateway.example/v1",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, CerebrasProvider)
    assert provider.litellm_model() == "cerebras/qwen-3.8-27b"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "api_key": "csk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_build_provider_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No configured key → ``CEREBRAS_API_KEY``; the endpoint falls back to the default."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-env")
    monkeypatch.delenv("CEREBRAS_API_BASE", raising=False)
    provider = build_provider(LLMClientConfig(provider=Provider.CEREBRAS))
    assert isinstance(provider, CerebrasProvider)
    assert provider.model == "gpt-oss-120b"
    assert provider.completion_kwargs() == {
        "api_key": "csk-env",
        "api_base": "https://api.cerebras.ai/v1",
    }


def test_build_provider_without_a_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key is loud, never an empty string handed to LiteLLM."""
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        _ = build_provider(LLMClientConfig(provider=Provider.CEREBRAS))


@pytest.mark.parametrize(
    "config",
    [
        LLMClientConfig(provider=Provider.CEREBRAS, api_key="k", aws_region_name="us-east-1"),
        LLMClientConfig(provider=Provider.CEREBRAS, api_key="k", vertex_project="p"),
        LLMClientConfig(provider=Provider.CEREBRAS, api_key="k", vertex_location="us-central1"),
        LLMClientConfig(provider=Provider.CEREBRAS, api_key="k", gemini_structured_output="json"),
    ],
    ids=["aws_region_name", "vertex_project", "vertex_location", "gemini_structured_output"],
)
def test_build_provider_rejects_unread_knobs(config: LLMClientConfig) -> None:
    """A provider-shaped knob Cerebras does not read is rejected, not ignored."""
    with pytest.raises(ValueError, match="Cerebras does not read"):
        _ = build_provider(config)


@pytest.mark.parametrize(
    ("effort", "model", "expected"),
    [
        ("disable", "qwen-3.8-27b", "none"),
        ("disable", "gpt-oss-120b", "low"),
        ("low", "gpt-oss-120b", "low"),
        ("high", "qwen-3.8-27b", "high"),
        ("medium", "gpt-oss-120b", "medium"),
    ],
)
def test_reasoning_kwargs_translate_disable(effort: str, model: str, expected: str) -> None:
    """``disable`` → ``none``, or ``low`` on gpt-oss, which rejects ``none`` (measured)."""
    provider = CerebrasProvider(api_key="csk-test")
    assert provider.reasoning_kwargs(effort, model) == {"reasoning_effort": expected}
