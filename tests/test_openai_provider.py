"""Unit tests for the direct OpenAI provider.

Parity with the other curated providers: the OpenAI provider must expose the
right LiteLLM model-prefix string (``openai/<model>``), pin OpenAI's native
strict structured-outputs mode (``Mode.TOOLS_STRICT``) rather than relying on
instructor's ``Mode.TOOLS`` fallback, and forward credentials correctly —
``api_key`` always, ``api_base`` only when a ``base_url`` is given (so the
default uses OpenAI's own endpoint). ``get_provider`` must map
``Provider.OPENAI`` explicitly and thread model / key / base_url /
reasoning_effort through.

These are mocked-shape assertions; a real round-trip lives in
``tests/integration/test_live_providers.py::test_openai_live``.
"""

from __future__ import annotations

import instructor

from llmkit import (
    LLMClientConfig,
    OpenAIProvider,
    Provider,
    get_provider,
)


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``openai/`` prefix and OpenAI's strict structured-outputs mode."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4.1-mini")
    assert provider.name == "OpenAI"
    assert provider.litellm_model() == "openai/gpt-4.1-mini"
    assert provider.litellm_model("o4-mini") == "openai/o4-mini"
    assert provider.instructor_mode is instructor.Mode.TOOLS_STRICT


def test_completion_kwargs_without_base_url() -> None:
    """No ``base_url`` → only ``api_key`` is sent, so LiteLLM hits OpenAI directly."""
    provider = OpenAIProvider(api_key="sk-test")
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_completion_kwargs_with_base_url() -> None:
    """A ``base_url`` (OpenAI-compatible gateway) is forwarded as ``api_base``."""
    provider = OpenAIProvider(api_key="sk-test", base_url="https://gateway.example/v1")
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_default_model() -> None:
    """A sane, structured-output-capable default when the host gives no model."""
    assert OpenAIProvider(api_key="sk-test").model == "gpt-4.1-mini"


def test_get_provider_maps_openai() -> None:
    """``get_provider`` constructs an ``OpenAIProvider`` and threads every field."""
    provider = get_provider(
        LLMClientConfig(
            provider=Provider.OPENAI,
            model="gpt-4.1",
            api_key="sk-test",
            base_url="https://gateway.example/v1",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.litellm_model() == "openai/gpt-4.1"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_get_provider_openai_defaults() -> None:
    """Without a base_url / reasoning_effort, the provider stays on OpenAI defaults."""
    provider = get_provider(
        LLMClientConfig(provider=Provider.OPENAI, model="gpt-4.1-mini", api_key="sk-test")
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.reasoning_effort is None
    assert provider.completion_kwargs() == {"api_key": "sk-test"}
