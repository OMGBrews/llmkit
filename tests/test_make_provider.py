"""Unit tests for one-liner provider construction and the model-default fallback.

``make_provider`` builds a provider straight from raw credentials for the
per-call ``provider=`` override — no :class:`LLMClientConfig`, no module-level
config source — so a multi-tenant host can construct a provider from a
per-request key in one line. Separately, a falsy ``model`` (``None`` or ``""``)
must resolve to the provider's *own* default model rather than emitting a
broken ``"<prefix>/"`` LiteLLM id; these tests pin both the ``make_provider``
path and the ``build_provider`` / config path against that footgun.

Mocked-shape assertions only; live round-trips live in
``tests/integration/test_live_providers.py``.
"""

from __future__ import annotations

import pytest

from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
    DeepSeekProvider,
    GoogleProvider,
    LLMClientConfig,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
    build_provider,
    make_provider,
)


def test_make_provider_builds_from_raw_creds() -> None:
    """``make_provider`` returns a usable provider from a bare key + model."""
    provider = make_provider(Provider.OPENAI, api_key="sk-test", model="gpt-4.1")
    assert isinstance(provider, OpenAIProvider)
    assert provider.litellm_model() == "openai/gpt-4.1"
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_make_provider_threads_optional_knobs() -> None:
    """Endpoint / reasoning knobs reach the provider when set."""
    provider = make_provider(
        Provider.OPENAI,
        api_key="sk-test",
        model="gpt-4.1",
        base_url="https://gateway.example/v1",
        reasoning_effort="low",
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_make_provider_openrouter_endpoint_default() -> None:
    """OpenRouter without a ``base_url`` falls back to its public endpoint."""
    provider = make_provider(Provider.OPENROUTER, api_key="sk-test", model="x/y")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://openrouter.ai/api/v1",
    }


def test_make_provider_ollama_local_endpoint_default() -> None:
    """Ollama ignores ``api_key`` and defaults to the local endpoint."""
    provider = make_provider(Provider.OLLAMA, model="llama3.2")
    assert isinstance(provider, OllamaProvider)
    assert provider.completion_kwargs() == {"api_base": "http://localhost:11434"}


def test_make_provider_bedrock_region_only() -> None:
    """Bedrock takes a region, never an ``api_key`` (ambient AWS chain signs)."""
    provider = make_provider(
        Provider.BEDROCK,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region_name="us-east-1",
    )
    assert isinstance(provider, BedrockProvider)
    assert provider.completion_kwargs() == {"aws_region_name": "us-east-1"}


@pytest.mark.parametrize(
    ("provider", "expected_prefix", "expected_model"),
    [
        (Provider.OPENROUTER, "openrouter/", "google/gemini-2.0-flash-001"),
        (Provider.OLLAMA, "ollama_chat/", "llama3.2"),
        (Provider.GOOGLE, "gemini/", "gemini-2.5-flash-lite"),
        (Provider.ANTHROPIC, "anthropic/", "claude-sonnet-4-6"),
        (Provider.OPENAI, "openai/", "gpt-4.1-mini"),
        (Provider.DEEPSEEK, "deepseek/", "deepseek-chat"),
        (Provider.BEDROCK, "bedrock/", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    ],
)
def test_make_provider_none_model_uses_provider_default(
    provider: Provider, expected_prefix: str, expected_model: str
) -> None:
    """A ``None`` model resolves to the provider default — never ``"<prefix>/"``."""
    built = make_provider(provider, api_key="k", model=None)
    litellm_model = built.litellm_model()
    assert litellm_model == f"{expected_prefix}{expected_model}"
    assert not litellm_model.endswith("/"), "dangling prefix means the model id is broken"


@pytest.mark.parametrize("falsy_model", [None, ""])
def test_build_provider_falsy_model_uses_default(falsy_model: str | None) -> None:
    """``build_provider`` with a falsy config model assembles a well-formed id."""
    provider = build_provider(
        LLMClientConfig(provider=Provider.ANTHROPIC, model=falsy_model, api_key="k")
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.litellm_model() == "anthropic/claude-sonnet-4-6"
    assert provider.litellm_model() != "anthropic/"


def test_config_model_defaults_to_none() -> None:
    """``LLMClientConfig.model`` is optional and defaults to ``None``."""
    config = LLMClientConfig(provider=Provider.GOOGLE, api_key="k")
    assert config.model is None
    provider = build_provider(config)
    assert isinstance(provider, GoogleProvider)
    assert provider.litellm_model() == "gemini/gemini-2.5-flash-lite"


def test_explicit_model_still_wins() -> None:
    """An explicit model overrides the provider default on the make_provider path."""
    provider = make_provider(Provider.DEEPSEEK, api_key="k", model="deepseek-reasoner")
    assert isinstance(provider, DeepSeekProvider)
    assert provider.litellm_model() == "deepseek/deepseek-reasoner"
