"""Unit tests for the direct DeepSeek provider.

Parity with the other curated providers: the DeepSeek provider must expose the
right LiteLLM model-prefix string (``deepseek/<model>``), pin DeepSeek's native
JSON mode (``Mode.JSON``) rather than relying on instructor's ``Mode.TOOLS``
fallback, and forward credentials correctly — ``api_key`` always, ``api_base``
only when a ``base_url`` is given (so the default uses DeepSeek's own
endpoint). ``build_provider`` must map ``Provider.DEEPSEEK`` explicitly and
thread model / key / base_url / reasoning_effort through.

``Mode.JSON`` (not the ``Mode.JSON_SCHEMA`` the OpenAI/Google/Ollama providers
pin) is deliberate: DeepSeek's API rejects strict ``response_format`` json-schema
with a ``BadRequestError``, while ``Mode.JSON`` validates on both
``deepseek-chat`` and ``deepseek-reasoner``. These are mocked-shape assertions; a
real round-trip lives in
``tests/integration/test_live_providers.py::test_deepseek_live``.
"""

from __future__ import annotations

import instructor

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import DeepSeekProvider


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``deepseek/`` prefix and DeepSeek's native JSON mode."""
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-chat")
    assert provider.name == "DeepSeek"
    assert provider.litellm_model() == "deepseek/deepseek-chat"
    assert provider.litellm_model("deepseek-reasoner") == "deepseek/deepseek-reasoner"
    assert provider.instructor_mode is instructor.Mode.JSON


def test_completion_kwargs_without_base_url() -> None:
    """No ``base_url`` → only ``api_key`` is sent, so LiteLLM hits DeepSeek directly."""
    provider = DeepSeekProvider(api_key="sk-test")
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_completion_kwargs_with_base_url() -> None:
    """A ``base_url`` (DeepSeek-compatible gateway) is forwarded as ``api_base``."""
    provider = DeepSeekProvider(api_key="sk-test", base_url="https://gateway.example/v1")
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_default_model() -> None:
    """A sane, structured-output-capable default when the host gives no model."""
    assert DeepSeekProvider(api_key="sk-test").model == "deepseek-chat"


def test_build_provider_maps_deepseek() -> None:
    """``build_provider`` constructs a ``DeepSeekProvider`` and threads every field."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.DEEPSEEK,
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://gateway.example/v1",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, DeepSeekProvider)
    assert provider.litellm_model() == "deepseek/deepseek-reasoner"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


def test_build_provider_deepseek_defaults() -> None:
    """Without a reasoning_effort, the provider stays on the DeepSeek default."""
    provider = build_provider(
        LLMClientConfig(provider=Provider.DEEPSEEK, model="deepseek-chat", api_key="sk-test")
    )
    assert isinstance(provider, DeepSeekProvider)
    assert provider.reasoning_effort is None
