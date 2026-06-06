"""Unit tests for the direct DeepSeek provider.

Parity with the other curated providers: the DeepSeek provider must expose the
right LiteLLM model-prefix string (``deepseek/<model>``), pin DeepSeek's native
JSON mode (``Mode.JSON``) rather than relying on instructor's ``Mode.TOOLS``
fallback, and forward credentials correctly (``api_key`` only). ``get_provider``
must map ``Provider.DEEPSEEK`` explicitly and thread model / key /
reasoning_effort through.

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
    DeepSeekProvider,
    LLMClientConfig,
    Provider,
    get_provider,
)


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``deepseek/`` prefix and DeepSeek's native JSON mode."""
    provider = DeepSeekProvider(api_key="sk-test", model="deepseek-chat")
    assert provider.name == "DeepSeek"
    assert provider.litellm_model() == "deepseek/deepseek-chat"
    assert provider.litellm_model("deepseek-reasoner") == "deepseek/deepseek-reasoner"
    assert provider.instructor_mode is instructor.Mode.JSON


def test_completion_kwargs() -> None:
    """Only ``api_key`` is sent — DeepSeek's endpoint is fixed (no base_url)."""
    provider = DeepSeekProvider(api_key="sk-test")
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_default_model() -> None:
    """A sane, structured-output-capable default when the host gives no model."""
    assert DeepSeekProvider(api_key="sk-test").model == "deepseek-chat"


def test_get_provider_maps_deepseek() -> None:
    """``get_provider`` constructs a ``DeepSeekProvider`` and threads every field."""
    provider = get_provider(
        LLMClientConfig(
            provider=Provider.DEEPSEEK,
            model="deepseek-reasoner",
            api_key="sk-test",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, DeepSeekProvider)
    assert provider.litellm_model() == "deepseek/deepseek-reasoner"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_get_provider_deepseek_defaults() -> None:
    """Without a reasoning_effort, the provider stays on the DeepSeek default."""
    provider = get_provider(
        LLMClientConfig(provider=Provider.DEEPSEEK, model="deepseek-chat", api_key="sk-test")
    )
    assert isinstance(provider, DeepSeekProvider)
    assert provider.reasoning_effort is None
