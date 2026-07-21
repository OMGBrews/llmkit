"""Unit tests for the direct DeepSeek provider.

Parity with the other curated providers: the DeepSeek provider must expose the
right LiteLLM model-prefix string (``deepseek/<model>``), pin DeepSeek's native
JSON mode (``Mode.JSON``) rather than relying on instructor's ``Mode.TOOLS``
fallback, and forward its wire kwargs correctly — ``api_key`` always, and
``api_base`` always too, resolved by the library itself: the configured
``base_url`` first, else ``DEEPSEEK_API_BASE``, else this provider's measured
default (``https://api.deepseek.com/beta`` — the ``/beta`` segment is LiteLLM's
own choice of DeepSeek base, not a typo). ``build_provider`` must map
``Provider.DEEPSEEK`` explicitly and thread model / key / base_url /
reasoning_effort through.

Why ``api_base`` is unconditional is worth stating precisely, because these
assertions are part of the record. Sending it makes the endpoint *llmkit's*
decision — readable straight off ``completion_kwargs()`` — instead of one
LiteLLM derives from sources the library neither reads nor documents (a
key-management backend behind ``get_secret``, whatever alias a future release
adds). What it deliberately does **not** do is make ``DEEPSEEK_API_BASE`` inert:
llmkit reads that alias itself, so a host that steers its traffic with it keeps
working — explicitly and testably, the same bargain ``resolve_api_key`` struck
for the credential. With nothing configured the default reproduces the pre-fix
request URL byte for byte, so the wire shape is unchanged.

Resolution order itself is pinned in ``test_base_url_resolution.py`` and the
resulting URL in ``test_endpoint_routing.py``; this file only pins what *this*
provider puts in its kwargs.

``Mode.JSON`` (not the ``Mode.JSON_SCHEMA`` the OpenAI/Google/Ollama providers
pin) is deliberate: DeepSeek's API rejects strict ``response_format`` json-schema
with a ``BadRequestError``, while ``Mode.JSON`` validates on both
``deepseek-chat`` and ``deepseek-reasoner``. These are mocked-shape assertions; a
real round-trip lives in
``tests/integration/test_live_providers.py::test_deepseek_live``.
"""

from __future__ import annotations

import instructor
import pytest

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


def test_completion_kwargs_uses_the_library_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``base_url`` and no alias → the library's own default rides along as ``api_base``.

    The default is DeepSeek's own endpoint, measured byte-for-byte against what
    LiteLLM resolved before the provider started sending ``api_base`` — ``/beta``
    included, which is why it has to be measured rather than reasoned about.

    ``DEEPSEEK_API_BASE`` is cleared even though the provider is built directly
    here, because resolution deliberately happens in ``completion_kwargs()``
    rather than in the constructor (a provider is routinely built before
    LiteLLM's import-time ``load_dotenv()`` has populated the environment), so
    every path through this provider reads the environment.
    """
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    provider = DeepSeekProvider(api_key="sk-test")
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://api.deepseek.com/beta",
    }


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
