"""Unit tests for the direct OpenAI provider.

Parity with the other curated providers: the OpenAI provider must expose the
right LiteLLM model-prefix string (``openai/<model>``), pin OpenAI's native
structured-outputs mode (``Mode.JSON_SCHEMA``) rather than relying on
instructor's ``Mode.TOOLS`` fallback, and forward its wire kwargs correctly —
``api_key`` always, and ``api_base`` always too, resolved by the library itself:
the configured ``base_url`` first, else ``OPENAI_BASE_URL``, else
``OPENAI_API_BASE``, else this provider's measured default
(``https://api.openai.com/v1``). ``build_provider`` must map ``Provider.OPENAI``
explicitly and thread model / key / base_url / reasoning_effort through.

Why ``api_base`` is unconditional is worth stating precisely, because these
assertions are part of the record. Sending it makes the endpoint *llmkit's*
decision — readable straight off ``completion_kwargs()`` — instead of one
LiteLLM derives from sources the library neither reads nor documents (its
``litellm.api_base`` module global, a key-management backend behind
``get_secret``, whatever alias a future release adds). What it deliberately does
**not** do is make the two environment aliases inert: llmkit reads them itself,
in LiteLLM's own measured precedence, so a host that steers its traffic with
``OPENAI_BASE_URL`` today keeps working — explicitly and testably, the same
bargain ``resolve_api_key`` struck for the credential. With nothing configured
the default reproduces the pre-fix request URL byte for byte, so the wire shape
is unchanged.

Resolution order itself is pinned in ``test_base_url_resolution.py`` and the
resulting URL in ``test_endpoint_routing.py``; this file only pins what *this*
provider puts in its kwargs. These are mocked-shape assertions; a real
round-trip lives in ``tests/integration/test_live_providers.py::test_openai_live``.
"""

from __future__ import annotations

import instructor
import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import OpenAIProvider


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``openai/`` prefix and OpenAI's native structured-outputs mode."""
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4.1-mini")
    assert provider.name == "OpenAI"
    assert provider.litellm_model() == "openai/gpt-4.1-mini"
    assert provider.litellm_model("o4-mini") == "openai/o4-mini"
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA


def test_completion_kwargs_uses_the_library_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``base_url`` and no alias → the library's own default rides along as ``api_base``.

    The default is OpenAI's own endpoint, measured byte-for-byte against what
    LiteLLM resolved before the provider started sending ``api_base`` — so a
    caller who configures nothing still reaches exactly the host they did before.

    The aliases are cleared even though the provider is built directly here.
    That is not belt-and-braces: resolution deliberately happens in
    ``completion_kwargs()`` rather than in the constructor (a provider is
    routinely built before LiteLLM's import-time ``load_dotenv()`` has populated
    the environment), so *every* path through this provider reads the
    environment, direct construction included.
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    provider = OpenAIProvider(api_key="sk-test")
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://api.openai.com/v1",
    }


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


def test_build_provider_maps_openai() -> None:
    """``build_provider`` constructs an ``OpenAIProvider`` and threads every field."""
    provider = build_provider(
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


def test_build_provider_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a base_url / reasoning_effort, the provider stays on OpenAI defaults.

    Unlike direct construction, this path runs ``resolve_base_url``, which *is*
    environment-sensitive by design — it honours ``OPENAI_BASE_URL`` and then
    ``OPENAI_API_BASE`` before falling back. Both are cleared here so the
    assertion is about the fallback rather than about the developer's machine: a
    maintainer who steers their own traffic with either variable (or whose
    ``.env`` supplies one — importing LiteLLM runs ``load_dotenv()``) would
    otherwise see a spurious failure. Their honoured-when-set behavior is pinned
    in ``test_base_url_resolution.py``, which is where that promise belongs.
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    provider = build_provider(
        LLMClientConfig(provider=Provider.OPENAI, model="gpt-4.1-mini", api_key="sk-test")
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.reasoning_effort is None
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://api.openai.com/v1",
    }
