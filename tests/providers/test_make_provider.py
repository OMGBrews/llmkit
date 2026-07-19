"""Unit tests for one-liner provider construction and the model-default fallback.

``make_provider`` builds a provider straight from raw credentials for the
per-call ``provider=`` override — no :class:`LLMClientConfig`, no module-level
config source — so a single call can run through a different provider, model,
or key in one line. Separately, a falsy ``model`` (``None`` or ``""``)
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
    VertexProvider,
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
    """OpenRouter without a ``base_url`` falls back to its public endpoint.

    ``require_parameters`` routing is on by default, so the kwargs also carry the
    ``extra_body`` schema-honoring routing preference (see
    ``tests/providers/test_openrouter_routing.py``).
    """
    provider = make_provider(Provider.OPENROUTER, api_key="sk-test", model="x/y")
    assert isinstance(provider, OpenRouterProvider)
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://openrouter.ai/api/v1",
        "extra_body": {"provider": {"require_parameters": True}},
    }


def test_make_provider_ollama_local_endpoint_default() -> None:
    """Ollama needs no key and defaults to the local endpoint."""
    provider = make_provider(Provider.OLLAMA, model="llama3.2")
    assert isinstance(provider, OllamaProvider)
    assert provider.completion_kwargs() == {"api_base": "http://localhost:11434"}


def test_make_provider_openai_no_base_url_stays_none() -> None:
    """OpenAI without a ``base_url`` injects no ``api_base`` literal.

    The endpoint-default normalization moved from ``make_provider``'s old per-arm
    switch into ``build()``. OpenRouter/Ollama substitute a concrete endpoint
    literal when ``base_url`` is omitted, but OpenAI must *not*: it leaves the
    endpoint unset so LiteLLM uses OpenAI's real default. This pins that
    distinction — ``completion_kwargs()`` carries only ``api_key`` (no
    ``api_base`` key at all), and the underlying ``base_url`` stays ``None``.
    """
    provider = make_provider(Provider.OPENAI, api_key="sk-test", model="gpt-4.1")
    assert isinstance(provider, OpenAIProvider)
    assert provider._base_url is None
    kwargs = provider.completion_kwargs()
    assert "api_base" not in kwargs
    assert kwargs == {"api_key": "sk-test"}


@pytest.mark.parametrize(
    "provider_enum",
    [Provider.ANTHROPIC, Provider.GOOGLE, Provider.DEEPSEEK],
)
def test_make_provider_forwards_base_url_as_api_base(provider_enum: Provider) -> None:
    """A ``base_url`` reaches LiteLLM as ``api_base`` — never silently dropped.

    These three providers historically ignored ``base_url`` under a false
    "endpoints are fixed" rationale, so a gateway-routed config silently sent
    every request to the public endpoint — a data-egress surprise for exactly
    the users who set an endpoint override. LiteLLM accepts ``api_base`` on
    the ``anthropic/`` / ``gemini/`` / ``deepseek/`` routes, so the override
    must land on the wire kwargs.
    """
    provider = make_provider(
        provider_enum, api_key="sk-test", base_url="https://gateway.example/v1"
    )
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://gateway.example/v1",
    }


@pytest.mark.parametrize(
    "provider_enum",
    [Provider.ANTHROPIC, Provider.GOOGLE, Provider.DEEPSEEK],
)
def test_make_provider_no_base_url_omits_api_base(provider_enum: Provider) -> None:
    """Without a ``base_url``, no ``api_base`` key is sent at all.

    Like OpenAI (and unlike OpenRouter/Ollama, which substitute an endpoint
    literal), these providers must leave the endpoint unset so LiteLLM uses
    each provider's real default — the pre-existing wire shape, byte-identical.
    """
    provider = make_provider(provider_enum, api_key="sk-test")
    assert provider.completion_kwargs() == {"api_key": "sk-test"}


def test_make_provider_bedrock_region_only() -> None:
    """Bedrock takes a region, never an ``api_key`` (ambient AWS chain signs)."""
    provider = make_provider(
        Provider.BEDROCK,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region_name="us-east-1",
    )
    assert isinstance(provider, BedrockProvider)
    assert provider.completion_kwargs() == {"aws_region_name": "us-east-1"}


def test_make_provider_vertex_project_and_location_only() -> None:
    """Vertex takes project + location, never an ``api_key`` (ambient ADC signs)."""
    provider = make_provider(
        Provider.VERTEX,
        model="gemini-2.5-flash",
        vertex_project="proj-123",
        vertex_location="europe-west4",
    )
    assert isinstance(provider, VertexProvider)
    assert provider.completion_kwargs() == {
        "vertex_project": "proj-123",
        "vertex_location": "europe-west4",
    }


def test_make_provider_bedrock_eager_sdk_checks_pass_in_dev_env() -> None:
    """Bedrock's eager boto3 check succeeds in the dev env.

    ``BedrockProvider.__init__`` calls ``require_boto3_sdk`` *eagerly* at
    construction. Reaching a constructed provider via ``make_provider`` at all
    proves boto3 is importable here; this asserts the region still threads
    through after that check passes.
    """
    provider = make_provider(Provider.BEDROCK, aws_region_name="us-east-1")
    assert isinstance(provider, BedrockProvider)
    assert provider.completion_kwargs() == {"aws_region_name": "us-east-1"}


@pytest.mark.parametrize(
    ("provider", "expected_prefix", "expected_model", "key"),
    [
        (Provider.OPENROUTER, "openrouter/", "google/gemini-2.5-flash-lite", "k"),
        (Provider.OLLAMA, "ollama_chat/", "llama3.2", None),
        (Provider.GOOGLE, "gemini/", "gemini-2.5-flash-lite", "k"),
        (Provider.ANTHROPIC, "anthropic/", "claude-sonnet-4-6", "k"),
        (Provider.OPENAI, "openai/", "gpt-4.1-mini", "k"),
        (Provider.DEEPSEEK, "deepseek/", "deepseek-chat", "k"),
        (Provider.BEDROCK, "bedrock/", "us.anthropic.claude-haiku-4-5-20251001-v1:0", None),
        (Provider.VERTEX, "vertex_ai/", "gemini-2.5-flash-lite", None),
    ],
)
def test_make_provider_none_model_uses_provider_default(
    provider: Provider, expected_prefix: str, expected_model: str, key: str | None
) -> None:
    """A ``None`` model resolves to the provider default — never ``"<prefix>/"``.

    Each provider gets only the knobs it reads (a bearer key where required,
    ``None`` for the ambient/local providers), since a knob a provider ignores now
    raises.
    """
    built = make_provider(provider, model=None, api_key=key)
    litellm_model = built.litellm_model()
    assert litellm_model == f"{expected_prefix}{expected_model}"
    assert not litellm_model.endswith("/"), "dangling prefix means the model id is broken"


@pytest.mark.parametrize(
    ("provider", "kwargs", "offender"),
    [
        (Provider.OLLAMA, {"api_key": "k"}, "api_key"),
        (Provider.BEDROCK, {"api_key": "k"}, "api_key"),
        (Provider.VERTEX, {"api_key": "k"}, "api_key"),
        (Provider.VERTEX, {"base_url": "https://x"}, "base_url"),
        (Provider.BEDROCK, {"base_url": "https://x"}, "base_url"),
        (Provider.OPENAI, {"aws_region_name": "us-east-1"}, "aws_region_name"),
        (Provider.OPENAI, {"gemini_structured_output": "json"}, "gemini_structured_output"),
        (Provider.ANTHROPIC, {"vertex_project": "p"}, "vertex_project"),
    ],
)
def test_make_provider_unread_knob_raises_naming_the_field(
    provider: Provider, kwargs: dict[str, str], offender: str
) -> None:
    """Passing a knob the selected provider does not read raises, naming the field —
    it is no longer silently dropped (the audit's ``make_provider`` footgun)."""
    with pytest.raises(ValueError, match=rf"does not read the config field.*{offender}"):
        _ = make_provider(provider, **kwargs)  # pyright: ignore[reportArgumentType]  # test feeds a deliberately-wrong knob


def test_build_provider_direct_config_with_unread_knob_raises() -> None:
    """The rejection lives at ``build_provider``, so a directly-constructed config
    (bypassing ``make_provider``) is validated too — the seam has no hole."""
    with pytest.raises(ValueError, match=r"AWS Bedrock does not read.*api_key"):
        _ = build_provider(LLMClientConfig(provider=Provider.BEDROCK, api_key="k"))


def test_make_provider_unread_knob_error_names_offenders_and_accepted_set() -> None:
    """The message names the offending field(s) and what the provider does read,
    so the fix is obvious from the error alone."""
    with pytest.raises(ValueError) as excinfo:
        _ = make_provider(Provider.OLLAMA, api_key="k")
    message = str(excinfo.value)
    assert "api_key" in message
    assert "Ollama reads" in message
    assert "base_url" in message


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
