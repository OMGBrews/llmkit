"""Unit tests for the direct Moonshot AI (Kimi) provider.

The Moonshot provider must expose the ``moonshot/<model>`` LiteLLM prefix, pin
``Mode.JSON``, not ``Mode.JSON_SCHEMA``, which Moonshot did not reliably enforce
(measured live, see the provider docstring), and forward ``api_key`` and ``api_base`` always: the configured
``base_url`` first, else ``MOONSHOT_API_BASE``, else the default
(``https://api.moonshot.ai/v1``), resolved per call rather than at construction.
``build_provider`` must map ``Provider.MOONSHOT`` explicitly and thread model /
key / base_url / reasoning_effort through, and the provider owns the translation
of the portable ``disable`` and ``lowest`` efforts to ``"none"``: Moonshot
accepted a forwarded ``"disable"`` and kept thinking.

Resolution order across providers is pinned in ``test_base_url_resolution.py``
and the resulting URL in ``test_endpoint_routing.py``. These are mocked-shape
assertions; real round-trips live in ``tests/integration/test_live_providers.py``.
"""

from __future__ import annotations

import instructor
import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
    capture_llm_records,
    make_provider,
    text_llm_call,
)
from llmkit.exceptions import ReasoningEffortUnsupportedError
from llmkit.providers import MoonshotProvider
from llmkit.rate_limiting import RateLimitPolicy, configure_rate_limit, get_rate_limit_policy
from llmkit.testing import Reply, install

_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
_ENV_KEY = "MOONSHOT_API_KEY"
_ENV_BASE = "MOONSHOT_API_BASE"


@pytest.fixture(autouse=True)
def _no_ambient_moonshot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variables under test, so no case passes off an ambient value."""
    monkeypatch.delenv(_ENV_KEY, raising=False)
    monkeypatch.delenv(_ENV_BASE, raising=False)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (None, "moonshot/kimi-k3.1-pro"),
        ("kimi-k2.6", "moonshot/kimi-k2.6"),
        ("kimi-k3.1-pro", "moonshot/kimi-k3.1-pro"),
    ],
)
def test_model_prefix(model: str | None, expected: str) -> None:
    provider = MoonshotProvider(api_key="sk-test", model=model)
    assert provider.litellm_model() == expected


def test_litellm_model_override_keeps_the_prefix() -> None:
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.litellm_model("kimi-k2.6") == "moonshot/kimi-k2.6"


def test_measured_traits() -> None:
    """Schema mode without strict; named tool_choice supported; compose not measured, so off."""
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.instructor_mode is instructor.Mode.JSON
    assert provider.strict_json_schema is False
    assert provider.compose_tools_schema is False
    assert provider.supports_tool_choice is True
    assert provider.single_tool_call_per_schema is False


def test_build_from_a_key_alone_uses_the_defaults() -> None:
    provider = make_provider(Provider.MOONSHOT, api_key="sk-only")
    assert isinstance(provider, MoonshotProvider)
    assert provider.model == "kimi-k3.1-pro"
    assert provider.name == "Moonshot"
    assert provider.completion_kwargs() == {"api_key": "sk-only", "api_base": _DEFAULT_BASE_URL}


@pytest.mark.parametrize("key", ["sk-env-one", "sk-env-two"])
def test_build_reads_the_key_from_the_documented_variable(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_KEY, key)
    provider = build_provider(LLMClientConfig(provider=Provider.MOONSHOT))
    assert isinstance(provider, MoonshotProvider)
    assert provider.completion_kwargs() == {"api_key": key, "api_base": _DEFAULT_BASE_URL}


def test_a_configured_key_wins_over_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_KEY, "sk-env")
    provider = build_provider(LLMClientConfig(provider=Provider.MOONSHOT, api_key="sk-config"))
    assert provider.completion_kwargs()["api_key"] == "sk-config"


def test_a_missing_key_raises_naming_the_variable() -> None:
    with pytest.raises(ValueError, match=r"Moonshot requires an API key.*MOONSHOT_API_KEY"):
        _ = build_provider(LLMClientConfig(provider=Provider.MOONSHOT))


def test_build_threads_every_field() -> None:
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.MOONSHOT,
            model="kimi-k2.6",
            api_key="sk-test",
            base_url="https://api.moonshot.cn/v1",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, MoonshotProvider)
    assert provider.litellm_model() == "moonshot/kimi-k2.6"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "api_key": "sk-test",
        "api_base": "https://api.moonshot.cn/v1",
    }


@pytest.mark.parametrize(
    ("base_url", "env_base", "expected"),
    [
        (None, None, _DEFAULT_BASE_URL),
        (None, "https://env.example/v1", "https://env.example/v1"),
        ("https://api.moonshot.cn/v1", None, "https://api.moonshot.cn/v1"),
        ("https://api.moonshot.cn/v1", "https://env.example/v1", "https://api.moonshot.cn/v1"),
        (None, "", _DEFAULT_BASE_URL),
    ],
    ids=["default", "env", "explicit", "explicit-beats-env", "empty-env-is-unset"],
)
def test_endpoint_precedence(
    base_url: str | None,
    env_base: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``base_url``, then ``MOONSHOT_API_BASE``, then the default."""
    if env_base is not None:
        monkeypatch.setenv(_ENV_BASE, env_base)
    provider = MoonshotProvider(api_key="sk-test", base_url=base_url)
    assert provider.completion_kwargs()["api_base"] == expected


def test_endpoint_is_resolved_per_call_not_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A variable set or changed after construction reaches the next call's kwargs."""
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.completion_kwargs()["api_base"] == _DEFAULT_BASE_URL
    monkeypatch.setenv(_ENV_BASE, "https://late.example/v1")
    assert provider.completion_kwargs()["api_base"] == "https://late.example/v1"
    monkeypatch.setenv(_ENV_BASE, "https://later.example/v1")
    assert provider.completion_kwargs()["api_base"] == "https://later.example/v1"
    monkeypatch.delenv(_ENV_BASE)
    assert provider.completion_kwargs()["api_base"] == _DEFAULT_BASE_URL


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "Moonshot"),
        (_DEFAULT_BASE_URL, "Moonshot"),
        ("https://API.MOONSHOT.AI/", "Moonshot"),
        ("https://api.moonshot.cn/v1", "Moonshot (api.moonshot.cn)"),
        ("https://gateway.example:8443/v1", "Moonshot (gateway.example:8443)"),
    ],
)
def test_name(base_url: str | None, expected: str) -> None:
    assert MoonshotProvider(api_key="sk-test", base_url=base_url).name == expected


def test_an_endpoint_from_the_environment_keeps_the_plain_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_BASE, "https://api.moonshot.cn/v1")
    assert MoonshotProvider(api_key="sk-test").name == "Moonshot"


@pytest.mark.parametrize("effort", ["disable", "lowest"])
@pytest.mark.parametrize("model", ["kimi-k3.1-pro", "kimi-k2.6", "moonshot/kimi-k3.1-pro"])
def test_disable_and_lowest_turn_thinking_off(effort: str, model: str) -> None:
    """Both portable values are sent as ``"none"``; a forwarded ``"disable"`` kept thinking."""
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.reasoning_kwargs(effort, model) == {"reasoning_effort": "none"}


@pytest.mark.parametrize("effort", ["disable", "lowest"])
@pytest.mark.parametrize(
    "model", ["moonshot-v1-8k", "moonshot-v1-auto", "moonshot/moonshot-v1-32k"]
)
def test_disable_and_lowest_send_nothing_to_a_listed_non_reasoning_model(
    effort: str, model: str
) -> None:
    """LiteLLM refuses ``reasoning_effort`` there before sending; the model does not think."""
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.reasoning_kwargs(effort, model) == {}


@pytest.mark.parametrize("effort", ["disable", "lowest"])
@pytest.mark.parametrize("model", ["kimi-unlisted-9", "moonshot/kimi-k99-preview"])
def test_disable_and_lowest_raise_on_an_unlisted_model(effort: str, model: str) -> None:
    """Nothing establishes how to turn thinking off, and LiteLLM would refuse the parameter."""
    provider = MoonshotProvider(api_key="sk-test")
    with pytest.raises(ReasoningEffortUnsupportedError, match=effort):
        _ = provider.reasoning_kwargs(effort, model)


@pytest.mark.parametrize("effort", ["disable", "lowest"])
@pytest.mark.parametrize(("model", "sent"), [("kimi-k3.1-pro", True), ("moonshot-v1-8k", False)])
def test_the_translation_is_accepted_by_litellms_moonshot_route(
    effort: str, model: str, sent: bool
) -> None:
    """LiteLLM's own parameter mapping accepts what the provider returns, without dropping it."""
    from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig

    # The route's own mapping, as the DeepSeek test reads DeepSeek's: a value it
    # does not list for the model is skipped here, and refused by LiteLLM's
    # public entry point, so "absent" below means nothing would be sent.
    kwargs = MoonshotProvider(api_key="sk-test").reasoning_kwargs(effort, model)
    params: dict[str, object] = MoonshotChatConfig().map_openai_params(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # raw-litellm: unparameterized upstream dict
        non_default_params=kwargs, optional_params={}, model=model, drop_params=False
    )
    assert params.get("reasoning_effort") == ("none" if sent else None)
    assert kwargs == ({"reasoning_effort": "none"} if sent else {})


@pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal", "max"])
@pytest.mark.parametrize("model", ["kimi-k3.1-pro", "kimi-k2.6"])
def test_other_efforts_pass_through_unchanged(effort: str, model: str) -> None:
    provider = MoonshotProvider(api_key="sk-test")
    assert provider.reasoning_kwargs(effort, model) == {"reasoning_effort": effort}


@pytest.mark.parametrize(
    "config",
    [
        LLMClientConfig(provider=Provider.MOONSHOT, api_key="k", aws_region_name="us-east-1"),
        LLMClientConfig(provider=Provider.MOONSHOT, api_key="k", vertex_project="p"),
        LLMClientConfig(provider=Provider.MOONSHOT, api_key="k", vertex_location="us-central1"),
        LLMClientConfig(provider=Provider.MOONSHOT, api_key="k", gemini_structured_output="json"),
    ],
    ids=["aws_region_name", "vertex_project", "vertex_location", "gemini_structured_output"],
)
def test_build_rejects_unread_knobs(config: LLMClientConfig) -> None:
    with pytest.raises(ValueError, match="Moonshot does not read"):
        _ = build_provider(config)


def test_moonshot_and_openai_on_moonshots_url_are_separate_routes() -> None:
    """The first-party class and the OpenAI class on Moonshot's URL are limited apart."""
    moonshot = make_provider(Provider.MOONSHOT, api_key="k")
    via_openai = make_provider(Provider.OPENAI, api_key="k", base_url=_DEFAULT_BASE_URL)
    assert moonshot.name == "Moonshot"
    assert via_openai.name == "OpenAI (api.moonshot.ai)"

    capped = RateLimitPolicy(max_concurrent=1)
    configure_rate_limit(overrides={moonshot: capped})
    assert get_rate_limit_policy(moonshot) == capped
    assert get_rate_limit_policy(Provider.MOONSHOT) == capped
    assert get_rate_limit_policy(via_openai) != capped


async def test_calls_are_logged_under_the_route_name() -> None:
    moonshot = make_provider(Provider.MOONSHOT, api_key="k")
    china = make_provider(Provider.MOONSHOT, api_key="k", base_url="https://api.moonshot.cn/v1")
    via_openai = make_provider(Provider.OPENAI, api_key="k", base_url=_DEFAULT_BASE_URL)
    with install(lambda _request: Reply.text("ok")) as fake, capture_llm_records() as records:
        for provider in (moonshot, china, via_openai):
            _ = await text_llm_call("hi", feature="moonshot-route", provider=provider)
    expected = ["Moonshot", "Moonshot (api.moonshot.cn)", "OpenAI (api.moonshot.ai)"]
    assert [record.provider for record in records] == expected
    assert [request.provider for request in fake.requests] == expected
    assert [request.model for request in fake.requests] == [
        "moonshot/kimi-k3.1-pro",
        "moonshot/kimi-k3.1-pro",
        "openai/gpt-4.1-mini",
    ]
