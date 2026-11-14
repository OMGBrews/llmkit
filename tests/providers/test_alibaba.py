"""Unit tests for the direct Alibaba (QwenCloud) provider.

The Alibaba provider must expose the ``qwencloud/<model>`` LiteLLM prefix, pin
``Mode.JSON_SCHEMA`` without the strict upgrade (measured live, see the provider
docstring), and forward ``api_key`` and ``api_base`` always: the configured
``base_url`` first, else ``QWENCLOUD_API_BASE``, else QwenCloud's Singapore
pay-as-you-go endpoint, resolved per call rather than at construction. That
default is not the endpoint LiteLLM's ``qwencloud/`` route picks with nothing
configured, so forwarding it always is what selects it. The key resolves only
from ``api_key`` or ``QWENCLOUD_API_KEY``, never ``DASHSCOPE_API_KEY``, and the
provider translates the portable ``disable`` and ``lowest`` efforts to
``"none"``: QwenCloud rejects ``"disable"``.

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
from llmkit.providers import AlibabaProvider
from llmkit.rate_limiting import RateLimitPolicy, configure_rate_limit, get_rate_limit_policy
from llmkit.testing import Reply, install

_DEFAULT_BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"
_ENV_KEY = "QWENCLOUD_API_KEY"
_ENV_BASE = "QWENCLOUD_API_BASE"
_GATEWAY = "https://qwen-gateway.example/compatible-mode/v1"


@pytest.fixture(autouse=True)
def _no_ambient_qwencloud_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variables under test, so no case passes off an ambient value."""
    for name in (_ENV_KEY, _ENV_BASE, "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (None, "qwencloud/qwen3.7-plus"),
        ("qwen-flash", "qwencloud/qwen-flash"),
        ("qwen3.7-plus", "qwencloud/qwen3.7-plus"),
    ],
)
def test_model_prefix(model: str | None, expected: str) -> None:
    provider = AlibabaProvider(api_key="sk-test", model=model)
    assert provider.litellm_model() == expected


def test_litellm_model_override_keeps_the_prefix() -> None:
    provider = AlibabaProvider(api_key="sk-test")
    assert provider.litellm_model("qwen-flash") == "qwencloud/qwen-flash"


def test_measured_traits() -> None:
    """Schema mode without strict; tool_choice supported; compose not measured, so off."""
    provider = AlibabaProvider(api_key="sk-test")
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA
    assert provider.strict_json_schema is False
    assert provider.compose_tools_schema is False
    assert provider.supports_tool_choice is True
    assert provider.single_tool_call_per_schema is False


def test_build_from_a_key_alone_uses_the_defaults() -> None:
    provider = make_provider(Provider.ALIBABA, api_key="sk-only")
    assert isinstance(provider, AlibabaProvider)
    assert provider.model == "qwen3.7-plus"
    assert provider.name == "Alibaba"
    assert provider.completion_kwargs() == {"api_key": "sk-only", "api_base": _DEFAULT_BASE_URL}


@pytest.mark.parametrize("key", ["sk-env-one", "sk-env-two"])
def test_build_reads_the_key_from_the_documented_variable(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV_KEY, key)
    provider = build_provider(LLMClientConfig(provider=Provider.ALIBABA))
    assert isinstance(provider, AlibabaProvider)
    assert provider.completion_kwargs() == {"api_key": key, "api_base": _DEFAULT_BASE_URL}


def test_a_configured_key_wins_over_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_KEY, "sk-env")
    provider = build_provider(LLMClientConfig(provider=Provider.ALIBABA, api_key="sk-config"))
    assert provider.completion_kwargs()["api_key"] == "sk-config"


def test_a_missing_key_raises_naming_the_variable() -> None:
    with pytest.raises(ValueError, match=r"Alibaba requires an API key.*QWENCLOUD_API_KEY"):
        _ = build_provider(LLMClientConfig(provider=Provider.ALIBABA))


def test_dashscope_key_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM's ``DASHSCOPE_API_KEY`` fallback is not llmkit's: setting only it still raises."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope")
    with pytest.raises(ValueError, match="QWENCLOUD_API_KEY"):
        _ = build_provider(LLMClientConfig(provider=Provider.ALIBABA))
    with pytest.raises(ValueError, match="QWENCLOUD_API_KEY"):
        _ = make_provider(Provider.ALIBABA)

    monkeypatch.setenv(_ENV_KEY, "sk-qwencloud")
    provider = build_provider(LLMClientConfig(provider=Provider.ALIBABA))
    assert provider.completion_kwargs()["api_key"] == "sk-qwencloud"


def test_build_threads_every_field() -> None:
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.ALIBABA,
            model="qwen-flash",
            api_key="sk-test",
            base_url=_GATEWAY,
            reasoning_effort="medium",
        )
    )
    assert isinstance(provider, AlibabaProvider)
    assert provider.litellm_model() == "qwencloud/qwen-flash"
    assert provider.reasoning_effort == "medium"
    assert provider.completion_kwargs() == {"api_key": "sk-test", "api_base": _GATEWAY}


@pytest.mark.parametrize(
    ("base_url", "env_base", "expected"),
    [
        (None, None, _DEFAULT_BASE_URL),
        (None, "https://env.example/v1", "https://env.example/v1"),
        (_GATEWAY, None, _GATEWAY),
        (_GATEWAY, "https://env.example/v1", _GATEWAY),
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
    """Explicit ``base_url``, then ``QWENCLOUD_API_BASE``, then the default."""
    if env_base is not None:
        monkeypatch.setenv(_ENV_BASE, env_base)
    provider = AlibabaProvider(api_key="sk-test", base_url=base_url)
    assert provider.completion_kwargs()["api_base"] == expected


def test_endpoint_is_resolved_per_call_not_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A variable set or changed after construction reaches the next call's kwargs."""
    provider = AlibabaProvider(api_key="sk-test")
    assert provider.completion_kwargs()["api_base"] == _DEFAULT_BASE_URL
    monkeypatch.setenv(_ENV_BASE, "https://late.example/v1")
    assert provider.completion_kwargs()["api_base"] == "https://late.example/v1"
    monkeypatch.setenv(_ENV_BASE, _GATEWAY)
    assert provider.completion_kwargs()["api_base"] == _GATEWAY
    monkeypatch.delenv(_ENV_BASE)
    assert provider.completion_kwargs()["api_base"] == _DEFAULT_BASE_URL


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "Alibaba"),
        (_DEFAULT_BASE_URL, "Alibaba"),
        ("https://AP-SOUTHEAST-1.maas.qwencloudapi.com:443/", "Alibaba"),
        (_GATEWAY, "Alibaba (qwen-gateway.example)"),
        (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "Alibaba (dashscope-intl.aliyuncs.com)",
        ),
    ],
)
def test_name(base_url: str | None, expected: str) -> None:
    assert AlibabaProvider(api_key="sk-test", base_url=base_url).name == expected


def test_an_endpoint_from_the_environment_keeps_the_plain_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_BASE, _GATEWAY)
    assert AlibabaProvider(api_key="sk-test").name == "Alibaba"


@pytest.mark.parametrize("effort", ["disable", "lowest"])
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen-flash", "qwencloud/qwen3.7-plus"])
def test_disable_and_lowest_turn_thinking_off(effort: str, model: str) -> None:
    """Both portable values are sent as ``"none"``; QwenCloud rejects ``"disable"``."""
    provider = AlibabaProvider(api_key="sk-test")
    assert provider.reasoning_kwargs(effort, model) == {"reasoning_effort": "none"}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal", "xhigh"])
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen-flash"])
def test_other_efforts_pass_through_unchanged(effort: str, model: str) -> None:
    provider = AlibabaProvider(api_key="sk-test")
    assert provider.reasoning_kwargs(effort, model) == {"reasoning_effort": effort}


@pytest.mark.parametrize(
    "config",
    [
        LLMClientConfig(provider=Provider.ALIBABA, api_key="k", aws_region_name="us-east-1"),
        LLMClientConfig(provider=Provider.ALIBABA, api_key="k", vertex_project="p"),
        LLMClientConfig(provider=Provider.ALIBABA, api_key="k", vertex_location="us-central1"),
        LLMClientConfig(provider=Provider.ALIBABA, api_key="k", gemini_structured_output="json"),
    ],
    ids=["aws_region_name", "vertex_project", "vertex_location", "gemini_structured_output"],
)
def test_build_rejects_unread_knobs(config: LLMClientConfig) -> None:
    with pytest.raises(ValueError, match="Alibaba does not read"):
        _ = build_provider(config)


def test_alibaba_moonshot_and_openai_on_the_vendor_url_are_separate_routes() -> None:
    """Each first-party class, and the OpenAI class on QwenCloud's URL, is limited apart."""
    alibaba = make_provider(Provider.ALIBABA, api_key="k")
    moonshot = make_provider(Provider.MOONSHOT, api_key="k")
    via_openai = make_provider(Provider.OPENAI, api_key="k", base_url=_DEFAULT_BASE_URL)
    names = [alibaba.name, moonshot.name, via_openai.name]
    assert names == ["Alibaba", "Moonshot", "OpenAI (ap-southeast-1.maas.qwencloudapi.com)"]

    capped = RateLimitPolicy(max_concurrent=1)
    other = RateLimitPolicy(max_concurrent=2)
    configure_rate_limit(overrides={alibaba: capped, moonshot: other})
    assert get_rate_limit_policy(alibaba) == capped
    assert get_rate_limit_policy(Provider.ALIBABA) == capped
    assert get_rate_limit_policy(moonshot) == other
    assert get_rate_limit_policy(via_openai) not in (capped, other)


async def test_calls_are_logged_under_the_route_name() -> None:
    alibaba = make_provider(Provider.ALIBABA, api_key="k")
    elsewhere = make_provider(Provider.ALIBABA, api_key="k", base_url=_GATEWAY)
    moonshot = make_provider(Provider.MOONSHOT, api_key="k")
    via_openai = make_provider(Provider.OPENAI, api_key="k", base_url=_DEFAULT_BASE_URL)
    with install(lambda _request: Reply.text("ok")) as fake, capture_llm_records() as records:
        for provider in (alibaba, elsewhere, moonshot, via_openai):
            _ = await text_llm_call("hi", feature="alibaba-route", provider=provider)
    expected = [
        "Alibaba",
        "Alibaba (qwen-gateway.example)",
        "Moonshot",
        "OpenAI (ap-southeast-1.maas.qwencloudapi.com)",
    ]
    assert [record.provider for record in records] == expected
    assert [request.provider for request in fake.requests] == expected
