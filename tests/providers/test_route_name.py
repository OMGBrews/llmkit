"""A provider built with an explicit non-default ``base_url`` reports a route name.

``BaseProvider.name`` is the rate-limit key and the ``provider`` value in call
records. Each class that accepts a ``base_url`` (the ``_ROUTES`` table below,
checked against every ``Provider`` member) reports its plain provider name on its
own endpoint and ``"<name> (<host>)"`` on any other, so a compatible endpoint
reached through another provider's class (Moonshot through ``OpenAIProvider``, a
proxy in front of Anthropic, a remote Ollama server) is limited and logged apart
from the provider itself.

Only a ``base_url`` passed at construction counts: an endpoint taken from an
environment variable keeps the plain name. The limiter-side consequences
(independent slots, overrides) are in ``tests/reliability/test_rate_limiting.py``.
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple, override

import instructor
import pytest

from llmkit import capture_llm_records, text_llm_call
from llmkit.providers import (
    AlibabaProvider,
    AnthropicProvider,
    BaseProvider,
    CerebrasProvider,
    DeepSeekProvider,
    GoogleProvider,
    LLMClientConfig,
    MoonshotProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
    describe_llm,
    make_provider,
    provider_display_name,
)
from llmkit.providers.base import route_name
from llmkit.testing import Reply, install


class _Route(NamedTuple):
    """One provider that accepts ``base_url``: its member, name and own endpoint."""

    member: Provider
    name: str
    default_base_url: str | None
    env_vars: tuple[str, ...]


_ROUTES = [
    _Route(
        Provider.OPENAI,
        "OpenAI",
        "https://api.openai.com/v1",
        ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    ),
    _Route(
        Provider.ANTHROPIC,
        "Anthropic",
        "https://api.anthropic.com",
        ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"),
    ),
    _Route(Provider.CEREBRAS, "Cerebras", "https://api.cerebras.ai/v1", ("CEREBRAS_API_BASE",)),
    _Route(Provider.MOONSHOT, "Moonshot", "https://api.moonshot.ai/v1", ("MOONSHOT_API_BASE",)),
    _Route(
        Provider.ALIBABA,
        "Alibaba",
        "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1",
        ("QWENCLOUD_API_BASE",),
    ),
    _Route(Provider.DEEPSEEK, "DeepSeek", "https://api.deepseek.com/beta", ("DEEPSEEK_API_BASE",)),
    _Route(Provider.GOOGLE, "Google AI Studio", None, ("GEMINI_API_BASE",)),
    _Route(Provider.OPENROUTER, "OpenRouter", "https://openrouter.ai/api/v1", ()),
    _Route(Provider.OLLAMA, "Ollama", "http://localhost:11434", ()),
]
_ROUTE_IDS = [str(route.member) for route in _ROUTES]
_ALL_ENV_VARS = [env_var for route in _ROUTES for env_var in route.env_vars]

_every_route = pytest.mark.parametrize("route", _ROUTES, ids=_ROUTE_IDS)


@pytest.fixture(autouse=True)
def _no_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every endpoint variable, so only the test's own setting is in play."""
    for env_var in _ALL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _build(route: _Route, base_url: str | None = None) -> BaseProvider:
    """Build ``route``'s provider offline through the public construction path."""
    if route.member is Provider.OLLAMA:
        return make_provider(route.member, base_url=base_url)
    return make_provider(route.member, api_key="k", base_url=base_url)


def test_the_table_covers_every_class_that_accepts_base_url() -> None:
    """The routes above are exactly the providers whose config reads ``base_url``."""
    accepting = {
        member
        for member in Provider
        if "base_url" in type(_build_any(member))._accepted_config_fields
    }
    assert accepting == {route.member for route in _ROUTES}


def _build_any(member: Provider) -> BaseProvider:
    if member is Provider.OLLAMA:
        return make_provider(member)
    if member is Provider.BEDROCK:
        return make_provider(member, aws_region_name="us-east-1")
    if member is Provider.VERTEX:
        return make_provider(member, vertex_project="p", vertex_location="us-central1")
    return make_provider(member, api_key="k")


@_every_route
def test_no_base_url_keeps_the_plain_name(route: _Route) -> None:
    assert _build(route).name == route.name


@_every_route
def test_the_declared_default_is_the_endpoint_the_provider_sends(route: _Route) -> None:
    """The endpoint ``name`` compares against is the one a default build calls."""
    provider = _build(route)
    assert type(provider)._default_base_url == route.default_base_url
    assert provider.completion_kwargs().get("api_base") == route.default_base_url


@_every_route
def test_an_explicit_default_endpoint_keeps_the_plain_name(route: _Route) -> None:
    """The default passed explicitly, in any spelling of its host and port, is the same route."""
    if route.default_base_url is None:
        pytest.skip("Google AI Studio has no default endpoint to pass explicitly")
    scheme, _, rest = route.default_base_url.partition("://")
    authority = rest.split("/", 1)[0]
    host, _, port = authority.partition(":")
    scheme_port = {"https": "443", "http": "80"}[scheme]
    spellings = [
        route.default_base_url,
        route.default_base_url + "/",
        f"{scheme}://{authority}",
        f"{scheme}://{authority}/some/other/path",
        f"{scheme}://{host.upper()}:{port or scheme_port}/",
    ]
    for base_url in spellings:
        assert _build(route, base_url).name == route.name, base_url


@_every_route
def test_another_host_gets_a_route_name(route: _Route) -> None:
    """A different host is a different route; the URL path is left out of its name."""
    provider = _build(route, "https://Gateway.Example/v1/chat")
    assert provider.name == f"{route.name} (gateway.example)"
    assert provider.completion_kwargs()["api_base"] == "https://Gateway.Example/v1/chat"


@_every_route
def test_a_port_in_the_url_is_part_of_the_route_name(route: _Route) -> None:
    assert _build(route, "http://gateway.example:8080/v1").name == (
        f"{route.name} (gateway.example:8080)"
    )
    # The scheme's own port says nothing the host does not, so it is dropped.
    assert _build(route, "https://gateway.example:443/v1").name == f"{route.name} (gateway.example)"
    assert _build(route, "http://gateway.example:80/v1").name == f"{route.name} (gateway.example)"


@_every_route
def test_the_default_host_on_another_port_is_a_different_route(route: _Route) -> None:
    if route.default_base_url is None:
        pytest.skip("Google AI Studio has no default endpoint")
    scheme, _, rest = route.default_base_url.partition("://")
    host = rest.split("/", 1)[0].partition(":")[0]
    assert _build(route, f"{scheme}://{host}:8443/v1").name == f"{route.name} ({host}:8443)"


@_every_route
def test_an_endpoint_set_only_by_environment_keeps_the_plain_name(
    route: _Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment moves the endpoint the call goes to, not the name."""
    if not route.env_vars:
        pytest.skip(f"{route.name} reads no endpoint environment variable")
    for env_var in route.env_vars:
        monkeypatch.setenv(env_var, "https://env-gateway.example/v1")
        provider = _build(route)
        assert provider.completion_kwargs()["api_base"] == "https://env-gateway.example/v1"
        assert provider.name == route.name
        monkeypatch.delenv(env_var)


@_every_route
def test_the_member_names_the_default_endpoint(route: _Route) -> None:
    """``display_name`` and the member's name stay the class constant on any instance."""
    provider = _build(route, "https://gateway.example/v1")
    assert type(provider).display_name() == route.name
    assert provider_display_name(route.member) == route.name


@_every_route
def test_describe_llm_reports_the_route_name(route: _Route) -> None:
    assert describe_llm(provider=_build(route)).provider_name == route.name
    info = describe_llm(provider=_build(route, "https://gateway.example/v1"))
    assert info.provider_name == f"{route.name} (gateway.example)"
    assert info.provider is route.member


@_every_route
async def test_a_call_is_logged_under_the_route_name(route: _Route) -> None:
    """The call record's ``provider`` and the fake's request both carry the route name."""
    provider = _build(route, "https://gateway.example/v1")
    with install(lambda _request: Reply.text("ok")) as installed, capture_llm_records() as records:
        _ = await text_llm_call("hi", feature="route-name", provider=provider)
        _ = await text_llm_call("hi", feature="route-name", provider=_build(route))
    expected = [f"{route.name} (gateway.example)", route.name]
    assert [record.provider for record in records] == expected
    assert [request.provider for request in installed.requests] == expected


def test_openai_class_pointed_at_compatible_vendors() -> None:
    """The cases the change exists for: compatible endpoints through the OpenAI class."""
    moonshot = OpenAIProvider(api_key="k", base_url="https://api.moonshot.ai/v1")
    cerebras = OpenAIProvider(api_key="k", base_url="https://api.cerebras.ai/v1")
    assert moonshot.name == "OpenAI (api.moonshot.ai)"
    assert cerebras.name == "OpenAI (api.cerebras.ai)"
    assert CerebrasProvider(api_key="k").name == "Cerebras"
    assert MoonshotProvider(api_key="k").name == "Moonshot"
    qwencloud = OpenAIProvider(
        api_key="k", base_url="https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"
    )
    assert qwencloud.name == "OpenAI (ap-southeast-1.maas.qwencloudapi.com)"
    assert AlibabaProvider(api_key="k").name == "Alibaba"


def test_moonshot_china_endpoint_is_its_own_route() -> None:
    """Moonshot's China endpoint, reached through its own class, is limited apart."""
    china = MoonshotProvider(api_key="k", base_url="https://api.moonshot.cn/v1")
    assert china.name == "Moonshot (api.moonshot.cn)"
    assert MoonshotProvider(api_key="k", base_url="https://api.moonshot.ai/v1").name == "Moonshot"


def test_deepseek_host_without_the_beta_path_is_the_default_route() -> None:
    assert DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com").name == "DeepSeek"


def test_ollama_hosts_are_compared_as_written() -> None:
    """No name resolution: the loopback address is not ``localhost``."""
    assert OllamaProvider().name == "Ollama"
    assert OllamaProvider(base_url="http://localhost:11434/").name == "Ollama"
    assert OllamaProvider(base_url="http://127.0.0.1:11434").name == "Ollama (127.0.0.1:11434)"
    assert OllamaProvider(base_url="http://gpu-box:11434").name == "Ollama (gpu-box:11434)"
    assert OllamaProvider(base_url="http://localhost").name == "Ollama (localhost)"
    assert OllamaProvider(base_url="http://[::1]:11434").name == "Ollama ([::1]:11434)"


def test_openrouter_and_anthropic_and_google_direct_construction() -> None:
    assert OpenRouterProvider(api_key="k").name == "OpenRouter"
    assert AnthropicProvider(api_key="k", base_url="https://proxy.example").name == (
        "Anthropic (proxy.example)"
    )
    assert GoogleProvider(api_key="k").name == "Google AI Studio"
    google = GoogleProvider(api_key="k", base_url="https://generativelanguage.googleapis.com")
    assert google.name == "Google AI Studio (generativelanguage.googleapis.com)"


def test_the_name_comes_from_config_built_providers_too() -> None:
    config = LLMClientConfig(
        provider=Provider.OPENAI, api_key="k", base_url="https://api.moonshot.ai/v1"
    )
    assert describe_llm(config).provider_name == "OpenAI (api.moonshot.ai)"


def test_an_empty_base_url_is_unset() -> None:
    assert OpenAIProvider(api_key="k", base_url="").name == "OpenAI"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://user:secret@gateway.example/v1", "OpenAI (gateway.example)"),
        ("gateway.example:8080/v1", "OpenAI (gateway.example:8080)"),
        ("https://gateway.example:notaport/v1", "OpenAI"),
        ("https:///v1", "OpenAI"),
    ],
)
def test_route_name_edge_urls(base_url: str, expected: str) -> None:
    """Userinfo never reaches the name; a URL with no readable host keeps the plain name."""
    assert route_name("OpenAI", base_url, "https://api.openai.com/v1") == expected


@pytest.mark.parametrize("member", [Provider.BEDROCK, Provider.VERTEX], ids=str)
def test_fixed_endpoint_providers_keep_the_plain_name(member: Provider) -> None:
    provider = _build_any(member)
    assert provider.name == type(provider).display_name()


class _HostProvider(BaseProvider):
    """A host-written provider that sets none of the route hooks."""

    _provider_name: ClassVar[str] = "Host"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "host-model"
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset()

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_base": "https://host.example/v1"}

    @classmethod
    @override
    def build(cls, config: LLMClientConfig) -> _HostProvider:
        return cls(config.model)


def test_a_host_subclass_that_sets_nothing_keeps_its_name() -> None:
    assert _HostProvider().name == "Host"
