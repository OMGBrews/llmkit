"""Endpoint providers resolve their base URL config-first, then env, else a pinned default.

The sibling :mod:`tests.providers.test_api_key_resolution` pins the *credential*
half of this problem; this file pins the *endpoint* half. Four providers
(OpenAI, Anthropic, Google AI Studio, DeepSeek) used to forward ``api_base``
only when a ``base_url`` was configured, leaving the endpoint to LiteLLM — which
reads its own environment aliases (``OPENAI_BASE_URL``, ``ANTHROPIC_API_BASE``,
``GEMINI_API_BASE``, ``DEEPSEEK_API_BASE``). A stray variable could therefore
redirect every request to another host with nothing in the library naming the
endpoint: the same invisible swap the key work closed, one layer down.

Endpoint resolution is now explicit library behavior: the configured
``base_url`` wins; otherwise the provider's own environment aliases are
consulted in a measured precedence order; otherwise the library's own pinned
default stands — so the endpoint a call will hit is readable from the provider
rather than inferred from the ambient environment.

Two shipped properties shape the tests below. Resolution happens **at the read
point** (``completion_kwargs()``), not at construction, because importing LiteLLM
is what runs ``load_dotenv()`` and ``import llmkit`` does not import LiteLLM — so
a provider built at startup must not freeze a default it read too early. And
**Google AI Studio owns no default**: its base carries an API version LiteLLM
derives from the model, so it sends no ``api_base`` at all when nothing is
configured. The other three always send one.

The defaults below are not transcribed from LiteLLM's source: each was measured
byte-for-byte on the wire against litellm 1.92.0 (2026-07-21), which is the only
way to catch the shape traps they encode — Anthropic's default must *omit*
``/v1`` (LiteLLM appends its own, so a pinned ``/v1`` yields ``/v1/v1/messages``),
and Gemini turned out not to admit a constant at all. The alias precedence was
measured the same way, with both aliases set to distinct sentinels.

Every test controls the environment explicitly (each clears every alias under
test before asserting), so a pass never depends on an ambient value — which
matters because importing LiteLLM runs ``load_dotenv()`` and can inject one into
the process. Configs pass ``api_key="sk-test"`` so key resolution, which these
tests are not about, never fails first.
"""

from __future__ import annotations

import pytest

from llmkit import LLMClientConfig, Provider
from llmkit.providers import build_provider

# (provider, endpoint env aliases in measured precedence order, library default).
#
# Google's default is ``None`` — alone among the four it owns no default endpoint,
# because the AI Studio base carries an API *version* LiteLLM derives from the
# model (``v1alpha`` for Gemini 3 and newer, ``v1beta`` otherwise). A static
# default would silently move some models to the wrong version, so with nothing
# configured that provider sends no ``api_base`` at all. See
# ``test_google_omits_api_base_when_nothing_is_configured`` below.
_ENDPOINT_PROVIDERS: list[tuple[Provider, tuple[str, ...], str | None]] = [
    (Provider.OPENAI, ("OPENAI_BASE_URL", "OPENAI_API_BASE"), "https://api.openai.com/v1"),
    (
        Provider.ANTHROPIC,
        ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"),
        "https://api.anthropic.com",
    ),
    (Provider.GOOGLE, ("GEMINI_API_BASE",), None),
    (Provider.DEEPSEEK, ("DEEPSEEK_API_BASE",), "https://api.deepseek.com/beta"),
]

# Flattened to one case per alias — six in all, since OpenAI and Anthropic each
# answer to two. Every alias gets its own assertion because "we kept honouring
# the variables LiteLLM already read" is a promise about each name, not about
# the provider in aggregate.
_PROVIDER_ALIAS_DEFAULTS: list[tuple[Provider, str, str | None]] = [
    (provider, alias, default)
    for provider, aliases, default in _ENDPOINT_PROVIDERS
    for alias in aliases
]

# The subset that asserts on the resolved endpoint rather than the default.
_PROVIDER_ALIASES = [(provider, alias) for provider, alias, _ in _PROVIDER_ALIAS_DEFAULTS]

# The three providers that *do* own a default endpoint — the fallback tests below
# are about that default, so Google (which owns none, by design) is excluded and
# gets its own dedicated test instead of a conditional inside these.
_PROVIDER_DEFAULTS = [
    (provider, default) for provider, _, default in _ENDPOINT_PROVIDERS if default is not None
]

# The same restriction, one case per (provider, alias).
_PROVIDER_ALIAS_DEFAULTS_WITH_FALLBACK = [
    (provider, alias, default)
    for provider, alias, default in _PROVIDER_ALIAS_DEFAULTS
    if default is not None
]

# (provider, winning alias, losing alias) for the two providers with a pair.
_ALIAS_PAIRS = [
    (provider, aliases[0], aliases[1])
    for provider, aliases, _ in _ENDPOINT_PROVIDERS
    if len(aliases) == 2
]

# Every endpoint alias any of the four providers consults.
_ALL_ALIASES = tuple(alias for _, alias, _ in _PROVIDER_ALIAS_DEFAULTS)


def _clear_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every endpoint alias from the environment.

    Clearing *all* of them rather than only the one under test keeps a case from
    passing off a sibling variable — the precedence tests would otherwise read
    an ambient second alias as if the test had set it.
    """
    for alias in _ALL_ALIASES:
        monkeypatch.delenv(alias, raising=False)


@pytest.mark.parametrize(("provider", "default"), _PROVIDER_DEFAULTS)
def test_no_config_no_env_uses_library_default(
    provider: Provider, default: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With neither config nor env, the library's own measured default endpoint
    reaches the wire kwargs — ``api_base`` is present unconditionally, so the
    endpoint is owned and readable here instead of being LiteLLM's implicit
    choice."""
    _clear_endpoint_env(monkeypatch)
    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))
    assert built.completion_kwargs()["api_base"] == default


@pytest.mark.parametrize(("provider", "alias"), _PROVIDER_ALIASES)
def test_env_var_resolves_endpoint_when_config_absent(
    provider: Provider, alias: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each alias LiteLLM already honoured still redirects the endpoint once the
    library resolves it — the no-silent-migration promise. Taking ownership of
    ``api_base`` would otherwise pin the default over a working deployment that
    steers its traffic with one of these variables today."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv(alias, "https://env.example/v1")
    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))
    assert built.completion_kwargs()["api_base"] == "https://env.example/v1"


@pytest.mark.parametrize(("provider", "winner", "loser"), _ALIAS_PAIRS)
def test_first_alias_wins_when_both_are_set(
    provider: Provider, winner: str, loser: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenAI and Anthropic each answer to two aliases, and LiteLLM's order
    between them is not alphabetical, symmetric, or documented — it was measured
    with both set to distinct sentinels. Pinning it keeps a caller who sets both
    (a leftover plus a new one) from landing on a different host than before."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv(winner, "https://first.example/v1")
    monkeypatch.setenv(loser, "https://second.example/v1")
    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))
    assert built.completion_kwargs()["api_base"] == "https://first.example/v1"


@pytest.mark.parametrize(("provider", "alias"), _PROVIDER_ALIASES)
def test_config_base_url_wins_over_env(
    provider: Provider, alias: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly configured endpoint beats the ambient one, exactly as an
    explicit ``api_key`` does. This is the precedence the whole feature rests
    on — reading the environment must never demote a value the host set on
    purpose — so it is pinned even though it already holds."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv(alias, "https://env.example/v1")
    built = build_provider(
        LLMClientConfig(provider=provider, api_key="sk-test", base_url="https://gateway.example/v1")
    )
    assert built.completion_kwargs()["api_base"] == "https://gateway.example/v1"


@pytest.mark.parametrize(("provider", "alias", "default"), _PROVIDER_ALIAS_DEFAULTS_WITH_FALLBACK)
def test_empty_env_var_is_treated_as_unset(
    provider: Provider, alias: str, default: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty alias is unset, not an empty endpoint — matching the
    falsy-is-unset convention ``api_key`` and ``model`` already follow. A blank
    line in a ``.env`` file must fall through to the default rather than send an
    empty ``api_base`` into the transport to fail at call time."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv(alias, "")
    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))
    assert built.completion_kwargs()["api_base"] == default


def test_google_omits_api_base_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google AI Studio alone owns no default endpoint, and must send no ``api_base``.

    Its three siblings pin a constant base. Google cannot, because the AI Studio
    base carries an API *version* that LiteLLM derives from the model —
    ``v1alpha`` for Gemini 3 and newer, ``v1beta`` otherwise — and applies only
    when ``api_base`` is absent. Measured against litellm 1.92.0 on 2026-07-21:
    pinning ``…/v1beta`` sends ``gemini-3-pro-preview`` to ``/v1beta`` where it
    had gone to ``/v1alpha``, and pinning the bare host loses the version
    segment altogether. Both are silent wire-shape changes, which is precisely
    what owning the endpoint was supposed to avoid — so here the library declines
    to name one, and lets LiteLLM choose the version.

    The scoped cost is real and deliberate: in this one case the endpoint can
    still come from a source llmkit does not read. Naming a ``base_url`` or
    ``GEMINI_API_BASE`` closes it, and both are pinned by the tests above.
    ``test_endpoint_routing.py`` proves the version is preserved on the wire.
    """
    _clear_endpoint_env(monkeypatch)
    built = build_provider(LLMClientConfig(provider=Provider.GOOGLE, api_key="sk-test"))
    assert "api_base" not in built.completion_kwargs()


@pytest.mark.parametrize(("provider", "alias"), _PROVIDER_ALIASES)
def test_endpoint_is_resolved_at_read_time_not_construction_time(
    provider: Provider, alias: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alias set *after* the provider is built still reaches the wire kwargs.

    Not a hypothetical ordering: ``import llmkit`` deliberately does not import
    LiteLLM (it is deferred to the first call), and importing LiteLLM is what
    runs ``load_dotenv()``. So in the library's own documented pattern — build a
    provider with ``make_provider`` at startup, pass it to a call later — the
    provider is constructed *before* the host's ``.env`` has been read into the
    process. Resolving the endpoint at construction would freeze the library
    default and then ignore the very ``GEMINI_API_BASE`` / ``OPENAI_BASE_URL``
    LiteLLM itself would have honoured, silently sending the call to the public
    endpoint instead of the host's gateway (measured 2026-07-21: it did exactly
    that, carrying the caller's pinned key). Resolution therefore happens in
    ``completion_kwargs()``, the read point, which the transport seam reaches
    only after LiteLLM is imported.

    This test fails against a construction-time implementation, which is the
    whole reason it exists — the sequencing here is the point, not incidental.
    """
    _clear_endpoint_env(monkeypatch)
    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))

    monkeypatch.setenv(alias, "https://late.example/v1")

    assert built.completion_kwargs()["api_base"] == "https://late.example/v1"


@pytest.mark.parametrize(
    ("provider", "api_key", "expected"),
    [
        (Provider.OPENROUTER, "sk-test", "https://openrouter.ai/api/v1"),
        (Provider.OLLAMA, None, "http://localhost:11434"),
    ],
)
def test_openrouter_and_ollama_ignore_litellm_endpoint_env_vars(
    provider: Provider,
    api_key: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two providers that already sent ``api_base`` unconditionally stay
    deaf to the environment, and that non-expansion is deliberate: their
    endpoints are a fixed gateway and a local server, so an ambient
    ``OPENROUTER_API_BASE`` / ``OLLAMA_API_BASE`` redirecting them buys nothing
    and widens the surface. Pinned here so a later "make all six providers
    consistent" tidy-up has to argue with a test rather than quietly grant two
    more variables the power to reroute traffic."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://hijack.example/v1")
    monkeypatch.setenv("OLLAMA_API_BASE", "https://hijack.example")
    built = build_provider(LLMClientConfig(provider=provider, api_key=api_key))
    assert built.completion_kwargs()["api_base"] == expected
