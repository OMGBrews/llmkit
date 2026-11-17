"""Each shipped provider's ``service_tier`` translation, and where the default comes from.

``BaseProvider.service_tier_kwargs`` turns the tier the host asked for into the
request arguments a route sends, or raises
:class:`~llmkit.ServiceTierUnsupportedError` before anything is sent. The
expected rows below come from live measurement on 2026-11-17 (LiteLLM 1.109.1);
the provider docstrings carry the evidence. The table is keyed by
:class:`~llmkit.Provider`, so a new provider fails here until its behaviour is
stated.

The second half checks that every provider carries a client-wide tier from
:class:`~llmkit.LLMClientConfig` and from ``make_provider``, and that the config
rejects a value that is not a non-empty string or ``None``.
"""

from __future__ import annotations

from typing import cast

import pytest

from llmkit import LLMClientConfig, Provider, ServiceTierUnsupportedError
from llmkit.providers import build_provider, make_provider
from llmkit.providers.base import BaseProvider

#: The minimal legal key per provider, as in ``test_registry_exhaustive.py``:
#: the bearer providers need one, the ambient and local ones must not get one.
_KEY: dict[Provider, str | None] = {
    Provider.OPENROUTER: "k",
    Provider.OLLAMA: None,
    Provider.GOOGLE: "k",
    Provider.ANTHROPIC: "k",
    Provider.OPENAI: "k",
    Provider.DEEPSEEK: "k",
    Provider.CEREBRAS: "k",
    Provider.MOONSHOT: "k",
    Provider.ALIBABA: "k",
    Provider.BEDROCK: None,
    Provider.VERTEX: None,
}

_REFUSED = "refused"
_MADE_UP = "turbo-lane-9"

_ANTHROPIC_FAST = "claude-opus-5-5"
_ANTHROPIC_NOT_FAST = "claude-sonnet-4-6"


def _forwarded(tier: str) -> tuple[str, object]:
    return (tier, {"service_tier": tier})


def _refused(tier: str) -> tuple[str, object]:
    return (tier, _REFUSED)


_OPENAI_COMPATIBLE_ROWS: list[tuple[str | None, list[tuple[str, object]]]] = [
    (
        None,
        [
            _forwarded("default"),
            _forwarded("flex"),
            _forwarded("priority"),
            _forwarded("auto"),
            _forwarded(_MADE_UP),
        ],
    )
]

#: Per provider: (model or ``None`` for the default model, [(tier, expected)]).
#: ``expected`` is the request arguments, or ``_REFUSED``. Anthropic is covered
#: separately below, because its rows depend on LiteLLM's model table.
_TABLE: dict[Provider, list[tuple[str | None, list[tuple[str, object]]]]] = {
    Provider.OPENAI: _OPENAI_COMPATIBLE_ROWS,
    Provider.OPENROUTER: _OPENAI_COMPATIBLE_ROWS,
    Provider.DEEPSEEK: _OPENAI_COMPATIBLE_ROWS,
    Provider.MOONSHOT: _OPENAI_COMPATIBLE_ROWS,
    Provider.ALIBABA: _OPENAI_COMPATIBLE_ROWS,
    Provider.BEDROCK: [
        (
            None,
            [
                _forwarded("default"),
                _forwarded("flex"),
                _forwarded("priority"),
                _refused("auto"),
                _refused(_MADE_UP),
            ],
        )
    ],
    Provider.GOOGLE: [
        (
            None,
            [
                _forwarded("default"),
                _forwarded("flex"),
                _forwarded("priority"),
                # Forwarded as written: LiteLLM lower-cases it on the wire.
                _forwarded("Priority"),
                _forwarded(_MADE_UP),
                _refused("auto"),
                _refused("AUTO"),
                _refused("Auto"),
            ],
        )
    ],
    Provider.VERTEX: [
        (
            model,
            [_refused(tier) for tier in ("default", "flex", "priority", "auto", _MADE_UP)],
        )
        for model in (None, "gemini-2.5-flash", "claude-sonnet-4-6")
    ],
    Provider.CEREBRAS: [
        (None, [_refused(tier) for tier in ("default", "flex", "priority", _MADE_UP)])
    ],
    Provider.OLLAMA: [
        (None, [_refused(tier) for tier in ("default", "flex", "priority", _MADE_UP)])
    ],
}


def _provider(member: Provider) -> BaseProvider:
    return make_provider(member, api_key=_KEY[member])


def _cases() -> list[object]:
    return [
        pytest.param(
            member, model, tier, expected, id=f"{member.value}-{model or 'default'}-{tier}"
        )
        for member, rows in _TABLE.items()
        for model, tiers in rows
        for tier, expected in tiers
    ]


def _assert_refusal_names(
    caught: pytest.ExceptionInfo[ServiceTierUnsupportedError],
    provider: BaseProvider,
    model: str,
    tier: str,
) -> None:
    message = str(caught.value)
    assert provider.name in message
    assert repr(model) in message
    assert repr(tier) in message


def test_every_provider_is_stated() -> None:
    """A new ``Provider`` member must state its translation here."""
    assert set(_TABLE) | {Provider.ANTHROPIC} == set(Provider)


@pytest.mark.parametrize(("member", "model", "tier", "expected"), _cases())
def test_service_tier_kwargs(
    member: Provider, model: str | None, tier: str, expected: object
) -> None:
    provider = _provider(member)
    effective = model or provider.model
    assert effective
    if expected == _REFUSED:
        with pytest.raises(ServiceTierUnsupportedError) as caught:
            _ = provider.service_tier_kwargs(tier, effective)
        _assert_refusal_names(caught, provider, effective, tier)
    else:
        assert provider.service_tier_kwargs(tier, effective) == expected


# --- Anthropic -------------------------------------------------------------


@pytest.fixture
def anthropic_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the LiteLLM facts Anthropic's translation reads, whichever map loaded.

    Fast mode is offered on ``_ANTHROPIC_FAST`` only, and LiteLLM lists
    ``service_tier`` among the Anthropic route's parameters, as LiteLLM 1.109.1
    does.
    """
    import litellm

    table = cast("dict[str, dict[str, object]]", litellm.model_cost)
    monkeypatch.setitem(
        table, _ANTHROPIC_FAST, {**table.get(_ANTHROPIC_FAST, {}), "supports_speed": True}
    )
    not_fast = {
        k: v for k, v in table.get(_ANTHROPIC_NOT_FAST, {}).items() if k != "supports_speed"
    }
    monkeypatch.setitem(table, _ANTHROPIC_NOT_FAST, not_fast)

    def _supported(model: str, custom_llm_provider: str | None = None) -> list[str]:
        del model, custom_llm_provider
        return ["temperature", "speed", "reasoning_effort", "service_tier"]

    monkeypatch.setattr(litellm, "get_supported_openai_params", _supported)


@pytest.mark.usefixtures("anthropic_table")
@pytest.mark.parametrize("model", [_ANTHROPIC_FAST, _ANTHROPIC_NOT_FAST])
@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("default", {"service_tier": "standard_only"}),
        ("auto", {"service_tier": "auto"}),
        ("standard_only", {"service_tier": "standard_only"}),
    ],
)
def test_anthropic_forwards_its_own_tiers(model: str, tier: str, expected: object) -> None:
    """``"default"`` becomes ``standard_only``; Anthropic's own names pass through."""
    assert _provider(Provider.ANTHROPIC).service_tier_kwargs(tier, model) == expected


@pytest.mark.usefixtures("anthropic_table")
def test_anthropic_priority_is_fast_mode_without_a_tier() -> None:
    assert _provider(Provider.ANTHROPIC).service_tier_kwargs("priority", _ANTHROPIC_FAST) == {
        "speed": "fast"
    }


@pytest.mark.usefixtures("anthropic_table")
@pytest.mark.parametrize(
    ("tier", "model"),
    [
        ("priority", _ANTHROPIC_NOT_FAST),
        ("priority", "claude-unlisted-9"),
        ("flex", _ANTHROPIC_FAST),
        ("flex", _ANTHROPIC_NOT_FAST),
        (_MADE_UP, _ANTHROPIC_FAST),
        ("Default", _ANTHROPIC_NOT_FAST),
    ],
)
def test_anthropic_refuses(tier: str, model: str) -> None:
    provider = _provider(Provider.ANTHROPIC)
    with pytest.raises(ServiceTierUnsupportedError) as caught:
        _ = provider.service_tier_kwargs(tier, model)
    _assert_refusal_names(caught, provider, model, tier)


@pytest.mark.parametrize("tier", ["default", "auto", "standard_only"])
def test_anthropic_refuses_a_tier_litellm_would_not_send(
    monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    """A LiteLLM that does not list ``service_tier`` for Anthropic gets no tier."""
    import litellm

    def _supported(model: str, custom_llm_provider: str | None = None) -> list[str]:
        del model, custom_llm_provider
        return ["temperature", "reasoning_effort"]

    monkeypatch.setattr(litellm, "get_supported_openai_params", _supported)
    provider = _provider(Provider.ANTHROPIC)
    with pytest.raises(ServiceTierUnsupportedError) as caught:
        _ = provider.service_tier_kwargs(tier, _ANTHROPIC_NOT_FAST)
    _assert_refusal_names(caught, provider, _ANTHROPIC_NOT_FAST, tier)


def test_anthropic_fast_mode_against_the_loaded_model_table() -> None:
    """The real LiteLLM table still marks the fast-mode model, and not the default one.

    No monkeypatching: if a LiteLLM upgrade drops ``supports_speed`` from
    ``claude-opus-5-5`` or adds it to ``claude-sonnet-4-6``, this fails and
    the rows above need re-measuring.
    """
    provider = _provider(Provider.ANTHROPIC)
    assert provider.service_tier_kwargs("priority", _ANTHROPIC_FAST) == {"speed": "fast"}
    with pytest.raises(ServiceTierUnsupportedError):
        _ = provider.service_tier_kwargs("priority", _ANTHROPIC_NOT_FAST)


# --- The configured default ------------------------------------------------


def _config(member: Provider, service_tier: str | None = None) -> LLMClientConfig:
    if service_tier is None:
        return LLMClientConfig(provider=member, api_key=_KEY[member])
    return LLMClientConfig(provider=member, api_key=_KEY[member], service_tier=service_tier)


@pytest.mark.parametrize("member", list(Provider), ids=[m.value for m in Provider])
def test_every_provider_carries_the_configured_tier(member: Provider) -> None:
    """``build_provider`` and ``make_provider`` both hand the tier to every provider.

    Parametrized over ``list(Provider)``, so a new member is covered, and fails
    here if its constructor or ``from_config`` drops the tier.
    """
    assert build_provider(_config(member, "flex")).service_tier == "flex"
    assert make_provider(member, api_key=_KEY[member], service_tier="flex").service_tier == "flex"
    assert build_provider(_config(member)).service_tier is None
    assert make_provider(member, api_key=_KEY[member]).service_tier is None


@pytest.mark.parametrize("bad", ["", 3, True], ids=["empty", "int", "bool"])
def test_config_rejects_a_tier_that_is_not_a_non_empty_string(bad: object) -> None:
    with pytest.raises(ValueError, match="service_tier"):
        _ = LLMClientConfig(
            provider=Provider.OPENAI,
            api_key="k",
            service_tier=cast("str", bad),
        )


@pytest.mark.parametrize("good", [None, "flex"])
def test_config_accepts_none_and_a_string(good: str | None) -> None:
    config = LLMClientConfig(provider=Provider.OPENAI, api_key="k", service_tier=good)
    assert config.service_tier == good
