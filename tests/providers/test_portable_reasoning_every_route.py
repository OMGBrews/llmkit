"""Portable ``disable`` and ``lowest`` on every route: honoured, or refused before transport.

``"disable"`` must turn thinking off or raise
:class:`~llmkit.exceptions.ReasoningEffortUnsupportedError` inside the provider's
``reasoning_kwargs``, which runs before transport, the cache and retries. It must
never become a lower effort that still reasons. ``"lowest"`` carries the least
reasoning each route supports: off where the model can stop thinking, else the
route's lowest effort.

Both tables are keyed by :class:`~llmkit.Provider`, so a new provider fails here
until its behaviour is stated. Each row also checks the value LiteLLM puts on
the wire where llmkit forwards a portable string LiteLLM interprets itself
(Google, Vertex and Ollama), because the kwarg alone does not show whether
thinking is off.
"""

from __future__ import annotations

from typing import cast

import pytest
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

from llmkit import Provider, ReasoningEffortUnsupportedError
from llmkit.exceptions import LLM_RECOVERABLE_ERRORS
from llmkit.providers import LLMProviderInterface, make_provider
from llmkit.providers.base import BaseProvider

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

_RAISES = "raises"
_OFF_FLAT = {"reasoning_effort": "none"}
_OFF_OPENROUTER = {"extra_body": {"reasoning": {"effort": "none"}}}
_FORWARD = {"reasoning_effort": "disable"}

#: ``disable`` on each route: (model, expected). ``None`` names the route's default.
#: ``_FORWARD`` is off only because LiteLLM maps it to off (asserted below).
_DISABLE: dict[Provider, list[tuple[str | None, object]]] = {
    Provider.OPENROUTER: [
        (None, _RAISES),  # google/gemini-3.1-flash-lite
        ("google/gemini-2.5-pro", _RAISES),
        ("google/gemini-2.5-flash-lite", _OFF_OPENROUTER),
        ("anthropic/claude-sonnet-4", _OFF_OPENROUTER),
    ],
    Provider.OLLAMA: [
        (None, _FORWARD),  # llama3.2
        ("deepseek-r1:8b", _FORWARD),
        ("gpt-oss:20b", _RAISES),
    ],
    Provider.GOOGLE: [
        (None, _RAISES),  # gemini-3.1-flash-lite
        ("gemini-2.5-pro", _RAISES),
        ("gemini-2.5-flash", _FORWARD),
    ],
    Provider.VERTEX: [
        (None, _RAISES),  # gemini-3.1-flash-lite
        ("gemini-2.5-pro", _RAISES),
        ("gemini-2.5-flash", _FORWARD),
    ],
    Provider.ANTHROPIC: [
        (None, _OFF_FLAT),  # claude-sonnet-4-6
        ("claude-opus-5-5", _RAISES),
    ],
    Provider.BEDROCK: [
        (None, _OFF_FLAT),
        ("us.anthropic.claude-opus-5-5", _RAISES),
    ],
    Provider.OPENAI: [
        (None, {}),  # gpt-4.1-mini: no reasoning, parameter omitted
        ("gpt-5.1", _OFF_FLAT),
        ("gpt-5", _RAISES),
    ],
    Provider.DEEPSEEK: [(None, _OFF_FLAT), ("deepseek-reasoner", _OFF_FLAT)],
    Provider.CEREBRAS: [
        (None, _RAISES),  # gpt-oss-120b
        ("qwen-3.8-27b", _OFF_FLAT),
    ],
    # Moonshot keeps thinking on a forwarded "disable"; QwenCloud rejects it.
    # Both turn thinking off on "none". LiteLLM's Moonshot route refuses the
    # parameter on a model its metadata lists without reasoning (omitted: it
    # does not think) and on an unlisted model (raises before transport).
    Provider.MOONSHOT: [
        (None, _OFF_FLAT),  # kimi-k3.1-pro
        ("kimi-k2.6", _OFF_FLAT),
        ("moonshot-v1-8k", {}),
        ("kimi-unlisted-9", _RAISES),
    ],
    Provider.ALIBABA: [(None, _OFF_FLAT), ("qwen-flash", _OFF_FLAT)],  # qwen3.7-plus
}

#: ``lowest`` on each route: off where the model can stop thinking, else its lowest effort.
_LOWEST: dict[Provider, list[tuple[str | None, object]]] = {
    Provider.OPENROUTER: [
        (None, {"extra_body": {"reasoning": {"effort": "minimal"}}}),
        ("google/gemini-2.5-pro", {"extra_body": {"reasoning": {"effort": "minimal"}}}),
        ("google/gemini-2.5-flash-lite", _OFF_OPENROUTER),
    ],
    Provider.OLLAMA: [
        (None, _FORWARD),
        ("gpt-oss:20b", {"reasoning_effort": "low"}),
    ],
    Provider.GOOGLE: [
        (None, _FORWARD),  # LiteLLM: lowest thinkingLevel, includeThoughts False
        ("gemini-2.5-pro", {"reasoning_effort": "minimal"}),
        ("gemini-2.5-flash", _FORWARD),
    ],
    Provider.VERTEX: [
        (None, _FORWARD),
        ("gemini-2.5-pro", {"reasoning_effort": "minimal"}),
        ("gemini-2.5-flash", _FORWARD),
    ],
    Provider.ANTHROPIC: [
        (None, _OFF_FLAT),
        ("claude-opus-5-5", {"reasoning_effort": "low"}),
    ],
    Provider.BEDROCK: [
        (None, _OFF_FLAT),
        ("us.anthropic.claude-opus-5-5", {"reasoning_effort": "low"}),
    ],
    Provider.OPENAI: [
        (None, {}),
        ("gpt-5.1", _OFF_FLAT),
        ("gpt-5", {"reasoning_effort": "minimal"}),
    ],
    Provider.DEEPSEEK: [(None, _OFF_FLAT), ("deepseek-reasoner", _OFF_FLAT)],
    Provider.CEREBRAS: [
        (None, {"reasoning_effort": "low"}),
        ("qwen-3.8-27b", _OFF_FLAT),
    ],
    Provider.MOONSHOT: [
        (None, _OFF_FLAT),
        ("kimi-k2.6", _OFF_FLAT),
        ("moonshot-v1-8k", {}),
        ("kimi-unlisted-9", _RAISES),
    ],
    Provider.ALIBABA: [(None, _OFF_FLAT), ("qwen-flash", _OFF_FLAT)],
}


def _provider(member: Provider) -> LLMProviderInterface:
    return make_provider(member, api_key=_KEY[member])


def _cases(table: dict[Provider, list[tuple[str | None, object]]]) -> list[object]:
    return [
        pytest.param(member, model, expected, id=f"{member.value}-{model or 'default'}")
        for member, rows in table.items()
        for model, expected in rows
    ]


@pytest.mark.parametrize("table", [_DISABLE, _LOWEST], ids=["disable", "lowest"])
def test_every_provider_is_stated(table: dict[Provider, list[tuple[str | None, object]]]) -> None:
    """A new ``Provider`` member must declare its behaviour, including its default model."""
    assert set(table) == set(Provider)
    for member, rows in table.items():
        assert any(model is None for model, _ in rows), f"{member!r} has no default-model row"


def _gemini_params(model: str, effort: str) -> dict[str, object]:
    # raw-llm: LiteLLM's Gemini transform helpers are loosely typed; cast once, here.
    return cast(
        "dict[str, object]",
        VertexGeminiConfig().map_openai_params(  # pyright: ignore[reportUnknownMemberType]
            {"reasoning_effort": effort}, {}, model, False
        ),
    )


def _ollama_params(model: str, effort: str) -> dict[str, object]:
    # raw-llm: as above, for LiteLLM's Ollama transform.
    return cast(
        "dict[str, object]",
        OllamaChatConfig().map_openai_params(  # pyright: ignore[reportUnknownMemberType]
            {"reasoning_effort": effort}, {}, model, False
        ),
    )


def _assert_litellm_turns_thinking_off(member: Provider, model: str) -> None:
    """The forwarded ``"disable"`` is off on the wire, not a lower level."""
    if member in (Provider.GOOGLE, Provider.VERTEX):
        params = _gemini_params(model, "disable")
        assert params["thinkingConfig"] == {"thinkingBudget": 0, "includeThoughts": False}
    else:
        assert member is Provider.OLLAMA
        assert _ollama_params(model, "disable") == {"think": False}


@pytest.mark.parametrize(("member", "model", "expected"), _cases(_DISABLE))
def test_disable_turns_thinking_off_or_raises(
    member: Provider, model: str | None, expected: object
) -> None:
    provider = _provider(member)
    effective = model or provider.model
    if expected == _RAISES:
        with pytest.raises(ReasoningEffortUnsupportedError) as caught:
            _ = provider.reasoning_kwargs("disable", effective)
        assert effective in str(caught.value)
        assert not isinstance(caught.value, LLM_RECOVERABLE_ERRORS)
        return
    result = provider.reasoning_kwargs("disable", effective)
    assert result == expected
    if result == _FORWARD:
        _assert_litellm_turns_thinking_off(member, effective)


@pytest.mark.parametrize(("member", "model", "expected"), _cases(_LOWEST))
def test_lowest_sends_the_least_reasoning_the_route_supports(
    member: Provider, model: str | None, expected: object
) -> None:
    provider = _provider(member)
    effective = model or provider.model
    if expected == _RAISES:
        # Only where nothing establishes the route's lowest effort (Moonshot's
        # unlisted models); it is never sent blind.
        with pytest.raises(ReasoningEffortUnsupportedError) as caught:
            _ = provider.reasoning_kwargs("lowest", effective)
        assert effective in str(caught.value)
        assert not isinstance(caught.value, LLM_RECOVERABLE_ERRORS)
        return
    assert provider.reasoning_kwargs("lowest", effective) == expected


def test_gemini_3_lowest_keeps_thought_summaries_off() -> None:
    """``lowest`` on Gemini 3 forwards ``disable``, LiteLLM's lowest level without summaries."""
    params = _gemini_params("gemini-3.1-flash-lite", "disable")
    assert params["thinkingConfig"] == {"thinkingLevel": "minimal", "includeThoughts": False}


def test_base_default_translates_lowest_and_leaves_disable() -> None:
    """Third-party routes are not made strict, but LiteLLM never sees the unknown ``lowest``."""
    # Every in-tree provider overrides the method, so call the base default directly.
    provider = cast("BaseProvider", _provider(Provider.OLLAMA))
    base = BaseProvider.reasoning_kwargs
    assert base(provider, "lowest", "m") == {"reasoning_effort": "disable"}
    assert base(provider, "disable", "m") == {"reasoning_effort": "disable"}
    assert base(provider, "high", "m") == {"reasoning_effort": "high"}


def _as_provider(value: object) -> LLMProviderInterface:
    return cast("LLMProviderInterface", value)


def test_providers_without_the_method_get_lowest_translated() -> None:
    """The fallback for a provider that predates ``reasoning_kwargs`` translates ``lowest``."""
    from llmkit import _litellm

    class _Old:
        model: str = "m"

    old = _as_provider(_Old())
    assert _litellm._reasoning_request_kwargs(old, "lowest", None) == {
        "reasoning_effort": "disable"
    }
    assert _litellm._reasoning_request_kwargs(old, "disable", None) == {
        "reasoning_effort": "disable"
    }
