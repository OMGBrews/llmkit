"""A per-call ``provider=`` given as a :class:`~llmkit.Provider` member, or refused.

A member runs the call on that provider, built from environment credentials
once per call. Every wrong way of naming the provider for one call fails before
any attempt is made or logged: a plain string is a ``TypeError`` naming the
fix, and a member that cannot be built raises its own error rather than
quietly running on the registered provider.

Every lane is driven through the public call functions with the round trip
faked through :mod:`llmkit.testing`, so the provider the request names is the
one the real request path built. The directory's autouse fixture registers
Ollama; tests that need another registration or none replace it.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Iterator
from typing import TypedDict, cast

import pytest
from pydantic import BaseModel

from llmkit import (
    ComposeUnsupportedError,
    LLMCallOptions,
    LLMClientConfig,
    Provider,
    capture_llm_records,
    configure_llm_client,
    make_provider,
    structured_llm_call_sync,
    text_llm_call_sync,
)
from llmkit.calls import (
    structured_llm_call,
    text_llm_call,
    text_llm_call_stream,
    tool_llm_call,
    tool_llm_call_stream,
)
from llmkit.testing import Reply, install
from llmkit.tools import ToolDefinition


class _Schema(BaseModel):
    ok: bool


_TOOLS = [ToolDefinition("lookup", "Look a thing up.", {"type": "object"})]


class _Keywords(TypedDict, total=False):
    provider: Provider
    options: LLMCallOptions


_Lane = Callable[[_Keywords], Awaitable[object]]


def _lanes() -> dict[str, _Lane]:
    """Every call lane, taking the ``provider``/``options`` keywords to pass."""

    async def structured(kw: _Keywords) -> object:
        return await structured_llm_call("hi", _Schema, feature="t", **kw)

    async def text(kw: _Keywords) -> object:
        return await text_llm_call("hi", feature="t", **kw)

    async def tool(kw: _Keywords) -> object:
        return await tool_llm_call("hi", _TOOLS, feature="t", **kw)

    async def compose(kw: _Keywords) -> object:
        return await tool_llm_call("hi", _TOOLS, feature="t", output_schema=_Schema, **kw)

    async def text_stream(kw: _Keywords) -> object:
        return [chunk async for chunk in text_llm_call_stream("hi", feature="t", **kw)]

    async def tool_stream(kw: _Keywords) -> object:
        return [event async for event in tool_llm_call_stream("hi", _TOOLS, feature="t", **kw)]

    return {
        "structured": structured,
        "text": text,
        "tool": tool,
        "compose": compose,
        "text_stream": text_stream,
        "tool_stream": tool_stream,
    }


def _passing(value: Provider | str, via: str) -> _Keywords:
    """The keywords that pass *value* as the call's provider *via* one route.

    A string is cast to the member type: the bare-string tests pass a value the
    signatures refuse, the run-time half of the typing pin in
    ``tests/typing/test_call_surface_types.py``.
    """
    member = cast("Provider", value)
    if via == "keyword":
        return {"provider": member}
    return {"options": LLMCallOptions(provider=member)}


def _answer(_request: object) -> Reply:
    # One reply every lane accepts: valid JSON for the structured and compose
    # lanes, plain content for the text and tool lanes.
    return Reply.json(_Schema(ok=True), cost=0.0)


@pytest.fixture
def _openai_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
    yield


# (member, what is registered while it runs). Each member differs from the
# registered provider, and the middle case registers nothing at all.
_MEMBER_CASES = [
    pytest.param(Provider.OPENAI, "ollama", id="openai-over-registered-ollama"),
    pytest.param(Provider.OPENAI, "none", id="openai-with-nothing-registered"),
    pytest.param(Provider.OLLAMA, "openai", id="ollama-over-registered-openai"),
]


def _register(which: str) -> None:
    if which == "none":
        _ = configure_llm_client(None)
    elif which == "openai":
        _ = configure_llm_client(
            LLMClientConfig(provider=Provider.OPENAI, api_key="sk-registered", model="gpt-x")
        )
    # "ollama": the directory fixture's registration stands.


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["keyword", "options"])
@pytest.mark.parametrize("lane", sorted(_lanes()))
@pytest.mark.parametrize(("member", "registered"), _MEMBER_CASES)
async def test_a_member_runs_the_call_on_that_provider(
    _openai_key: None, member: Provider, registered: str, lane: str, via: str
) -> None:
    """The request names the member's provider and routes the member's default
    model, whatever is registered, on the keyword and through options."""
    _register(registered)
    expected = make_provider(member)

    if lane == "compose" and not expected.compose_tools_schema:
        # Ollama refuses compose before any request; that it is Ollama's refusal,
        # with OpenAI (which composes) registered, shows the member was used.
        with install(_answer) as installed, pytest.raises(ComposeUnsupportedError, match="Ollama"):
            _ = await _lanes()[lane](_passing(member, via))
        assert installed.requests == []
        return

    with install(_answer) as installed, capture_llm_records() as records:
        _ = await _lanes()[lane](_passing(member, via))

    assert [r.provider for r in installed.requests] == [expected.name]
    assert [r.model for r in installed.requests] == [expected.litellm_model()]
    assert [r.provider for r in records] == [expected.name]


@pytest.mark.parametrize("via", ["keyword", "options"])
def test_a_member_runs_a_sync_call_on_that_provider(_openai_key: None, via: str) -> None:
    expected = make_provider(Provider.OPENAI)
    with install(_answer) as installed:
        result = structured_llm_call_sync(
            "hi", _Schema, feature="t", **_passing(Provider.OPENAI, via)
        )
        _ = text_llm_call_sync("hi", feature="t", provider=Provider.OPENAI)

    assert result.ok is True
    assert [r.provider for r in installed.requests] == [expected.name, expected.name]
    assert [r.model for r in installed.requests] == [expected.litellm_model()] * 2


@pytest.fixture
def _no_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove the Anthropic key for good.

    LiteLLM's import runs ``load_dotenv()``, which would put a key from a
    ``.env`` file above the package back into the environment; importing it
    first means that has already happened when the variable is removed.
    """
    _ = importlib.import_module("llmkit._litellm")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", sorted(_lanes()))
async def test_a_member_that_cannot_be_built_fails_before_reaching_the_registered_provider(
    _no_anthropic_key: None, lane: str
) -> None:
    """With Ollama registered, an Anthropic member and no Anthropic key raises the
    missing-key error before any attempt: the call never runs on Ollama."""
    with install(_answer) as installed:
        with (
            capture_llm_records() as records,
            pytest.raises(ValueError, match="requires an API key"),
        ):
            _ = await _lanes()[lane]({"provider": Provider.ANTHROPIC})

    assert installed.requests == []
    assert records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["keyword", "options"])
@pytest.mark.parametrize("lane", sorted(_lanes()))
async def test_a_provider_string_is_refused_before_any_attempt(lane: str, via: str) -> None:
    """``provider="openai"`` names the member and ``make_provider`` instead of
    failing inside the first attempt."""
    with install(_answer) as installed:
        with (
            capture_llm_records() as records,
            pytest.raises(TypeError, match=r"Provider\.OPENAI.*make_provider\(Provider\.OPENAI"),
        ):
            _ = await _lanes()[lane](_passing("openai", via))

    assert installed.requests == []
    assert records == []


def test_a_string_in_another_case_names_the_member_it_parses_to() -> None:
    with install(_answer), pytest.raises(TypeError, match=r"'Ollama'.*Provider\.OLLAMA"):
        _ = text_llm_call_sync("hi", feature="t", provider="Ollama")  # pyright: ignore[reportArgumentType]


def test_an_unknown_provider_string_lists_every_member() -> None:
    with install(_answer), pytest.raises(TypeError, match="make_provider") as caught:
        _ = text_llm_call_sync("hi", feature="t", provider="olama")  # pyright: ignore[reportArgumentType]

    message = str(caught.value)
    assert all(f"Provider.{member.name}" in message for member in Provider)
