"""Tests for fully per-call hosts that never register a global config source.

The per-call ``provider=`` override is a standout feature for multi-tenant
hosts that inject a different credential per request. A host that passes
``provider=`` on *every* call should not have to register a global
:func:`~llmkit.configure_llm_client` source it never uses — the per-call
pattern must stand on its own: the call runs, and logging resolves the
*effective* provider from the per-call provider rather than from the global
config.

These tests pin that with the global config source explicitly unset (so a
spurious read would raise), while confirming the global-config path is
unchanged — a call with neither a per-call provider nor a registered source
still raises the clear "not configured" ``RuntimeError``.

The provider round trip is faked through :mod:`llmkit.testing`; the per-call
provider objects are real, so they run the real request path up to the fake.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator

import pytest
from pydantic import BaseModel

from llmkit import (
    LLMClientConfig,
    LLMClientNotConfiguredError,
    LlmkitError,
    Provider,
    capture_llm_records,
    configure_llm_client,
    make_provider,
)
from llmkit.calls import (
    structured_llm_call,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_sync,
    tool_llm_call,
    tool_llm_call_stream,
)
from llmkit.providers.base import active_config, client_configured
from llmkit.testing import Reply, install
from llmkit.tools import ToolDefinition


class _Schema(BaseModel):
    ok: bool


_TOOLS = [ToolDefinition("lookup", "Look a thing up.", {"type": "object"})]


@pytest.fixture
def _no_global_config() -> Iterator[None]:
    """Clear the client registration for the duration of the test.

    Other tests may have registered a source in this process; clearing it
    guarantees the per-call path is exercised with genuinely no global config
    available, so a spurious fallback read would surface as the "not
    configured" ``RuntimeError``. Applied on top of the directory's autouse
    Ollama setup, which mutes logging and restores the previous config source
    afterwards.
    """
    _ = configure_llm_client(None)
    yield


@pytest.mark.asyncio
async def test_structured_per_call_only_runs_and_logs_effective_provider(
    _no_global_config: None,
) -> None:
    """A structured call with only a per-call ``provider=`` (no global source)
    runs, and the log records the effective per-call provider + model."""
    provider = make_provider(Provider.OPENAI, api_key="sk-test", model="gpt-4.1")

    with (
        install(lambda _request: Reply.json(_Schema(ok=True), cost=0.0)) as installed,
        capture_llm_records() as records,
    ):
        result = await structured_llm_call("hi", _Schema, feature="t", provider=provider)

    assert result.ok is True
    # The round trip itself went out through the per-call provider.
    assert [r.provider for r in installed.requests] == ["OpenAI"]
    assert len(records) == 1
    # Logging resolved the effective provider purely from the per-call provider —
    # no global source was consulted (there was none).
    assert records[0].provider == "OpenAI"
    assert records[0].model == "gpt-4.1"


@pytest.mark.asyncio
async def test_text_per_call_only_runs_and_logs_effective_provider(
    _no_global_config: None,
) -> None:
    """The text path is per-call-only too: it runs and logs the per-call provider."""
    provider = make_provider(Provider.OPENROUTER, api_key="sk-test", model="x/y")

    with (
        install(lambda _request: Reply.text("hello", cost=0.0)) as installed,
        capture_llm_records() as records,
    ):
        result = await text_llm_call("hi", feature="t", provider=provider)

    assert result == "hello"
    assert [r.provider for r in installed.requests] == ["OpenRouter"]
    assert records[0].provider == "OpenRouter"
    assert records[0].model == "x/y"


@pytest.mark.asyncio
async def test_no_provider_and_no_global_config_still_raises_clearly(
    _no_global_config: None,
) -> None:
    """The global-config path is unchanged: a call with neither a per-call
    provider nor a registered source raises the clear "not configured" error
    (the transport reaches ``build_provider`` -> ``active_config``). The error
    is non-transient, so it propagates immediately rather than being retried."""
    with pytest.raises(LLMClientNotConfiguredError, match="not configured"):
        _ = await structured_llm_call("hi", _Schema, feature="t")


def _unconfigured_lanes() -> dict[str, Callable[[], Awaitable[object]]]:
    async def text_stream() -> object:
        return [chunk async for chunk in text_llm_call_stream("hi", feature="t")]

    async def tool_stream() -> object:
        return [event async for event in tool_llm_call_stream("hi", _TOOLS, feature="t")]

    return {
        "structured": lambda: structured_llm_call("hi", _Schema, feature="t"),
        "text": lambda: text_llm_call("hi", feature="t"),
        "tool": lambda: tool_llm_call("hi", _TOOLS, feature="t"),
        "compose": lambda: tool_llm_call("hi", _TOOLS, feature="t", output_schema=_Schema),
        "text_stream": text_stream,
        "tool_stream": tool_stream,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", sorted(_unconfigured_lanes()))
async def test_unconfigured_call_fails_before_any_attempt(
    _no_global_config: None, lane: str
) -> None:
    """Forgetting to configure fails before any round trip or log record, and
    says so, on every lane — the compose lane included, which used to report
    an unsupported provider instead."""
    with install(lambda _request: Reply.text("unreachable")) as installed:
        with capture_llm_records() as records, pytest.raises(LLMClientNotConfiguredError):
            _ = await _unconfigured_lanes()[lane]()

    assert installed.requests == []
    assert records == []


def test_unconfigured_sync_call_fails_with_the_same_error(_no_global_config: None) -> None:
    with pytest.raises(LLMClientNotConfiguredError, match="configure_llm_client"):
        _ = text_llm_call_sync("hi", feature="t")


def test_not_configured_error_keeps_its_runtime_error_base() -> None:
    assert issubclass(LLMClientNotConfiguredError, RuntimeError)
    assert issubclass(LLMClientNotConfiguredError, LlmkitError)


@pytest.mark.asyncio
async def test_configure_accepts_a_config_instance(_no_global_config: None) -> None:
    """The plain instance is the primary registration form."""
    _ = configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA, model="from-instance"))

    with install(lambda _request: Reply.text("hello")) as installed:
        assert await text_llm_call("hi", feature="t") == "hello"

    assert [r.model for r in installed.requests] == ["ollama_chat/from-instance"]
    assert active_config().model == "from-instance"


def test_configure_still_accepts_a_callable_read_on_each_build(_no_global_config: None) -> None:
    models = iter(["first", "second"])
    _ = configure_llm_client(lambda: LLMClientConfig(provider=Provider.OLLAMA, model=next(models)))

    assert active_config().model == "first"
    assert active_config().model == "second"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registration",
    [
        LLMClientConfig(provider=Provider.OLLAMA, model="instance"),
        lambda: LLMClientConfig(provider=Provider.OLLAMA, model="callable"),
    ],
    ids=["instance", "callable"],
)
async def test_configure_none_clears_the_registration(
    registration: LLMClientConfig | Callable[[], LLMClientConfig],
) -> None:
    """``configure_llm_client(None)`` takes a registration back: a call without
    ``provider=`` then fails as unconfigured before any attempt is made."""
    _ = configure_llm_client(registration)
    assert client_configured()

    _ = configure_llm_client(None)

    assert not client_configured()
    with install(lambda _request: Reply.text("unreachable")) as installed:
        with pytest.raises(LLMClientNotConfiguredError):
            _ = await text_llm_call("hi", feature="t")
    assert installed.requests == []


def test_configure_returns_none_when_nothing_was_registered(_no_global_config: None) -> None:
    assert configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA, model="a")) is None
    assert configure_llm_client(None) is not None
    assert configure_llm_client(None) is None


@pytest.mark.parametrize("form", ["instance", "callable"])
def test_configure_returns_the_previous_source_for_restoring(
    _no_global_config: None, form: str
) -> None:
    """Registering A, then B, then passing back what B's call returned resolves
    A again, whether A was registered as an instance or as a callable."""
    a = LLMClientConfig(provider=Provider.OLLAMA, model="a")
    _ = configure_llm_client(a if form == "instance" else lambda: a)
    returned = configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA, model="b"))
    assert active_config().model == "b"

    assert callable(returned)
    assert not isinstance(returned, LLMClientConfig)
    restored_b = configure_llm_client(returned)

    assert active_config().model == "a"
    assert restored_b is not None
    assert restored_b().model == "b"


def test_restoring_a_none_previous_clears_again(_no_global_config: None) -> None:
    """Passing back a ``None`` previous restores "nothing registered"."""
    previous = configure_llm_client(LLMClientConfig(provider=Provider.OLLAMA, model="a"))
    assert previous is None

    _ = configure_llm_client(previous)

    assert not client_configured()
