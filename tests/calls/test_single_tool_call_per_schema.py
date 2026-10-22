"""Offline wire tests: one schema, one tool call, on the Claude routes only.

On Bedrock models without native structured output, LiteLLM carries a schema
as a synthetic ``json_tool_call`` tool, and with thinking on the model may
call it twice, splitting one answer across two calls (measured live at 2 in
300 on Sonnet 5.5, 2026-10-21). The Anthropic and Bedrock providers therefore
opt into ``single_tool_call_per_schema``, and ``acompletion_structured`` sends
``parallel_tool_calls=False`` for them.

These tests pin the three properties that make the change safe:

* the parameter reaches LiteLLM on a structured request for both Claude
  providers, through the real instructor client rather than a stub of it;
* no other provider's structured request carries it, swept over every
  ``Provider`` so a new one is covered by default;
* the text, stream and tool-calling transports never send it, because real
  tools exist there and parallel use is the caller's choice.

Mutation check: with the ``parallel_tool_calls`` entry removed from
``acompletion_structured``, the two structured Claude-route tests fail; with
the gate replaced by an unconditional ``{"parallel_tool_calls": False}``, the
sweep fails on every non-Claude provider; with it moved into the shared
``_completion_request_kwargs`` helper, every non-structured lane test fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import pytest

from llmkit import Provider, _litellm, make_provider
from llmkit import calls as llm_calls
from llmkit.providers import LLMProviderInterface
from llmkit.tools import ToolDefinition
from tests._support import OkSchema, quiet_logging

_CLAUDE_ROUTES = frozenset({Provider.ANTHROPIC, Provider.BEDROCK})

_PROVIDER_BUILDERS: dict[Provider, Callable[[], LLMProviderInterface]] = {
    Provider.ANTHROPIC: lambda: make_provider(Provider.ANTHROPIC, api_key="k"),
    Provider.BEDROCK: lambda: make_provider(Provider.BEDROCK, aws_region_name="us-east-1"),
    Provider.DEEPSEEK: lambda: make_provider(Provider.DEEPSEEK, api_key="k"),
    Provider.GOOGLE: lambda: make_provider(Provider.GOOGLE, api_key="k"),
    Provider.OLLAMA: lambda: make_provider(Provider.OLLAMA),
    Provider.OPENAI: lambda: make_provider(Provider.OPENAI, api_key="k"),
    Provider.OPENROUTER: lambda: make_provider(Provider.OPENROUTER, api_key="k"),
    Provider.VERTEX: lambda: make_provider(
        Provider.VERTEX, vertex_project="p", vertex_location="us-central1"
    ),
}


def test_provider_build_table_covers_every_provider() -> None:
    """A new ``Provider`` member must join the sweep below."""
    assert set(_PROVIDER_BUILDERS) == set(Provider)


def _capture_structured_request(provider: LLMProviderInterface) -> dict[str, object]:
    """The kwargs instructor's ``create_with_completion`` received."""
    seen: dict[str, object] = {}

    async def _fake_create_with_completion(**kwargs: object) -> tuple[OkSchema, MagicMock]:
        seen.update(kwargs)
        return OkSchema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat = MagicMock(
        completions=MagicMock(create_with_completion=_fake_create_with_completion)
    )
    with (
        quiet_logging(),
        patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client),
    ):
        parsed = asyncio.run(
            llm_calls.structured_llm_call("hi", OkSchema, feature="test", provider=provider)
        )
    assert parsed.ok is True
    assert "messages" in seen  # the request reached the seam
    return seen


@pytest.mark.parametrize("provider_enum", list(Provider))
def test_structured_request_disables_parallel_tool_calls_only_on_claude_routes(
    provider_enum: Provider,
) -> None:
    """Claude routes send ``parallel_tool_calls=False``; every other route sends nothing."""
    seen = _capture_structured_request(_PROVIDER_BUILDERS[provider_enum]())

    if provider_enum in _CLAUDE_ROUTES:
        assert seen["parallel_tool_calls"] is False
    else:
        assert "parallel_tool_calls" not in seen


def _ok_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"ok": true}'},
            }
        ]
    )


@pytest.mark.parametrize("provider_enum", sorted(_CLAUDE_ROUTES))
def test_parallel_tool_calls_survives_instructor_to_litellm(provider_enum: Provider) -> None:
    """Through the *real* instructor client, the parameter reaches ``litellm.acompletion``.

    The sweep above stops at instructor's entry point; this one stubs only the
    LiteLLM transport, so it would fail if instructor's mode handler dropped
    the kwarg on the way down.
    """
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return _ok_response()

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        parsed, _cost, _usage = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                OkSchema,
                temperature=None,
                model=None,
                provider=_PROVIDER_BUILDERS[provider_enum](),
            )
        )

    assert parsed.ok is True
    assert captured["parallel_tool_calls"] is False


def _tool() -> ToolDefinition:
    return ToolDefinition("lookup", "Look something up.", {"type": "object"})


class _EmptyStream:
    def __aiter__(self) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            for _ in ():
                yield _

        return _gen()


async def _drain(gen: AsyncIterator[object]) -> None:
    async for _ in gen:
        pass


def _text(provider: LLMProviderInterface) -> Awaitable[object]:
    return _litellm.acompletion_text("hi", temperature=None, model=None, provider=provider)


def _stream(provider: LLMProviderInterface) -> Awaitable[object]:
    return _drain(_litellm.astream_text("hi", temperature=None, model=None, provider=provider))


def _tools(provider: LLMProviderInterface) -> Awaitable[object]:
    return _litellm.acompletion_tools(
        "hi", [_tool()], tool_choice=None, temperature=None, model=None, provider=provider
    )


def _compose(provider: LLMProviderInterface) -> Awaitable[object]:
    return _litellm.acompletion_tools(
        "hi",
        [_tool()],
        tool_choice=None,
        temperature=None,
        model=None,
        provider=provider,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "OkSchema", "schema": OkSchema.model_json_schema()},
        },
    )


def _tool_stream(provider: LLMProviderInterface) -> Awaitable[object]:
    return _drain(
        _litellm.astream_tools(
            "hi", [_tool()], tool_choice=None, temperature=None, model=None, provider=provider
        )
    )


_NON_STRUCTURED: dict[str, tuple[Callable[[LLMProviderInterface], Awaitable[object]], bool]] = {
    "text": (_text, False),
    "stream": (_stream, True),
    "tools": (_tools, False),
    "compose": (_compose, False),
    "tool-stream": (_tool_stream, True),
}


@pytest.mark.parametrize("provider_enum", sorted(_CLAUDE_ROUTES))
@pytest.mark.parametrize("lane", list(_NON_STRUCTURED))
def test_non_structured_lanes_never_send_parallel_tool_calls(
    provider_enum: Provider, lane: str
) -> None:
    """Text, stream and every tool-calling transport leave parallel use alone."""
    drive, streams = _NON_STRUCTURED[lane]
    captured: dict[str, object] = {}
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=None,
        _hidden_params={},
    )

    async def _fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        return _EmptyStream() if streams else reply

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        asyncio.run(_run(drive, provider_enum))

    assert "messages" in captured  # the request reached the seam
    assert "parallel_tool_calls" not in captured


async def _run(
    drive: Callable[[LLMProviderInterface], Awaitable[object]], provider_enum: Provider
) -> None:
    _ = await drive(_PROVIDER_BUILDERS[provider_enum]())
