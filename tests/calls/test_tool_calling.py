"""Offline contract tests for the public single-turn tool-call primitive.

The provider is faked through :mod:`llmkit.testing`, so the real tool-argument
parsing runs over the arguments the fake sends. Tests that need tool choice or
compose use a real OpenAI provider object (a dummy key; the fake answers before
any network call) because the configured Ollama provider supports neither.
``test_compose_requests_a_json_schema_response_format`` stays on the private
``acompletion_tools`` seam: the fake's request does not carry
``response_format``.
"""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ComposeUnsupportedError,
    Provider,
    ToolArgumentError,
    ToolCallResult,
    ToolComposeResult,
    ToolDefinition,
    ToolName,
    make_provider,
    tool_llm_call,
    tool_result_message,
)
from llmkit.providers import LLMProviderInterface
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from llmkit.tools import TokenUsage
from tests._support import capturing_sink, provider_mock


class _WeatherArgs(BaseModel):
    city: str


class _Forecast(BaseModel):
    summary: str


def _openai() -> LLMProviderInterface:
    """A real provider that supports tool choice and compose."""
    return make_provider(Provider.OPENAI, api_key="k")


def _calls(*calls: FakeToolCall) -> Reply:
    return Reply(tool_calls=calls)


async def test_tool_call_is_parsed_validated_logged_and_round_trips_history() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reply = Reply(
        tool_calls=[FakeToolCall("weather", '{"city":"Ottawa"}', id="call_1")],
        usage=TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        cost=0.01,
    )
    with install(lambda _request: reply) as installed, capturing_sink() as records:
        result = await tool_llm_call(
            [{"role": "user", "content": "weather?"}],
            [definition],
            feature="assistant",
            provider=_openai(),
            tool_choice=ToolName("weather"),
        )
    assert isinstance(result, ToolCallResult)
    assert result.tool_calls[0].name == "weather"
    assert result.tool_calls[0].validated == _WeatherArgs(city="Ottawa")
    assert result.usage.total_tokens == 7
    assert result.to_message()["tool_calls"][0]["id"] == "call_1"
    assert tool_result_message("call_1", "sunny") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "sunny",
    }
    assert records[0].schema == "tools"
    assert records[0].tools == [definition.to_litellm()]
    assert records[0].tool_calls is not None
    assert [r.tool_choice for r in installed.requests] == [ToolName("weather")]


async def test_unknown_tool_is_a_repairable_tool_error() -> None:
    with install(lambda _request: _calls(FakeToolCall("unoffered", "{}"))):
        with pytest.raises(ToolArgumentError, match="unknown tool"):
            _ = await tool_llm_call(
                "hi",
                [ToolDefinition("weather", "", {"type": "object"})],
                feature="assistant",
                retry=NO_RETRY,
            )


async def test_one_malformed_call_no_longer_discards_its_well_formed_siblings() -> None:
    """Nothing has executed when parsing happens, so dropping the whole round
    for one bad argument string was lossy, not safe: with parallel calls it
    cost every good call and forced a re-ask. The survivors stay, the failure
    is reported beside them, and — the part that keeps the history valid — the
    assistant turn names only the calls the caller can answer."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reply = _calls(
        FakeToolCall("weather", '{"city":"Ottawa"}', id="call_good"),
        FakeToolCall("weather", "{not json", id="call_bad"),
    )
    with install(lambda _request: reply), capturing_sink() as records:
        result = await tool_llm_call("weather?", [definition], feature="assistant", retry=NO_RETRY)
    assert [call.id for call in result.tool_calls] == ["call_good"]
    assert [error.call_id for error in result.invalid_calls] == ["call_bad"]
    assert "not valid JSON" in str(result.invalid_calls[0])
    # The wire history must not name a call the caller was never handed.
    assert [call["id"] for call in result.to_message()["tool_calls"]] == ["call_good"]
    # A round that quietly lost a call must not read as a clean one in the log.
    logged = cast("dict[str, object]", records[0].response)
    assert cast("list[dict[str, object]]", logged["invalid_calls"])[0]["id"] == "call_bad"


async def test_a_round_of_only_malformed_calls_still_raises_for_the_whole_round() -> None:
    """The re-ask contract, unchanged: when nothing in the round survived there
    is nothing to salvage, so it stays a whole-round ``ToolArgumentError`` on
    the validation budget rather than becoming an empty, successful-looking
    turn the caller has to inspect ``invalid_calls`` to notice."""
    reply = _calls(
        FakeToolCall("weather", "{not json", id="call_bad"),
        FakeToolCall("weather", "[]", id="call_worse"),
    )
    with install(lambda _request: reply):
        with pytest.raises(ToolArgumentError, match="not valid JSON"):
            _ = await tool_llm_call(
                "weather?",
                [ToolDefinition.from_model("weather", _WeatherArgs)],
                feature="assistant",
                retry=NO_RETRY,
            )


async def test_compose_re_asks_a_round_of_only_malformed_calls() -> None:
    """Compose keeps the tool lane's re-ask contract: an all-malformed round is
    charged to the validation budget and re-asked, not raised on the first
    attempt because the final-answer budget happened to omit
    ``ToolArgumentError``."""

    def fake(request: TransportRequest) -> Reply:
        if request.attempt == 1:
            return _calls(FakeToolCall("weather", "{not json", id="call_bad"))
        return Reply.text('{"summary":"sunny"}')

    with install(fake) as installed:
        result = await tool_llm_call(
            "weather?",
            [ToolDefinition.from_model("weather", _WeatherArgs)],
            feature="assistant",
            provider=_openai(),
            output_schema=_Forecast,
        )
    assert len(installed.requests) == 2
    assert isinstance(result, ToolComposeResult)
    assert result.parsed == _Forecast(summary="sunny")


async def test_a_clean_round_carries_no_invalid_calls_and_logs_the_old_shape() -> None:
    """The negative half of the pair: the new field is empty and the record's
    ``response`` keeps exactly the keys it had, so the salvage path cannot be
    confirmed by a test that would also pass with the field always populated."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reply = _calls(FakeToolCall("weather", '{"city":"Ottawa"}', id="call_1"))
    with install(lambda _request: reply), capturing_sink() as records:
        result = await tool_llm_call("weather?", [definition], feature="assistant")
    assert result.invalid_calls == []
    assert sorted(cast("dict[str, object]", records[0].response)) == ["text", "tool_calls"]


async def test_a_cancelled_round_is_recorded_as_cancelled_not_as_an_empty_success() -> None:
    """``asyncio.CancelledError`` is a ``BaseException``, so it used to skip the
    error handler while the ``finally`` still wrote a record — leaving
    ``error=None, response=None``, byte-identical to a successful round that
    requested nothing. Both halves are asserted: the record exists (the write
    still happens under cancellation) *and* it names the cancellation, which is
    the half a test for "an error was recorded" would miss."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reached_transport = asyncio.Event()

    async def _hang(_request: TransportRequest) -> Reply:
        reached_transport.set()
        await asyncio.sleep(3600)
        raise AssertionError("cancelled before answering")  # pragma: no cover

    with install(_hang), capturing_sink() as records:
        call = asyncio.ensure_future(
            tool_llm_call("weather?", [definition], feature="assistant", retry=NO_RETRY)
        )
        _ = await reached_transport.wait()
        _ = call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert len(records) == 1
        assert records[0].error == "CancelledError"
        assert cast("object", records[0].response) is None
        assert records[0].schema == "tools"


async def test_compose_validates_final_answer_and_logs_distinct_schema() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reply = Reply.text(
        '{"summary":"sunny"}',
        usage=TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        cost=0.01,
    )
    with install(lambda _request: reply) as installed, capturing_sink() as records:
        result = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=_openai(),
            output_schema=_Forecast,
        )
    assert isinstance(result, ToolComposeResult)
    assert result.parsed == _Forecast(summary="sunny")
    assert result.tool_calls == []
    assert records[0].schema == "tools+_Forecast"
    assert cast("dict[str, object]", records[0].response) == {
        "text": '{"summary":"sunny"}',
        "tool_calls": [],
        "parsed": {"summary": "sunny"},
    }
    assert [r.output_schema for r in installed.requests] == [_Forecast]


@pytest.mark.asyncio
async def test_compose_requests_a_json_schema_response_format() -> None:
    """Compose asks the provider for a JSON-schema ``response_format``. Stays on
    the private seam: the public fake's request does not carry
    ``response_format``. The caller-visible half is
    ``test_compose_validates_final_answer_and_logs_distinct_schema``."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    with patch(
        "llmkit._litellm.acompletion_tools",
        return_value=('{"summary":"sunny"}', [], "stop", (3, 4, 7), 0.01),
    ) as transport:
        _ = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=provider_mock(compose_tools_schema=True),
            output_schema=_Forecast,
        )
    response_format = cast("dict[str, object]", transport.call_args.kwargs["response_format"])
    assert response_format["type"] == "json_schema"


async def test_compose_returns_tool_turn_without_parsing_accompanying_text() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    reply = Reply(
        content="not JSON",
        tool_calls=[FakeToolCall("weather", '{"city":"Ottawa"}', id="call_1")],
    )
    with install(lambda _request: reply):
        result = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=_openai(),
            output_schema=_Forecast,
        )
    assert isinstance(result, ToolComposeResult)
    assert result.parsed is None
    assert result.text == "not JSON"


async def test_compose_refuses_unsupported_provider_before_transport() -> None:
    # The configured Ollama provider does not support compose.
    with install(lambda _request: Reply.text("unreachable")) as installed:
        with pytest.raises(ComposeUnsupportedError, match="two-step"):
            _ = await tool_llm_call(
                "weather?",
                [ToolDefinition("weather", "", {"type": "object"})],
                feature="assistant",
                output_schema=_Forecast,
            )
    assert installed.requests == []
