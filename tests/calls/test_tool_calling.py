"""Offline contract tests for the public single-turn tool-call primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ComposeUnsupportedError,
    ToolArgumentError,
    ToolCallResult,
    ToolComposeResult,
    ToolDefinition,
    ToolName,
    tool_llm_call,
    tool_result_message,
)
from tests._support import capturing_sink, provider_mock


class _WeatherArgs(BaseModel):
    city: str


class _Forecast(BaseModel):
    summary: str


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _RawCall:
    id: str
    function: _Function


@pytest.mark.asyncio
async def test_tool_call_is_parsed_validated_logged_and_round_trips_history() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    raw = _RawCall("call_1", _Function("weather", '{"city":"Ottawa"}'))
    with (
        patch(
            "llmkit._litellm.acompletion_tools",
            return_value=(None, [raw], "tool_calls", (3, 4, 7), 0.01),
        ) as transport,
        capturing_sink() as records,
    ):
        result = await tool_llm_call(
            [{"role": "user", "content": "weather?"}],
            [definition],
            feature="assistant",
            provider=provider_mock(supports_tool_choice=True),
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
    assert transport.call_args.kwargs["tool_choice"] == ToolName("weather")


@pytest.mark.asyncio
async def test_unknown_tool_is_a_repairable_tool_error() -> None:
    raw = _RawCall("call_1", _Function("unoffered", "{}"))
    with patch(
        "llmkit._litellm.acompletion_tools",
        return_value=(None, [raw], "tool_calls", (None, None, None), None),
    ):
        with pytest.raises(ToolArgumentError, match="unknown tool"):
            _ = await tool_llm_call(
                "hi",
                [ToolDefinition("weather", "", {"type": "object"})],
                feature="assistant",
                provider=provider_mock(),
                retry=NO_RETRY,
            )


@pytest.mark.asyncio
async def test_one_malformed_call_no_longer_discards_its_well_formed_siblings() -> None:
    """Nothing has executed when parsing happens, so dropping the whole round
    for one bad argument string was lossy, not safe: with parallel calls it
    cost every good call and forced a re-ask. The survivors stay, the failure
    is reported beside them, and — the part that keeps the history valid — the
    assistant turn names only the calls the caller can answer."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    good = _RawCall("call_good", _Function("weather", '{"city":"Ottawa"}'))
    bad = _RawCall("call_bad", _Function("weather", "{not json"))
    with (
        patch(
            "llmkit._litellm.acompletion_tools",
            return_value=(None, [good, bad], "tool_calls", (None, None, None), None),
        ),
        capturing_sink() as records,
    ):
        result = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=provider_mock(),
            retry=NO_RETRY,
        )
    assert [call.id for call in result.tool_calls] == ["call_good"]
    assert [error.call_id for error in result.invalid_calls] == ["call_bad"]
    assert "not valid JSON" in str(result.invalid_calls[0])
    # The wire history must not name a call the caller was never handed.
    assert [call["id"] for call in result.to_message()["tool_calls"]] == ["call_good"]
    # A round that quietly lost a call must not read as a clean one in the log.
    logged = cast("dict[str, object]", records[0].response)
    assert cast("list[dict[str, object]]", logged["invalid_calls"])[0]["id"] == "call_bad"


@pytest.mark.asyncio
async def test_a_round_of_only_malformed_calls_still_raises_for_the_whole_round() -> None:
    """The re-ask contract, unchanged: when nothing in the round survived there
    is nothing to salvage, so it stays a whole-round ``ToolArgumentError`` on
    the validation budget rather than becoming an empty, successful-looking
    turn the caller has to inspect ``invalid_calls`` to notice."""
    bad = _RawCall("call_bad", _Function("weather", "{not json"))
    worse = _RawCall("call_worse", _Function("weather", "[]"))
    with patch(
        "llmkit._litellm.acompletion_tools",
        return_value=(None, [bad, worse], "tool_calls", (None, None, None), None),
    ):
        with pytest.raises(ToolArgumentError, match="not valid JSON"):
            _ = await tool_llm_call(
                "weather?",
                [ToolDefinition.from_model("weather", _WeatherArgs)],
                feature="assistant",
                provider=provider_mock(),
                retry=NO_RETRY,
            )


@pytest.mark.asyncio
async def test_a_clean_round_carries_no_invalid_calls_and_logs_the_old_shape() -> None:
    """The negative half of the pair: the new field is empty and the record's
    ``response`` keeps exactly the keys it had, so the salvage path cannot be
    confirmed by a test that would also pass with the field always populated."""
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    raw = _RawCall("call_1", _Function("weather", '{"city":"Ottawa"}'))
    with (
        patch(
            "llmkit._litellm.acompletion_tools",
            return_value=(None, [raw], "tool_calls", (None, None, None), None),
        ),
        capturing_sink() as records,
    ):
        result = await tool_llm_call(
            "weather?", [definition], feature="assistant", provider=provider_mock()
        )
    assert result.invalid_calls == []
    assert sorted(cast("dict[str, object]", records[0].response)) == ["text", "tool_calls"]


@pytest.mark.asyncio
async def test_compose_validates_final_answer_and_logs_distinct_schema() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    with (
        patch(
            "llmkit._litellm.acompletion_tools",
            return_value=(
                '{"summary":"sunny"}',
                [],
                "stop",
                (3, 4, 7),
                0.01,
            ),
        ) as transport,
        capturing_sink() as records,
    ):
        result = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=provider_mock(compose_tools_schema=True),
            output_schema=_Forecast,
        )
    assert isinstance(result, ToolComposeResult)
    assert result.parsed == _Forecast(summary="sunny")
    assert result.tool_calls == []
    assert records[0].schema == "tools+_Forecast"
    assert cast("dict[str, object]", records[0].response) == {
        "text": '{"summary":"sunny"}',
        "tool_calls": [],
        "parsed": _Forecast(summary="sunny"),
    }
    response_format = cast("dict[str, object]", transport.call_args.kwargs["response_format"])
    assert response_format["type"] == "json_schema"


@pytest.mark.asyncio
async def test_compose_returns_tool_turn_without_parsing_accompanying_text() -> None:
    definition = ToolDefinition.from_model("weather", _WeatherArgs, "Look up weather")
    raw = _RawCall("call_1", _Function("weather", '{"city":"Ottawa"}'))
    with patch(
        "llmkit._litellm.acompletion_tools",
        return_value=("not JSON", [raw], "tool_calls", (None, None, None), None),
    ):
        result = await tool_llm_call(
            "weather?",
            [definition],
            feature="assistant",
            provider=provider_mock(compose_tools_schema=True),
            output_schema=_Forecast,
        )
    assert isinstance(result, ToolComposeResult)
    assert result.parsed is None
    assert result.text == "not JSON"


@pytest.mark.asyncio
async def test_compose_refuses_unsupported_provider_before_transport() -> None:
    with patch("llmkit._litellm.acompletion_tools") as transport:
        with pytest.raises(ComposeUnsupportedError, match="two-step"):
            _ = await tool_llm_call(
                "weather?",
                [ToolDefinition("weather", "", {"type": "object"})],
                feature="assistant",
                provider=provider_mock(compose_tools_schema=False),
                output_schema=_Forecast,
            )
    transport.assert_not_called()
