"""Offline contract tests for the public single-turn tool-call primitive."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ToolArgumentError,
    ToolCallResult,
    ToolDefinition,
    ToolName,
    tool_llm_call,
    tool_result_message,
)
from tests._support import capturing_sink, provider_mock


class _WeatherArgs(BaseModel):
    city: str


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
