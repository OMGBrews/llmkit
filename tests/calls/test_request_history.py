"""Original prompts and attempted dispatches are separate, owned log data."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from litellm import exceptions as litellm_errors
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ChatMessage,
    RetryPolicy,
    ToolDefinition,
    calls,
    capture_llm_records,
)
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.rate_limiting import GlobalRateLimiter, RateLimitSlot
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from tests._support import OkSchema
from tests.calls.test_input_snapshots import (
    LANES,
    begin,
    finish,
    inputs,
    mutate,
    prepared_prompt,
    provider,
    response,
)


async def test_instructor_repair_history_keeps_each_prepared_request_and_original_prompt() -> None:
    prompt: list[ChatMessage] = [{"role": "user", "content": "original question"}]
    expected = deepcopy(prompt)
    dispatched: list[dict[str, object]] = []

    async def dispatch(**kwargs: object) -> object:
        dispatched.append(deepcopy(kwargs))
        messages = cast("list[dict[str, object]]", kwargs["messages"])
        prompt[0]["content"] = "caller mutation during repair"
        if len(dispatched) == 1:
            # A provider adapter may rewrite its request in place. Instructor
            # must build repair from its own prepared data, not that working copy.
            messages[0]["content"] = "SDK destroyed the schema instructions"
            messages[1]["content"] = "SDK replaced the question"
        else:
            first_messages = cast("list[dict[str, object]]", dispatched[0]["messages"])
            assert messages[:2] == first_messages[:2]
        return response('{"ok": "not a boolean"}' if len(dispatched) == 1 else '{"ok": true}')

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        result = await calls.structured_llm_call(
            prompt, OkSchema, feature="history", provider=provider(), retry=NO_RETRY
        )
    assert result == OkSchema(ok=True)
    assert len(records) == 1
    record = records[0]
    assert record.prompt == expected
    assert [r["repair"] for r in record.requests] == [False, True]
    assert [r["messages"] for r in record.requests] == [r["messages"] for r in dispatched]
    assert len(record.requests[1]["messages"]) > len(record.requests[0]["messages"])
    assert record.requests[0]["messages"][0]["role"] == "system"
    # Both histories are separately owned even where their messages are equal.
    record.requests[1]["messages"][0]["content"] = "edited repair record"
    assert record.requests[0]["messages"] == dispatched[0]["messages"]
    assert record.prompt == expected


class Count(BaseModel):
    n: int


async def test_tool_repair_logs_original_prompt_and_schema_despite_caller_mutation() -> None:
    prompt: list[ChatMessage] = [{"role": "user", "content": "count"}]
    tool = ToolDefinition.from_model("count", Count)
    tools = [tool]
    expected_tools = [deepcopy(tool.to_litellm())]
    answers = 0

    def answer(_request: TransportRequest) -> Reply:
        nonlocal answers
        answers += 1
        if answers == 1:
            prompt[0]["content"] = "changed"
            tools.clear()
            tool.parameters.clear()
            return Reply(tool_calls=[FakeToolCall("count", '{"n": "1"}', id="bad")])
        return Reply(tool_calls=[FakeToolCall("count", '{"n": 1}', id="good")])

    with install(answer) as fake, capture_llm_records() as records:
        result = await calls.tool_llm_call(
            prompt,
            tools,
            feature="history",
            provider=provider(),
            retry=RetryPolicy(max_attempts=2, backoff_base_seconds=0),
        )
    assert result.tool_calls[0].validated == Count(n=1)
    assert [r.prompt for r in records] == [[{"role": "user", "content": "count"}]] * 2
    assert [r.requests[0]["repair"] for r in records] == [False, True]
    assert [r.requests[0]["messages"] for r in records] == [list(r.messages) for r in fake.requests]
    assert [r.requests[0]["tools"] for r in records] == [expected_tools, expected_tools]
    error_content = records[1].requests[0]["messages"][2]["content"]
    assert isinstance(error_content, str)
    assert error_content.startswith("ERROR: ")


async def test_temperature_reissue_history_records_drop_directive_and_allowlisted_settings() -> (
    None
):
    dispatched: list[dict[str, object]] = []
    working: list[dict[str, object]] = []
    p = provider()
    p.completion_kwargs = MagicMock(
        return_value={
            "api_key": "secret",
            "extra_headers": {"Authorization": "secret"},
            "custom_provider_data": "secret",
        }
    )

    async def dispatch(**kwargs: object) -> object:
        dispatched.append(deepcopy(kwargs))
        working.append(kwargs)
        assert kwargs["messages"] == [{"role": "user", "content": "original"}]
        if len(dispatched) == 1:
            cast("list[dict[str, object]]", kwargs["messages"])[0]["content"] = "SDK mutation"
            raise litellm_errors.BadRequestError(
                message="`temperature` is not supported for this model.",
                model="snapshot-model",
                llm_provider="openai",
            )
        return response("done")

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        assert (
            await calls.text_llm_call(
                "original",
                feature="history",
                provider=p,
                temperature=0.25,
                max_tokens=50,
                timeout=10,
                retry=NO_RETRY,
            )
            == "done"
        )
    assert len(dispatched) == 2
    (record,) = records
    assert record.prompt == "original"
    assert len(record.requests) == 2
    before, after = record.requests
    assert before["settings"]["temperature"] == after["settings"]["temperature"] == 0.25
    assert before["settings"]["model"] == "openai/snapshot-model"
    assert before["settings"]["max_tokens"] == 50
    assert before["settings"]["timeout"] == 10
    assert "additional_drop_params" not in before["settings"]
    assert after["settings"]["additional_drop_params"] == ["temperature"]
    assert before["repair"] is after["repair"] is False
    assert "secret" not in str(record.requests)
    assert set(before["settings"]) <= {
        "model",
        "stream",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "timeout",
        "tool_choice",
        "response_format",
        "additional_drop_params",
    }
    cast("list[object]", working[1]["additional_drop_params"]).clear()
    cast("list[dict[str, object]]", working[1]["messages"])[0]["content"] = "late SDK mutation"
    assert after["settings"]["additional_drop_params"] == ["temperature"]
    assert after["messages"] == [{"role": "user", "content": "original"}]


@pytest.mark.parametrize("lane", LANES)
async def test_failure_retains_attempted_dispatch_history(lane: str) -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)

    async def dispatch(**_kwargs: object) -> object:
        mutate(prompt, tools, "after dispatch")
        raise RuntimeError("offline provider failure")

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        with pytest.raises(Exception, match="offline provider failure"):
            _ = await finish(begin(lane, prompt, tools))
    assert len(records) == 1
    assert records[0].error is not None
    assert records[0].prompt == expected
    assert len(records[0].requests) == 1
    assert records[0].requests[0]["messages"][-3:] == prepared_prompt(lane, expected)


@pytest.mark.parametrize("lane", LANES)
async def test_predispatch_failure_has_empty_history(lane: str) -> None:
    original = GlobalRateLimiter.acquire_async

    @asynccontextmanager
    async def refused(name: str) -> AsyncGenerator[RateLimitSlot]:
        if name == "OpenAI":
            raise RuntimeError("admission refused")
        async with original(name) as slot:
            yield slot

    prompt, tools = inputs()
    with (
        patch.object(GlobalRateLimiter, "acquire_async", refused),
        capture_llm_records() as records,
    ):
        with pytest.raises(RuntimeError, match="admission refused"):
            _ = await finish(begin(lane, prompt, tools))
    assert len(records) == 1
    assert records[0].requests == []
    assert records[0].prompt == prompt


@pytest.mark.parametrize("lane", ("text", "structured", "tool"))
async def test_cancellation_retains_started_dispatch(lane: str) -> None:
    prompt, tools = inputs()
    entered = asyncio.Event()
    never = asyncio.Event()

    async def dispatch(**_kwargs: object) -> object:
        entered.set()
        _ = await never.wait()
        return response()

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        task = asyncio.create_task(finish(begin(lane, prompt, tools)))
        _ = await asyncio.wait_for(entered.wait(), 5)
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await task
    assert records[0].error == "CancelledError"
    assert len(records[0].requests) == 1
    assert records[0].requests[0]["messages"][-3:] == prepared_prompt(lane, prompt)


async def test_zero_timeout_keeps_dispatch_started_before_first_suspension() -> None:
    never = asyncio.Event()

    async def dispatch(**_kwargs: object) -> object:
        _ = await never.wait()
        return response()

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        with pytest.raises(TimeoutError):
            _ = await calls.text_llm_call(
                "original", feature="history", provider=provider(), timeout=0, retry=NO_RETRY
            )
    assert records[0].error is not None
    assert records[0].error.startswith("TimeoutError:")
    assert len(records[0].requests) == 1


@pytest.mark.parametrize("lane", ("text_stream", "tool_stream"))
async def test_mutation_between_chunks_and_abandonment_preserve_history(lane: str) -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)
    with (
        install(lambda _request: Reply(chunks=["first", "second"])),
        capture_llm_records() as records,
    ):
        if lane == "text_stream":
            stream = calls.text_llm_call_stream(
                prompt, feature="history", provider=provider(), retry=NO_RETRY
            )
        else:
            stream = calls.tool_llm_call_stream(
                prompt, tools, feature="history", provider=provider(), retry=NO_RETRY
            )
        _ = await anext(stream)
        mutate(prompt, tools, "between chunks")
        await stream.aclose()
    assert records[0].error == STREAM_ABANDONED_ERROR
    assert records[0].prompt == expected
    assert len(records[0].requests) == 1
    assert records[0].requests[0]["messages"] == expected


async def test_compose_history_carries_wire_schema_and_keeps_original_prompt() -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)
    with install(lambda _request: Reply.json(OkSchema(ok=True))), capture_llm_records() as records:
        result = await calls.tool_llm_call(
            prompt,
            tools,
            output_schema=OkSchema,
            feature="history",
            provider=provider(),
            retry=NO_RETRY,
        )
    assert result.parsed == OkSchema(ok=True)
    assert records[0].prompt == expected
    assert "response_format" in records[0].requests[0]["settings"]
    assert records[0].requests[0]["messages"] == expected
