"""Input ownership is fixed at execution, before any asynchronous boundary.

The low-level LiteLLM seam deliberately bypasses testing._public_request's
copying: a fake that copies requests must not conceal call-layer aliasing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import cast, override
from unittest.mock import MagicMock, patch

import instructor
import pytest
from litellm.types.utils import ModelResponse

from llmkit import (
    NO_RETRY,
    AssistantToolMessage,
    ChatMessage,
    InMemoryLLMCache,
    LLMCallRecord,
    RetryPolicy,
    ToolDefinition,
    calls,
    capture_llm_records,
    configure_llm_cache,
)
from llmkit.cache import LLMCacheEntry
from llmkit.rate_limiting import GlobalRateLimiter, RateLimitSlot
from tests._support import OkSchema

LANES = ("text", "structured", "tool", "text_stream", "tool_stream")


def provider() -> MagicMock:
    """Every trait read by call preparation, cache keys and transports is explicit."""
    result = MagicMock()
    result.name = "OpenAI"
    result.model = "snapshot-model"
    result.litellm_model = MagicMock(return_value="openai/snapshot-model")
    result.completion_kwargs = MagicMock(return_value={"api_key": "secret-not-for-history"})
    result.native_kwargs = MagicMock(return_value={})
    result.reasoning_kwargs = MagicMock(return_value={})
    result.reasoning_effort = None
    result.instructor_mode = instructor.Mode.JSON
    result.strict_json_schema = False
    result.single_tool_call_per_schema = False
    result.supports_tool_choice = True
    result.compose_tools_schema = True
    return result


def inputs() -> tuple[list[ChatMessage], list[ToolDefinition]]:
    prompt: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "original"},
                {"type": "image_url", "image_url": {"url": "https://example.test/original.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "previous",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "previous", "content": "previous result"},
    ]
    tools = [
        ToolDefinition(
            "lookup", "Look up", {"type": "object", "properties": {"q": {"type": "string"}}}
        )
    ]
    return prompt, tools


def prepared_prompt(lane: str, prompt: Sequence[ChatMessage]) -> list[dict[str, object]]:
    """The fixture's prepared tail, excluding Instructor's schema system turn.

    Instructor converts scalar/null content to text parts in multimodal history.
    The original captured prompt must still preserve its explicit null.
    """
    prepared: list[dict[str, object]] = deepcopy([dict(message) for message in prompt])
    if lane == "structured":
        for message in prepared:
            if message["content"] is None or isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"] or ""}]
    return prepared


def mutate(prompt: list[ChatMessage], tools: list[ToolDefinition], value: str) -> None:
    parts = cast("list[dict[str, object]]", prompt[0]["content"])
    parts[0]["text"] = value
    cast("dict[str, object]", parts[1]["image_url"])["url"] = value
    assistant = cast("AssistantToolMessage", prompt[1])
    previous = assistant["tool_calls"][0]
    cast("dict[str, object]", previous["function"])["arguments"] = value
    tools[0].parameters["properties"] = {value: {"type": "number"}}
    tools.append(ToolDefinition(value, "added", {"type": "object"}))


def response(content: str = '{"ok": true}') -> ModelResponse:
    return ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )


class Stream:
    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        async def frames() -> AsyncIterator[SimpleNamespace]:
            for text in ("first", "second"):
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=None
                        )
                    ],
                    usage=None,
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop"
                    )
                ],
                usage=None,
            )

        return frames()


def begin(
    lane: str,
    prompt: Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    retry: RetryPolicy = NO_RETRY,
) -> Coroutine[object, object, object] | AsyncIterator[object]:
    p = provider()
    if lane == "text":
        return calls.text_llm_call(prompt, feature="snapshot", provider=p, retry=retry)
    if lane == "structured":
        return calls.structured_llm_call(
            prompt, OkSchema, feature="snapshot", provider=p, retry=retry
        )
    if lane == "tool":
        return calls.tool_llm_call(prompt, tools, feature="snapshot", provider=p, retry=retry)
    if lane == "text_stream":
        return calls.text_llm_call_stream(prompt, feature="snapshot", provider=p, retry=retry)
    return calls.tool_llm_call_stream(prompt, tools, feature="snapshot", provider=p, retry=retry)


async def finish(pending: Coroutine[object, object, object] | AsyncIterator[object]) -> object:
    if isinstance(pending, AsyncIterator):
        return [item async for item in pending]
    return await pending


@pytest.mark.parametrize("lane", LANES)
async def test_snapshot_is_taken_at_execution_and_isolated_from_caller_and_transport(
    lane: str,
) -> None:
    prompt, tools = inputs()
    pending = begin(lane, prompt, tools)
    # Constructing a native coroutine/generator does not take ownership yet.
    mutate(prompt, tools, "before execution")
    expected = deepcopy(prompt)
    expected_tools = [deepcopy(tool.to_litellm()) for tool in tools]
    entered, release = asyncio.Event(), asyncio.Event()
    seen: list[dict[str, object]] = []

    async def dispatch(**kwargs: object) -> object:
        seen.append(deepcopy(kwargs))
        entered.set()
        _ = await release.wait()
        # Provider-side working-container mutation must not rewrite history.
        messages = cast("list[dict[str, object]]", kwargs["messages"])
        messages[-1]["content"] = "transport mutation"
        if "tools" in kwargs:
            cast("list[dict[str, object]]", kwargs["tools"]).clear()
        return Stream() if kwargs.get("stream") else response()

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        task = asyncio.create_task(finish(pending))
        _ = await asyncio.wait_for(entered.wait(), 5)
        mutate(prompt, tools, "after execution")
        caller_after = deepcopy(prompt)
        release.set()
        _ = await asyncio.wait_for(task, 5)
    assert prompt == caller_after, "transport touched caller-owned messages"
    assert len(seen) == len(records) == 1
    record = records[0]
    assert record.prompt == expected
    (request,) = record.requests
    assert request["messages"] == seen[0]["messages"]
    assert request["messages"][-3:] == prepared_prompt(lane, expected)
    assert request["repair"] is False
    assert "api_key" not in request["settings"]
    if lane in ("tool", "tool_stream"):
        assert request["tools"] == expected_tools
        assert record.tools == expected_tools
    # Publication and the collector do not retain the caller's live objects.
    mutate(prompt, tools, "after completion")
    assert record.prompt == expected
    assert request["messages"] == seen[0]["messages"]


@pytest.mark.parametrize("lane", LANES)
def test_sync_forms_own_data_before_transport_and_streams_wait_for_iteration(lane: str) -> None:
    prompt, tools = inputs()
    p = provider()
    seen: list[dict[str, object]] = []

    async def dispatch(**kwargs: object) -> object:
        seen.append(deepcopy(kwargs))
        mutate(prompt, tools, "during dispatch")
        return Stream() if kwargs.get("stream") else response()

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        if lane == "text_stream":
            stream = calls.text_llm_call_stream_sync(
                prompt, feature="snapshot", provider=p, retry=NO_RETRY
            )
            mutate(prompt, tools, "before first iteration")
            expected = deepcopy(prompt)
            _ = list(stream)
        elif lane == "tool_stream":
            stream = calls.tool_llm_call_stream_sync(
                prompt, tools, feature="snapshot", provider=p, retry=NO_RETRY
            )
            mutate(prompt, tools, "before first iteration")
            expected = deepcopy(prompt)
            _ = list(stream)
        else:
            expected = deepcopy(prompt)
            if lane == "text":
                _ = calls.text_llm_call_sync(prompt, feature="snapshot", provider=p, retry=NO_RETRY)
            elif lane == "structured":
                _ = calls.structured_llm_call_sync(
                    prompt, OkSchema, feature="snapshot", provider=p, retry=NO_RETRY
                )
            else:
                _ = calls.tool_llm_call_sync(
                    prompt, tools, feature="snapshot", provider=p, retry=NO_RETRY
                )
    assert len(records) == len(seen) == 1
    assert records[0].prompt == expected
    assert records[0].requests[0]["messages"] == seen[0]["messages"]
    assert cast("list[ChatMessage]", seen[0]["messages"])[-3:] == prepared_prompt(lane, expected)


@pytest.mark.parametrize("lane", LANES)
async def test_retry_transport_and_previous_record_cannot_change_the_private_baseline(
    lane: str,
) -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)
    expected_tools = [deepcopy(t.to_litellm()) for t in tools]
    sends = 0
    records: list[LLMCallRecord]

    async def dispatch(**kwargs: object) -> object:
        nonlocal sends
        sends += 1
        assert cast("list[ChatMessage]", kwargs["messages"])[-3:] == prepared_prompt(lane, expected)
        if lane in ("tool", "tool_stream"):
            assert kwargs["tools"] == expected_tools
        if sends == 1:
            mutate(prompt, tools, "caller mutation")
            cast("list[dict[str, object]]", kwargs["messages"])[-1]["content"] = (
                "transport mutation"
            )
            if "tools" in kwargs:
                cast("list[object]", kwargs["tools"]).clear()
        elif sends == 2:
            assert records[0].prompt == expected
            cast("list[ChatMessage]", records[0].prompt).clear()
            records[0].requests[0]["messages"].clear()
            if records[0].tools is not None:
                records[0].tools.clear()
        if sends < 3:
            raise TimeoutError("transient offline failure")
        return Stream() if kwargs.get("stream") else response()

    retry = RetryPolicy(max_attempts=3, validation_max_attempts=1, backoff_base_seconds=0)
    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        _ = await finish(begin(lane, prompt, tools, retry=retry))
    assert sends == 3
    assert [record.attempt for record in records] == [1, 2, 3]
    assert all(record.prompt == expected for record in records[1:])
    assert all(len(record.requests) == 1 for record in records)
    assert records[2].requests[0]["messages"][-3:] == prepared_prompt(lane, expected)


@pytest.mark.parametrize("lane", ("text", "structured", "text_stream"))
async def test_cache_lookup_uses_one_snapshot_for_key_dispatch_and_hit_record(lane: str) -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedCache(InMemoryLLMCache):
        @override
        async def get(self, key: str) -> LLMCacheEntry | None:
            entered.set()
            _ = await release.wait()
            return await super().get(key)

    cache = PausedCache()
    configure_llm_cache(cache)
    sends: list[dict[str, object]] = []

    async def dispatch(**kwargs: object) -> object:
        sends.append(deepcopy(kwargs))
        return Stream() if kwargs.get("stream") else response()

    with patch("llmkit._litellm.litellm.acompletion", dispatch), capture_llm_records() as records:
        task = asyncio.create_task(finish(begin(lane, prompt, tools)))
        _ = await asyncio.wait_for(entered.wait(), 5)
        mutate(prompt, tools, "during cache lookup")
        release.set()
        first = await asyncio.wait_for(task, 5)
        second = await finish(begin(lane, deepcopy(expected), tools))
    if lane == "text_stream":
        assert "".join(cast("list[str]", first)) == "".join(cast("list[str]", second))
    else:
        assert first == second
    assert len(cache) == len(sends) == 1
    assert [record.cache_hit for record in records] == [False, True]
    assert [record.prompt for record in records] == [expected, expected]
    assert records[0].requests[0]["messages"][-3:] == prepared_prompt(lane, expected)
    assert records[1].requests == []


@pytest.mark.parametrize("lane", LANES)
async def test_rate_limit_wait_precedes_dispatch_but_follows_snapshot(lane: str) -> None:
    prompt, tools = inputs()
    expected = deepcopy(prompt)
    expected_tools = [deepcopy(t.to_litellm()) for t in tools]
    entered, release = asyncio.Event(), asyncio.Event()
    original = GlobalRateLimiter.acquire_async

    @asynccontextmanager
    async def paused(name: str) -> AsyncGenerator[RateLimitSlot]:
        entered.set()
        _ = await release.wait()
        async with original(name) as slot:
            yield slot

    async def dispatch(**kwargs: object) -> object:
        assert cast("list[ChatMessage]", kwargs["messages"])[-3:] == prepared_prompt(lane, expected)
        if lane in ("tool", "tool_stream"):
            assert kwargs["tools"] == expected_tools
        return Stream() if kwargs.get("stream") else response()

    with (
        patch.object(GlobalRateLimiter, "acquire_async", paused),
        patch("llmkit._litellm.litellm.acompletion", dispatch),
        capture_llm_records() as records,
    ):
        task = asyncio.create_task(finish(begin(lane, prompt, tools)))
        _ = await asyncio.wait_for(entered.wait(), 5)
        mutate(prompt, tools, "while waiting for admission")
        release.set()
        _ = await asyncio.wait_for(task, 5)
    assert records[0].prompt == expected
    assert len(records[0].requests) == 1


@pytest.mark.parametrize("lane", ("text", "structured"))
async def test_coalesced_follower_keeps_its_snapshot_after_joining(lane: str) -> None:
    from llmkit.cache import read_through
    from llmkit.cache._flight import lead_or_follow

    configure_llm_cache(InMemoryLLMCache())
    leader_prompt, tools = inputs()
    follower_prompt = deepcopy(leader_prompt)
    expected = deepcopy(leader_prompt)
    dispatched, joined, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_join = lead_or_follow
    sends = 0

    def join(key: str) -> tuple[bool, asyncio.Future[LLMCacheEntry | None]]:
        result = original_join(key)
        if not result[0]:
            joined.set()
        return result

    async def dispatch(**_kwargs: object) -> object:
        nonlocal sends
        sends += 1
        dispatched.set()
        _ = await release.wait()
        return response()

    with (
        patch.object(read_through, "lead_or_follow", join),
        patch("llmkit._litellm.litellm.acompletion", dispatch),
        capture_llm_records() as records,
    ):
        leader = asyncio.create_task(finish(begin(lane, leader_prompt, tools)))
        _ = await asyncio.wait_for(dispatched.wait(), 5)
        follower = asyncio.create_task(finish(begin(lane, follower_prompt, tools)))
        _ = await asyncio.wait_for(joined.wait(), 5)
        mutate(follower_prompt, tools, "after joining")
        release.set()
        first, second = await asyncio.wait_for(asyncio.gather(leader, follower), 5)
    assert first == second
    assert sends == 1
    assert [record.cache_hit for record in records] == [False, True]
    assert [record.prompt for record in records] == [expected, expected]
    assert records[1].requests == []
