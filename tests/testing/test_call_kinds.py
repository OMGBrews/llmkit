"""Every call kind reaches an installed fake through the public call functions.

Each test drives a public call function, sync or async, over a real Ollama
provider config with a fake installed through :mod:`llmkit.testing` and
nothing patched: the same setup a downstream project uses. The streaming kinds
have no public sync function, so their sync form runs the stream on the
:func:`llmkit.sync.run_sync` bridge the sync wrappers use.
"""

from __future__ import annotations

import dataclasses
import threading

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    OutputLimitError,
    TextDeltaEvent,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
    capture_llm_records,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_sync,
    tool_llm_call,
    tool_llm_call_stream,
    tool_llm_call_sync,
)
from llmkit.sync import run_sync
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install


class Summary(BaseModel):
    title: str
    score: int


class AddArgs(BaseModel):
    left: int
    right: int


ADD = ToolDefinition.from_model("add", AddArgs, description="Add two integers.")


def _answer(request: TransportRequest) -> Reply:
    """A fake that answers each kind the way a provider would."""
    if request.kind == "structured":
        return Reply.json(Summary(title="t", score=3), usage=TokenUsage(5, 7, 12), cost=0.25)
    if request.kind in ("tool", "tool_stream"):
        return Reply(
            content="Adding.",
            tool_calls=[FakeToolCall("add", {"left": 1, "right": 2}, id="call_add")],
            usage=TokenUsage(4, 6, 10),
        )
    return Reply.text("hello there world", usage=TokenUsage(1, 3, 4))


async def _collect_text_stream() -> list[str]:
    return [chunk async for chunk in text_llm_call_stream("say hi", feature="greet")]


async def _collect_tool_stream() -> list[TextDeltaEvent | ToolCallResult]:
    return [event async for event in tool_llm_call_stream("add", [ADD], feature="math")]


def _assert_tool_result(result: ToolCallResult) -> None:
    assert result.text == "Adding."
    assert [(call.id, call.name, call.validated) for call in result.tool_calls] == [
        ("call_add", "add", AddArgs(left=1, right=2))
    ]
    assert result.stop_reason == "tool_calls"
    assert result.usage == TokenUsage(4, 6, 10)


async def test_structured_async() -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        result = await structured_llm_call("summarize", output_schema=Summary, feature="sum")
    assert result.model_dump() == {"title": "t", "score": 3}
    assert [request.kind for request in fake.requests] == ["structured"]
    assert records[0].usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert records[0].approximate_cost == 0.25


def test_structured_sync() -> None:
    with install(_answer) as fake:
        result = structured_llm_call_sync("summarize", output_schema=Summary, feature="sum")
    assert result.model_dump() == {"title": "t", "score": 3}
    assert [request.kind for request in fake.requests] == ["structured"]


async def test_text_async() -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        assert await text_llm_call("say hi", feature="greet") == "hello there world"
    assert [request.kind for request in fake.requests] == ["text"]
    assert records[0].usage == {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}


def test_text_sync() -> None:
    with install(_answer) as fake:
        assert text_llm_call_sync("say hi", feature="greet") == "hello there world"
    assert [request.kind for request in fake.requests] == ["text"]


async def test_tool_async() -> None:
    with install(_answer) as fake:
        result = await tool_llm_call("add", [ADD], feature="math")
    _assert_tool_result(result)
    assert [request.kind for request in fake.requests] == ["tool"]


def test_tool_sync() -> None:
    with install(_answer) as fake:
        result = tool_llm_call_sync("add", [ADD], feature="math")
    _assert_tool_result(result)
    assert [request.kind for request in fake.requests] == ["tool"]


async def test_text_stream_async() -> None:
    with install(_answer) as fake:
        chunks = await _collect_text_stream()
    assert chunks == ["hello ", "there ", "world"]
    assert [request.kind for request in fake.requests] == ["text_stream"]


async def test_text_stream_records_usage_and_cost() -> None:
    """A streamed text call's record carries the reply's usage and cost."""
    reply = Reply(chunks=["hel", "lo"], usage=TokenUsage(2, 5, 7), cost=0.125)
    with install(lambda _request: reply), capture_llm_records() as records:
        chunks = await _collect_text_stream()
    assert chunks == ["hel", "lo"]
    assert records[0].schema == "stream"
    assert records[0].usage == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
    assert records[0].approximate_cost == 0.125


def test_text_stream_sync() -> None:
    with install(_answer) as fake:
        chunks = run_sync(_collect_text_stream())
    assert chunks == ["hello ", "there ", "world"]
    assert [request.kind for request in fake.requests] == ["text_stream"]


async def test_tool_stream_async() -> None:
    with install(_answer) as fake:
        events = await _collect_tool_stream()
    assert events[:-1] == [TextDeltaEvent("Adding.")]
    result = events[-1]
    assert isinstance(result, ToolCallResult)
    _assert_tool_result(result)
    assert [request.kind for request in fake.requests] == ["tool_stream"]


def test_tool_stream_sync() -> None:
    with install(_answer) as fake:
        events = run_sync(_collect_tool_stream())
    result = events[-1]
    assert isinstance(result, ToolCallResult)
    _assert_tool_result(result)
    assert [request.kind for request in fake.requests] == ["tool_stream"]


async def test_explicit_stream_chunks() -> None:
    with install(lambda _request: Reply(chunks=["a", "bc", "d"])):
        chunks = await _collect_text_stream()
        text = await text_llm_call("x", feature="f")
    assert chunks == ["a", "bc", "d"]
    assert text == "abcd"


async def test_async_fake() -> None:
    async def fake(_request: TransportRequest) -> Reply:
        return Reply.text("from a coroutine")

    async with install(fake):
        assert await text_llm_call("x", feature="f") == "from a coroutine"


async def test_request_carries_identity_inputs_and_settings() -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        _ = await tool_llm_call(
            [{"role": "user", "content": "add 1 and 2"}],
            [ADD],
            feature="math",
            label="adder",
            temperature=0.5,
            max_tokens=64,
            reasoning_effort="low",
        )
    (request,) = fake.requests
    assert request.kind == "tool"
    assert (request.feature, request.label) == ("math", "adder")
    assert request.call_id == records[0].call_id
    assert request.attempt == records[0].attempt == 1
    assert request.repair is False
    assert request.prompt == ({"role": "user", "content": "add 1 and 2"},)
    assert request.messages == ({"role": "user", "content": "add 1 and 2"},)
    assert request.tools == (ADD,)
    assert request.tool_choice is None  # Ollama routes accept no tool_choice
    assert request.output_schema is None
    assert request.model == "ollama_chat/fake-model"
    assert request.provider == "Ollama"
    assert (request.temperature, request.max_tokens, request.reasoning_effort) == (0.5, 64, "low")
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.feature = "other"  # pyright: ignore[reportAttributeAccessIssue]  # proving it is frozen


async def test_structured_request_carries_the_schema_and_prompt() -> None:
    with install(_answer) as fake:
        _ = await structured_llm_call("summarize", output_schema=Summary, feature="sum")
    (request,) = fake.requests
    assert request.output_schema is Summary
    assert request.prompt == "summarize"
    assert request.tools == ()


async def test_request_carries_no_credentials() -> None:
    names = {field.name for field in dataclasses.fields(TransportRequest)}
    assert not names & {"api_key", "api_base", "kwargs", "extra_body"}


async def test_schema_repair_is_flagged_within_one_attempt() -> None:
    answers = iter([Reply.text('{"title": "t"}'), Reply.json(Summary(title="t", score=1))])
    with install(lambda _request: next(answers)) as fake:
        result = await structured_llm_call("summarize", output_schema=Summary, feature="sum")
    assert result.score == 1
    assert [(r.attempt, r.repair) for r in fake.requests] == [(1, False), (1, True)]
    assert len(fake.requests[1].messages) > len(fake.requests[0].messages)


async def test_length_truncation_per_kind() -> None:
    truncated = Reply.text('{"title": "t', finish_reason="length")
    with install(lambda _request: truncated):
        with pytest.raises(OutputLimitError):
            _ = await structured_llm_call("x", output_schema=Summary, feature="f", retry=NO_RETRY)
        with pytest.raises(OutputLimitError) as caught:
            _ = await text_llm_call("x", feature="f")
        result = await tool_llm_call("x", [ADD], feature="f")
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == '{"title": "t'
    assert result.stop_reason == "length"


async def test_malformed_tool_arguments_run_the_real_parser() -> None:
    reply = Reply(
        tool_calls=[
            FakeToolCall("add", {"left": 1, "right": 2}),
            FakeToolCall("add", '{"left": 1', id="broken"),
        ]
    )
    with install(lambda _request: reply):
        result = await tool_llm_call("x", [ADD], feature="f")
    assert [call.id for call in result.tool_calls] == ["call_0"]
    assert [error.call_id for error in result.invalid_calls] == ["broken"]


def test_a_thread_the_host_starts_reaches_the_fake() -> None:
    results: list[str] = []

    def worker() -> None:
        results.append(text_llm_call_sync("x", feature="from-thread"))

    with install(lambda _request: Reply.text("threaded")) as fake:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=30)
    assert results == ["threaded"]
    assert [request.feature for request in fake.requests] == ["from-thread"]


async def test_a_fake_must_return_a_reply() -> None:
    with install(lambda _request: "not a reply"), pytest.raises(TypeError, match="Reply"):  # pyright: ignore[reportArgumentType]  # the wrong return is the point
        _ = await text_llm_call("x", feature="f", retry=NO_RETRY)
