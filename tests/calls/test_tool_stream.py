"""Offline contract tests for the streaming tool-call primitive.

The invariants that separate this lane from the two it borrows from:

* it yields a **union** — text events, then exactly one terminal
  ``ToolCallResult`` — where the text stream yields ``str`` and the buffered
  tool lane yields nothing at all;
* argument fragments spread across chunks reassemble by their cumulative index;
* the salvage contract (`invalid_calls`) is the buffered lane's, because both
  lanes now go through one parser;
* a consumer that takes the result and stops has **completed** the call, while
  one that leaves earlier has abandoned it, and the log record tells them apart.

Mutation check (recorded per the resolved-sibling convention): with the
``if not completed:`` guard removed from ``_stream_tools_once``'s abandonment
arm, ``test_taking_the_result_and_stopping_is_a_completed_call_not_an_abandonment``
fails — the idiomatic consumer's record carries ``STREAM_ABANDONED_ERROR``
beside populated ``tool_calls``; restored, it passes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import cast, final
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    TextDeltaEvent,
    ToolArgumentError,
    ToolCallResult,
    ToolDefinition,
    ToolName,
    tool_llm_call_stream,
)
from llmkit._litellm import StreamedToolTurn
from llmkit.calls import STREAM_ABANDONED_ERROR
from tests._support import capturing_sink, provider_mock, quiet_logging


class _AdditionArgs(BaseModel):
    left: int
    right: int


def _add_tool() -> ToolDefinition:
    return ToolDefinition.from_model("add", _AdditionArgs, "Add two integer values.")


@dataclass
class _Function:
    name: str | None
    arguments: str


@dataclass
class _RawCall:
    id: str | None
    function: _Function


def _turn(
    *,
    calls: list[object] | None = None,
    stop_reason: str | None = "tool_calls",
    usage: tuple[int | None, int | None, int | None] = (3, 4, 7),
    cost: float | None = 0.01,
) -> StreamedToolTurn:
    return StreamedToolTurn(
        raw_calls=calls
        if calls is not None
        else [_RawCall("call_1", _Function("add", '{"left": 1, "right": 1}'))],
        stop_reason=stop_reason,
        usage=usage,
        approximate_cost=cost,
    )


def _fake_transport(items: list[object]) -> Callable[..., AsyncIterator[object]]:
    """Patch ``astream_tools`` with a generator yielding *items* in order."""

    def _factory(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            for item in items:
                yield item

        return _gen()

    return _factory


async def test_yields_text_events_then_the_completed_result() -> None:
    """The shape committed to the consumer: deltas, then one terminal result."""
    raw = _RawCall("call_1", _Function("add", '{"left": 2, "right": 3}'))
    with patch(
        "llmkit._litellm.astream_tools",
        side_effect=_fake_transport(["I will ", "add them.", _turn(calls=[raw])]),
    ):
        events = [
            event
            async for event in tool_llm_call_stream(
                "add 2 and 3", [_add_tool()], feature="chat", provider=provider_mock()
            )
        ]

    assert events[:-1] == [
        TextDeltaEvent("I will "),
        TextDeltaEvent("add them."),
    ]
    result = events[-1]
    assert isinstance(result, ToolCallResult)
    assert result.text == "I will add them."
    assert result.stop_reason == "tool_calls"
    assert result.usage.total_tokens == 7
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].validated == _AdditionArgs(left=2, right=3)


async def test_a_tool_only_turn_yields_no_text_events_and_records_no_text() -> None:
    """A forced tool call produces no prose; ``text`` is ``None``, not ``""``."""
    raw = _RawCall("call_1", _Function("add", '{"left": 1, "right": 1}'))
    with patch("llmkit._litellm.astream_tools", side_effect=_fake_transport([_turn(calls=[raw])])):
        events = [
            event
            async for event in tool_llm_call_stream(
                "add", [_add_tool()], feature="chat", provider=provider_mock()
            )
        ]

    assert len(events) == 1
    result = events[0]
    assert isinstance(result, ToolCallResult)
    assert result.text is None


async def test_log_record_carries_the_tool_lane_fields_under_its_own_schema() -> None:
    raw = _RawCall("call_1", _Function("add", '{"left": 2, "right": 3}'))
    with (
        patch(
            "llmkit._litellm.astream_tools", side_effect=_fake_transport(["hi", _turn(calls=[raw])])
        ),
        capturing_sink() as records,
    ):
        _ = [
            event
            async for event in tool_llm_call_stream(
                "add 2 and 3",
                [_add_tool()],
                feature="chat",
                label="panel",
                provider=provider_mock(),
            )
        ]

    assert len(records) == 1
    record = records[0]
    assert record.schema == "tools-stream"
    assert record.label == "panel"
    assert record.error is None
    # The three fields ``build_text_record`` cannot carry, which is why this
    # family builds its record by hand.
    assert record.tools is not None and len(record.tools) == 1
    assert record.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"left": 2, "right": 3}'},
        }
    ]
    assert record.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert record.approximate_cost == 0.01


async def test_taking_the_result_and_stopping_is_a_completed_call_not_an_abandonment() -> None:
    """The completion latch.

    Breaking out after the terminal result is *the* idiomatic consumer shape.
    Without the latch the ``GeneratorExit`` thrown into the suspended terminal
    yield reaches the abandonment arm, and every well-behaved consumer logs an
    error beside populated ``tool_calls`` — a self-contradictory record. Assert
    the positive property (a clean record naming the calls), not merely the
    absence of the marker.
    """
    raw = _RawCall("call_1", _Function("add", '{"left": 2, "right": 3}'))
    with (
        patch(
            "llmkit._litellm.astream_tools", side_effect=_fake_transport(["hi", _turn(calls=[raw])])
        ),
        capturing_sink() as records,
    ):
        result: ToolCallResult | None = None
        stream = tool_llm_call_stream(
            "add 2 and 3", [_add_tool()], feature="chat", provider=provider_mock()
        )
        async for event in stream:
            if isinstance(event, ToolCallResult):
                result = event
                break
        await stream.aclose()

    assert result is not None and len(result.tool_calls) == 1
    assert len(records) == 1
    assert records[0].error is None
    assert records[0].tool_calls is not None and len(records[0].tool_calls) == 1


async def test_abandoning_before_the_result_records_the_partial_transcript() -> None:
    """Leaving mid-prose is a real abandonment and still says so."""
    raw = _RawCall("call_1", _Function("add", "{}"))
    with (
        quiet_logging(),
        patch(
            "llmkit._litellm.astream_tools",
            side_effect=_fake_transport(["one ", "two ", "three", _turn(calls=[raw])]),
        ),
        capturing_sink() as records,
    ):
        stream = tool_llm_call_stream("go", [_add_tool()], feature="chat", provider=provider_mock())
        async for _event in stream:
            break
        await stream.aclose()

    assert len(records) == 1
    assert records[0].error == STREAM_ABANDONED_ERROR
    assert cast("object", records[0].response) == {"text": "one ", "tool_calls": []}
    # No usage is claimed for a turn that never completed.
    assert records[0].usage is None


async def test_partly_malformed_round_keeps_the_good_calls() -> None:
    """The salvage contract reaches the streamed lane through the shared parser."""
    good = _RawCall("call_ok", _Function("add", '{"left": 1, "right": 2}'))
    bad = _RawCall("call_bad", _Function("add", "{not json"))
    with (
        quiet_logging(),
        patch(
            "llmkit._litellm.astream_tools", side_effect=_fake_transport([_turn(calls=[good, bad])])
        ),
    ):
        events = [
            event
            async for event in tool_llm_call_stream(
                "add", [_add_tool()], feature="chat", provider=provider_mock()
            )
        ]

    result = events[-1]
    assert isinstance(result, ToolCallResult)
    assert [call.id for call in result.tool_calls] == ["call_ok"]
    assert [error.call_id for error in result.invalid_calls] == ["call_bad"]


async def test_all_malformed_round_raises_and_is_not_retried() -> None:
    """The whole-round contract holds, but the stream cannot re-ask.

    ``with_retries_stream`` is a transport-only budget, so ``ToolArgumentError``
    reaches the caller after exactly one transport call — the documented
    divergence from the buffered lane's ``tool_validation_budget`` re-ask.
    """
    bad = _RawCall("call_bad", _Function("add", "{not json"))
    factory = _fake_transport([_turn(calls=[bad])])
    calls = 0

    def _counting(*args: object, **kwargs: object) -> AsyncIterator[object]:
        nonlocal calls
        calls += 1
        return factory(*args, **kwargs)

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_tools", side_effect=_counting),
        pytest.raises(ToolArgumentError),
    ):
        _ = [
            event
            async for event in tool_llm_call_stream(
                "add", [_add_tool()], feature="chat", provider=provider_mock()
            )
        ]

    assert calls == 1


async def test_a_failed_attempt_records_the_provider_error() -> None:
    def _boom(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            raise RuntimeError("provider exploded")
            yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

        return _gen()

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_tools", side_effect=_boom),
        capturing_sink() as records,
        pytest.raises(RuntimeError),
    ):
        _ = [
            event
            async for event in tool_llm_call_stream(
                "add", [_add_tool()], feature="chat", provider=provider_mock(), retry=NO_RETRY
            )
        ]

    assert len(records) == 1
    assert records[0].error == "RuntimeError: provider exploded"


# --------------------------------------------------------------------------
# Transport-level: the real ``astream_tools`` over a faked LiteLLM stream.
# --------------------------------------------------------------------------


@final
class _FakeChunk:
    """A stream frame in the shape LiteLLM produces (loose by design)."""

    def __init__(
        self,
        *,
        content: str | None = None,
        tool_calls: list[object] | None = None,
        finish_reason: str | None = None,
        usage: object = None,
    ) -> None:
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason
        self.choices = [choice] if (content or tool_calls or finish_reason) else []
        self.usage = usage
        self._hidden_params: dict[str, object] = {}


@dataclass
class _DeltaFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _DeltaCall:
    index: int
    id: str | None = None
    function: _DeltaFunction = field(default_factory=_DeltaFunction)


@final
class _FakeUsage:
    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


def _stream_provider() -> MagicMock:
    provider = MagicMock()
    provider.name = "openai"
    provider.model = "fake-model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.litellm_model = MagicMock(return_value="openai/fake")
    provider.reasoning_effort = None
    provider.supports_tool_choice = True
    return provider


def _readback(raw: object) -> tuple[object, object, object]:
    """Read an assembled call the way the parser does: id, name, arguments.

    Deliberately ``getattr`` over the attribute shape rather than the private
    ``_StreamedToolCall`` type — the contract the transport owes the parser is
    the shape, not the class, and asserting on the shape is what would catch a
    transport that started emitting dicts.
    """
    function = getattr(raw, "function", None)
    return (
        getattr(raw, "id", None),
        getattr(function, "name", None),
        getattr(function, "arguments", None),
    )


async def _drive_transport(
    chunks: list[_FakeChunk], **kwargs: object
) -> tuple[list[str | StreamedToolTurn], dict[str, object]]:
    """Run the real ``astream_tools`` over *chunks*; return items and request kwargs."""
    from llmkit import _litellm

    seen: dict[str, object] = {}

    @final
    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[_FakeChunk]:
            async def _gen() -> AsyncIterator[_FakeChunk]:
                for chunk in chunks:
                    yield chunk

            return _gen()

    async def _fake_acompletion(**request: object) -> _FakeStream:
        seen.update(request)
        return _FakeStream()

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        items = [
            item
            async for item in _litellm.astream_tools(
                "hi",
                [_add_tool()],
                tool_choice=None,
                temperature=None,
                model=None,
                provider=_stream_provider(),
                **kwargs,  # pyright: ignore[reportArgumentType]
            )
        ]
    return items, seen


async def test_transport_requests_usage_on_the_stream() -> None:
    """The decided divergence from ``astream_text``, asserted positively.

    Measured 2026-08-29 on live Vertex, Anthropic, OpenAI, OpenRouter, Google
    AI Studio and DeepSeek: every one accepts the parameter and reports real
    token counts back, which is what keeps the terminal result's ``usage``, the
    log record and the TPM debit populated.
    """
    _items, request = await _drive_transport([_FakeChunk(content="hi", finish_reason="stop")])

    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    tools_sent = request["tools"]
    assert isinstance(tools_sent, list)
    assert len(cast("list[object]", tools_sent)) == 1


async def test_transport_assembles_argument_fragments_by_cumulative_index() -> None:
    """Two calls, arguments split across frames, ids and names arriving once."""
    chunks = [
        _FakeChunk(tool_calls=[_DeltaCall(0, "call_a", _DeltaFunction("add", '{"left":'))]),
        _FakeChunk(tool_calls=[_DeltaCall(0, None, _DeltaFunction(None, ' 1, "right": 2}'))]),
        _FakeChunk(tool_calls=[_DeltaCall(1, "call_b", _DeltaFunction("add", '{"left": 3,'))]),
        _FakeChunk(tool_calls=[_DeltaCall(1, None, _DeltaFunction(None, ' "right": 4}'))]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    items, _request = await _drive_transport(chunks)

    turn = items[-1]
    assert isinstance(turn, StreamedToolTurn)
    assert turn.stop_reason == "tool_calls"
    assembled = [_readback(call) for call in turn.raw_calls]
    assert assembled == [
        ("call_a", "add", '{"left": 1, "right": 2}'),
        ("call_b", "add", '{"left": 3, "right": 4}'),
    ]


async def test_transport_normalises_a_no_argument_call_to_an_empty_object() -> None:
    """A route that streams no argument fragments still parses like the buffered one."""
    chunks = [
        _FakeChunk(tool_calls=[_DeltaCall(0, "call_a", _DeltaFunction("add", None))]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    items, _request = await _drive_transport(chunks)

    turn = items[-1]
    assert isinstance(turn, StreamedToolTurn)
    assert _readback(turn.raw_calls[0])[2] == "{}"


async def test_transport_reads_usage_from_the_final_choiceless_frame() -> None:
    """``include_usage`` puts usage on a frame that carries no choices at all."""
    chunks = [
        _FakeChunk(content="hi"),
        _FakeChunk(finish_reason="stop"),
        _FakeChunk(usage=_FakeUsage(11, 22, 33)),
    ]
    items, _request = await _drive_transport(chunks)

    turn = items[-1]
    assert isinstance(turn, StreamedToolTurn)
    assert turn.usage == (11, 22, 33)


async def test_transport_rejects_tool_choice_on_a_route_without_support() -> None:
    """The guard duplicated from ``acompletion_tools``.

    Without it the streaming surface would silently accept what the buffered
    one rejects before any request.
    """
    from llmkit import _litellm

    provider = _stream_provider()
    provider.supports_tool_choice = False

    with pytest.raises(ValueError, match="does not support tool_choice"):
        _ = [
            item
            async for item in _litellm.astream_tools(
                "hi",
                [_add_tool()],
                tool_choice=ToolName("add"),
                temperature=None,
                model=None,
                provider=provider,
            )
        ]


async def test_transport_ignores_choiceless_keepalive_frames() -> None:
    """The Gemini empty-``choices`` frame must not raise out of the loop."""
    chunks = [_FakeChunk(), _FakeChunk(content="hi"), _FakeChunk()]
    items, _request = await _drive_transport(chunks)

    assert items[0] == "hi"
    assert isinstance(items[-1], StreamedToolTurn)


async def test_cancelling_the_consumer_records_the_abandonment() -> None:
    """Cancellation mid-stream takes the same honest-record path as a break."""
    park = asyncio.Event()
    first = asyncio.Event()

    def _parking(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            yield "a"
            _ = await park.wait()
            yield "b"  # pragma: no cover - the consumer is cancelled first

        return _gen()

    async def _consume() -> None:
        async for _event in tool_llm_call_stream(
            "go", [_add_tool()], feature="chat", provider=provider_mock()
        ):
            first.set()

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_tools", side_effect=_parking),
        capturing_sink() as records,
    ):
        task = asyncio.create_task(_consume())
        async with asyncio.timeout(2):
            _ = await first.wait()
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert any(record.error == STREAM_ABANDONED_ERROR for record in records)
