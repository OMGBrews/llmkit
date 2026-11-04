"""The per-request ``timeout`` is a deadline llmkit enforces itself.

One deadline per provider round trip, on every call function: the buffered
send, the refused-temperature re-issue, a stream's opening request, and each
wait for a stream's next wire frame. Most tests answer through an
:mod:`llmkit.testing` fake, whose ``Reply(delay=...)`` the deadline now cuts
off. The stream tests that need a frame to arrive and the next one to stall,
or a frame with no text, patch ``litellm.acompletion`` instead: the fake can
delay only before a stream's first chunk.

The records are part of the contract: a cut attempt records a
``TimeoutError`` naming the deadline, never the consumer-abandonment marker,
and a cancelled buffered attempt records ``CancelledError`` rather than
reading as a success that returned nothing.

The structured lane is the one wrinkle, and an existing one: instructor wraps
every transport error that escapes it in ``InstructorRetryException``, so the
cut-off ``TimeoutError`` reaches the caller and the record inside that
wrapper, as a LiteLLM timeout always has. The retry layer unwraps it and
charges it to the transport budget.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import litellm
import pytest
from litellm import exceptions as litellm_errors

from llmkit import (
    NO_RETRY,
    UNSET,
    InMemoryLLMCache,
    LLMCallOptions,
    RetryPolicy,
    ToolDefinition,
    Unset,
    calls,
    capture_llm_records,
    configure_llm_cache,
)
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.exceptions import underlying_provider_error
from llmkit.logging import LLMCallRecord
from llmkit.providers import OpenAIProvider
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from tests._support import OkSchema

_TOOLS = [ToolDefinition("lookup", "", {"type": "object"})]
#: Long enough to be unmistakably past every deadline below, short enough that
#: a broken deadline fails the test in seconds rather than hanging it.
_STALL = 2.0
_DEADLINE = 0.05
_BUFFERED = ("text", "structured", "tool")
_STREAMED = ("text_stream", "tool_stream", "text_stream_sync", "tool_stream_sync")


def _answer(kind: str, *, delay: float = 0.0) -> Reply:
    if kind == "structured":
        return Reply.json(OkSchema(ok=True), delay=delay)
    return Reply.text("ok", delay=delay)


def _slow_first_attempt(request: TransportRequest) -> Reply:
    """Stall attempt 1 past every deadline; answer attempt 2 at once."""
    return _answer(request.kind, delay=_STALL if request.attempt == 1 else 0.0)


async def _run(
    lane: str,
    *,
    timeout: float | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> object:
    """Call *lane*; a stream is drained and its items returned."""
    knobs = {"timeout": timeout, "retry": retry, "options": options}
    if lane == "text":
        return await calls.text_llm_call("hi", feature="deadline", **knobs)  # pyright: ignore[reportArgumentType]
    if lane == "structured":
        return await calls.structured_llm_call("hi", OkSchema, feature="deadline", **knobs)  # pyright: ignore[reportArgumentType]
    if lane == "tool":
        return await calls.tool_llm_call(
            "hi", _TOOLS, feature="deadline", timeout=timeout, retry=retry, options=options
        )
    if lane == "text_stream":
        return [c async for c in calls.text_llm_call_stream("hi", feature="deadline", **knobs)]  # pyright: ignore[reportArgumentType]
    if lane == "tool_stream":
        return [
            e
            async for e in calls.tool_llm_call_stream("hi", _TOOLS, feature="deadline", **knobs)  # pyright: ignore[reportArgumentType]
        ]
    if lane == "text_stream_sync":
        return list(calls.text_llm_call_stream_sync("hi", feature="deadline", **knobs))  # pyright: ignore[reportArgumentType]
    return list(calls.tool_llm_call_stream_sync("hi", _TOOLS, feature="deadline", **knobs))  # pyright: ignore[reportArgumentType]


def _cut(record: LLMCallRecord, deadline: float = _DEADLINE) -> bool:
    """Whether *record* says its attempt was cut off at *deadline*."""
    message = f"no provider response within {deadline:g}s"
    if record.schema == OkSchema.__name__:
        # The structured lane records instructor's wrapper around the timeout.
        return record.error == f"InstructorRetryException: {message}"
    return record.error == f"TimeoutError: {message}"


def _timed_out(exc: BaseException) -> bool:
    root = underlying_provider_error(exc)
    return isinstance(root, TimeoutError) and "no provider response within" in str(root)


# --- Every call function, under the fake ------------------------------------


@pytest.mark.parametrize("lane", _BUFFERED + _STREAMED)
async def test_an_attempt_past_its_timeout_is_cut_off_and_raised(lane: str) -> None:
    with install(lambda r: _answer(r.kind, delay=_STALL)) as installed:
        started = time.monotonic()
        with pytest.raises(Exception) as caught:
            _ = await _run(lane, timeout=_DEADLINE, retry=NO_RETRY)
        elapsed = time.monotonic() - started
    assert _timed_out(caught.value)
    assert elapsed < _STALL / 2, "the call waited for the reply instead of its deadline"
    assert len(installed.requests) == 1


@pytest.mark.parametrize("lane", _BUFFERED + _STREAMED)
async def test_the_default_policy_retries_a_cut_attempt(lane: str) -> None:
    """Builtin ``TimeoutError`` is a transport error; the retry gets a fresh deadline."""
    with install(_slow_first_attempt) as installed, capture_llm_records() as records:
        result = await _run(lane, timeout=_DEADLINE)
    assert result
    assert [r.attempt for r in installed.requests] == [1, 2]
    assert len({r.call_id for r in installed.requests}) == 1
    assert [_cut(record) for record in records] == [True, False]
    assert records[1].error is None


@pytest.mark.parametrize("lane", _BUFFERED + _STREAMED)
async def test_no_timeout_sets_no_deadline(lane: str) -> None:
    with install(lambda r: _answer(r.kind, delay=0.2)):
        assert await _run(lane, retry=NO_RETRY)
        assert await _run(lane, timeout=None, retry=NO_RETRY)


@pytest.mark.parametrize("lane", _BUFFERED)
async def test_options_timeout_is_enforced(lane: str) -> None:
    with install(lambda r: _answer(r.kind, delay=_STALL)), pytest.raises(Exception) as caught:
        _ = await _run(lane, options=LLMCallOptions(timeout=_DEADLINE), retry=NO_RETRY)
    assert _timed_out(caught.value)


async def test_a_provider_completion_kwargs_timeout_is_the_effective_deadline() -> None:
    """The deadline reads the request as sent, so a provider-supplied timeout counts."""
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    original = OpenAIProvider.completion_kwargs

    def _with_timeout(self: OpenAIProvider) -> dict[str, object]:
        return {**original(self), "timeout": _DEADLINE}

    with (
        patch.object(OpenAIProvider, "completion_kwargs", _with_timeout),
        install(lambda r: _answer(r.kind, delay=_STALL)) as installed,
        pytest.raises(TimeoutError),
    ):
        _ = await calls.text_llm_call("hi", feature="deadline", provider=provider, retry=NO_RETRY)
    # The call itself set none: the deadline came from the provider's kwargs.
    assert [r.timeout for r in installed.requests] == [None]


def test_a_sync_call_honours_the_deadline() -> None:
    with install(lambda r: _answer(r.kind, delay=_STALL)), pytest.raises(TimeoutError):
        _ = calls.text_llm_call_sync("hi", feature="deadline", timeout=_DEADLINE, retry=NO_RETRY)


async def test_zero_is_accepted_and_fires_at_the_first_suspension() -> None:
    with install(lambda r: _answer(r.kind, delay=0.01)), pytest.raises(TimeoutError):
        _ = await calls.text_llm_call("hi", feature="deadline", timeout=0.0, retry=NO_RETRY)


# --- Each round trip gets its own budget ------------------------------------


def _model_response(content: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )


async def test_the_refused_temperature_reissue_gets_its_own_deadline() -> None:
    """Each send takes 60% of the deadline: pooled, the two would breach it."""
    deadline = 0.3
    sends: list[dict[str, object]] = []

    async def _acompletion(**kwargs: object) -> object:
        sends.append(kwargs)
        await asyncio.sleep(deadline * 0.6)
        if len(sends) == 1:
            raise litellm_errors.BadRequestError(
                message="`temperature` is not supported for this model.",
                model="gpt-4.1-mini",
                llm_provider="openai",
            )
        return _model_response("ok")

    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    with patch("llmkit._litellm.litellm.acompletion", _acompletion):
        text = await calls.text_llm_call(
            "hi",
            feature="deadline",
            provider=provider,
            temperature=0.2,
            timeout=deadline,
            retry=NO_RETRY,
        )
    assert text == "ok"
    assert len(sends) == 2


async def test_a_litellm_send_that_never_answers_is_cut_off() -> None:
    async def _hang(**_kwargs: object) -> object:
        await asyncio.sleep(_STALL)
        return _model_response("late")

    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    with (
        patch("llmkit._litellm.litellm.acompletion", _hang),
        pytest.raises(Exception) as caught,
    ):
        _ = await calls.structured_llm_call(
            "hi", OkSchema, feature="deadline", provider=provider, timeout=_DEADLINE, retry=NO_RETRY
        )
    assert _timed_out(caught.value)


# --- Streams: first chunk, later chunks, and nothing else -------------------


def _frame(text: str | None, *, finish: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=finish
            )
        ],
        usage=None,
    )


class _ScriptedStream:
    """A LiteLLM stream whose frames arrive after the given waits.

    ``frames`` pairs each frame with the seconds to wait before it. ``closed``
    records whether llmkit closed the stream, as it must after cutting one.
    """

    def __init__(self, frames: list[tuple[float, SimpleNamespace]]) -> None:
        self._frames: list[tuple[float, SimpleNamespace]] = frames
        self.closed: bool = False

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self._frames:
            raise StopAsyncIteration
        wait, frame = self._frames.pop(0)
        await asyncio.sleep(wait)
        return frame

    async def aclose(self) -> None:
        self.closed = True


def _serving(*streams: _ScriptedStream) -> Callable[..., object]:
    queue = list(streams)

    async def _acompletion(**_kwargs: object) -> object:
        return queue.pop(0)

    return _acompletion


@pytest.mark.parametrize("lane", ["text_stream", "tool_stream"])
async def test_a_stream_whose_opening_request_never_answers_is_cut_off(lane: str) -> None:
    opened: list[object] = []

    async def _hang(**kwargs: object) -> object:
        opened.append(kwargs.get("stream"))
        await asyncio.sleep(_STALL)
        return _ScriptedStream([])

    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    started = time.monotonic()
    with (
        patch("llmkit._litellm.litellm.acompletion", _hang),
        capture_llm_records() as records,
        pytest.raises(TimeoutError),
    ):
        if lane == "text_stream":
            async for _chunk in calls.text_llm_call_stream(
                "hi", feature="deadline", provider=provider, timeout=_DEADLINE, retry=NO_RETRY
            ):
                pass
        else:
            async for _event in calls.tool_llm_call_stream(
                "hi",
                _TOOLS,
                feature="deadline",
                provider=provider,
                timeout=_DEADLINE,
                retry=NO_RETRY,
            ):
                pass
    assert time.monotonic() - started < _STALL / 2
    assert opened == [True]
    assert [_cut(record) for record in records] == [True]
    assert records[0].first_chunk_ms is None


async def test_a_stall_after_the_first_chunk_ends_the_stream_unretried() -> None:
    stalled = _ScriptedStream([(0.0, _frame("first")), (_STALL, _frame(" late"))])
    spare = _ScriptedStream([(0.0, _frame("never", finish="stop"))])
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    received: list[str] = []
    with (
        patch("llmkit._litellm.litellm.acompletion", _serving(stalled, spare)),
        capture_llm_records() as records,
    ):
        with pytest.raises(TimeoutError):
            async for chunk in calls.text_llm_call_stream(
                "hi", feature="deadline", provider=provider, timeout=_DEADLINE
            ):
                received.append(chunk)
    assert received == ["first"]
    assert [record.attempt for record in records] == [1]
    assert _cut(records[0])
    assert records[0].error != STREAM_ABANDONED_ERROR
    assert cast("object", records[0].response) == "first"
    assert stalled.closed, "the stalled stream was left open"
    assert spare.closed is False


async def test_a_stall_before_the_first_chunk_is_retried_as_a_fresh_attempt() -> None:
    stalled = _ScriptedStream([(_STALL, _frame("late"))])
    healthy = _ScriptedStream([(0.0, _frame("fresh", finish="stop"))])
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    with (
        patch("llmkit._litellm.litellm.acompletion", _serving(stalled, healthy)),
        capture_llm_records() as records,
    ):
        chunks = [
            c
            async for c in calls.text_llm_call_stream(
                "hi", feature="deadline", provider=provider, timeout=_DEADLINE
            )
        ]
    assert chunks == ["fresh"]
    assert [record.attempt for record in records] == [1, 2]
    assert _cut(records[0])
    assert records[0].first_chunk_ms is None
    assert records[1].error is None
    assert stalled.closed


async def test_a_slow_stream_with_short_gaps_and_a_slow_consumer_is_never_cut() -> None:
    """The deadline bounds each frame wait, not the total or the consumer's time."""
    gap = _DEADLINE * 0.5
    frames = [(gap, _frame(word)) for word in ("a", "b", "c", "d", "e")]
    frames.append((gap, _frame(None, finish="stop")))
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    received: list[str] = []
    started = time.monotonic()
    with patch("llmkit._litellm.litellm.acompletion", _serving(_ScriptedStream(frames))):
        async for chunk in calls.text_llm_call_stream(
            "hi", feature="deadline", provider=provider, timeout=_DEADLINE, retry=NO_RETRY
        ):
            received.append(chunk)
            # The consumer holds each chunk past the deadline.
            await asyncio.sleep(_DEADLINE * 1.5)
    assert received == ["a", "b", "c", "d", "e"]
    assert time.monotonic() - started > _DEADLINE * 4


def test_a_sync_stream_takes_a_deadline_per_step() -> None:
    """Each sync step runs as its own task, so no scope may span a step."""
    gap = _DEADLINE * 0.5
    frames = [(gap, _frame(word)) for word in ("a", "b", "c")]
    frames.append((gap, _frame(None, finish="stop")))
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    received: list[str] = []
    with patch("llmkit._litellm.litellm.acompletion", _serving(_ScriptedStream(frames))):
        for chunk in calls.text_llm_call_stream_sync(
            "hi", feature="deadline", provider=provider, timeout=_DEADLINE, retry=NO_RETRY
        ):
            received.append(chunk)
            time.sleep(_DEADLINE * 1.5)
    assert received == ["a", "b", "c"]


async def test_a_consumer_cancel_still_records_abandonment() -> None:
    """The deadline changes nothing about a stream its own consumer gives up on."""
    with install(lambda _r: Reply(chunks=["a", "b"])), capture_llm_records() as records:
        stream = calls.text_llm_call_stream("hi", feature="deadline", timeout=5.0)
        assert await anext(stream) == "a"
        await stream.aclose()
    assert [record.error for record in records] == [STREAM_ABANDONED_ERROR]


# --- Cancelled buffered attempts --------------------------------------------


@pytest.mark.parametrize("lane", _BUFFERED)
async def test_a_cancelled_buffered_attempt_is_recorded_as_cancelled(lane: str) -> None:
    with install(lambda r: _answer(r.kind, delay=_STALL)), capture_llm_records() as records:
        task = asyncio.create_task(_run(lane, retry=NO_RETRY))
        await asyncio.sleep(0.05)
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(records) == 1
    assert records[0].error == "CancelledError"
    assert cast("object", records[0].response) is None


@pytest.mark.parametrize("lane", _BUFFERED)
async def test_a_host_timeout_around_a_buffered_call_is_recorded(lane: str) -> None:
    """The host-side deadline PIA Maker wrote: its cancellation is not a success."""
    with install(lambda r: _answer(r.kind, delay=_STALL)), capture_llm_records() as records:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                _ = await _run(lane, retry=NO_RETRY)
    assert [record.error for record in records] == ["CancelledError"]


# --- Time to first chunk ----------------------------------------------------


@pytest.mark.parametrize("lane", ["text_stream", "tool_stream"])
async def test_a_streamed_record_carries_time_to_first_chunk(lane: str) -> None:
    with install(lambda _r: Reply.text("ok", delay=0.1)), capture_llm_records() as records:
        assert await _run(lane, retry=NO_RETRY)
    (record,) = records
    assert record.first_chunk_ms is not None
    assert 100 <= record.first_chunk_ms <= record.duration_ms


async def test_a_tool_only_round_carries_time_to_first_chunk() -> None:
    reply = Reply(tool_calls=[FakeToolCall("lookup", {})], delay=0.05)
    with install(lambda _r: reply), capture_llm_records() as records:
        _ = await _run("tool_stream", retry=NO_RETRY)
    (record,) = records
    assert record.first_chunk_ms is not None
    assert record.first_chunk_ms >= 50


async def test_time_to_first_chunk_is_the_first_wire_frame_not_the_first_text() -> None:
    frames = [
        (0.0, _frame(None)),  # role-only: no text, but the provider has answered
        (0.2, _frame("late text", finish="stop")),
    ]
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    with (
        patch("llmkit._litellm.litellm.acompletion", _serving(_ScriptedStream(frames))),
        capture_llm_records() as records,
    ):
        chunks = [
            c
            async for c in calls.text_llm_call_stream(
                "hi", feature="deadline", provider=provider, retry=NO_RETRY
            )
        ]
    assert chunks == ["late text"]
    (record,) = records
    assert record.first_chunk_ms is not None
    assert record.first_chunk_ms < 100 <= record.duration_ms


@pytest.mark.parametrize("lane", _BUFFERED)
async def test_a_buffered_record_has_no_time_to_first_chunk(lane: str) -> None:
    with install(lambda r: _answer(r.kind)), capture_llm_records() as records:
        assert await _run(lane, retry=NO_RETRY)
    assert [record.first_chunk_ms for record in records] == [None]


async def test_a_streamed_cache_hit_has_no_time_to_first_chunk() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _r: Reply.text("ok")), capture_llm_records() as records:
        assert await _run("text_stream")
        assert await _run("text_stream")
    assert [record.cache_hit for record in records] == [False, True]
    assert records[0].first_chunk_ms is not None
    assert records[1].first_chunk_ms is None
