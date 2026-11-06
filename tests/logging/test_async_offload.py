"""Fast sink controls for managed delivery, capture and stream finalization.

Blocked sinks and delivery limits are exercised in isolated subprocesses in
``test_delivery.py``. These controls prove normal writes still work.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest

from llmkit import LLMCallRecord, LocalYamlLogSink, configure_llm_logging, flush_llm_logs_async
from llmkit import calls as llm_calls
from llmkit.capture import capture_llm_log_paths, record_call_async
from llmkit.logging._health import WriteOutcome
from llmkit.testing import FakeToolCall, Reply, install
from tests._support import OkSchema
from tests._support import make_record as _record


class _ThreadRecordingSink:
    """Sink that records which thread each ``write`` ran on."""

    def __init__(self) -> None:
        self.write_threads: list[int] = []
        self.records: list[LLMCallRecord] = []

    def write(self, record: LLMCallRecord) -> None:
        self.write_threads.append(threading.get_ident())
        self.records.append(record)


@pytest.mark.asyncio
async def test_structured_call_writes_log_off_the_event_loop() -> None:
    """The structured path's sink write runs on a worker thread, not the
    thread driving the event loop."""

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with install(lambda _request: Reply.json(OkSchema(ok=True))):
            _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert len(sink.write_threads) == 1
    assert sink.write_threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_text_call_writes_log_off_the_event_loop() -> None:
    """Same off-loop guarantee for the buffered text path."""

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with install(lambda _request: Reply.text("hello")):
            _ = await llm_calls.text_llm_call("hi", feature="test")
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert sink.write_threads == [w for w in sink.write_threads if w != threading.get_ident()]
    assert len(sink.write_threads) == 1


@pytest.mark.asyncio
async def test_clean_and_abandoned_text_streams_write_off_loop() -> None:
    """Abandonment submits without awaiting; an explicit flush observes delivery."""

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with install(lambda _request: Reply(chunks=["a", "b"])):
            # Clean finish → off-loop write.
            _ = [c async for c in llm_calls.text_llm_call_stream("hi", feature="test")]
            # Abandonment submits to the same worker without waiting.
            stream = llm_calls.text_llm_call_stream("hi", feature="test")
            async for _chunk in stream:
                break
            await stream.aclose()
            assert await flush_llm_logs_async()
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(sink.records) == 2
    clean_thread, abandoned_thread = sink.write_threads
    assert clean_thread != threading.get_ident()
    assert abandoned_thread != threading.get_ident()
    assert sink.records[0].error is None
    assert sink.records[1].error == llm_calls.STREAM_ABANDONED_ERROR


@pytest.mark.asyncio
async def test_completed_and_abandoned_tool_streams_write_off_loop() -> None:
    """A terminal result remains successful even when the consumer then closes."""
    from pydantic import BaseModel

    from llmkit import ToolCallResult, ToolDefinition

    class _Args(BaseModel):
        left: int

    tools = [ToolDefinition.from_model("add", _Args, "Add.")]
    reply = Reply(chunks=["a", "b"], tool_calls=[FakeToolCall("add", {"left": 1})])

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with install(lambda _request: reply):
            # Took the result and stopped → a completed call, off-loop write.
            stream = llm_calls.tool_llm_call_stream("hi", tools, feature="test")
            async for event in stream:
                if isinstance(event, ToolCallResult):
                    break
            await stream.aclose()
            # Left mid-prose → a real abandonment, submitted without waiting.
            early = llm_calls.tool_llm_call_stream("hi", tools, feature="test")
            async for _event in early:
                break
            await early.aclose()
            assert await flush_llm_logs_async()
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(sink.records) == 2
    completed_thread, abandoned_thread = sink.write_threads
    assert completed_thread != threading.get_ident()
    assert abandoned_thread != threading.get_ident()
    assert sink.records[0].error is None
    assert sink.records[1].error == llm_calls.STREAM_ABANDONED_ERROR


@pytest.mark.asyncio
async def test_loop_stays_responsive_during_a_slow_write() -> None:
    """A slow sink write no longer stalls the loop: a heartbeat task keeps
    ticking while the write blocks its worker thread."""

    class _SlowSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            import time

            time.sleep(0.15)

    ticks = [0]
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while not stop.is_set():
            ticks[0] += 1
            await asyncio.sleep(0.01)

    configure_llm_logging(_SlowSink())
    try:
        with install(lambda _request: Reply.json(OkSchema(ok=True))):
            beat = asyncio.create_task(_heartbeat())
            _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
            stop.set()
            await beat
            assert await flush_llm_logs_async()
    finally:
        configure_llm_logging(LocalYamlLogSink())
    # The call waits at most 100 ms while this sink needs 150 ms; the
    # heartbeat progresses during that wait and explicit flush observes completion.
    assert ticks[0] >= 7


@pytest.mark.asyncio
async def test_log_path_is_captured_before_the_call_returns(tmp_path: Path) -> None:
    """A fast file write is captured when observed inside the delivery wait."""

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with install(lambda _request: Reply.json(OkSchema(ok=True))):
            with capture_llm_log_paths() as paths:
                _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
                assert len(paths) == 1
                assert paths[0].exists()
    finally:
        configure_llm_logging(LocalYamlLogSink())


@pytest.mark.asyncio
async def test_cancellation_mid_write_still_lands_the_record(tmp_path: Path) -> None:
    """Cancelling the caller while the write is in flight doesn't lose the
    record: the worker thread isn't cancelled, so the file still lands."""
    write_started = threading.Event()
    release_write = threading.Event()

    class _BlockingSink(LocalYamlLogSink):
        @override
        def write_with_outcome(self, record: LLMCallRecord) -> WriteOutcome:
            write_started.set()
            _ = release_write.wait(timeout=5)
            return super().write_with_outcome(record)

    configure_llm_logging(_BlockingSink(tmp_path))
    try:
        with install(lambda _request: Reply.json(OkSchema(ok=True))):
            call = asyncio.create_task(
                llm_calls.structured_llm_call("hi", OkSchema, feature="test")
            )
            while not write_started.is_set():
                await asyncio.sleep(0.005)
            _ = call.cancel()
            release_write.set()
            with pytest.raises(asyncio.CancelledError):
                _ = await call
        # The worker thread finished the write despite the cancelled awaiter.
        deadline = 50
        while not list(tmp_path.glob("*.yaml")) and deadline:
            await asyncio.sleep(0.01)
            deadline -= 1
        assert len(list(tmp_path.glob("*.yaml"))) == 1
    finally:
        configure_llm_logging(LocalYamlLogSink())


@pytest.mark.asyncio
async def test_delivery_does_not_need_the_default_executor(tmp_path: Path) -> None:
    """The dedicated delivery thread works when executor submission is unavailable."""

    async def _boom_to_thread(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("cannot schedule new futures after shutdown")

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with patch("asyncio.to_thread", side_effect=_boom_to_thread) as offload:
            path = await record_call_async(_record())
        offload.assert_not_called()
        assert path is not None
        assert path.exists()
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_sync_bridge_fast_write_captures_before_return(tmp_path: Path) -> None:
    """A fast managed write preserves path capture across the sync bridge."""

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with install(lambda _request: Reply.json(OkSchema(ok=True))):
            with capture_llm_log_paths() as paths:
                result = llm_calls.structured_llm_call_sync("hi", OkSchema, feature="test")
        assert result.ok is True
        assert len(paths) == 1
        assert paths[0].exists()
    finally:
        configure_llm_logging(LocalYamlLogSink())
