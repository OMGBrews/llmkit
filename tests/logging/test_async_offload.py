"""Tests for the off-loop offload of sink I/O (``record_call_async``).

Pins the July-2026 audit fix for blocking log I/O on the async hot path:

* every buffered call's record — and a *cleanly finished* stream's — is
  written on a worker thread, never the event loop, while the abandoned
  stream deliberately stays synchronous (suspending while ``GeneratorExit``
  unwinds risks losing the truncation-witness record). Both streaming
  families are covered, because each carries its own copy of that branch;
* the ordering contracts survive the offload: the record and the written
  path are captured before the call returns, and a caller cancelled
  mid-write still gets the record on disk;
* the ``RuntimeError`` fallback keeps late writes (no loop / executor shut
  down) blocking-but-written instead of lost.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest

from llmkit import LLMCallRecord, LocalYamlLogSink, configure_llm_logging
from llmkit import calls as llm_calls
from llmkit.capture import capture_llm_log_paths, record_call_async
from tests._support import OkSchema, provider_mock
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

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert len(sink.write_threads) == 1
    assert sink.write_threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_text_call_writes_log_off_the_event_loop() -> None:
    """Same off-loop guarantee for the buffered text path."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "hello", None

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with (
            patch("llmkit._litellm.acompletion_text", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            _ = await llm_calls.text_llm_call("hi", feature="test")
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert sink.write_threads == [w for w in sink.write_threads if w != threading.get_ident()]
    assert len(sink.write_threads) == 1


@pytest.mark.asyncio
async def test_clean_stream_finish_writes_off_loop_but_abandonment_stays_sync() -> None:
    """The two-branch stream ``finally``: a cleanly consumed stream offloads
    its (largest-payload) write to a worker thread, while an abandoned stream
    writes synchronously on the consumer's thread — suspending while
    ``GeneratorExit`` unwinds could lose the truncation-witness record."""

    def _fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield "a"
            yield "b"

        return _gen()

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with (
            patch("llmkit._litellm.astream_text", side_effect=_fake_stream),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            # Clean finish → off-loop write.
            _ = [c async for c in llm_calls.text_llm_call_stream("hi", feature="test")]
            # Abandonment → synchronous write on this (the consumer's) thread.
            stream = llm_calls.text_llm_call_stream("hi", feature="test")
            async for _chunk in stream:
                break
            await stream.aclose()
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(sink.records) == 2
    clean_thread, abandoned_thread = sink.write_threads
    assert clean_thread != threading.get_ident()
    assert abandoned_thread == threading.get_ident()
    assert sink.records[0].error is None
    assert sink.records[1].error == llm_calls.STREAM_ABANDONED_ERROR


@pytest.mark.asyncio
async def test_tool_stream_offloads_a_completed_turn_and_syncs_an_abandoned_one() -> None:
    """The same two branches on the streaming *tool* lane, which copies them.

    ``calls/tool_stream.py`` does not share ``_stream_once``'s ``finally`` — it
    has its own — so the text lane's evidence says nothing about it. The extra
    thing pinned here is the completion latch's effect on the *record path*: a
    consumer that takes the terminal result and stops must land on the
    off-loop, error-free branch, not the synchronous abandonment one.
    """
    from pydantic import BaseModel

    from llmkit import ToolCallResult, ToolDefinition
    from llmkit._litellm import StreamedToolTurn

    class _Args(BaseModel):
        left: int

    tools = [ToolDefinition.from_model("add", _Args, "Add.")]

    @dataclass
    class _Function:
        name: str
        arguments: str

    @dataclass
    class _RawCall:
        id: str
        function: _Function

    def _fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            yield "a"
            yield "b"
            yield StreamedToolTurn(
                raw_calls=[_RawCall("call_1", _Function("add", '{"left": 1}'))],
                stop_reason="tool_calls",
                usage=(1, 2, 3),
                approximate_cost=None,
            )

        return _gen()

    sink = _ThreadRecordingSink()
    configure_llm_logging(sink)
    try:
        with (
            patch("llmkit._litellm.astream_tools", side_effect=_fake_stream),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            # Took the result and stopped → a completed call, off-loop write.
            stream = llm_calls.tool_llm_call_stream("hi", tools, feature="test")
            async for event in stream:
                if isinstance(event, ToolCallResult):
                    break
            await stream.aclose()
            # Left mid-prose → a real abandonment, synchronous write.
            early = llm_calls.tool_llm_call_stream("hi", tools, feature="test")
            async for _event in early:
                break
            await early.aclose()
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(sink.records) == 2
    completed_thread, abandoned_thread = sink.write_threads
    assert completed_thread != threading.get_ident()
    assert abandoned_thread == threading.get_ident()
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

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    ticks = [0]
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while not stop.is_set():
            ticks[0] += 1
            await asyncio.sleep(0.01)

    configure_llm_logging(_SlowSink())
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            beat = asyncio.create_task(_heartbeat())
            _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
            stop.set()
            await beat
    finally:
        configure_llm_logging(LocalYamlLogSink())
    # A blocked loop would have frozen the heartbeat for the whole 150 ms
    # write; off-loop it keeps ticking (~15 ticks; assert half to be safe).
    assert ticks[0] >= 7


@pytest.mark.asyncio
async def test_log_path_is_captured_before_the_call_returns(tmp_path: Path) -> None:
    """The documented capture contract survives the offload: the written
    path is in the capture list at the moment the call returns."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
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
        def write_returning_path(self, record: LLMCallRecord) -> Path | None:
            write_started.set()
            _ = release_write.wait(timeout=5)
            return super().write_returning_path(record)

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    configure_llm_logging(_BlockingSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
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
async def test_offload_falls_back_to_sync_when_to_thread_unavailable(tmp_path: Path) -> None:
    """A late write hitting a shut-down executor (``RuntimeError``) degrades
    to the blocking seam instead of losing the record."""

    async def _boom_to_thread(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("cannot schedule new futures after shutdown")

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with patch("llmkit.capture.asyncio.to_thread", side_effect=_boom_to_thread):
            path = await record_call_async(_record())
        assert path is not None
        assert path.exists()
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_sync_bridge_call_writes_and_captures_through_the_offload(tmp_path: Path) -> None:
    """A ``*_sync`` call driven over the persistent loop still writes the
    YAML and captures its path before returning — the offload composes with
    the bridge's context propagation."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            patch("llmkit.providers.build_provider", return_value=provider_mock()),
        ):
            with capture_llm_log_paths() as paths:
                result = llm_calls.structured_llm_call_sync("hi", OkSchema, feature="test")
        assert result.ok is True
        assert len(paths) == 1
        assert paths[0].exists()
    finally:
        configure_llm_logging(LocalYamlLogSink())
