"""Sink faults preserve the answer/error and captures on every write path."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from llmkit import (
    NO_RETRY,
    LLMCallRecord,
    LocalYamlLogSink,
    ToolDefinition,
    capture_llm_log_paths,
    capture_llm_records,
    configure_llm_logging,
    flush_llm_logs_async,
    get_logging_health,
    structured_llm_call,
    text_llm_call,
    text_llm_call_stream,
    tool_llm_call,
    tool_llm_call_stream,
)
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.exceptions import underlying_provider_error
from llmkit.testing import Reply, TransportRequest, install
from tests._support import OkSchema


class RaisingSink:
    def write(self, record: LLMCallRecord) -> None:
        _ = record
        raise OSError("sink unavailable")


class InvalidPathSink:
    def write(self, record: LLMCallRecord) -> None:
        _ = record

    def write_returning_path(self, _record: LLMCallRecord) -> Path | None:
        return "invalid"  # pyright: ignore[reportReturnType]  # intentional contract breach


async def unavailable_offload(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("executor shut down")


async def call_lane(lane: str) -> object:
    tools = [ToolDefinition.from_model("check", OkSchema, "Check.")]
    if lane == "structured":
        return await structured_llm_call("hi", OkSchema, feature="audit", retry=NO_RETRY)
    if lane == "text":
        return await text_llm_call("hi", feature="audit", retry=NO_RETRY)
    if lane == "tool":
        return await tool_llm_call("hi", tools, feature="audit", retry=NO_RETRY)
    if lane == "text_stream":
        return [c async for c in text_llm_call_stream("hi", feature="audit", retry=NO_RETRY)]
    return [c async for c in tool_llm_call_stream("hi", tools, feature="audit", retry=NO_RETRY)]


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["structured", "text", "tool", "text_stream", "tool_stream"])
@pytest.mark.parametrize("fault", ["raises", "invalid_path", "filesystem"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_sink_fault_preserves_call_and_capture(
    lane: str, fault: str, fallback: bool, provider_fails: bool, tmp_path: Path
) -> None:
    if fault == "filesystem":
        blocked = tmp_path / "not-a-directory"
        _ = blocked.write_text("occupied")
        configure_llm_logging(LocalYamlLogSink(blocked))
    else:
        configure_llm_logging(RaisingSink() if fault == "raises" else InvalidPathSink())
    original = ValueError("provider rejected request")
    reply = Reply.json(OkSchema(ok=True)) if lane == "structured" else Reply.text("answer")

    def respond(_request: TransportRequest) -> Reply:
        if provider_fails:
            raise original
        return reply

    with install(respond):
        with capture_llm_records() as records, capture_llm_log_paths() as paths:
            with (
                patch("llmkit.capture.asyncio.to_thread", side_effect=unavailable_offload)
                if fallback
                else nullcontext()
            ):
                if provider_fails:
                    with pytest.raises(Exception) as raised:
                        _ = await call_lane(lane)
                    assert underlying_provider_error(raised.value) is original
                else:
                    result = await call_lane(lane)
                    if lane == "structured":
                        assert result == OkSchema(ok=True)
                    elif lane == "text":
                        assert result == "answer"
                    elif lane == "text_stream":
                        assert result == ["answer"]
                    else:
                        assert result is not None
    assert len(records) == 1
    assert (records[0].error is not None) is provider_fails
    assert paths == []
    assert get_logging_health().failed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["text", "tool"])
@pytest.mark.parametrize("fault", ["raises", "invalid_path", "filesystem"])
async def test_abandoned_stream_preserves_witness_with_broken_sink(
    lane: str, fault: str, tmp_path: Path
) -> None:
    if fault == "filesystem":
        blocked = tmp_path / "not-a-directory"
        _ = blocked.write_text("occupied")
        configure_llm_logging(LocalYamlLogSink(blocked))
    else:
        configure_llm_logging(RaisingSink() if fault == "raises" else InvalidPathSink())
    tools = [ToolDefinition.from_model("check", OkSchema, "Check.")]
    with install(lambda _request: Reply(chunks=["a", "b"])):
        with capture_llm_records() as records, capture_llm_log_paths() as paths:
            stream = (
                text_llm_call_stream("hi", feature="audit", retry=NO_RETRY)
                if lane == "text"
                else tool_llm_call_stream("hi", tools, feature="audit", retry=NO_RETRY)
            )
            _ = await anext(stream)
            await stream.aclose()
    assert len(records) == 1
    assert records[0].error == STREAM_ABANDONED_ERROR
    assert paths == []
    assert await flush_llm_logs_async()
    assert get_logging_health().failed == 1
