"""Text calls raise :class:`~llmkit.OutputLimitError` on a length truncation.

A buffered text call, its sync wrapper and a text stream raise only when the
provider explicitly reports ``finish_reason="length"``. The partial text rides
the error's diagnostics and the call log, nothing truncated is cached, the
default policy sends one request, and a stream that has yielded text is never
replayed. Most tests answer through the public :mod:`llmkit.testing` fake.

The fake turns a ``None`` finish reason into ``"stop"`` and puts the reason on
a frame of its own, so the frame-shape tests patch the wire boundary
(``llmkit._litellm._send`` / ``_stream``) instead, and one test drives the
LiteLLM adapter with a real ``ModelResponse``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import ModelResponse, Usage

from llmkit import (
    InMemoryLLMCache,
    OutputLimitError,
    RetryPolicy,
    TokenUsage,
    _litellm,
    capture_llm_records,
    configure_llm_cache,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_sync,
)
from llmkit._transport import WireChunk, WireRequest, WireResponse, WireStreamEnd, WireUsage
from llmkit.testing import Reply, TransportRequest, install

_OPT_IN = RetryPolicy(retry_on=(OutputLimitError,), backoff_base_seconds=0.0)


async def _drain(collected: list[str], **kwargs: object) -> None:
    stream = text_llm_call_stream("q", feature="f", **kwargs)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat
    async for chunk in stream:
        collected.append(chunk)


@pytest.mark.parametrize("max_tokens", [5, None])
async def test_buffered_truncation_raises_with_partial_text(max_tokens: int | None) -> None:
    reply = Reply.text("half an ans", finish_reason="length", usage=TokenUsage(3, 5, 8))
    with install(lambda _request: reply) as fake, capture_llm_records() as records:
        with pytest.raises(OutputLimitError) as caught:
            _ = await text_llm_call("q", feature="f", max_tokens=max_tokens)
    exc = caught.value
    assert exc.max_tokens == max_tokens
    assert exc.completion_tokens == 5
    assert exc.diagnostics is not None
    assert exc.diagnostics.partial_text == "half an ans"
    assert exc.diagnostics.finish_reason == "length"
    assert (exc.diagnostics.prompt_tokens, exc.diagnostics.total_tokens) == (3, 8)
    # Fail-fast under the default policy: one request, one record.
    assert len(fake.requests) == 1
    [record] = records
    assert cast("object", record.response) is None
    assert record.error is not None and record.error.startswith("OutputLimitError: ")
    assert record.partial_text == "half an ans"
    assert record.output_limit == {
        "finish_reason": "length",
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "reasoning_tokens": None,
    }
    assert record.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


async def test_buffered_truncation_with_no_text_keeps_an_empty_partial() -> None:
    with install(lambda _request: Reply.text("", finish_reason="length")):
        with pytest.raises(OutputLimitError) as caught:
            _ = await text_llm_call("q", feature="f")
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == ""


def test_sync_wrapper_raises_too() -> None:
    with install(lambda _request: Reply.text("cut", finish_reason="length")):
        with pytest.raises(OutputLimitError) as caught:
            _ = text_llm_call_sync("q", feature="f")
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == "cut"


@pytest.mark.parametrize("reason", ["stop", "content_filter", "tool_calls", None])
async def test_other_finish_reasons_return_normally(reason: str | None) -> None:
    reply = Reply.text("whole", finish_reason=reason)
    with install(lambda _request: reply):
        assert await text_llm_call("q", feature="f") == "whole"
        collected: list[str] = []
        await _drain(collected)
    assert "".join(collected) == "whole"


@pytest.mark.parametrize("max_tokens", [4, None])
async def test_stream_yields_everything_then_raises(max_tokens: int | None) -> None:
    reply = Reply(chunks=["par", "tial"], finish_reason="length", usage=TokenUsage(2, 4, 6))
    collected: list[str] = []
    with install(lambda _request: reply) as fake, capture_llm_records() as records:
        with pytest.raises(OutputLimitError) as caught:
            await _drain(collected, max_tokens=max_tokens)
    assert collected == ["par", "tial"]
    exc = caught.value
    assert exc.max_tokens == max_tokens
    assert exc.diagnostics is not None
    assert exc.diagnostics.partial_text == "partial"
    assert exc.diagnostics.completion_tokens == 4
    assert len(fake.requests) == 1
    [record] = records
    # The stream's record keeps the transcript the consumer already received.
    assert cast("object", record.response) == "partial"
    assert record.partial_text == "partial"
    assert record.error is not None and record.error.startswith("OutputLimitError: ")
    assert record.output_limit is not None
    assert record.output_limit["finish_reason"] == "length"
    assert record.usage == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}


async def test_empty_truncated_stream_raises_once_under_the_default_policy() -> None:
    collected: list[str] = []
    with install(lambda _request: Reply(chunks=[], finish_reason="length")) as fake:
        with pytest.raises(OutputLimitError) as caught:
            await _drain(collected)
    assert collected == []
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == ""
    assert len(fake.requests) == 1


async def test_truncated_answers_are_not_cached() -> None:
    configure_llm_cache(InMemoryLLMCache())
    answers = iter(
        [
            Reply.text("cut", finish_reason="length"),
            Reply(chunks=["cu", "t"], finish_reason="length"),
            Reply.text("whole answer"),
        ]
    )
    with install(lambda _request: next(answers)) as fake:
        with pytest.raises(OutputLimitError):
            _ = await text_llm_call("q", feature="f")
        with pytest.raises(OutputLimitError):
            await _drain([])
        assert await text_llm_call("q", feature="f") == "whole answer"
        # The finished answer is stored and replayed across lanes.
        replayed: list[str] = []
        await _drain(replayed)
    assert replayed == ["whole answer"]
    assert len(fake.requests) == 3


async def test_explicit_opt_in_retries_a_buffered_truncation() -> None:
    answers = iter([Reply.text("cut", finish_reason="length"), Reply.text("whole")])
    with install(lambda _request: next(answers)) as fake:
        assert await text_llm_call("q", feature="f", retry=_OPT_IN) == "whole"
    assert len(fake.requests) == 2


async def test_explicit_opt_in_retries_a_stream_that_yielded_nothing() -> None:
    answers = iter([Reply(chunks=[], finish_reason="length"), Reply(chunks=["whole"])])
    collected: list[str] = []
    with install(lambda _request: next(answers)) as fake:
        await _drain(collected, retry=_OPT_IN)
    assert collected == ["whole"]
    assert len(fake.requests) == 2


async def test_a_stream_that_yielded_text_is_never_replayed() -> None:
    def _answer(_request: TransportRequest) -> Reply:
        return Reply(chunks=["half"], finish_reason="length")

    collected: list[str] = []
    with install(_answer) as fake:
        with pytest.raises(OutputLimitError):
            await _drain(collected, retry=_OPT_IN)
    assert collected == ["half"]
    assert len(fake.requests) == 1


# --- Below the fake: frame shapes the public fake cannot produce. ---


def _wire_stream(
    *items: WireChunk | WireStreamEnd,
) -> Callable[[WireRequest], AsyncGenerator[WireChunk | WireStreamEnd]]:
    async def _stream(_request: WireRequest) -> AsyncGenerator[WireChunk | WireStreamEnd]:
        for item in items:
            yield item

    return _stream


async def _wire_drain(items: Sequence[WireChunk | WireStreamEnd]) -> list[str]:
    collected: list[str] = []
    with patch.object(_litellm, "_stream", _wire_stream(*items)):
        await _drain(collected)
    return collected


async def test_a_missing_finish_reason_is_not_a_truncation() -> None:
    with patch.object(_litellm, "_send", AsyncMock(return_value=WireResponse(text="t"))):
        assert await text_llm_call("q", feature="f") == "t"
    assert await _wire_drain([WireChunk(text="t"), WireStreamEnd()]) == ["t"]


async def test_finish_reason_on_the_last_text_frame() -> None:
    items = [WireChunk(text="a"), WireChunk(text="b", finish_reason="length"), WireStreamEnd()]
    collected: list[str] = []
    with patch.object(_litellm, "_stream", _wire_stream(*items)):
        with pytest.raises(OutputLimitError) as caught:
            await _drain(collected)
    assert collected == ["a", "b"]
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == "ab"


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        # Usage on a trailing frame after the finish-only frame wins.
        (
            [
                WireChunk(text="a"),
                WireChunk(finish_reason="length"),
                WireChunk(usage=WireUsage(1, 2, 3)),
                WireStreamEnd(usage=WireUsage(9, 9, 18)),
            ],
            (1, 2, 3),
        ),
        # With none on a frame, the stream's own usage is the fallback.
        (
            [
                WireChunk(text="a"),
                WireChunk(finish_reason="length"),
                WireStreamEnd(WireUsage(4, 5, 9)),
            ],
            (4, 5, 9),
        ),
        # Nothing reported: the counts stay unknown.
        (
            [WireChunk(text="a"), WireChunk(finish_reason="length"), WireStreamEnd()],
            (None, None, None),
        ),
    ],
)
async def test_stream_diagnostics_carry_the_reported_usage(
    items: list[WireChunk | WireStreamEnd], expected: tuple[int | None, ...]
) -> None:
    with patch.object(_litellm, "_stream", _wire_stream(*items)):
        with pytest.raises(OutputLimitError) as caught:
            await _drain([])
    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.partial_text == "a"
    assert (
        diagnostics.prompt_tokens,
        diagnostics.completion_tokens,
        diagnostics.total_tokens,
    ) == expected


async def test_a_later_stop_reason_supersedes_an_earlier_one() -> None:
    items = [WireChunk(text="a", finish_reason="length"), WireChunk(finish_reason="stop")]
    assert await _wire_drain([*items, WireStreamEnd()]) == ["a"]


async def test_the_litellm_adapter_reports_a_length_truncation() -> None:
    response = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "cut off"},
            }
        ],
        usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
    )
    with patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=response)):
        with pytest.raises(OutputLimitError) as caught:
            _ = await text_llm_call("q", feature="f", max_tokens=2)
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.partial_text == "cut off"
    assert caught.value.completion_tokens == 2
