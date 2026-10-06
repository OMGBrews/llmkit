"""A call function re-run by an outer ``with_retries`` joins the loop's logical call.

``LLMCallRecord`` promises one ``call_id`` per *logical* call and a 1-based
``attempt`` within it. The documented wrapper pattern — ``with_retries`` around
a ``retry=NO_RETRY`` call — invokes the call function afresh on every pass, so
these pin that its records still meet the contract internally retried calls
meet:

* one ``call_id`` and attempts ``1..n`` across passes, for every buffered
  family, through the sync bridge, and for both stream families when the
  failure arrives after the first yielded item;
* a call from another task never joins (the nested-retry guard's ownership
  rule);
* two calls per pass stay two logical calls, matched by ``(feature, label)``
  and occurrence within the pass, so a call reached only on a later pass
  starts at attempt 1.

The outside-a-loop baseline lives in ``tests/logging/test_record_enrichment.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llmkit import NO_RETRY, LLMCallRecord, ToolDefinition, capture_llm_records
from llmkit import calls as llm_calls
from llmkit._litellm import StreamedToolTurn
from llmkit.retry import with_retries
from llmkit.sync import run_sync
from tests._support import OkSchema, provider_mock, quiet_logging


def _assert_one_logical_call(records: list[LLMCallRecord], attempts: int) -> None:
    assert [r.attempt for r in records] == list(range(1, attempts + 1))
    call_ids = {r.call_id for r in records}
    assert len(call_ids) == 1
    (call_id,) = call_ids
    assert call_id is not None and len(call_id) == 32


class _Args(BaseModel):
    left: int


_TOOLS = [ToolDefinition.from_model("add", _Args, "Add.")]


def _failing_twice[T](result: T) -> Callable[..., Awaitable[T]]:
    """A transport fake that times out on its first two calls, then returns."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> T:
        calls[0] += 1
        if calls[0] <= 2:
            raise TimeoutError("transient")
        return result

    return _transport


async def _structured() -> object:
    return await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=NO_RETRY)


async def _text() -> object:
    return await llm_calls.text_llm_call("hi", feature="test", retry=NO_RETRY)


async def _tool() -> object:
    return await llm_calls.tool_llm_call("hi", _TOOLS, feature="test", retry=NO_RETRY)


_BUFFERED_FAMILIES: list[tuple[str, object, Callable[[], Awaitable[object]]]] = [
    ("acompletion_structured", (OkSchema(ok=True), None), _structured),
    ("acompletion_text", ("hello", None), _text),
    ("acompletion_tools", ("done", [], "stop", (None, None, None), None), _tool),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "result", "fn"),
    _BUFFERED_FAMILIES,
    ids=["structured", "text", "tool"],
)
async def test_outer_with_retries_joins_passes_under_one_call_id(
    transport: str, result: object, fn: Callable[[], Awaitable[object]]
) -> None:
    """Two transport failures, then a success, under an outer
    ``with_retries(max_attempts=3)``: three records, one ``call_id``,
    attempts 1, 2, 3."""
    with (
        quiet_logging(),
        patch(f"llmkit._litellm.{transport}", side_effect=_failing_twice(result)),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await with_retries(fn, max_attempts=3, retry_on=(TimeoutError,))

    assert len(records) == 3
    _assert_one_logical_call(records, 3)
    assert [r.error is None for r in records] == [False, False, True]


def test_outer_with_retries_joins_passes_through_the_sync_bridge() -> None:
    """``run_sync(with_retries(fn))`` runs the loop and ``fn`` in one bridge
    task, so the passes join exactly as they do on the async path."""
    with (
        quiet_logging(),
        patch(
            "llmkit._litellm.acompletion_structured",
            side_effect=_failing_twice((OkSchema(ok=True), None)),
        ),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = run_sync(with_retries(_structured, max_attempts=3, retry_on=(TimeoutError,)))

    _assert_one_logical_call(records, 3)


@pytest.mark.asyncio
async def test_text_stream_joins_passes_when_failing_after_the_first_chunk() -> None:
    """A mid-stream failure propagates out of the collapsed stream to the outer
    loop, whose next pass opens a new stream that rejoins the same call."""
    calls = [0]

    def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        fail = calls[0] <= 2

        async def _gen() -> AsyncIterator[str]:
            yield "he"
            if fail:
                raise TimeoutError("mid-stream")
            yield "llo"

        return _gen()

    async def _fn() -> list[str]:
        return [
            chunk
            async for chunk in llm_calls.text_llm_call_stream("hi", feature="test", retry=NO_RETRY)
        ]

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_text", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        chunks = await with_retries(_fn, max_attempts=3, retry_on=(TimeoutError,))

    assert chunks == ["he", "llo"]
    _assert_one_logical_call(records, 3)


@pytest.mark.asyncio
async def test_tool_stream_joins_passes_when_failing_after_the_first_event() -> None:
    """The same for the streamed tool lane."""
    calls = [0]

    def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        calls[0] += 1
        fail = calls[0] <= 2

        async def _gen() -> AsyncIterator[object]:
            yield "some prose"
            if fail:
                raise TimeoutError("mid-stream")
            yield StreamedToolTurn(
                raw_calls=[], stop_reason="stop", usage=(None, None, None), approximate_cost=None
            )

        return _gen()

    async def _fn() -> list[object]:
        return [
            event
            async for event in llm_calls.tool_llm_call_stream(
                "hi", _TOOLS, feature="test", retry=NO_RETRY
            )
        ]

    with (
        quiet_logging(),
        patch("llmkit._litellm.astream_tools", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await with_retries(_fn, max_attempts=3, retry_on=(TimeoutError,))

    _assert_one_logical_call(records, 3)


@pytest.mark.asyncio
async def test_call_from_another_task_keeps_its_own_call_id() -> None:
    """A call spawned with ``create_task`` inside ``fn`` runs in its own task, so
    it never joins the loop's logical call — even with the same feature and
    label as the call that does."""
    passes = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    async def _fn() -> OkSchema:
        passes[0] += 1
        _ = await asyncio.create_task(_structured_in_child())
        result = await _structured()
        if passes[0] == 1:
            raise TimeoutError("fail the pass after both calls ran")
        return cast("OkSchema", result)

    async def _structured_in_child() -> OkSchema:
        return await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="test", label="child", retry=NO_RETRY
        )

    with (
        quiet_logging(),
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await with_retries(_fn, max_attempts=2, retry_on=(TimeoutError,))

    children = [r for r in records if r.label == "child"]
    owned = [r for r in records if r.label != "child"]
    assert len(children) == 2
    assert children[0].call_id != children[1].call_id
    assert [r.attempt for r in children] == [1, 1]
    _assert_one_logical_call(owned, 2)
    assert not {r.call_id for r in children} & {r.call_id for r in owned}


@pytest.mark.asyncio
async def test_two_calls_per_pass_stay_two_logical_calls() -> None:
    """Two calls per pass — here with the *same* feature and label — form two
    logical calls, matched across passes by occurrence within the pass."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        calls[0] += 1
        if calls[0] == 2:
            raise TimeoutError("second call of pass 1")
        return "ok", None

    async def _fn() -> None:
        _ = await llm_calls.text_llm_call("first", feature="test", retry=NO_RETRY)
        _ = await llm_calls.text_llm_call("second", feature="test", retry=NO_RETRY)

    with (
        quiet_logging(),
        patch("llmkit._litellm.acompletion_text", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        await with_retries(_fn, max_attempts=2, retry_on=(TimeoutError,))

    # Pass 1: first, second (failed). Pass 2: first, second.
    assert len(records) == 4
    first = [records[0], records[2]]
    second = [records[1], records[3]]
    _assert_one_logical_call(first, 2)
    _assert_one_logical_call(second, 2)
    assert first[0].call_id != second[0].call_id


@pytest.mark.asyncio
async def test_call_first_reached_on_a_later_pass_starts_at_attempt_one() -> None:
    """``attempt`` counts tries of the logical call, not outer passes: a call
    that pass 1 never reached records attempt 1 on pass 2. A differently
    labelled call that runs on only some passes does not shift the others."""
    passes = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "ok", None

    async def _fn() -> None:
        passes[0] += 1
        _ = await llm_calls.text_llm_call("draft", feature="report", label="draft", retry=NO_RETRY)
        if passes[0] == 1:
            raise TimeoutError("rejected before the review call")
        _ = await llm_calls.text_llm_call(
            "review", feature="report", label="review", retry=NO_RETRY
        )

    with (
        quiet_logging(),
        patch("llmkit._litellm.acompletion_text", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        await with_retries(_fn, max_attempts=2, retry_on=(TimeoutError,))

    drafts = [r for r in records if r.label == "draft"]
    reviews = [r for r in records if r.label == "review"]
    _assert_one_logical_call(drafts, 2)
    _assert_one_logical_call(reviews, 1)
    assert drafts[0].call_id != reviews[0].call_id
