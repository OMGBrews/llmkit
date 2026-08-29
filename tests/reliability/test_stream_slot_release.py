"""Offline tests that streaming frees the rate-limiter slot when it should.

``astream_text`` holds one per-provider concurrency slot for the whole lifetime
of a stream (``async with GlobalRateLimiter.acquire_async(...)``), released in
``acquire_async``'s ``finally``. Both ways a stream can be abandoned — a consumer
that **breaks out** and one whose **task is cancelled** — must free that slot.

``astream_tools`` adds a third case with the opposite shape: the stream has
*not* been abandoned, but the consumer is parked on the terminal
``ToolCallResult`` running its tools, and holding a per-provider slot across
that is exactly the self-inflicted deadlock the limiter exists to prevent. So
its ``async with`` ends when the provider stream is exhausted, before the
terminal yield — asserted here while the generator is still suspended at it.

The pre-existing abandonment tests patched ``astream_text`` itself — the very
function that holds the slot — so no slot ever existed in them and a leak would
pass. Here the fake is injected one level *deeper*, at ``litellm.acompletion``,
so the real ``astream_text`` runs and holds a real slot; under
``max_concurrent=1`` a leak becomes structurally detectable (a subsequent acquire
wedges).

Mutation check (recorded per the resolved-sibling convention): with the
``finally: gate.release()`` in ``GlobalRateLimiter.acquire_async`` commented out,
both text-stream tests fail (the slot never frees, the cap-1 probe times out);
restored, both pass. For the tool-stream test the mutation is moving the
terminal ``yield StreamedToolTurn(...)`` back *inside* ``astream_tools``'s
``async with``: the consumer parked on the result then holds the slot and the
cap-1 probe times out.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import final
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import ToolCallResult, ToolDefinition
from llmkit import calls as llm_calls
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.rate_limiting import GlobalRateLimiter, configure_rate_limit
from tests._support import capturing_sink, quiet_logging


def _stream_provider() -> MagicMock:
    """A fake provider shaped for the streaming transport seam.

    ``reasoning_effort`` is set to ``None`` *explicitly*: a bare MagicMock
    attribute is truthy and would forward a bogus ``reasoning_effort`` kwarg.
    """
    provider = MagicMock()
    provider.name = "openai"
    provider.model = "fake-model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k"})
    provider.litellm_model = MagicMock(return_value="openai/fake")
    provider.reasoning_effort = None
    return provider


@final
class _FakeStream:
    """A plain-class async stream (deliberately *not* a MagicMock: its auto
    attributes would feed ``_total_tokens`` on clean completion).

    Yields one chunk; when *park* is given, it then blocks forever on that unset
    event before a second chunk — so a consumer awaiting the next pull is parked
    inside ``astream_text``, holding the slot, until it is cancelled.
    """

    def __init__(self, *, park: asyncio.Event | None = None) -> None:
        self._park = park

    def __aiter__(self) -> AsyncIterator[MagicMock]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[MagicMock]:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="a"))])
        if self._park is not None:
            _ = await self._park.wait()  # unset → parks until the consumer is cancelled
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="b"))])


async def _pump(iterations: int = 100) -> None:
    """Give asyncio's async-generator finalizer hook time to run.

    ``_stream_once`` iterates ``astream_text`` with a plain ``async for`` (no
    ``aclosing``), so on break-abandonment the slot-holding ``astream_text`` is
    finalized *asynchronously* a few loop iterations after ``aclose()`` returns.
    If this bound is ever hit, do **not** raise it past ~100 — investigate a
    genuine leak (an ``aclosing`` gap in ``_stream_once``) and report it.
    """
    for _ in range(iterations):
        await asyncio.sleep(0)


async def test_abandoned_stream_releases_rate_limit_slot() -> None:
    """A consumer that breaks out of the stream releases the slot it held."""
    configure_rate_limit(max_concurrent=1)
    provider = _stream_provider()

    async def _fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream()

    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        stream = llm_calls.text_llm_call_stream("hi", feature="test", provider=provider)
        first: str | None = None
        async for chunk in stream:
            first = chunk
            break
        assert first == "a"
        # Same-object discipline: capture the exact gate the stream acquired on
        # this loop *before* asserting (a later fetch could build a fresh one).
        gate = GlobalRateLimiter._get_async_gate("openai")
        assert gate._in_flight == 1  # the slot is genuinely held
        await stream.aclose()
        await _pump()
        assert gate._in_flight == 0
        # Positive property: a fresh acquire under cap 1 must not wedge on a
        # leaked slot.
        async with asyncio.timeout(2):
            async with GlobalRateLimiter.acquire_async("openai"):
                pass


async def test_cancelled_stream_consumer_releases_rate_limit_slot() -> None:
    """A cancelled stream *consumer* releases the slot and records the
    abandonment via the ``CancelledError`` arm of ``_stream_once``."""
    configure_rate_limit(max_concurrent=1)
    provider = _stream_provider()
    park = asyncio.Event()
    first_arrived = asyncio.Event()

    async def _fake_acompletion(**_kwargs: object) -> _FakeStream:
        return _FakeStream(park=park)

    async def _consume() -> None:
        async for _chunk in llm_calls.text_llm_call_stream("hi", feature="test", provider=provider):
            first_arrived.set()  # next pull parks on the unset event in the fake

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        task = asyncio.create_task(_consume())
        async with asyncio.timeout(2):
            _ = await first_arrived.wait()
        gate = GlobalRateLimiter._get_async_gate("openai")
        assert gate._in_flight == 1  # slot genuinely held — kills the vacuous pass
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _pump()
        assert gate._in_flight == 0
        async with asyncio.timeout(2):
            async with GlobalRateLimiter.acquire_async("openai"):
                pass

    assert any(record.error == STREAM_ABANDONED_ERROR for record in captured)


class _AdditionArgs(BaseModel):
    left: int
    right: int


@dataclass
class _DeltaFunction:
    name: str | None
    arguments: str | None


@dataclass
class _DeltaCall:
    index: int
    id: str | None
    function: _DeltaFunction


@final
class _FakeToolStream:
    """A stream that emits prose, one complete tool call, then a usage frame.

    A plain class, not a MagicMock, for the reason the text fake gives: auto
    attributes would feed ``_total_tokens`` and make the usage assertions
    meaningless. The final frame carries ``usage`` and **no choices**, which is
    the shape ``stream_options={"include_usage": True}`` actually produces.
    """

    def __aiter__(self) -> AsyncIterator[MagicMock]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[MagicMock]:
        yield _frame(content="thinking")
        # Plain dataclasses, not MagicMocks: ``MagicMock(name=...)`` names the
        # mock instead of setting the attribute, so ``function.name`` would come
        # back a mock and the call would parse as malformed.
        yield _frame(
            tool_calls=[_DeltaCall(0, "call_1", _DeltaFunction("add", '{"left": 2, "right": 3}'))]
        )
        yield _frame(finish_reason="tool_calls")
        usage_frame = MagicMock(choices=[], _hidden_params={})
        usage_frame.usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        yield usage_frame


def _frame(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    frame = MagicMock(choices=[choice], _hidden_params={})
    frame.usage = None
    return frame


async def test_tool_stream_releases_the_slot_before_the_terminal_result() -> None:
    """A consumer holding the assembled turn holds no provider slot.

    The positive property, asserted from *inside* the ``async for`` while the
    generator is suspended at its terminal yield: a fresh acquire under cap 1
    must not wedge. Assert that rather than the mere absence of a leak, because
    a stream that never acquired at all would satisfy the negative form.
    """
    configure_rate_limit(max_concurrent=1)
    provider = _stream_provider()
    add = ToolDefinition.from_model("add", _AdditionArgs, "Add two integer values.")

    async def _fake_acompletion(**_kwargs: object) -> _FakeToolStream:
        return _FakeToolStream()

    with (
        quiet_logging(),
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        gate = GlobalRateLimiter._get_async_gate("openai")
        held_during_text: int | None = None
        held_at_result: int | None = None
        async for event in llm_calls.tool_llm_call_stream(
            "add 2 and 3", [add], feature="test", provider=provider
        ):
            if isinstance(event, ToolCallResult):
                held_at_result = gate._in_flight
                # The whole point: the tools this consumer is about to run can
                # themselves call the provider without deadlocking on cap 1.
                async with asyncio.timeout(2):
                    async with GlobalRateLimiter.acquire_async("openai"):
                        pass
                assert event.usage.total_tokens == 7
            else:
                held_during_text = gate._in_flight

    # Kills the vacuous pass: the slot was genuinely held while text streamed.
    assert held_during_text == 1
    assert held_at_result == 0
