"""Offline tests that stream abandonment releases the rate-limiter slot.

``astream_text`` holds one per-provider concurrency slot for the whole lifetime
of a stream (``async with GlobalRateLimiter.acquire_async(...)``), released in
``acquire_async``'s ``finally``. Both ways a stream can be abandoned — a consumer
that **breaks out** and one whose **task is cancelled** — must free that slot.

The pre-existing abandonment tests patched ``astream_text`` itself — the very
function that holds the slot — so no slot ever existed in them and a leak would
pass. Here the fake is injected one level *deeper*, at ``litellm.acompletion``,
so the real ``astream_text`` runs and holds a real slot; under
``max_concurrent=1`` a leak becomes structurally detectable (a subsequent acquire
wedges).

Mutation check (recorded per the resolved-sibling convention): with the
``finally: gate.release()`` in ``GlobalRateLimiter.acquire_async`` commented out,
both tests fail (the slot never frees, the cap-1 probe times out); restored, both
pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import final
from unittest.mock import MagicMock, patch

import pytest

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
