"""The sync stream forms are the async streams, iterated synchronously.

``text_llm_call_stream_sync`` and ``tool_llm_call_stream_sync`` hand the async
generator to :func:`llmkit.sync.iterate_sync`, which runs one generator step
per ``next()`` as a task on llmkit's persistent loop. These tests pin what that
bridge owes the caller beyond the signature and the retry path (both in
``test_sync_parity.py``) and the shutdown behavior (in
``test_sync_bridge_drain.py``):

* **Lazy.** A ``next()`` pulls exactly one item; nothing is fetched ahead.
* **Honest abandonment.** Closing the iterator, breaking out of a ``for`` loop
  over it, or dropping it writes the abandoned-stream record the async form
  writes, inside the caller's capture, before the next statement runs.
* **No loop-on-loop.** Creating or advancing a sync stream on the persistent
  loop's own thread raises instead of deadlocking.
* **One context.** The caller's context is captured once, at creation.

The fakes that must suspend mid-stream patch the transport seams
(``litellm.acompletion`` or ``_litellm.astream_tools``), because the public
:mod:`llmkit.testing` fake emits every delta without a suspension point.

Mutation checks, recorded when the bridge landed: copying a fresh context per
step fails the record-parity and capture-at-creation tests (the queue-wait
stamp and the capture list are lost); calling the generator method on the
consumer's thread rather than inside the loop task fails
:func:`test_a_foreign_loop_does_not_adopt_the_stream`; removing the
iterator's finalizer fails the break and drop cases; and dropping the
creation-time loop-thread check fails the creation test. The no-ceiling
mutation (waiting with ``run_sync``'s default) fails the stream lanes of
``test_sync_form_adds_no_whole_call_ceiling`` in ``test_sync_parity.py``.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import cast, final
from unittest.mock import patch

import pytest

from llmkit import (
    STREAM_ABANDONED_ERROR,
    InMemoryLLMCache,
    LLMCallRecord,
    TextDeltaEvent,
    ToolCallResult,
    ToolDefinition,
    capture_llm_records,
    configure_llm_cache,
    text_llm_call_stream,
    text_llm_call_stream_sync,
    tool_llm_call_stream,
    tool_llm_call_stream_sync,
)
from llmkit import sync as sync_module
from llmkit._litellm import StreamedToolTurn
from llmkit.sync import run_sync
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from tests._support import ollama_call_defaults

_TOOLS = [ToolDefinition("lookup", "Look a thing up.", {"type": "object"})]


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """A keyless Ollama config, no cache, muted logging, default limits."""
    with ollama_call_defaults():
        yield


def _answer(request: TransportRequest) -> Reply:
    """Two text deltas on every stream; the tool stream also requests a call."""
    if request.kind == "tool_stream":
        return Reply(content="ab", chunks=["a", "b"], tool_calls=[FakeToolCall("lookup", {})])
    return Reply(content="ab", chunks=["a", "b"])


def _text() -> Generator[object]:
    return text_llm_call_stream_sync("hi", feature="sync-stream")


def _tool() -> Generator[object]:
    return tool_llm_call_stream_sync("hi", _TOOLS, feature="sync-stream")


_STREAMS: dict[str, Callable[[], Generator[object]]] = {"text": _text, "tool": _tool}


def _chunk(text: str, finish_reason: str | None = None) -> SimpleNamespace:
    """One LiteLLM-shaped streaming chunk carrying a text delta."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


@final
class _ProviderStream:
    """A provider stream that records each chunk as it is produced.

    With *park* set, it produces one chunk and then waits on *park* before the
    rest, so a consumer can be caught blocked inside ``next()``.
    """

    def __init__(self, texts: list[str], *, park: asyncio.Event | None = None) -> None:
        self.texts = texts
        self.produced: list[str] = []
        self.parked = threading.Event()
        self._park = park

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[SimpleNamespace]:
        for index, text in enumerate(self.texts):
            if index == 1 and self._park is not None:
                self.parked.set()
                _ = await self._park.wait()
            self.produced.append(text)
            last = index == len(self.texts) - 1
            yield _chunk(text, "stop" if last else None)


def _serving(stream: _ProviderStream) -> AbstractContextManager[object]:
    async def _acompletion(**_kwargs: object) -> _ProviderStream:
        return stream

    return patch("llmkit._litellm.litellm.acompletion", side_effect=_acompletion)


# --- Lazy ------------------------------------------------------------------------


def test_the_text_stream_pulls_one_provider_chunk_per_next() -> None:
    stream = _ProviderStream(["a", "b", "c"])
    with _serving(stream), capture_llm_records() as records:
        chunks = text_llm_call_stream_sync("hi", feature="lazy")
        assert stream.produced == [], "creating the stream ran it"
        assert next(chunks) == "a"
        assert stream.produced == ["a"]
        assert next(chunks) == "b"
        assert stream.produced == ["a", "b"]
        assert list(chunks) == ["c"]
    assert [(record.error, cast("object", record.response)) for record in records] == [
        (None, "abc")
    ]


def test_the_tool_stream_pulls_one_item_per_next() -> None:
    produced: list[str] = []

    def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            for text in ("a", "b"):
                produced.append(text)
                yield text
            produced.append("turn")
            yield StreamedToolTurn([], "stop", (None, None, None), None)

        return _gen()

    with (
        patch("llmkit._litellm.astream_tools", side_effect=_transport),
        capture_llm_records() as records,
    ):
        events = tool_llm_call_stream_sync("hi", _TOOLS, feature="lazy")
        assert produced == []
        assert next(events) == TextDeltaEvent("a")
        assert produced == ["a"]
        assert next(events) == TextDeltaEvent("b")
        assert produced == ["a", "b"]
        rest = list(events)
    assert len(rest) == 1 and isinstance(rest[0], ToolCallResult)
    assert rest[0].text == "ab"
    assert [record.error for record in records] == [None]


def test_a_second_thread_cannot_advance_a_stream_in_flight() -> None:
    """Like a generator, the iterator refuses a concurrent ``next()``."""
    park = asyncio.Event()
    stream = _ProviderStream(["a", "b"], park=park)
    received: list[object] = []
    with _serving(stream):
        chunks = text_llm_call_stream_sync("hi", feature="lazy")
        assert next(chunks) == "a"
        worker = threading.Thread(target=lambda: received.append(next(chunks)))
        worker.start()
        assert stream.parked.wait(timeout=5.0), "the second pull never reached the provider"
        with pytest.raises(ValueError, match="already executing"):
            _ = next(chunks)
        loop = sync_module._loop
        assert loop is not None
        _ = loop.call_soon_threadsafe(park.set)
        worker.join(timeout=5.0)
        assert not worker.is_alive()
    assert received == ["b"]


# --- Honest abandonment ----------------------------------------------------------


def _abandon_by_close(make: Callable[[], Generator[object]]) -> None:
    items = make()
    _ = next(items)
    items.close()


def _abandon_by_break(make: Callable[[], Generator[object]]) -> None:
    for _item in make():
        break


def _abandon_by_drop(make: Callable[[], Generator[object]]) -> None:
    items = make()
    _ = next(items)
    del items
    _ = gc.collect()


_ABANDON: dict[str, Callable[[Callable[[], Generator[object]]], None]] = {
    "close": _abandon_by_close,
    "break": _abandon_by_break,
    "drop": _abandon_by_drop,
}


@pytest.mark.parametrize("how", sorted(_ABANDON))
@pytest.mark.parametrize("stream", sorted(_STREAMS))
def test_abandoning_the_sync_stream_records_the_abandonment(stream: str, how: str) -> None:
    """The record is written, and captured, before the next statement runs."""
    with install(_answer), capture_llm_records() as records:
        _ABANDON[how](_STREAMS[stream])
        errors = [record.error for record in records]
    assert errors == [STREAM_ABANDONED_ERROR]


def _async_items(stream: str) -> AsyncIterator[object]:
    if stream == "text":
        return text_llm_call_stream("hi", feature="sync-stream")
    return tool_llm_call_stream("hi", _TOOLS, feature="sync-stream")


async def _abandon_async(stream: str) -> None:
    async for _item in _async_items(stream):
        break


async def _consume_async(stream: str) -> None:
    async for _item in _async_items(stream):
        pass


def _comparable(record: LLMCallRecord) -> tuple[object, ...]:
    # The queue-wait stamp is written by the first step and read by the last,
    # so it survives only if every step runs in the one captured context.
    return (
        record.schema,
        record.error,
        cast("object", record.response),
        record.attempt,
        record.tool_calls,
        record.queue_wait_ms is None,
    )


@pytest.mark.parametrize("stream", sorted(_STREAMS))
def test_a_completed_sync_stream_records_what_the_async_one_does(stream: str) -> None:
    with install(_answer), capture_llm_records() as async_records:
        asyncio.run(_consume_async(stream))
    with install(_answer), capture_llm_records() as sync_records:
        _ = list(_STREAMS[stream]())
    assert [_comparable(r) for r in sync_records] == [_comparable(r) for r in async_records]
    assert sync_records[0].error is None
    assert sync_records[0].queue_wait_ms is not None


@pytest.mark.parametrize("stream", sorted(_STREAMS))
def test_a_sync_abandonment_records_what_the_async_one_does(stream: str) -> None:
    with install(_answer), capture_llm_records() as async_records:
        asyncio.run(_abandon_async(stream))
    with install(_answer), capture_llm_records() as sync_records:
        _abandon_by_break(_STREAMS[stream])
    assert [_comparable(r) for r in sync_records] == [_comparable(r) for r in async_records]
    assert sync_records[0].error == STREAM_ABANDONED_ERROR


def test_stopping_at_the_tool_result_is_a_completed_call() -> None:
    with install(_answer), capture_llm_records() as records:
        for event in tool_llm_call_stream_sync("hi", _TOOLS, feature="sync-stream"):
            if isinstance(event, ToolCallResult):
                break
    assert [record.error for record in records] == [None]


def test_throw_forwards_into_the_stream() -> None:
    """``throw`` mirrors ``athrow``: the stream unwinds and the error propagates."""
    with install(_answer), capture_llm_records() as records:
        chunks = text_llm_call_stream_sync("hi", feature="sync-stream")
        assert next(chunks) == "a"
        with pytest.raises(KeyError, match="from the consumer"):
            _ = chunks.throw(KeyError("from the consumer"))
        with pytest.raises(StopIteration):
            _ = next(chunks)
    assert [record.error for record in records] == [STREAM_ABANDONED_ERROR]


def test_closing_a_stream_that_never_started_writes_nothing() -> None:
    with install(_answer) as transport, capture_llm_records() as records:
        chunks = text_llm_call_stream_sync("hi", feature="sync-stream")
        chunks.close()
        with pytest.raises(StopIteration):
            _ = next(chunks)
    assert records == []
    assert transport.requests == []


# --- No loop-on-loop -------------------------------------------------------------


def test_creating_a_sync_stream_on_the_sync_loop_thread_raises() -> None:
    async def _create() -> None:
        _ = text_llm_call_stream_sync("hi", feature="sync-stream")

    with pytest.raises(RuntimeError, match="sync loop thread"):
        run_sync(_create(), timeout=5.0)


def test_advancing_a_sync_stream_on_the_sync_loop_thread_raises() -> None:
    with install(_answer) as transport, capture_llm_records() as records:
        chunks = tool_llm_call_stream_sync("hi", _TOOLS, feature="sync-stream")

        async def _advance() -> object:
            return next(chunks)

        with pytest.raises(RuntimeError, match="sync loop thread"):
            _ = run_sync(_advance(), timeout=5.0)
        # Refused before anything ran, so the stream is still whole.
        assert transport.requests == []
        assert next(chunks) == TextDeltaEvent("a")
        _ = list(chunks)
    assert [record.error for record in records] == [None]


def test_a_sync_stream_works_from_a_foreign_running_loop() -> None:
    """Any loop but the sync loop is allowed, blocking it per item like ``run_sync``."""
    loops: list[int] = []

    def _fake(request: TransportRequest) -> Reply:
        loops.append(id(asyncio.get_running_loop()))
        return _answer(request)

    async def _host() -> list[object]:
        return list(text_llm_call_stream_sync("hi", feature="sync-stream"))

    async def _loop_id() -> int:
        return id(asyncio.get_running_loop())

    with install(_fake):
        assert asyncio.run(_host()) == ["a", "b"]
    assert loops == [run_sync(_loop_id())]


def test_a_foreign_loop_does_not_adopt_the_stream() -> None:
    """The stream is bound to the sync loop, whichever loop the caller runs.

    Every generator method is first called inside a task on the sync loop, so
    a foreign host loop's teardown (``asyncio.run`` finalizing the async
    generators it saw) never reaches the stream; only its own close does.
    """

    async def _host() -> Generator[object]:
        chunks = text_llm_call_stream_sync("hi", feature="sync-stream")
        assert next(chunks) == "a"
        return chunks

    with install(_answer), capture_llm_records() as records:
        chunks = asyncio.run(_host())
        assert records == [], "the host loop's teardown closed the stream"
        chunks.close()
    assert [record.error for record in records] == [STREAM_ABANDONED_ERROR]


# --- One context -----------------------------------------------------------------


def test_the_context_is_captured_when_the_stream_is_created() -> None:
    """A capture entered after creation does not see the stream's record."""
    with install(_answer):
        with capture_llm_records() as at_creation:
            chunks = text_llm_call_stream_sync("hi", feature="sync-stream")
        with capture_llm_records() as at_iteration:
            assert list(chunks) == ["a", "b"]
    assert [record.error for record in at_creation] == [None]
    assert at_iteration == []


def test_a_cache_hit_replays_as_one_chunk_through_the_sync_form() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(_answer) as transport, capture_llm_records() as records:
        assert list(text_llm_call_stream_sync("hi", feature="cached")) == ["a", "b"]
        assert list(text_llm_call_stream_sync("hi", feature="cached")) == ["ab"]
    assert len(transport.requests) == 1
    assert [record.cache_hit for record in records] == [False, True]
