"""The configured cache ``timeout``: a stalled store ``get`` or ``set`` is cut off.

The store under test is gated on an :class:`asyncio.Event`, never on a sleep:
its ``get`` or ``set`` waits until the event is set, which nothing does except
the fixture's ``finally``. So a call that returns at all has been released by
the bound, and a call that would block forever without it is caught by an outer
guard (:func:`_guarded`) that fails the test instead of hanging the suite.

The follower's wait on the single flight has no bound of its own; its tests
live in ``test_read_through`` and are deliberately untouched by this one.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from collections.abc import Awaitable, Iterator
from typing import cast, override

import pytest
from pydantic import BaseModel

from llmkit import (
    InMemoryLLMCache,
    LLMCacheEntry,
    configure_llm_cache,
    get_llm_cache,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_stream_sync,
    text_llm_call_sync,
)
from llmkit.cache.registry import DEFAULT_CACHE_TIMEOUT, cache_timeout
from llmkit.testing import Reply, TransportRequest, install
from tests.cache.conftest import CALL_MODE, cache_provider
from tests.cache.test_cache_config import _DictStore

#: The budget the gated tests configure: short, so a test takes milliseconds.
BUDGET = 0.02
#: How long the outer guard waits before declaring the call stuck in the store.
#: Two orders of magnitude above the budget, so a slow machine is not a failure.
GUARD = 2.0


class _Answer(BaseModel):
    verdict: str


def _answer(request: TransportRequest) -> Reply:
    """``answer`` as structured JSON, plain text, or one streamed chunk."""
    if request.kind == "structured":
        return Reply.json(_Answer(verdict="answer"))
    if request.kind == "text_stream":
        return Reply(chunks=["answer"])
    return Reply.text("answer")


class _Gated(_DictStore):
    """A store whose ``get`` or ``set`` (per *gate*) waits on an event nobody sets.

    Each gated operation records itself in ``entered`` and, if it is cancelled
    while waiting, in ``cancelled``. :meth:`release` sets every gate from any
    thread, so a call stuck on the sync wrappers' loop can still be let go.
    """

    def __init__(self, gate: str) -> None:
        super().__init__()
        self.gate: str = gate
        self.entered: list[str] = []
        self.cancelled: list[str] = []
        self._gates: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    async def _wait(self, what: str) -> None:
        gate = asyncio.Event()
        self._gates.append((asyncio.get_running_loop(), gate))
        self.entered.append(what)
        try:
            _ = await gate.wait()
        except asyncio.CancelledError:
            self.cancelled.append(what)
            raise

    @override
    async def get(self, key: str) -> LLMCacheEntry | None:
        if self.gate == "get":
            await self._wait("get")
        return await super().get(key)

    @override
    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        if self.gate == "set":
            await self._wait("set")
        await super().set(key, entry)

    def release(self) -> None:
        for loop, gate in self._gates:
            if not loop.is_closed():
                _ = loop.call_soon_threadsafe(gate.set)


@pytest.fixture
def gated() -> Iterator[list[_Gated]]:
    """Every gated store a test makes, released on the way out whatever happened."""
    made: list[_Gated] = []
    try:
        yield made
    finally:
        for store in made:
            store.release()


def _store(gated: list[_Gated], gate: str) -> _Gated:
    store = _Gated(gate)
    gated.append(store)
    return store


async def _guarded[T](call: Awaitable[T]) -> T:
    """Await *call*, failing the test if it is still blocked after :data:`GUARD`."""
    try:
        return await asyncio.wait_for(call, GUARD)
    except TimeoutError:
        pytest.fail(f"the call was still blocked in the cache store after {GUARD}s")


def _cache_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "llmkit.cache" and r.levelno == logging.WARNING]


def _assert_one_timeout_warning(caplog: pytest.LogCaptureFixture, what: str) -> None:
    """One WARNING naming *what* and its outcome; the budget is in the exception."""
    (warning,) = _cache_warnings(caplog)
    message = warning.getMessage()
    assert message.startswith(f"LLM response cache {what} failed; ")
    if what == "get":
        assert "the call proceeds as a miss" in message
    else:
        assert "the call returns its answer and stores nothing" in message
    assert warning.exc_info is not None
    exc = warning.exc_info[1]
    assert type(exc) is TimeoutError
    assert str(exc) == f"cache {what} took longer than {BUDGET:g}s"


async def _text(provider: object) -> object:
    return await text_llm_call("q", feature="f", provider=provider)  # pyright: ignore[reportArgumentType]  # a provider double


async def _structured(provider: object) -> object:
    return await structured_llm_call("q", _Answer, feature="f", provider=provider)  # pyright: ignore[reportArgumentType]  # a provider double


async def _stream(provider: object) -> object:
    stream = text_llm_call_stream("q", feature="f", provider=provider)  # pyright: ignore[reportArgumentType]  # a provider double
    return "".join([chunk async for chunk in stream])


_LANES = {"text": _text, "structured": _structured, "stream": _stream}
_EXPECTED = {"text": "answer", "structured": _Answer(verdict="answer"), "stream": "answer"}


# --- the bound, on every lane that reads the cache --------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["text", "structured", "stream"])
async def test_a_stalled_get_is_a_miss_within_the_budget(
    lane: str, gated: list[_Gated], caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(gated, "get")
    configure_llm_cache(store, timeout=BUDGET)
    with install(_answer) as installed:
        got = await _guarded(_LANES[lane](cache_provider(instructor_mode=CALL_MODE)))

    assert got == _EXPECTED[lane]
    assert len(installed.requests) == 1
    assert store.cancelled == ["get"]
    # The miss went on to store the answer the provider gave.
    assert len(store.entries) == 1
    _assert_one_timeout_warning(caplog, "get")


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["text", "structured", "stream"])
async def test_a_stalled_set_returns_the_answer_and_stores_nothing(
    lane: str, gated: list[_Gated], caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(gated, "set")
    configure_llm_cache(store, timeout=BUDGET)
    with install(_answer) as installed:
        got = await _guarded(_LANES[lane](cache_provider(instructor_mode=CALL_MODE)))

    assert got == _EXPECTED[lane]
    assert len(installed.requests) == 1
    assert store.cancelled == ["set"]
    assert store.entries == {}
    _assert_one_timeout_warning(caplog, "set")


def _text_sync(provider: object) -> object:
    return text_llm_call_sync("q", feature="f", provider=provider)  # pyright: ignore[reportArgumentType]  # a provider double


def _structured_sync(provider: object) -> object:
    return structured_llm_call_sync("q", _Answer, feature="f", provider=provider)  # pyright: ignore[reportArgumentType]  # a provider double


def _stream_sync(provider: object) -> object:
    return "".join(text_llm_call_stream_sync("q", feature="f", provider=provider))  # pyright: ignore[reportArgumentType]  # a provider double


_SYNC_LANES = {"text": _text_sync, "structured": _structured_sync, "stream": _stream_sync}


@pytest.mark.parametrize("gate", ["get", "set"])
@pytest.mark.parametrize("lane", ["text", "structured", "stream"])
def test_the_sync_wrappers_are_bounded_too(
    lane: str, gate: str, gated: list[_Gated], caplog: pytest.LogCaptureFixture
) -> None:
    """The sync forms run the same helpers on llmkit's own loop. The call runs
    on a worker thread so a missing bound fails the guard instead of blocking
    the test thread; the fixture's release lets that thread finish."""
    store = _store(gated, gate)
    configure_llm_cache(store, timeout=BUDGET)
    provider = cache_provider(instructor_mode=CALL_MODE)
    with (
        install(_answer) as installed,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(_SYNC_LANES[lane], provider)
        try:
            got = future.result(timeout=GUARD)
        except concurrent.futures.TimeoutError:
            store.release()
            pytest.fail(f"the sync call was still blocked in the cache store after {GUARD}s")

    assert got == _EXPECTED[lane]
    assert len(installed.requests) == 1
    assert store.cancelled == [gate]
    assert len(store.entries) == (1 if gate == "get" else 0)
    _assert_one_timeout_warning(caplog, gate)


@pytest.mark.asyncio
async def test_a_stalled_store_warns_once_across_calls(
    gated: list[_Gated], caplog: pytest.LogCaptureFixture
) -> None:
    configure_llm_cache(_store(gated, "get"), timeout=BUDGET)
    with install(_answer) as installed:
        for _ in range(3):
            _ = await _guarded(_text(cache_provider()))

    assert len(installed.requests) == 3
    assert len(_cache_warnings(caplog)) == 1


# --- what the bound leaves alone ---------------------------------------------


@pytest.mark.asyncio
async def test_a_store_timeout_error_before_the_deadline_is_its_own(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A store's own ``TimeoutError`` (a client timeout firing first) is
    reported as raised, not relabelled as llmkit's budget."""
    own = TimeoutError("redis read timed out")

    class _OwnTimeout(_DictStore):
        @override
        async def get(self, key: str) -> LLMCacheEntry | None:
            raise own

    configure_llm_cache(_OwnTimeout(), timeout=BUDGET)
    with install(_answer) as installed:
        assert await _guarded(_text(cache_provider())) == "answer"

    assert len(installed.requests) == 1
    (warning,) = _cache_warnings(caplog)
    assert warning.exc_info is not None and warning.exc_info[1] is own


class _ScopeSpy:
    """Stands in for the ``asyncio`` module inside ``read_through`` and records
    every ``asyncio.timeout`` scope the cache helpers open."""

    def __init__(self) -> None:
        self.scopes: list[float | None] = []

    def __getattr__(self, name: str) -> object:
        return cast("object", getattr(asyncio, name))

    def timeout(self, delay: float | None) -> asyncio.Timeout:
        self.scopes.append(delay)
        return asyncio.timeout(delay)


@pytest.mark.asyncio
async def test_none_turns_the_bound_off(
    gated: list[_Gated], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``timeout=None`` no scope is opened around the store at all, so a
    stalled ``get`` holds the call until the store answers; releasing the gate
    is what lets it finish. The spy, not the wait, is what tells "off" from a
    longer budget the gate happened to be released inside."""
    spy = _ScopeSpy()
    monkeypatch.setattr("llmkit.cache.read_through.asyncio", spy)
    store = _store(gated, "get")
    configure_llm_cache(store, timeout=BUDGET)
    configure_llm_cache(store, timeout=None)
    with install(_answer) as installed:
        task = asyncio.create_task(_text(cache_provider()))
        # Past the budget configured just before ``None``.
        done, _ = await asyncio.wait({task}, timeout=BUDGET * 5)
        assert not done
        assert store.entered == ["get"]
        store.release()
        assert await _guarded(task) == "answer"

    assert spy.scopes == []
    assert store.cancelled == []
    assert len(installed.requests) == 1


@pytest.mark.asyncio
async def test_cancelling_the_call_during_a_store_operation_propagates(
    gated: list[_Gated],
) -> None:
    """A host cancellation is not a cache failure: it is not turned into a
    miss, and it reaches the caller as ``CancelledError``."""
    store = _store(gated, "get")
    configure_llm_cache(store)  # the default budget, far longer than this test
    with install(_answer) as installed:
        task = asyncio.create_task(_text(cache_provider()))
        try:
            async with asyncio.timeout(GUARD):
                while not store.entered:
                    await asyncio.sleep(0)
        except TimeoutError:
            pytest.fail(f"the call did not reach the cache store within {GUARD}s")
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert store.cancelled == ["get"]
    assert installed.requests == []


@pytest.mark.asyncio
async def test_a_zero_budget_still_serves_the_in_memory_store(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``0`` fires at the store's first suspension; the in-memory store never
    suspends, so it still stores and answers."""
    configure_llm_cache(cache := InMemoryLLMCache(), timeout=0)
    provider = cache_provider()
    with install(_answer) as installed:
        first = await _text(provider)
        second = await _text(provider)

    assert first == second == "answer"
    assert len(installed.requests) == 1
    assert len(cache) == 1
    assert not [r for r in caplog.records if r.name == "llmkit.cache"]


@pytest.mark.asyncio
async def test_a_zero_budget_cuts_off_a_store_that_suspends(
    gated: list[_Gated], caplog: pytest.LogCaptureFixture
) -> None:
    """The other half of ``0``: a store that suspends at all is cancelled at
    that first suspension, so ``0`` is a bound, not "off"."""
    store = _store(gated, "get")
    configure_llm_cache(store, timeout=0)
    with install(_answer) as installed:
        assert await _guarded(_text(cache_provider())) == "answer"

    assert store.cancelled == ["get"]
    assert len(installed.requests) == 1
    (warning,) = _cache_warnings(caplog)
    assert warning.exc_info is not None
    assert str(warning.exc_info[1]) == "cache get took longer than 0s"


def test_each_failure_names_what_the_call_does_about_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A read or decode failure is a miss; a write or encode failure keeps the
    answer and stores nothing; a key failure bypasses the cache."""
    from llmkit.cache.registry import report_cache_failure

    outcomes = {
        "get": "the call proceeds as a miss",
        "decode": "the call proceeds as a miss",
        "set": "the call returns its answer and stores nothing",
        "encode": "the call returns its answer and stores nothing",
        "key": "the call bypasses the cache and stores nothing",
    }
    for what, outcome in outcomes.items():
        configure_llm_cache(None)  # re-arms the latch, so each one warns
        caplog.clear()
        report_cache_failure(what, ValueError(what))
        (warning,) = _cache_warnings(caplog)
        assert warning.getMessage().startswith(f"LLM response cache {what} failed; {outcome} (")


# --- configuration -------------------------------------------------------------


def test_the_default_budget_is_one_second() -> None:
    assert DEFAULT_CACHE_TIMEOUT == 1.0
    assert cache_timeout() == DEFAULT_CACHE_TIMEOUT
    configure_llm_cache(InMemoryLLMCache())
    assert cache_timeout() == 1.0


def test_omitting_the_timeout_keeps_the_configured_one() -> None:
    configure_llm_cache(InMemoryLLMCache(), timeout=2.5)
    configure_llm_cache(InMemoryLLMCache())
    assert cache_timeout() == 2.5
    configure_llm_cache(None)
    assert cache_timeout() == 2.5
    configure_llm_cache(None, timeout=None)
    assert cache_timeout() is None
    configure_llm_cache(InMemoryLLMCache())
    assert cache_timeout() is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.5, True, False])
def test_a_timeout_that_is_not_a_finite_nonnegative_number_is_rejected(bad: float) -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache, timeout=3.0)
    with pytest.raises(ValueError, match="timeout must be finite and nonnegative"):
        configure_llm_cache(InMemoryLLMCache(), timeout=bad)
    # Nothing changed: neither the store nor the budget.
    assert get_llm_cache() is cache
    assert cache_timeout() == 3.0
