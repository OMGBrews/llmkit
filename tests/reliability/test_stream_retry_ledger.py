"""A host-driven stream retry joins one logical call.

A host that opts a stream function out of its own retries and re-runs it from
an attempt factory under :func:`~llmkit.retry.with_retries_stream` calls the
function afresh on every attempt. The loop's call ledger is what makes those
attempts one logical call, as :func:`~llmkit.retry.with_retries` does for the
buffered functions: one ``call_id``, a rising ``attempt``, and a response-cache
read on the first attempt only (a later attempt exists because the earlier one
failed, so there is nothing new to find).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterator
from typing import final

import pytest

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    RetryPolicy,
    capture_llm_records,
    configure_llm_cache,
    text_llm_call_stream,
)
from llmkit.cache import LLMCacheEntry
from llmkit.retry import with_retries_stream
from llmkit.testing import Reply, TransportRequest, install, server_error
from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _ollama() -> Iterator[None]:
    with ollama_call_defaults():
        yield


@final
class _CountingCache:
    """An in-memory cache that counts its lookups."""

    def __init__(self) -> None:
        self._inner: InMemoryLLMCache = InMemoryLLMCache()
        self.lookups: int = 0

    async def get(self, key: str) -> LLMCacheEntry | None:
        self.lookups += 1
        return await self._inner.get(key)

    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        await self._inner.set(key, entry)


def _fail_first(request: TransportRequest) -> Reply:
    if request.attempt == 1:
        raise server_error()
    return Reply(chunks=["fresh", " answer"])


def _factory(label: str | None = None) -> Callable[[int], AsyncGenerator[str]]:
    def _attempt(_n: int) -> AsyncGenerator[str]:
        return text_llm_call_stream("hi", feature="host", label=label, retry=NO_RETRY)

    return _attempt


async def test_a_host_stream_retry_shares_one_call_id_and_reads_the_cache_once() -> None:
    cache = _CountingCache()
    configure_llm_cache(cache)
    with install(_fail_first) as installed, capture_llm_records() as records:
        chunks = [
            chunk
            async for chunk in with_retries_stream(
                _factory(), policy=RetryPolicy(), label="host", surface="host_stream"
            )
        ]
    assert chunks == ["fresh", " answer"]
    assert [record.attempt for record in records] == [1, 2]
    assert len({record.call_id for record in records}) == 1
    assert [request.attempt for request in installed.requests] == [1, 2]
    assert len({request.call_id for request in installed.requests}) == 1
    assert cache.lookups == 1


async def test_separate_host_streams_keep_separate_call_ids() -> None:
    """The ledger belongs to one loop: two loops are two logical calls."""
    with install(lambda _r: Reply(chunks=["ok"])), capture_llm_records() as records:
        for _ in range(2):
            async for _chunk in with_retries_stream(
                _factory(), policy=RetryPolicy(), label="host", surface="host_stream"
            ):
                pass
    assert [record.attempt for record in records] == [1, 1]
    assert len({record.call_id for record in records}) == 2


async def test_a_consumer_call_between_chunks_does_not_join_the_stream() -> None:
    """The scope, and with it the ledger, is released around every yield."""
    with install(lambda _r: Reply(chunks=["a", "b"])), capture_llm_records() as records:
        async for _chunk in with_retries_stream(
            _factory("outer"), policy=RetryPolicy(), label="host", surface="host_stream"
        ):
            async for _inner in text_llm_call_stream(
                "between", feature="host", label="outer", retry=NO_RETRY
            ):
                pass
    stream_ids = {record.call_id for record in records if record.prompt == "hi"}
    consumer_ids = {record.call_id for record in records if record.prompt == "between"}
    assert len(stream_ids) == 1
    assert len(consumer_ids) == 2
    assert not stream_ids & consumer_ids
    assert {record.attempt for record in records} == {1}
