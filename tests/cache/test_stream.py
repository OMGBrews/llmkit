"""The cache seen through ``text_llm_call_stream``.

A stream shares the buffered text key, replays a hit as one chunk outside the
retry loop, stores only a stream that ran to completion, and takes no part in
the single flight. As in ``test_read_through``, each test answers the provider
through an :mod:`llmkit.testing` fake and counts the requests it received; the
cache sits above the fake, so "answered from the cache" is a positive
observation rather than the absence of an error.

Three tests stay below the fake: the mid-stream failure, because a fake can
fail a stream only before its first chunk; and the slot-and-retry and
byte-identical request tests, which patch ``litellm.acompletion`` below the
LiteLLM adapter by design.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast, final
from unittest.mock import MagicMock, patch

import pytest

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCallOptions,
    ResultValidationError,
    TokenUsage,
    capture_llm_records,
    configure_llm_cache,
    text_llm_call,
    text_llm_call_stream,
)
from llmkit.calls import STREAM_ABANDONED_ERROR
from llmkit.rate_limiting import GlobalRateLimiter
from llmkit.retry import with_retries, with_retries_stream
from llmkit.testing import Reply, TransportRequest, install
from tests.cache.conftest import cache_provider


@dataclass
class _InTurn:
    """A fake answering *answers* in turn, streamed or buffered.

    A streamed answer arrives as one chunk per word, each later word with its
    leading space, so a replay's single chunk is distinguishable from the live
    stream it stored. ``len(installed.requests)`` counts the calls that paid.
    """

    answers: list[str] = field(default_factory=lambda: ["one two", "three four", "five"])
    served: int = 0

    def __call__(self, request: TransportRequest) -> Reply:
        self.served += 1
        answer = self.answers[self.served - 1]
        if request.kind == "text_stream":
            words = answer.split(" ")
            return Reply(chunks=[words[0], *(f" {word}" for word in words[1:])])
        return Reply.text(answer, cost=0.002, usage=TokenUsage(10, 5, 15))


def _no_cache_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The key computed and the store answered — no silent bypass."""
    assert not [r for r in caplog.records if r.name == "llmkit.cache"]


async def _chunks(**kwargs: object) -> list[str]:
    stream = text_llm_call_stream("q", feature="f", **kwargs)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_hit_replays_the_stored_text_as_one_chunk(caplog: pytest.LogCaptureFixture) -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed:
        live = await _chunks(provider=provider)
        replayed = await _chunks(provider=provider)

    assert live == ["one", " two"]
    assert replayed == ["one two"]
    assert len(installed.requests) == 1
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_a_buffered_answer_serves_a_stream(caplog: pytest.LogCaptureFixture) -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed:
        paid = await text_llm_call("q", feature="f", provider=provider)
        replayed = await _chunks(provider=provider)

    assert replayed == [paid] == ["one two"]
    assert len(installed.requests) == 1
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_a_streamed_answer_serves_a_buffered_call(caplog: pytest.LogCaptureFixture) -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed, capture_llm_records() as records:
        _ = await _chunks(provider=provider)
        served = await text_llm_call("q", feature="f", provider=provider)

    assert served == "one two"
    assert len(installed.requests) == 1
    assert len(cache) == 1
    paid, hit = records
    # The hit names the lane that answered; the call that paid is the stream.
    assert (paid.schema, hit.schema, hit.cache_hit) == ("stream", "text", True)
    assert hit.source_call_id == paid.call_id
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_an_abandoned_stream_is_not_stored() -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed, capture_llm_records() as records:
        stream = text_llm_call_stream("q", feature="f", provider=provider)
        async for _chunk in stream:
            break
        await stream.aclose()
        assert len(cache) == 0
        later = await _chunks(provider=provider)

    assert records[0].error == STREAM_ABANDONED_ERROR
    assert later == ["three", " four"]
    assert len(installed.requests) == 2
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_a_stream_failing_before_its_first_chunk_is_not_stored() -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    provider = cache_provider()

    def _failing(_request: TransportRequest) -> Reply:
        raise ValueError("provider broke")

    seen: list[str] = []
    with install(_failing) as installed, pytest.raises(ValueError, match="provider broke"):
        async for chunk in text_llm_call_stream(
            "q", feature="f", provider=provider, retry=NO_RETRY
        ):
            seen.append(chunk)

    assert seen == []
    assert len(installed.requests) == 1
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_a_stream_failing_mid_stream_is_not_stored() -> None:
    """Stays on the private ``astream_text`` seam: a public fake can fail a
    stream only before its first chunk."""
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    provider = cache_provider()

    def _failing(*_a: object, **_k: object) -> AsyncIterator[str]:
        async def _chunks() -> AsyncIterator[str]:
            yield "partial"
            raise ValueError("provider broke")

        return _chunks()

    seen: list[str] = []
    with (
        patch("llmkit._litellm.astream_text", side_effect=_failing),
        pytest.raises(ValueError, match="provider broke"),
    ):
        async for chunk in text_llm_call_stream(
            "q", feature="f", provider=provider, retry=NO_RETRY
        ):
            seen.append(chunk)

    assert seen == ["partial"]
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_stream_hit_record_fields() -> None:
    """The hit record names the stream lane, pays nothing, points at the call
    that paid, and does not inherit the paid attempt's queue-wait stamp (the
    real limiter's near-zero wait, stamped once the paid stream held its slot)."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()

    with install(lambda _request: Reply.text("yes")), capture_llm_records() as records:
        _ = await _chunks(provider=provider)
        _ = await _chunks(provider=provider)

    paid, hit = records
    assert (paid.cache_hit, paid.approximate_cost) == (False, None)
    assert paid.queue_wait_ms is not None and paid.queue_wait_ms >= 0
    assert (hit.cache_hit, hit.schema, hit.approximate_cost) == (True, "stream", 0.0)
    assert (hit.queue_wait_ms, hit.error, hit.attempt) == (None, None, 1)
    assert cast("object", hit.response) == "yes"
    assert hit.source_call_id == paid.call_id
    assert hit.call_id != paid.call_id


@pytest.mark.asyncio
async def test_an_abandoned_replay_is_recorded_as_abandoned() -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed, capture_llm_records() as records:
        _ = await _chunks(provider=provider)
        stream = text_llm_call_stream("q", feature="f", provider=provider)
        async for _chunk in stream:
            break
        await stream.aclose()

    hit = records[1]
    assert (hit.cache_hit, hit.error, hit.approximate_cost) == (True, STREAM_ABANDONED_ERROR, 0.0)
    assert len(installed.requests) == 1


@pytest.mark.asyncio
async def test_a_stored_empty_answer_replays_as_no_chunks() -> None:
    """An empty buffered answer is stored; replayed, it yields nothing — as a
    live empty stream does — and the hit is still recorded."""
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn(answers=[""])
    provider = cache_provider()
    with install(transport) as installed, capture_llm_records() as records:
        assert await text_llm_call("q", feature="f", provider=provider) == ""
        assert await _chunks(provider=provider) == []

    assert len(installed.requests) == 1
    assert (records[1].cache_hit, records[1].schema) == (True, "stream")


@final
class _FakeStream:
    """A plain-class LiteLLM stream (a MagicMock's auto attributes would feed
    the transport's token accounting)."""

    def __aiter__(self) -> AsyncIterator[MagicMock]:
        async def _gen() -> AsyncIterator[MagicMock]:
            for delta in ("ans", "wer"):
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])

        return _gen()


@pytest.mark.asyncio
async def test_a_stream_hit_takes_no_slot_and_enters_no_retry_loop() -> None:
    """Driven through the real ``astream_text`` over a faked
    ``litellm.acompletion``, so the limiter is genuinely acquired on the miss
    and observably skipped on the hit."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()

    async def _respond(**_kwargs: object) -> _FakeStream:
        return _FakeStream()

    acompletion = MagicMock(side_effect=_respond)
    with (
        patch("llmkit._litellm.litellm.acompletion", acompletion),
        patch.object(
            GlobalRateLimiter, "acquire_async", side_effect=GlobalRateLimiter.acquire_async
        ) as acquire,
        patch(
            "llmkit.calls.stream.with_retries_stream", side_effect=with_retries_stream
        ) as retries,
    ):
        first = await _chunks(provider=provider)
        assert (acquire.call_count, retries.call_count, acompletion.call_count) == (1, 1, 1)
        second = await _chunks(provider=provider)

    assert (first, second) == (["ans", "wer"], ["answer"])
    assert (acquire.call_count, retries.call_count, acompletion.call_count) == (1, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["keyword", "options"])
async def test_cache_false_bypasses_lookup_and_store(via: str) -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    transport = _InTurn()
    provider = cache_provider()
    opt_out: dict[str, object] = (
        {"cache": False} if via == "keyword" else {"options": LLMCallOptions(cache=False)}
    )
    with install(transport) as installed:
        _ = await _chunks(provider=provider)  # stores "one two"
        bypassed = await _chunks(provider=provider, **opt_out)
        assert len(cache) == 1
        served = await _chunks(provider=provider)

    # The opted-out stream neither read "one two" nor replaced it.
    assert bypassed == ["three", " four"]
    assert served == ["one two"]
    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_identical_streams_in_flight_both_reach_the_transport() -> None:
    """Streams take no part in the single flight: a second identical stream
    started while the first is mid-consumption does not wait on it."""
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    transport = _InTurn(answers=["a b", "a b"])
    provider = cache_provider()
    with install(transport) as installed:
        first = text_llm_call_stream("q", feature="f", provider=provider)
        second = text_llm_call_stream("q", feature="f", provider=provider)
        async with asyncio.timeout(2):
            assert await anext(first) == "a"
            assert await anext(second) == "a"
            assert len(installed.requests) == 2
            assert [c async for c in first] == [" b"]
            assert [c async for c in second] == [" b"]

    assert len(cache) == 1


@pytest.mark.asyncio
async def test_a_stream_does_not_block_a_buffered_call_on_the_same_key() -> None:
    """A buffered call made between a stream's chunks — the stream's own task —
    neither waits on the stream nor is served a half-finished answer."""
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn()
    provider = cache_provider()
    with install(transport) as installed:
        async with asyncio.timeout(2):
            stream = text_llm_call_stream("q", feature="f", provider=provider)
            assert await anext(stream) == "one"
            buffered = await text_llm_call("q", feature="f", provider=provider)
            rest = [c async for c in stream]

    assert (buffered, rest) == ("three four", [" two"])
    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_outer_retry_reroll_skips_the_lookup() -> None:
    """A stream re-run by an enclosing ``with_retries`` pass asks for a new
    sample: it skips the lookup and its answer replaces the stored one."""
    configure_llm_cache(InMemoryLLMCache())
    transport = _InTurn()
    provider = cache_provider()
    seen: list[str] = []

    async def _draft() -> str:
        text = "".join(await _chunks(provider=provider, retry=NO_RETRY))
        seen.append(text)
        if text == "one two":
            raise ResultValidationError("host rejects the first draft")
        return text

    with install(transport) as installed:
        _ = await _chunks(provider=provider)  # stores "one two"
        accepted = await with_retries(
            _draft, max_attempts=3, validation_retry_on=(ResultValidationError,)
        )
        after = await _chunks(provider=provider)

    # Pass 1 hit the stored "one two"; pass 2 skipped the lookup and paid.
    assert seen == ["one two", "three four"]
    assert accepted == "three four"
    assert after == ["three four"]
    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_an_unkeyable_stream_bypasses_the_cache(caplog: pytest.LogCaptureFixture) -> None:
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    transport = _InTurn()
    provider = cache_provider()
    provider.completion_kwargs = MagicMock(return_value={"api_base": object()})
    with (
        caplog.at_level(logging.WARNING, logger="llmkit.cache"),
        install(transport) as installed,
    ):
        _ = await _chunks(provider=provider)
        _ = await _chunks(provider=provider)

    assert (len(installed.requests), len(cache)) == (2, 0)
    assert [r for r in caplog.records if r.name == "llmkit.cache"]


def _stream_request_kwargs(*, configure: bool) -> dict[str, object]:
    """The kwargs ``litellm.acompletion`` receives for one streamed call."""
    seen: dict[str, object] = {}

    async def _acompletion(**kwargs: object) -> _FakeStream:
        seen.update(kwargs)
        return _FakeStream()

    async def _drive() -> list[str]:
        return [
            chunk
            async for chunk in text_llm_call_stream(
                "hi", feature="f", provider=cache_provider(), temperature=0.5, max_tokens=64
            )
        ]

    configure_llm_cache(InMemoryLLMCache() if configure else None)
    with patch("llmkit._litellm.litellm.acompletion", side_effect=_acompletion):
        assert asyncio.run(_drive()) == ["ans", "wer"]
    configure_llm_cache(None)
    return seen


def test_stream_requests_are_byte_identical_with_and_without_a_cache() -> None:
    without = _stream_request_kwargs(configure=False)
    with_cache = _stream_request_kwargs(configure=True)
    assert without == with_cache
    assert set(without) == {
        "model",
        "messages",
        "temperature",
        "stream",
        "stream_options",
        "api_key",
        "api_base",
        "max_tokens",
    }
    assert repr(without) == repr(with_cache)
