"""A stored answer the caller's ``on_result`` hook rejects is a recorded attempt.

Every entry the hook sees is a pass of the call, so it writes a record with
``cache_hit`` true, ``source_call_id`` naming the call that stored it, and the
hook's own exception in ``error``; the live attempt that follows is numbered
after it. A hook exception outside the validation budget is recorded the same
way and then raised. A cancellation before any entry reaches the hook (while
the store's ``get`` or a single-flight leader is awaited) writes nothing.

These live outside ``tests/cache``, whose conftest mutes records, and read
them through :func:`~llmkit.capture_llm_records`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast, override

import pytest

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCacheEntry,
    ResultValidationError,
    RetryPolicy,
    capture_llm_records,
    configure_llm_cache,
    structured_llm_call,
    text_llm_call,
)
from llmkit.cache import _flight
from llmkit.retry import with_retries
from llmkit.testing import Reply, TransportRequest, install, server_error
from tests._support import OkSchema


def _in_turn(*answers: str) -> Callable[[TransportRequest], Reply]:
    served = [0]

    def fake(_request: TransportRequest) -> Reply:
        served[0] += 1
        return Reply.text(answers[served[0] - 1])

    return fake


async def _seed(answer: str = "stored") -> str:
    """Store *answer* through a paid text call; return that call's id."""
    with install(lambda _r: Reply.text(answer)), capture_llm_records() as records:
        _ = await text_llm_call("q", feature="f", label="seed")
    (paid,) = records
    assert paid.call_id is not None
    return paid.call_id


@pytest.mark.asyncio
async def test_a_rejected_hit_is_recorded_before_the_live_attempt() -> None:
    configure_llm_cache(InMemoryLLMCache())
    source = await _seed()

    def only_fresh(text: str) -> None:
        if text == "stored":
            raise ResultValidationError("stale")

    with install(_in_turn("fresh")) as installed, capture_llm_records() as records:
        answer = await text_llm_call("q", feature="f", label="pick", on_result=only_fresh)

    assert (answer, len(installed.requests)) == ("fresh", 1)
    rejected, live = records
    assert rejected.cache_hit is True
    assert rejected.source_call_id == source
    assert rejected.error == "ResultValidationError: stale"
    assert cast("object", rejected.response) == "stored"
    assert (rejected.approximate_cost, rejected.usage, rejected.queue_wait_ms) == (0.0, None, None)
    assert live.cache_hit is False and live.error is None
    assert cast("object", live.response) == "fresh"
    assert rejected.call_id == live.call_id
    assert (rejected.attempt, live.attempt) == (1, 2)


@pytest.mark.asyncio
async def test_a_validation_budget_exception_records_the_hooks_own_type() -> None:
    """``run_cached`` turns the hook's exception into a ResultValidationError to
    make the hit a miss; the record keeps what the hook actually raised."""
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _r: Reply.json(OkSchema(ok=True))):
        _ = await structured_llm_call("q", OkSchema, feature="f", label="seed")
    seen: list[OkSchema] = []

    def reject_first(value: OkSchema) -> None:
        seen.append(value)
        if len(seen) == 1:
            raise ValueError("not this one")

    policy = RetryPolicy(max_attempts=1, validation_retry_on=(ValueError,))
    with (
        install(lambda _r: Reply.json(OkSchema(ok=True))) as installed,
        capture_llm_records() as records,
    ):
        _ = await structured_llm_call(
            "q", OkSchema, feature="f", label="pick", retry=policy, on_result=reject_first
        )

    assert len(installed.requests) == 1
    rejected, live = records
    assert (rejected.cache_hit, rejected.error, rejected.attempt) == (
        True,
        "ValueError: not this one",
        1,
    )
    assert (live.cache_hit, live.error, live.attempt) == (False, None, 2)


@pytest.mark.asyncio
async def test_a_hook_exception_outside_the_budget_is_recorded_and_raised() -> None:
    configure_llm_cache(InMemoryLLMCache())
    source = await _seed()

    def broken(_text: str) -> None:
        raise RuntimeError("hook bug")

    with install(_in_turn()) as installed, capture_llm_records() as records:
        with pytest.raises(RuntimeError, match="hook bug"):
            _ = await text_llm_call("q", feature="f", label="pick", on_result=broken)

    assert len(installed.requests) == 0
    (record,) = records
    assert (record.cache_hit, record.source_call_id) == (True, source)
    assert (record.error, record.attempt) == ("RuntimeError: hook bug", 1)


@pytest.mark.asyncio
async def test_a_follower_rejecting_both_entries_writes_two_records() -> None:
    """The stored entry and then the in-flight leader's entry each reach the
    follower's hook; each rejection is its own attempt, then the follower pays."""
    configure_llm_cache(InMemoryLLMCache())
    seed_id = await _seed()
    following = asyncio.Event()
    release = asyncio.Event()

    def watch(key: str) -> tuple[bool, asyncio.Future[LLMCacheEntry | None]]:
        leader, flight = _flight.lead_or_follow(key)
        if not leader:
            following.set()
        return leader, flight

    async def gated(_request: TransportRequest) -> Reply:
        # The leader's request is the first the fake sees; the follower's own
        # comes after it has rejected the leader's entry.
        if len(installed.requests) == 1:
            _ = await release.wait()
            return Reply.text("leader's")
        return Reply.text("own")

    def leader_hook(text: str) -> None:
        if text == "stored":
            raise ResultValidationError("leader wants fresh")

    def follower_hook(text: str) -> None:
        if text != "own":
            raise ResultValidationError(f"follower rejects {text}")

    with (
        pytest.MonkeyPatch.context() as mp,
        install(gated) as installed,
        capture_llm_records() as records,
    ):
        mp.setattr("llmkit.cache.read_through.lead_or_follow", watch)
        leader = asyncio.create_task(
            text_llm_call("q", feature="f", label="leader", on_result=leader_hook)
        )
        follower = asyncio.create_task(
            text_llm_call("q", feature="f", label="follower", on_result=follower_hook)
        )
        _ = await asyncio.wait_for(following.wait(), timeout=5)
        release.set()
        results = await asyncio.gather(leader, follower)

    assert results == ["leader's", "own"]
    mine = [r for r in records if r.label == "follower"]
    assert [(r.cache_hit, r.error, r.attempt) for r in mine] == [
        (True, "ResultValidationError: follower rejects stored", 1),
        (True, "ResultValidationError: follower rejects leader's", 2),
        (False, None, 3),
    ]
    leader_record = next(r for r in records if r.label == "leader" and not r.cache_hit)
    assert [r.source_call_id for r in mine] == [seed_id, leader_record.call_id, None]
    assert len({r.call_id for r in mine}) == 1


@pytest.mark.asyncio
async def test_an_outer_retry_pass_numbers_after_a_rejected_hit() -> None:
    """Pass 1 rejects the stored answer (attempt 1) and its live attempt fails
    (attempt 2); the ledger counted both, so pass 2 is attempt 3."""
    configure_llm_cache(InMemoryLLMCache())
    _ = await _seed()
    served = [0]

    def fail_then_answer(_request: TransportRequest) -> Reply:
        served[0] += 1
        if served[0] == 1:
            raise server_error()
        return Reply.text("fresh")

    def only_fresh(text: str) -> None:
        if text == "stored":
            raise ResultValidationError("stale")

    async def once() -> str:
        return await text_llm_call(
            "q", feature="f", label="pick", retry=NO_RETRY, on_result=only_fresh
        )

    with install(fail_then_answer), capture_llm_records() as records:
        answer = await with_retries(
            once, policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0)
        )

    assert answer == "fresh"
    assert [(r.cache_hit, r.error is None, r.attempt) for r in records] == [
        (True, False, 1),
        (False, False, 2),
        (False, True, 3),
    ]
    assert len({r.call_id for r in records}) == 1


class _BlockingGet(InMemoryLLMCache):
    """A store whose ``get`` never answers."""

    def __init__(self, entered: asyncio.Event) -> None:
        super().__init__()
        self._entered: asyncio.Event = entered

    @override
    async def get(self, key: str) -> LLMCacheEntry | None:
        del key
        self._entered.set()
        _ = await asyncio.Event().wait()
        return None


async def _cancel_when(started: asyncio.Event, call: Awaitable[str]) -> None:
    task = asyncio.ensure_future(call)
    _ = await asyncio.wait_for(started.wait(), timeout=5)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancelled_while_the_store_get_is_awaited_writes_no_record() -> None:
    entered = asyncio.Event()
    configure_llm_cache(_BlockingGet(entered))
    with install(_in_turn("never")) as installed, capture_llm_records() as records:
        await _cancel_when(entered, text_llm_call("q", feature="f", on_result=lambda _t: None))

    assert (len(installed.requests), records) == (0, [])


@pytest.mark.asyncio
async def test_cancelled_while_following_a_leader_writes_no_record() -> None:
    configure_llm_cache(InMemoryLLMCache())
    following = asyncio.Event()
    release = asyncio.Event()

    def watch(key: str) -> tuple[bool, asyncio.Future[LLMCacheEntry | None]]:
        leader, flight = _flight.lead_or_follow(key)
        if not leader:
            following.set()
        return leader, flight

    async def gated(_request: TransportRequest) -> Reply:
        _ = await release.wait()
        return Reply.text("leader's")

    with (
        pytest.MonkeyPatch.context() as mp,
        install(gated),
        capture_llm_records() as records,
    ):
        mp.setattr("llmkit.cache.read_through.lead_or_follow", watch)
        leader = asyncio.create_task(text_llm_call("q", feature="f", label="leader"))
        await _cancel_when(
            following,
            text_llm_call("q", feature="f", label="follower", on_result=lambda _t: None),
        )
        release.set()
        _ = await leader

    assert [r.label for r in records] == ["leader"]
