"""The cache seen through the buffered call functions, over an installed fake.

Each test answers the provider round trip with an :mod:`llmkit.testing` fake
and counts the requests it received. The cache sits above the fake, so
"answered from the cache" is a positive observation — the request never
reached the provider — rather than the absence of an error.

Four structured-call tests stay on the private ``acompletion_structured``
seam: ``test_structured_hit_skips_the_transport``, ``test_sync_wrappers_hit``,
``test_identical_requests_in_flight_share_one_transport_call`` and
``test_a_truncated_answer_is_not_stored``. Through the real parser a paid
structured answer is an instance of instructor's response-model subclass of
the schema, not of the schema itself, so it compares unequal to the cached
copy and to a freshly built instance, which these tests assert. That is
reported as a likely defect, not rewritten around.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCallOptions,
    OutputLimitError,
    ResultValidationError,
    configure_llm_cache,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_sync,
)
from llmkit.retry import with_retries
from llmkit.testing import Reply, TransportRequest, install
from tests._support import NO_USAGE, UsageCounts
from tests.cache.conftest import CALL_MODE, cache_provider


class _Answer(BaseModel):
    verdict: str


@dataclass
class _InTurn:
    """A fake provider answering *answers* in turn, one per round trip.

    A structured call gets the answer as ``_Answer(verdict=...)`` JSON, a text
    call as plain text. The cache sits above the fake, so a hit or a joined
    in-flight request never reaches it: ``len(installed.requests)`` counts the
    calls that paid.
    """

    answers: list[str] = field(default_factory=lambda: ["first", "second", "third"])
    served: int = 0

    def __call__(self, request: TransportRequest) -> Reply:
        self.served += 1
        answer = self.answers[self.served - 1]
        if request.kind == "structured":
            return Reply.json(_Answer(verdict=answer))
        return Reply.text(answer)


@dataclass
class _Transport:
    """A fake transport seam for the tests still on it: returns *answers* in
    turn and counts calls."""

    answers: list[str] = field(default_factory=lambda: ["first", "second", "third"])
    calls: int = 0

    async def text(self, *_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        self.calls += 1
        return self.answers[self.calls - 1], 0.002, (10, 5, 15)

    async def structured(
        self, *_a: object, **_k: object
    ) -> tuple[_Answer, float | None, UsageCounts]:
        text, cost, usage = await self.text()
        return _Answer(verdict=text), cost, usage


def _no_cache_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The key computed and the store answered — no silent bypass."""
    assert not [r for r in caplog.records if r.name == "llmkit.cache"]


@pytest.mark.asyncio
async def test_structured_hit_skips_the_transport(caplog: pytest.LogCaptureFixture) -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_structured", side_effect=transport.structured):
        first = await structured_llm_call("q", _Answer, feature="f", provider=provider)
        second = await structured_llm_call("q", _Answer, feature="f", provider=provider)

    assert transport.calls == 1
    assert first == second == _Answer(verdict="first")
    assert first is not second
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_text_hit_skips_the_transport(caplog: pytest.LogCaptureFixture) -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    with install(_InTurn()) as installed:
        first = await text_llm_call("q", feature="f", provider=provider)
        second = await text_llm_call("q", feature="f", provider=provider)

    assert len(installed.requests) == 1
    assert first == second == "first"
    _no_cache_warning(caplog)


def test_sync_wrappers_hit(caplog: pytest.LogCaptureFixture) -> None:
    """The sync bridge runs on its own loop; a hit works there too."""
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()
    with (
        patch("llmkit._litellm.acompletion_text", side_effect=transport.text),
        patch("llmkit._litellm.acompletion_structured", side_effect=transport.structured),
    ):
        texts = [text_llm_call_sync("q", feature="f", provider=provider) for _ in range(2)]
        parsed = [
            structured_llm_call_sync("q", _Answer, feature="f", provider=provider) for _ in range(2)
        ]

    assert transport.calls == 2
    assert texts == ["first", "first"]
    assert parsed == [_Answer(verdict="second"), _Answer(verdict="second")]
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_different_request_misses() -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    with install(_InTurn()) as installed:
        a = await text_llm_call("q", feature="f", provider=provider)
        b = await text_llm_call("q", feature="f", provider=provider, temperature=0.2)

    assert (a, b, len(installed.requests)) == ("first", "second", 2)


@pytest.mark.asyncio
async def test_feature_and_label_do_not_split_hits() -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    with install(_InTurn()) as installed:
        _ = await text_llm_call("q", feature="drafting", provider=provider)
        hit = await text_llm_call("q", feature="review", label="x", provider=provider)

    assert (hit, len(installed.requests)) == ("first", 1)


@pytest.mark.asyncio
async def test_no_cache_configured_always_reaches_the_transport() -> None:
    provider = cache_provider()
    with install(_InTurn()) as installed:
        a = await text_llm_call("q", feature="f", provider=provider)
        b = await text_llm_call("q", feature="f", provider=provider)

    assert (a, b, len(installed.requests)) == ("first", "second", 2)


# --- in flight at once -----------------------------------------------------


@pytest.mark.asyncio
async def test_identical_requests_in_flight_share_one_transport_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()
    calls = [0]

    async def _gated(*_a: object, **_k: object) -> tuple[_Answer, float | None, UsageCounts]:
        calls[0] += 1
        _ = await release.wait()
        return _Answer(verdict=f"answer-{calls[0]}"), 0.01, NO_USAGE

    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_structured", side_effect=_gated):
        leader = asyncio.create_task(
            structured_llm_call("q", _Answer, feature="coach", provider=provider)
        )
        follower = asyncio.create_task(
            structured_llm_call("q", _Answer, feature="coach", provider=provider)
        )
        # Both are in flight before the transport answers.
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(leader, follower)

    assert calls[0] == 1
    assert results[0] == results[1] == _Answer(verdict="answer-1")
    _no_cache_warning(caplog)


@pytest.mark.asyncio
async def test_follower_runs_itself_when_the_leader_raises() -> None:
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()
    calls = [0]

    async def _first_fails(_request: TransportRequest) -> Reply:
        calls[0] += 1
        if calls[0] == 1:
            _ = await release.wait()
            raise ValueError("provider rejected the request")
        return Reply.text("recovered")

    provider = cache_provider()
    with install(_first_fails) as installed:
        leader = asyncio.create_task(
            text_llm_call("q", feature="f", provider=provider, retry=NO_RETRY)
        )
        follower = asyncio.create_task(
            text_llm_call("q", feature="f", provider=provider, retry=NO_RETRY)
        )
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(leader, follower, return_exceptions=True)

    assert isinstance(results[0], ValueError)
    assert results[1] == "recovered"
    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_follower_runs_itself_when_the_leader_is_cancelled() -> None:
    """Cancellation is a BaseException; followers must not hang on it."""
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()
    calls = [0]

    async def _gated(_request: TransportRequest) -> Reply:
        calls[0] += 1
        if calls[0] == 1:
            _ = await asyncio.Event().wait()  # never answers
        _ = await release.wait()
        return Reply.text("follower's own")

    provider = cache_provider()
    with install(_gated) as installed:
        leader = asyncio.create_task(text_llm_call("q", feature="f", provider=provider))
        follower = asyncio.create_task(text_llm_call("q", feature="f", provider=provider))
        for _ in range(5):
            await asyncio.sleep(0)
        _ = leader.cancel()
        release.set()
        result = await asyncio.wait_for(follower, timeout=5)

    assert result == "follower's own"
    assert leader.cancelled()
    assert len(installed.requests) == 2


# --- what is never stored --------------------------------------------------


@pytest.mark.asyncio
async def test_a_raised_attempt_is_not_stored() -> None:
    configure_llm_cache(cache := InMemoryLLMCache())

    replies: list[Exception | Reply] = [ValueError("boom"), Reply.text("ok")]

    def _fails_once(_request: TransportRequest) -> Reply:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    provider = cache_provider()
    with install(_fails_once) as installed:
        with pytest.raises(ValueError, match="boom"):
            _ = await text_llm_call("q", feature="f", provider=provider, retry=NO_RETRY)
        assert len(cache) == 0
        assert await text_llm_call("q", feature="f", provider=provider) == "ok"

    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_an_on_result_rejection_is_not_stored() -> None:
    configure_llm_cache(cache := InMemoryLLMCache())
    provider = cache_provider()

    def _reject(_text: str) -> None:
        raise ResultValidationError("not this one")

    with install(_InTurn()):
        with pytest.raises(ResultValidationError):
            _ = await text_llm_call(
                "q", feature="f", provider=provider, retry=NO_RETRY, on_result=_reject
            )
        assert len(cache) == 0
        assert await text_llm_call("q", feature="f", provider=provider) == "second"


@pytest.mark.asyncio
async def test_a_truncated_answer_is_not_stored() -> None:
    configure_llm_cache(cache := InMemoryLLMCache())
    truncated = OutputLimitError(model="gemini/x", max_tokens=8, completion_tokens=8)
    transport = AsyncMock(side_effect=[truncated, (_Answer(verdict="whole"), None, NO_USAGE)])
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_structured", transport):
        with pytest.raises(OutputLimitError):
            _ = await structured_llm_call("q", _Answer, feature="f", provider=provider)
        assert len(cache) == 0
        whole = await structured_llm_call("q", _Answer, feature="f", provider=provider)

    assert whole == _Answer(verdict="whole")
    assert transport.await_count == 2


# --- opting out ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["keyword", "options"])
async def test_cache_false_bypasses_lookup_and_store(via: str) -> None:
    configure_llm_cache(cache := InMemoryLLMCache())
    provider = cache_provider()
    opt_out: dict[str, object] = (
        {"cache": False} if via == "keyword" else {"options": LLMCallOptions(cache=False)}
    )
    with install(_InTurn()) as installed:
        # Not stored...
        a = await text_llm_call("q", feature="f", provider=provider, **opt_out)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat
        assert len(cache) == 0
        b = await text_llm_call("q", feature="f", provider=provider)
        # ...and not looked up, though an entry now exists.
        c = await text_llm_call("q", feature="f", provider=provider, **opt_out)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat

    assert (a, b, c) == ("first", "second", "third")
    assert len(installed.requests) == 3


@pytest.mark.asyncio
async def test_keyword_overrides_options_cache() -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    off = LLMCallOptions(cache=False)
    with install(_InTurn()) as installed:
        _ = await text_llm_call("q", feature="f", provider=provider, options=off, cache=True)
        hit = await text_llm_call("q", feature="f", provider=provider, options=off, cache=True)

    assert (hit, len(installed.requests)) == ("first", 1)


@pytest.mark.asyncio
async def test_cache_false_skips_single_flight() -> None:
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()

    async def _gated(_request: TransportRequest) -> Reply:
        _ = await release.wait()
        return Reply.text("x")

    provider = cache_provider()
    with install(_gated) as installed:
        tasks = [
            asyncio.create_task(text_llm_call("q", feature="f", provider=provider, cache=False))
            for _ in range(2)
        ]
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        _ = await asyncio.gather(*tasks)

    assert len(installed.requests) == 2


# --- the caller rejecting a stored answer ----------------------------------


@pytest.mark.asyncio
async def test_on_result_rejecting_a_cached_answer_is_a_miss() -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()

    def _only_second(text: str) -> None:
        if text != "second":
            raise ResultValidationError(f"{text} rejected")

    with install(_InTurn()) as installed:
        _ = await text_llm_call("q", feature="f", provider=provider)
        picky = await text_llm_call("q", feature="f", provider=provider, on_result=_only_second)
        # The accepted answer replaced the stored one.
        later = await text_llm_call("q", feature="f", provider=provider)

    assert (picky, later, len(installed.requests)) == ("second", "second", 2)


@pytest.mark.asyncio
async def test_outer_retry_reroll_skips_the_lookup() -> None:
    """The documented re-roll pattern — ``with_retries`` around a call with
    ``retry=NO_RETRY`` — asks for a new sample on each later pass. Those passes
    skip the lookup, so the stored answer cannot become a fixed point."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    seen: list[str] = []

    async def _draft() -> str:
        text = await text_llm_call("q", feature="draft", provider=provider, retry=NO_RETRY)
        seen.append(text)
        if text == "first":
            raise ResultValidationError("host rejects the first draft")
        return text

    with install(_InTurn()) as installed:
        _ = await text_llm_call("q", feature="draft", provider=provider)  # stores "first"
        accepted = await with_retries(
            _draft, max_attempts=3, validation_retry_on=(ResultValidationError,)
        )
        after = await text_llm_call("q", feature="draft", provider=provider)

    # Pass 1 hit the stored "first"; pass 2 skipped the lookup and paid.
    assert seen == ["first", "second"]
    assert accepted == "second"
    assert after == "second"
    assert len(installed.requests) == 2


@pytest.mark.asyncio
async def test_aliases_survive_the_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    from pydantic import ConfigDict, Field

    class _Aliased(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=False)
        score: int = Field(alias="Score")

    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider(instructor_mode=CALL_MODE)
    with (
        caplog.at_level(logging.DEBUG, logger="llmkit.cache"),
        install(lambda _request: Reply.json(_Aliased(Score=3))) as installed,
    ):
        _ = await structured_llm_call("q", _Aliased, feature="f", provider=provider)
        hit = await structured_llm_call("q", _Aliased, feature="f", provider=provider)

    assert hit.score == 3
    assert len(installed.requests) == 1
    _no_cache_warning(caplog)
