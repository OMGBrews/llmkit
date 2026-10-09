"""The cache seen through the buffered call functions, over the patched seam.

Each test patches the transport seam in :mod:`llmkit._litellm` and counts its
invocations, so "answered from the cache" is a positive observation — the
transport was not awaited — rather than the absence of an error.
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
from tests._support import NO_USAGE, UsageCounts
from tests.cache.conftest import cache_provider


class _Answer(BaseModel):
    verdict: str


@dataclass
class _Transport:
    """A fake transport seam: returns *answers* in turn and counts calls."""

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
    transport = _Transport()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        first = await text_llm_call("q", feature="f", provider=provider)
        second = await text_llm_call("q", feature="f", provider=provider)

    assert transport.calls == 1
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
    transport = _Transport()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        a = await text_llm_call("q", feature="f", provider=provider)
        b = await text_llm_call("q", feature="f", provider=provider, temperature=0.2)

    assert (a, b, transport.calls) == ("first", "second", 2)


@pytest.mark.asyncio
async def test_feature_and_label_do_not_split_hits() -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        _ = await text_llm_call("q", feature="drafting", provider=provider)
        hit = await text_llm_call("q", feature="review", label="x", provider=provider)

    assert (hit, transport.calls) == ("first", 1)


@pytest.mark.asyncio
async def test_no_cache_configured_always_reaches_the_transport() -> None:
    transport = _Transport()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        a = await text_llm_call("q", feature="f", provider=provider)
        b = await text_llm_call("q", feature="f", provider=provider)

    assert (a, b, transport.calls) == ("first", "second", 2)


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

    async def _first_fails(*_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        calls[0] += 1
        if calls[0] == 1:
            _ = await release.wait()
            raise ValueError("provider rejected the request")
        return "recovered", None, NO_USAGE

    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=_first_fails):
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
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_follower_runs_itself_when_the_leader_is_cancelled() -> None:
    """Cancellation is a BaseException; followers must not hang on it."""
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()
    calls = [0]

    async def _gated(*_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        calls[0] += 1
        if calls[0] == 1:
            _ = await asyncio.Event().wait()  # never answers
        _ = await release.wait()
        return "follower's own", None, NO_USAGE

    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=_gated):
        leader = asyncio.create_task(text_llm_call("q", feature="f", provider=provider))
        follower = asyncio.create_task(text_llm_call("q", feature="f", provider=provider))
        for _ in range(5):
            await asyncio.sleep(0)
        _ = leader.cancel()
        release.set()
        result = await asyncio.wait_for(follower, timeout=5)

    assert result == "follower's own"
    assert leader.cancelled()
    assert calls[0] == 2


# --- what is never stored --------------------------------------------------


@pytest.mark.asyncio
async def test_a_raised_attempt_is_not_stored() -> None:
    configure_llm_cache(cache := InMemoryLLMCache())
    transport = AsyncMock(side_effect=[ValueError("boom"), ("ok", None, NO_USAGE)])
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", transport):
        with pytest.raises(ValueError, match="boom"):
            _ = await text_llm_call("q", feature="f", provider=provider, retry=NO_RETRY)
        assert len(cache) == 0
        assert await text_llm_call("q", feature="f", provider=provider) == "ok"

    assert transport.await_count == 2


@pytest.mark.asyncio
async def test_an_on_result_rejection_is_not_stored() -> None:
    configure_llm_cache(cache := InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()

    def _reject(_text: str) -> None:
        raise ResultValidationError("not this one")

    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
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
    transport = _Transport()
    provider = cache_provider()
    opt_out: dict[str, object] = (
        {"cache": False} if via == "keyword" else {"options": LLMCallOptions(cache=False)}
    )
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        # Not stored...
        a = await text_llm_call("q", feature="f", provider=provider, **opt_out)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat
        assert len(cache) == 0
        b = await text_llm_call("q", feature="f", provider=provider)
        # ...and not looked up, though an entry now exists.
        c = await text_llm_call("q", feature="f", provider=provider, **opt_out)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat

    assert (a, b, c) == ("first", "second", "third")
    assert transport.calls == 3


@pytest.mark.asyncio
async def test_keyword_overrides_options_cache() -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()
    off = LLMCallOptions(cache=False)
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        _ = await text_llm_call("q", feature="f", provider=provider, options=off, cache=True)
        hit = await text_llm_call("q", feature="f", provider=provider, options=off, cache=True)

    assert (hit, transport.calls) == ("first", 1)


@pytest.mark.asyncio
async def test_cache_false_skips_single_flight() -> None:
    configure_llm_cache(InMemoryLLMCache())
    release = asyncio.Event()
    calls = [0]

    async def _gated(*_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        calls[0] += 1
        _ = await release.wait()
        return "x", None, NO_USAGE

    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=_gated):
        tasks = [
            asyncio.create_task(text_llm_call("q", feature="f", provider=provider, cache=False))
            for _ in range(2)
        ]
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        _ = await asyncio.gather(*tasks)

    assert calls[0] == 2


# --- the caller rejecting a stored answer ----------------------------------


@pytest.mark.asyncio
async def test_on_result_rejecting_a_cached_answer_is_a_miss() -> None:
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()

    def _only_second(text: str) -> None:
        if text != "second":
            raise ResultValidationError(f"{text} rejected")

    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        _ = await text_llm_call("q", feature="f", provider=provider)
        picky = await text_llm_call("q", feature="f", provider=provider, on_result=_only_second)
        # The accepted answer replaced the stored one.
        later = await text_llm_call("q", feature="f", provider=provider)

    assert (picky, later, transport.calls) == ("second", "second", 2)


@pytest.mark.asyncio
async def test_outer_retry_reroll_skips_the_lookup() -> None:
    """The documented re-roll pattern — ``with_retries`` around a call with
    ``retry=NO_RETRY`` — asks for a new sample on each later pass. Those passes
    skip the lookup, so the stored answer cannot become a fixed point."""
    configure_llm_cache(InMemoryLLMCache())
    transport = _Transport()
    provider = cache_provider()
    seen: list[str] = []

    async def _draft() -> str:
        text = await text_llm_call("q", feature="draft", provider=provider, retry=NO_RETRY)
        seen.append(text)
        if text == "first":
            raise ResultValidationError("host rejects the first draft")
        return text

    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        _ = await text_llm_call("q", feature="draft", provider=provider)  # stores "first"
        accepted = await with_retries(
            _draft, max_attempts=3, validation_retry_on=(ResultValidationError,)
        )
        after = await text_llm_call("q", feature="draft", provider=provider)

    # Pass 1 hit the stored "first"; pass 2 skipped the lookup and paid.
    assert seen == ["first", "second"]
    assert accepted == "second"
    assert after == "second"
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_aliases_survive_the_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    from pydantic import ConfigDict, Field

    class _Aliased(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=False)
        score: int = Field(alias="Score")

    configure_llm_cache(InMemoryLLMCache())
    transport = AsyncMock(return_value=(_Aliased(Score=3), None, NO_USAGE))
    provider = cache_provider()
    with (
        caplog.at_level(logging.DEBUG, logger="llmkit.cache"),
        patch("llmkit._litellm.acompletion_structured", transport),
    ):
        _ = await structured_llm_call("q", _Aliased, feature="f", provider=provider)
        hit = await structured_llm_call("q", _Aliased, feature="f", provider=provider)

    assert hit.score == 3
    assert transport.await_count == 1
    _no_cache_warning(caplog)
