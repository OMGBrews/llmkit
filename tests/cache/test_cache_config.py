"""Configuring the cache, the shipped store, and failures that must not fail a call."""

from __future__ import annotations

import logging
from typing import cast, override
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    ChatMessage,
    InMemoryLLMCache,
    LLMCache,
    LLMCacheEntry,
    configure_llm_cache,
    get_llm_cache,
    structured_llm_call,
    text_llm_call,
)
from tests._support import UsageCounts
from tests.cache.conftest import cache_provider


def _entry(response: str = "x") -> LLMCacheEntry:
    return LLMCacheEntry(response=response, schema="text", provider="p", model="m", call_id="c")


class _DictStore:
    """A minimal third-party store; the failing stores below override one half."""

    def __init__(self) -> None:
        self.entries: dict[str, LLMCacheEntry] = {}

    async def get(self, key: str) -> LLMCacheEntry | None:
        return self.entries.get(key)

    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        self.entries[key] = entry


def _cache_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "llmkit.cache" and r.levelno == logging.WARNING]


class _Counting:
    def __init__(self) -> None:
        self.calls: int = 0

    async def text(self, *_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        self.calls += 1
        return f"answer-{self.calls}", None, (None, None, None)


# --- configure_llm_cache ---------------------------------------------------


def test_configure_and_read_back() -> None:
    assert get_llm_cache() is None
    cache = InMemoryLLMCache()
    configure_llm_cache(cache)
    assert get_llm_cache() is cache
    assert isinstance(cache, LLMCache)
    configure_llm_cache(None)
    assert get_llm_cache() is None


@pytest.mark.parametrize("bogus", [object(), "redis://", MagicMock(), {"get": 1}])
def test_configure_rejects_a_non_cache(bogus: object) -> None:
    with pytest.raises(TypeError, match="must implement LLMCache"):
        configure_llm_cache(bogus)  # pyright: ignore[reportArgumentType]  # the runtime guard under test
    assert get_llm_cache() is None


def test_configure_rejects_the_class_object() -> None:
    with pytest.raises(TypeError, match=r"did you mean InMemoryLLMCache\(\)\?"):
        configure_llm_cache(InMemoryLLMCache)  # pyright: ignore[reportArgumentType]  # the runtime guard under test


def test_a_third_party_store_is_accepted() -> None:
    configure_llm_cache(store := _DictStore())
    assert get_llm_cache() is store


# --- InMemoryLLMCache ------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_store_evicts_least_recently_used() -> None:
    cache = InMemoryLLMCache(max_entries=2)
    await cache.set("a", _entry("a"))
    await cache.set("b", _entry("b"))
    assert await cache.get("a") == _entry("a")  # "a" is now most recent
    await cache.set("c", _entry("c"))

    assert len(cache) == 2
    assert await cache.get("b") is None
    assert await cache.get("a") == _entry("a")
    assert await cache.get("c") == _entry("c")


@pytest.mark.asyncio
async def test_in_memory_store_replaces_an_entry() -> None:
    cache = InMemoryLLMCache()
    await cache.set("k", _entry("old"))
    await cache.set("k", _entry("new"))
    assert (len(cache), await cache.get("k")) == (1, _entry("new"))


@pytest.mark.parametrize("bad", [0, -1])
def test_in_memory_store_rejects_a_useless_bound(bad: int) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        _ = InMemoryLLMCache(max_entries=bad)


# --- failures degrade to a miss --------------------------------------------


class _GetRaises(_DictStore):
    @override
    async def get(self, key: str) -> LLMCacheEntry | None:
        raise ConnectionError(f"store unreachable reading {key[:8]}")


class _SetRaises(_DictStore):
    @override
    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        raise ConnectionError(f"store unreachable writing {entry.schema} to {key[:8]}")


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [_GetRaises, _SetRaises])
async def test_a_raising_store_degrades_to_a_miss_and_warns_once(
    store: type[LLMCache], caplog: pytest.LogCaptureFixture
) -> None:
    """A store whose ``get`` keeps failing while its ``set`` works still warns
    once in total, not once per call: a success does not re-arm the latch."""
    configure_llm_cache(store())
    transport = _Counting()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        results = [await text_llm_call("q", feature="f", provider=provider) for _ in range(3)]

    assert results == ["answer-1", "answer-2", "answer-3"]
    (warning,) = _cache_warnings(caplog)
    assert warning.exc_info is not None and warning.exc_info[0] is ConnectionError


@pytest.mark.asyncio
async def test_reconfiguring_rearms_the_warning(caplog: pytest.LogCaptureFixture) -> None:
    transport = _Counting()
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        configure_llm_cache(_GetRaises())
        _ = await text_llm_call("q", feature="f", provider=provider)
        _ = await text_llm_call("q", feature="f", provider=provider)
        configure_llm_cache(_GetRaises())
        _ = await text_llm_call("q", feature="f", provider=provider)

    assert len(_cache_warnings(caplog)) == 2


@pytest.mark.asyncio
async def test_a_store_returning_junk_is_a_miss(caplog: pytest.LogCaptureFixture) -> None:
    class _Junk(_DictStore):
        @override
        async def get(self, key: str) -> LLMCacheEntry | None:
            return cast("LLMCacheEntry", cast("object", f"not an entry for {key}"))

    configure_llm_cache(_Junk())
    transport = _Counting()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        assert await text_llm_call("q", feature="f", provider=cache_provider()) == "answer-1"
    assert len(_cache_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_an_undecodable_entry_is_a_miss(caplog: pytest.LogCaptureFixture) -> None:
    class _Answer(BaseModel):
        n: int

    class _Corrupt(_DictStore):
        @override
        async def get(self, key: str) -> LLMCacheEntry | None:
            return LLMCacheEntry("{not json", "_Answer", None, None, call_id=key)

    configure_llm_cache(_Corrupt())
    transport = AsyncMock(return_value=(_Answer(n=1), None, (None, None, None)))
    with patch("llmkit._litellm.acompletion_structured", transport):
        got = await structured_llm_call("q", _Answer, feature="f", provider=cache_provider())

    assert got == _Answer(n=1)
    assert transport.await_count == 1
    assert len(_cache_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_an_unkeyable_request_bypasses_the_cache(caplog: pytest.LogCaptureFixture) -> None:
    """A content part that is not JSON-serialisable cannot be fingerprinted:
    the call still runs, nothing is stored, and one warning says why."""
    configure_llm_cache(cache := InMemoryLLMCache())
    transport = _Counting()
    # Not a ChatMessage the types admit, which is the point: bytes reach the
    # call from an untyped caller and cannot be JSON-serialised for the key.
    blob = cast("object", {"role": "user", "content": [{"type": "blob", "data": b"\x00"}]})
    prompt = [cast("ChatMessage", blob)]
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        a = await text_llm_call(prompt, feature="f", provider=cache_provider())
        b = await text_llm_call(prompt, feature="f", provider=cache_provider())

    assert (a, b, len(cache)) == ("answer-1", "answer-2", 0)
    assert len(_cache_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_a_provider_the_key_cannot_read_bypasses_the_cache(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare ``MagicMock`` provider — every attribute an unserialisable mock —
    is the shape most third-party test doubles take."""
    configure_llm_cache(cache := InMemoryLLMCache())
    transport = _Counting()
    with patch("llmkit._litellm.acompletion_text", side_effect=transport.text):
        _ = await text_llm_call("q", feature="f", provider=MagicMock())
    assert (transport.calls, len(cache)) == (1, 0)
    assert len(_cache_warnings(caplog)) == 1


# --- no cache, no change ---------------------------------------------------


def _request_kwargs(*, configure: bool) -> dict[str, object]:
    """The kwargs ``litellm.acompletion`` receives for one ``text_llm_call``."""
    import asyncio

    seen: dict[str, object] = {}
    response = MagicMock(_hidden_params={})
    response.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def _acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        return response

    configure_llm_cache(InMemoryLLMCache() if configure else None)
    with patch("llmkit._litellm.litellm.acompletion", side_effect=_acompletion):
        _ = asyncio.run(
            text_llm_call(
                "hi", feature="f", provider=cache_provider(), temperature=0.5, max_tokens=64
            )
        )
    configure_llm_cache(None)
    return seen


def test_requests_are_byte_identical_with_and_without_a_cache() -> None:
    """The cache adds nothing to the request: the kwargs reaching LiteLLM are
    the same with no cache configured as on a miss with one."""
    without = _request_kwargs(configure=False)
    with_cache = _request_kwargs(configure=True)
    assert without == with_cache
    assert set(without) == {"model", "messages", "temperature", "api_key", "api_base", "max_tokens"}
    assert repr(without) == repr(with_cache)
