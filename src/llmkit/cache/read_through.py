"""The cache hook the call families share, written once.

Both buffered lanes reach the store through :func:`read_through`, parameterised
by what differs between them (how to run the call, how to encode an answer, how
to accept a stored one), so the structured and text paths cannot drift. The
streamed text lane cannot be an awaitable *run* — its answer is complete only
when its consumer pulls the last chunk — so it takes no part in the single
flight and calls :func:`cached_entry` and :func:`store_entry` directly. Every
cache-side failure degrades to a miss and reports to the one warn-once latch in
:mod:`llmkit.cache.registry`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.cache._flight import land, lead_or_follow
from llmkit.cache.key import llm_cache_key
from llmkit.cache.registry import logger, report_cache_failure
from llmkit.cache.store import LLMCache, LLMCacheEntry
from llmkit.exceptions import ResultValidationError
from llmkit.providers import LLMProviderInterface


def cache_key_or_none(
    *,
    provider: LLMProviderInterface,
    prompt: str | Sequence[ChatMessage],
    output_schema: type[BaseModel] | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
) -> str | None:
    """:func:`~llmkit.cache.llm_cache_key`, or ``None`` when it cannot be computed.

    A prompt with a content part that is not JSON-serialisable, or a
    third-party provider missing an attribute the key reads, must not fail the
    call: it bypasses the cache, stores nothing, and warns once.
    """
    try:
        return llm_cache_key(
            provider=provider,
            prompt=prompt,
            output_schema=output_schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        report_cache_failure("key", exc)
        return None


async def read_through[T](
    cache: LLMCache,
    key: str,
    *,
    lookup: bool,
    run: Callable[[], Awaitable[T]],
    encode: Callable[[T], LLMCacheEntry],
    accept: Callable[[LLMCacheEntry], Awaitable[T]],
) -> T:
    """Answer from *cache* when it can, otherwise run the call once and store it.

    *run* is the call's whole retry pass; only a result that leaves it cleanly
    is encoded and stored, which keeps a raised attempt, an ``on_result``
    rejection and a truncated answer out of the store. *accept* turns a stored
    entry into the call's result and writes the hit record; a
    :class:`~llmkit.ResultValidationError` from it (the caller's ``on_result``
    rejected the stored answer) or any decode failure makes it a miss.

    With *lookup* false — a re-roll pass of an enclosing
    :func:`~llmkit.retry.with_retries` loop, whose caller rejected the earlier
    answer — the store is not read and no in-flight leader is followed: the
    call runs and its new answer replaces the stored one.

    Otherwise a miss joins the single flight for *key*: the first caller runs,
    and identical callers that arrive while it is in flight wait for its entry
    and accept it as a hit, or run themselves if it produced none.
    """
    if not lookup:
        return await _run_and_store(cache, key, run, encode)
    stored = await cached_entry(cache, key)
    if stored is not None:
        hit = await _accept(accept, stored)
        if hit is not None:
            return hit[0]
    leader, flight = lead_or_follow(key)
    if not leader:
        # Shielded so a cancelled follower does not cancel the shared future
        # under every other follower.
        shared = await asyncio.shield(flight)
        if shared is not None:
            hit = await _accept(accept, shared)
            if hit is not None:
                return hit[0]
        return await _run_and_store(cache, key, run, encode)
    entry: LLMCacheEntry | None = None
    try:
        result = await run()
        entry = _encode(encode, result)
    finally:
        # ``finally``, not ``except Exception``: cancellation is a
        # BaseException, and the sync bridge cancels on timeout and shutdown.
        # Followers must never be left waiting on a leader that is gone.
        land(key, flight, entry)
    if entry is not None:
        await store_entry(cache, key, entry)
    return result


async def _run_and_store[T](
    cache: LLMCache,
    key: str,
    run: Callable[[], Awaitable[T]],
    encode: Callable[[T], LLMCacheEntry],
) -> T:
    result = await run()
    entry = _encode(encode, result)
    if entry is not None:
        await store_entry(cache, key, entry)
    return result


async def cached_entry(cache: LLMCache, key: str) -> LLMCacheEntry | None:
    """The entry *cache* holds for *key*, or ``None`` on a miss or any failure."""
    try:
        entry = await cache.get(key)
        # A structural protocol match promises the method exists, never that it
        # honours its return annotation.
        if entry is not None and not isinstance(entry, LLMCacheEntry):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard on a third-party return
            raise TypeError(
                f"LLMCache.get must return LLMCacheEntry | None, got {type(entry).__name__}"
            )
    except Exception as exc:
        report_cache_failure("get", exc)
        return None
    return entry


async def store_entry(cache: LLMCache, key: str, entry: LLMCacheEntry) -> None:
    """Store *entry* under *key*; a failing store is reported, never raised."""
    try:
        await cache.set(key, entry)
    except Exception as exc:
        report_cache_failure("set", exc)


def _encode[T](encode: Callable[[T], LLMCacheEntry], result: T) -> LLMCacheEntry | None:
    try:
        return encode(result)
    except Exception as exc:
        # A custom serializer on the schema raised: the caller still gets its
        # answer, it is just not stored.
        report_cache_failure("encode", exc)
        return None


async def _accept[T](
    accept: Callable[[LLMCacheEntry], Awaitable[T]], entry: LLMCacheEntry
) -> tuple[T] | None:
    try:
        return (await accept(entry),)
    except ResultValidationError:
        # Not a cache failure: this caller's ``on_result`` rejects the stored
        # answer, so it asks the provider for its own.
        logger.debug("on_result rejected a cached answer; treating it as a miss")
        return None
    except Exception as exc:
        report_cache_failure("decode", exc)
        return None
