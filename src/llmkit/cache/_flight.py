"""Single flight: identical requests in flight at once share one provider call.

The registry holds in-flight work only — an entry exists from the moment a
leader starts a miss until it finishes, and the store owns everything after
that. It is keyed per event loop, as the rate limiter keys its gates, because
an :class:`asyncio.Future` cannot be awaited from a loop other than its own
and llmkit's sync wrappers run on a loop of their own.
"""

from __future__ import annotations

import asyncio
import threading

from llmkit.cache.store import LLMCacheEntry

type _Flight = asyncio.Future[LLMCacheEntry | None]

_lock = threading.Lock()
_in_flight: dict[tuple[asyncio.AbstractEventLoop, str], _Flight] = {}


def lead_or_follow(key: str) -> tuple[bool, _Flight]:
    """Join the flight for *key* on the running loop: ``(True, fut)`` makes the
    caller its leader, ``(False, fut)`` a follower of an existing one."""
    loop = asyncio.get_running_loop()
    with _lock:
        for stale in [k for k in _in_flight if k[0].is_closed()]:
            del _in_flight[stale]
        flight = _in_flight.get((loop, key))
        if flight is not None:
            return False, flight
        flight = loop.create_future()
        _in_flight[(loop, key)] = flight
        return True, flight


def land(key: str, flight: _Flight, entry: LLMCacheEntry | None) -> None:
    """End the leader's flight: deregister it and hand *entry* to its followers.

    ``None`` is the "no value" marker — the leader raised, was cancelled, or
    could not encode its answer — and sends every follower to the provider on
    its own retry budget.
    """
    with _lock:
        registry_key = (flight.get_loop(), key)
        if _in_flight.get(registry_key) is flight:
            del _in_flight[registry_key]
    if not flight.done():
        flight.set_result(entry)
