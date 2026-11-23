"""The shipped reference store: a bounded in-process LRU."""

from __future__ import annotations

import threading
from collections import OrderedDict

from llmkit.cache.store import LLMCacheEntry


class InMemoryLLMCache:
    """An :class:`~llmkit.cache.LLMCache` that keeps the most recent entries in memory.

    Holds at most ``max_entries`` entries and evicts the least recently used
    one past that bound. Nothing expires by age, and nothing survives the
    process: this is the reference store and the right one for a script or a
    test run, while a host that wants persistence, sharing between workers, or
    tenant scoping supplies its own store.

    Guarded by a :class:`threading.Lock` rather than an asyncio lock, because
    one instance is reached from more than one event loop on more than one
    thread (llmkit's sync wrappers run on their own persistent loop).
    """

    def __init__(self, max_entries: int = 256) -> None:
        """Create an empty store.

        Raises:
            ValueError: if ``max_entries`` is not a positive non-bool integer.
        """
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError(f"max_entries must be an integer >= 1, got {max_entries!r}")
        self.max_entries: int = max_entries
        self._entries: OrderedDict[str, LLMCacheEntry] = OrderedDict()
        # A forked child gets a fresh copy of this lock only for the configured
        # store (see ``reset_after_fork``); a store the host holds elsewhere is
        # the host's to reset, as with a custom log sink.
        self._lock: threading.Lock = threading.Lock()

    async def get(self, key: str) -> LLMCacheEntry | None:
        """Return the entry stored under *key*, marking it most recently used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        """Store *entry* under *key*, evicting the least recently used past the bound."""
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                _ = self._entries.popitem(last=False)

    def __len__(self) -> int:
        """The number of entries currently held."""
        with self._lock:
            return len(self._entries)


def reset_after_fork(cache: InMemoryLLMCache) -> None:
    """Give a forked child's copy of *cache* an unheld lock; its entries stay.

    Internal: the cache's fork hook calls this for the configured store,
    between ``fork`` and the child's first line of code, where the child's
    copy would otherwise stay held forever if a parent thread held it. Takes
    no lock and cannot raise.
    """
    cache._lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]  # the module that owns the class
