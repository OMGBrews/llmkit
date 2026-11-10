"""What a response cache must be, and what it stores.

Two async methods, deliberately: the shared contract says nothing about where
entries live, how long they last, or who else can read them, so a host's store
(a database table, a shared key-value service, a per-tenant map) is a
two-method object. Retention is the store's policy, as it is the log sink's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LLMCacheEntry:
    """One stored answer, as a store receives it from ``set`` and returns it from ``get``.

    Plain strings only, so any store can persist it without knowing a Pydantic
    model: ``response`` is the answer text the provider sent, which a
    structured hit validates with ``model_validate_json(text, strict=True)``,
    or the plain text of a text call. ``schema`` is the output schema's
    class name, or ``"text"``. ``provider`` and ``model`` name what produced the
    answer, and ``call_id`` is the paid call that stored it, which a hit record
    reports as its ``source_call_id``.

    There is no expiry or timestamp field: a store that wants time-based expiry
    records the time itself when ``set`` is called.
    """

    response: str
    schema: str
    provider: str | None
    model: str | None
    call_id: str | None


@runtime_checkable
class LLMCache(Protocol):
    """A host-supplied store for provider responses, keyed by request fingerprint.

    ``get`` returns the entry stored under *key*, or ``None`` on a miss.
    ``set`` stores *entry* under *key*, replacing any earlier one. llmkit owns
    the key (see :func:`~llmkit.cache.llm_cache_key`), when to look up and
    when to store, request coalescing, and the hit record; the store owns
    persistence and retention.

    A store that raises never fails the call: llmkit treats a failed ``get``
    as a miss and a failed ``set`` as a dropped write, and logs one warning per
    failure signature until the cache is reconfigured.

    What a store owes llmkit: bound your own I/O, and let ``get`` and ``set``
    be cancelled. llmkit cancels an operation that overruns the ``timeout``
    configured with :func:`~llmkit.cache.configure_llm_cache` (1.0 second
    unless configured) and treats it as failed, as above. That bound works only
    through cancellation: a store that blocks the event loop synchronously, or
    that catches the cancellation and carries on, is outside it.

    ``@runtime_checkable`` so :func:`~llmkit.cache.configure_llm_cache` can
    reject a non-cache at configuration time. The check is structural and
    tests attribute *presence* only, as :class:`~llmkit.LogSink`'s does.
    """

    async def get(self, key: str) -> LLMCacheEntry | None: ...

    async def set(self, key: str, entry: LLMCacheEntry) -> None: ...
