"""A host-pluggable response cache for the buffered calls and the text stream.

With a cache configured, a :func:`~llmkit.structured_llm_call`,
:func:`~llmkit.text_llm_call` or :func:`~llmkit.text_llm_call_stream` whose
resolved request was answered before is answered from the store: no provider
call, no rate-limiter slot, no retry budget — and still one
:class:`~llmkit.LLMCallRecord`, marked ``cache_hit=True``. A streamed and a
buffered text request share one key; a streamed hit replays the stored text as
one chunk. Identical buffered requests in flight at once in one process share
one provider call; streams take no part in that. The host supplies the store; llmkit owns everything that
needs the resolved request.

Off by default: with no cache configured every request is byte-identical to a
library without this package. ``cache=False`` on a call or on
:class:`~llmkit.LLMCallOptions` opts one call out — for a caller that re-sends
a prompt to get a different sample. The tool families do not read the cache.

Module layout
-------------

* :mod:`~llmkit.cache.store` — the :class:`LLMCache` protocol and the
  :class:`LLMCacheEntry` it stores;
* :mod:`~llmkit.cache.registry` — the configured cache, its
  :func:`get_llm_cache` reader, and the warn-once latch every cache-side
  failure reports to;
* :mod:`~llmkit.cache.key` — :func:`llm_cache_key`, the request fingerprint;
* :mod:`~llmkit.cache.memory` — :class:`InMemoryLLMCache`, the reference store;
* :mod:`~llmkit.cache.read_through` — the hook the lanes that read the cache
  call;
* ``_flight`` — the per-loop single-flight registry.
"""

from llmkit.cache.key import LLM_CACHE_KEY_VERSION, llm_cache_key
from llmkit.cache.memory import InMemoryLLMCache
from llmkit.cache.registry import configure_llm_cache, get_llm_cache
from llmkit.cache.store import LLMCache, LLMCacheEntry

__all__ = [
    "LLM_CACHE_KEY_VERSION",
    "InMemoryLLMCache",
    "LLMCache",
    "LLMCacheEntry",
    "configure_llm_cache",
    "get_llm_cache",
    "llm_cache_key",
]
