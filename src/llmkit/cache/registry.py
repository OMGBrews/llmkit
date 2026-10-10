"""The configured cache, and the one latch every cache-side failure reports to.

The process-global cache and the functions that mutate and read it are one
unit, for the reason :mod:`llmkit.logging.registry` gives for its sink: a
``from ... import _cache`` in a second module would bind a snapshot that
:func:`configure_llm_cache` could never update.
"""

from __future__ import annotations

import logging

from llmkit._latch import OnceLatch
from llmkit.cache.store import LLMCache

# Named explicitly so every module in the package emits under one logger name.
logger = logging.getLogger("llmkit.cache")

# ``None`` is the default: no cache, and every request byte-identical to a
# library without one.
_cache: LLMCache | None = None

# One warn-once latch for every cache-side failure — computing the key,
# ``get``, ``set``, encoding or decoding an entry. Unlike the sink's latch it
# is re-armed only by :func:`configure_llm_cache`, never by a success: a store
# whose ``get`` keeps raising while its ``set`` works would otherwise warn on
# every call, re-armed each time by the other half.
_cache_latch = OnceLatch()


def configure_llm_cache(cache: LLMCache | None) -> None:
    """Set the response cache the call functions read through.

    Pass ``None`` (the default state) to turn caching off. With a cache
    configured, :func:`~llmkit.structured_llm_call`,
    :func:`~llmkit.text_llm_call` (and their sync wrappers) and
    :func:`~llmkit.text_llm_call_stream` answer a request whose fingerprint
    was stored by an earlier successful call from the store and store each new
    successful answer; the buffered calls also coalesce identical requests in
    flight at once into one provider call. ``cache=False`` on a call or on
    :class:`~llmkit.LLMCallOptions` opts that call out.

    Re-arms the warn-once latch, so a newly configured store gets a fresh loud
    first warning if it too turns out to be broken.

    Raises:
        TypeError: if *cache* is neither ``None`` nor an :class:`LLMCache`
            instance — including passing the class rather than an instance of
            it. The structural check tests attribute presence only, so, as
            with :func:`~llmkit.configure_llm_logging`, a bare ``MagicMock()``
            does not match; stub with a small fake class.
    """
    global _cache
    if isinstance(cache, type):
        # A class object carries ``get`` and ``set`` as attributes, so it passes
        # the structural check below while failing every call on the missing
        # ``self``.
        raise TypeError(
            "cache must be an LLMCache instance, not the class itself — "
            + f"did you mean {cache.__name__}()?"
        )
    if cache is not None and not isinstance(cache, LLMCache):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard at public boundary
        raise TypeError(  # pyright: ignore[reportUnreachable]  # reachable from untyped callers
            "cache must implement LLMCache (async get(key) and set(key, entry)) or be None, "
            + f"got {type(cache).__name__}"
        )
    _cache = cache
    _cache_latch.succeeded()


def get_llm_cache() -> LLMCache | None:
    """Return the cache currently configured, or ``None`` when caching is off.

    The read half of :func:`configure_llm_cache`, so a host that installs a
    cache temporarily can restore what was there before.
    """
    return _cache


def report_cache_failure(what: str, exc: BaseException) -> None:
    """Log a cache-side failure: WARNING for a new signature, DEBUG for a repeat.

    *what* names the step that failed (``"get"``, ``"set"``, ``"key"``,
    ``"decode"``, ``"encode"``); the call proceeds as a miss either way.
    """
    if _cache_latch.should_warn(exc):
        logger.warning(
            "LLM response cache %s failed; the call proceeds as a miss "
            + "(further identical failures logged at DEBUG)",
            what,
            exc_info=exc,
        )
    else:
        logger.debug("LLM response cache %s failed", what, exc_info=exc)
