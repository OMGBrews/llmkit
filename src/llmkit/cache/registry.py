"""The configured cache, and the one latch every cache-side failure reports to.

The process-global cache and the functions that mutate and read it are one
unit, for the reason :mod:`llmkit.logging.registry` gives for its sink: a
``from ... import _cache`` in a second module would bind a snapshot that
:func:`configure_llm_cache` could never update.
"""

from __future__ import annotations

import logging

from llmkit._diagnostics import report
from llmkit._durations import duration
from llmkit._latch import OnceLatch
from llmkit.cache.store import LLMCache
from llmkit.options import UNSET, Unset

# Named explicitly so every module in the package emits under one logger name.
logger = logging.getLogger("llmkit.cache")

# ``None`` is the default: no cache, and every request byte-identical to a
# library without one.
_cache: LLMCache | None = None

#: Seconds one store ``get`` or ``set`` may take before llmkit cancels it: a
#: read that overruns is a miss, a write that overruns is dropped.
DEFAULT_CACHE_TIMEOUT = 1.0

# ``None`` turns the bound off. Kept apart from ``_cache`` so a later
# ``configure_llm_cache(store)`` that omits ``timeout`` keeps the configured one.
_timeout: float | None = DEFAULT_CACHE_TIMEOUT

# One warn-once latch for every cache-side failure — computing the key,
# ``get``, ``set``, encoding or decoding an entry. Unlike the sink's latch it
# is re-armed only by :func:`configure_llm_cache`, never by a success: a store
# whose ``get`` keeps raising while its ``set`` works would otherwise warn on
# every call, re-armed each time by the other half.
_cache_latch = OnceLatch()


def configure_llm_cache(cache: LLMCache | None, *, timeout: float | None | Unset = UNSET) -> None:
    """Set the response cache the call functions read through, and its time budget.

    Pass ``None`` (the default state) to turn caching off. With a cache
    configured, :func:`~llmkit.structured_llm_call`,
    :func:`~llmkit.text_llm_call` (and their sync wrappers) and
    :func:`~llmkit.text_llm_call_stream` (and its sync form) answer a request whose fingerprint
    was stored by an earlier successful call from the store and store each new
    successful answer; the buffered calls also coalesce identical requests in
    flight at once into one provider call. ``cache=False`` on a call or on
    :class:`~llmkit.LLMCallOptions` opts that call out.

    *timeout* is the seconds one store ``get`` or ``set`` may take,
    :data:`DEFAULT_CACHE_TIMEOUT` (1.0) unless configured. llmkit cancels an
    operation that overruns it: a read becomes a miss and the call asks the
    provider, a write is dropped and the call returns its answer unchanged,
    each reported like any other cache failure. ``None`` turns the bound off.
    ``0`` cancels at the store's first suspension, so a store that never
    suspends — :class:`~llmkit.InMemoryLLMCache` among them — still answers.
    Omitting *timeout* keeps the value configured before, so a later call that
    only swaps the store, or turns caching off, does not reset it. The bound
    is cooperative: it works by cancellation, so a store that blocks the event
    loop synchronously or swallows the cancellation is outside it (see
    :class:`~llmkit.cache.LLMCache`). It never bounds the wait of a call that
    joined an identical call already in flight: that wait ends when the
    leading call's provider pass answers, fails or is cancelled, before the
    leader writes to the cache, so what bounds that pass (the leader's
    per-request ``timeout`` and retry budget) bounds the wait.

    Re-arms the warn-once latch, so a newly configured store — or the same
    store under a new *timeout* — gets a fresh loud first warning if it turns
    out to be broken.

    Raises:
        TypeError: if *cache* is neither ``None`` nor an :class:`LLMCache`
            instance — including passing the class rather than an instance of
            it. The structural check tests attribute presence only, so, as
            with :func:`~llmkit.configure_llm_logging`, a bare ``MagicMock()``
            does not match; stub with a small fake class.
        ValueError: if *timeout* is a ``bool``, nonfinite or negative.
            Nothing is changed when either check fails.
    """
    global _cache, _timeout
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
    if not isinstance(timeout, Unset) and timeout is not None:
        timeout = duration(timeout, "timeout")
    _cache = cache
    if not isinstance(timeout, Unset):
        _timeout = timeout
    _cache_latch.succeeded()


def get_llm_cache() -> LLMCache | None:
    """Return the cache currently configured, or ``None`` when caching is off.

    The read half of :func:`configure_llm_cache`, so a host that installs a
    cache temporarily can restore what was there before.
    """
    return _cache


def cache_timeout() -> float | None:
    """The configured budget for one store ``get`` or ``set``; ``None`` when off."""
    return _timeout


# What the call does after each step fails; anything unlisted proceeds as a miss.
_OUTCOMES = {
    "set": "the call returns its answer and stores nothing",
    "encode": "the call returns its answer and stores nothing",
    "key": "the call bypasses the cache and stores nothing",
}


def report_cache_failure(what: str, exc: BaseException) -> None:
    """Log a cache-side failure: WARNING for a new signature, DEBUG for a repeat.

    *what* names the step that failed (``"get"``, ``"set"``, ``"key"``,
    ``"decode"``, ``"encode"``), and the warning says what the call does about
    it: a failed ``set`` or ``encode`` returns the answer and stores nothing, a
    failed ``key`` bypasses the cache, and a failed ``get`` or ``decode``
    proceeds as a miss.

    Never raises an ordinary exception: the record goes through
    :func:`llmkit._diagnostics.report`, so a host handler that raises while
    handling it is contained and counted in
    :func:`~llmkit.get_diagnostics_health` rather than replacing the call's
    answer or the provider's error. A handler that blocks is not contained.
    """
    if _cache_latch.should_warn(exc):
        outcome = _OUTCOMES.get(what, "the call proceeds as a miss")
        _ = report(
            logger,
            logging.WARNING,
            "LLM response cache %s failed; %s (further identical failures logged at DEBUG)",
            what,
            outcome,
            exc_info=exc,
        )
    else:
        _ = report(logger, logging.DEBUG, "LLM response cache %s failed", what, exc_info=exc)
