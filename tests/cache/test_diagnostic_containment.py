"""A raising ``llmkit.cache`` handler never replaces a call's answer or error.

Every cache-side failure degrades to a miss and is reported on the
``llmkit.cache`` logger. A host handler whose ``emit`` raises used to turn
that miss into the handler's exception, with no provider request made. These
tests install such a handler and check that the call still reaches the
provider once, returns its answer or raises its error, that each lost record
is counted in :func:`llmkit.get_diagnostics_health`, and that the seam never
re-logs its own failure on the broken logger. Cancellation and keyboard
interrupt raised by the handler are not contained.
"""

from __future__ import annotations

import asyncio
import logging
from typing import override

import pytest
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCacheEntry,
    ResultValidationError,
    configure_llm_cache,
    get_diagnostics_health,
    structured_llm_call,
    text_llm_call,
)
from llmkit.cache.read_through import CacheDecodeError
from llmkit.testing import Reply, TransportRequest, install
from tests._support.diagnostics import RaisingHandler, lost_diagnostics, raising_handler
from tests.cache.conftest import CALL_MODE, cache_provider

_CACHE = "llmkit.cache"


class _GetRaises:
    """A store whose ``get`` always fails; ``set`` works."""

    def __init__(self) -> None:
        self.entries: dict[str, LLMCacheEntry] = {}

    async def get(self, key: str) -> LLMCacheEntry | None:
        raise ConnectionError(f"store unreachable reading {key[:8]}")

    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        self.entries[key] = entry


class _ProviderDown(Exception):
    """The provider's own error, which must reach the caller unchanged."""


def _answers(request: TransportRequest) -> Reply:
    del request
    return Reply.text("answer")


def _fails(request: TransportRequest) -> Reply:
    del request
    raise _ProviderDown("provider down")


def _levels(handler: RaisingHandler) -> list[int]:
    return [r.levelno for r in handler.records]


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
async def test_broken_get_and_raising_handler_still_returns_the_answer(calls: int) -> None:
    configure_llm_cache(_GetRaises())
    provider = cache_provider()
    before = lost_diagnostics()
    with raising_handler(_CACHE) as handler, install(_answers) as installed:
        results = [await text_llm_call("q", feature="f", provider=provider) for _ in range(calls)]

    assert results == ["answer"] * calls
    assert len(installed.requests) == calls
    assert lost_diagnostics() - before == calls
    # One record per call: WARNING first, DEBUG for the repeat behind the
    # latch, and nothing further from the seam about its own failure.
    assert _levels(handler) == [logging.WARNING, logging.DEBUG][:calls]
    assert all(r.exc_info is not None and r.exc_info[0] is ConnectionError for r in handler.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
async def test_broken_get_and_raising_handler_still_raises_the_provider_error(
    calls: int,
) -> None:
    configure_llm_cache(_GetRaises())
    provider = cache_provider()
    before = lost_diagnostics()
    with raising_handler(_CACHE) as handler, install(_fails) as installed:
        for _ in range(calls):
            with pytest.raises(_ProviderDown):
                _ = await text_llm_call("q", feature="f", provider=provider, retry=NO_RETRY)

    assert len(installed.requests) == calls
    assert lost_diagnostics() - before == calls
    assert _levels(handler) == [logging.WARNING, logging.DEBUG][:calls]


@pytest.mark.asyncio
async def test_lost_cache_warning_is_described_in_diagnostics_health() -> None:
    configure_llm_cache(_GetRaises())
    with raising_handler(_CACHE), install(_answers):
        _ = await text_llm_call("q", feature="f", provider=cache_provider())

    loss = get_diagnostics_health().last_loss
    assert loss is not None
    assert (loss.logger, loss.level) == (_CACHE, "WARNING")
    assert loss.error == "DiagnosticHandlerError: diagnostic handler failed"


@pytest.mark.asyncio
async def test_decode_site_forwards_the_cause_as_exc_info() -> None:
    """The decode site reports ``exc.__cause__``, not the CacheDecodeError
    wrapper it caught; the seam must forward that object unchanged."""

    class _Answer(BaseModel):
        n: int

    class _Corrupt(_GetRaises):
        @override
        async def get(self, key: str) -> LLMCacheEntry | None:
            return LLMCacheEntry("{not json", "_Answer", None, None, call_id=key)

    configure_llm_cache(_Corrupt())
    provider = cache_provider(instructor_mode=CALL_MODE)
    before = lost_diagnostics()
    with (
        raising_handler(_CACHE) as handler,
        install(lambda _request: Reply.json(_Answer(n=1))) as installed,
    ):
        got = await structured_llm_call("q", _Answer, feature="f", provider=provider)

    assert got == _Answer(n=1)
    assert len(installed.requests) == 1
    assert lost_diagnostics() - before == 1
    (record,) = handler.records
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    exc_type, exc, _tb = record.exc_info
    assert exc_type is not None and exc is not None
    assert not issubclass(exc_type, CacheDecodeError)
    assert type(exc) is exc_type


@pytest.mark.asyncio
async def test_on_result_rejection_debug_is_contained() -> None:
    """``_accept``'s DEBUG when ``on_result`` rejects a stored answer."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    answers = iter(["first", "second"])

    def _next(request: TransportRequest) -> Reply:
        del request
        return Reply.text(next(answers))

    def _only_second(text: str) -> None:
        if text != "second":
            raise ResultValidationError(f"{text} rejected")

    with install(_next) as installed:
        _ = await text_llm_call("q", feature="f", provider=provider)
        before = lost_diagnostics()
        with raising_handler(_CACHE) as handler:
            picky = await text_llm_call("q", feature="f", provider=provider, on_result=_only_second)

    assert picky == "second"
    assert len(installed.requests) == 2
    assert lost_diagnostics() - before == 1
    (record,) = handler.records
    assert record.levelno == logging.DEBUG
    assert record.getMessage().startswith("on_result rejected a cached answer")


@pytest.mark.asyncio
async def test_record_names_the_real_reporting_site() -> None:
    configure_llm_cache(_GetRaises())
    with raising_handler(_CACHE) as handler, install(_answers):
        _ = await text_llm_call("q", feature="f", provider=cache_provider())

    (record,) = handler.records
    assert record.funcName == "report_cache_failure"
    assert record.pathname.endswith("registry.py")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [asyncio.CancelledError, KeyboardInterrupt], ids=["cancelled", "keyboard-interrupt"]
)
async def test_cancellation_from_the_handler_propagates(error: type[BaseException]) -> None:
    configure_llm_cache(_GetRaises())
    before = lost_diagnostics()
    with raising_handler(_CACHE, error=error) as handler, install(_answers) as installed:
        with pytest.raises(error):
            _ = await text_llm_call("q", feature="f", provider=cache_provider())

    assert len(installed.requests) == 0
    assert lost_diagnostics() == before
    assert len(handler.records) == 1
