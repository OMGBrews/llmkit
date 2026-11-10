"""A raising ``llmkit.retry`` handler never replaces a retried call's outcome.

The retry loops report each non-final failure at WARNING, a raising progress
callback at ERROR (with the callback's exception attached), and exhaustion at
ERROR immediately before re-raising the provider's error. A host handler
whose ``emit`` raises used to turn a transient-then-success call into the
handler's exception after attempt 1, and an exhausted call into the handler's
exception in place of the provider's. These tests install such a handler on
``llmkit.retry`` and check that:

* the original result, or the original provider exception (by identity where
  the loop is driven directly), still reaches the caller;
* each lost record is counted exactly once in
  :func:`llmkit.get_diagnostics_health`, on the first occurrence and again on
  a second occurrence in the same process (the retry seams have no latch);
* the handler sees exactly the expected records and nothing further, so the
  seam never re-logs its own failure; and
* ``asyncio.CancelledError`` and ``KeyboardInterrupt`` raised by the handler
  are not contained.

Both loops are driven directly with plain awaitables and async generators,
and the buffered and streaming text calls through an :mod:`llmkit.testing`
fake, so everything runs offline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import override

import openai
import pytest

from llmkit import RetryPolicy, text_llm_call, text_llm_call_stream
from llmkit.retry import retry_progress_callback, with_retries, with_retries_stream
from llmkit.testing import Fake, Reply, TransportRequest, install, timeout_error
from tests._support.diagnostics import RaisingHandler, lost_diagnostics, raising_handler

_RETRY = "llmkit.retry"
_TWO_PASSES = RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,))
_TEXT_TWO_PASSES = RetryPolicy(max_attempts=2, backoff_base_seconds=0.0)

_ATTEMPT_FAILED = "%s: attempt %d/%d failed: %s"
_CALLBACK_RAISED = "%s: retry progress callback raised"
_STOPPED = "%s: stopped after %d attempts: %s"


class _Transient(ValueError):
    """The provider's own retryable error, which must reach the caller unchanged."""


class _CallbackBroke(Exception):
    """What the raising progress callback raises."""


def _shape(handler: RaisingHandler) -> list[tuple[int, str]]:
    return [(r.levelno, r.msg) for r in handler.records]


def _flaky(failures: int, errors: list[_Transient]) -> Callable[[], Awaitable[str]]:
    """An awaitable factory failing *failures* times, recording each error raised."""
    calls = [0]

    async def run() -> str:
        calls[0] += 1
        if calls[0] <= failures:
            error = _Transient(f"transient {calls[0]}")
            errors.append(error)
            raise error
        return "ok"

    return run


def _flaky_stream(failures: int, errors: list[_Transient]) -> Callable[[int], AsyncGenerator[str]]:
    """An attempt factory whose first *failures* attempts fail before any chunk."""

    async def attempt(number: int) -> AsyncGenerator[str]:
        if number <= failures:
            error = _Transient(f"transient {number}")
            errors.append(error)
            raise error
        yield "he"
        yield "llo"

    return attempt


async def _run_buffered(failures: int, errors: list[_Transient]) -> str:
    return await with_retries(_flaky(failures, errors), policy=_TWO_PASSES, label="x")


async def _run_stream(failures: int, errors: list[_Transient]) -> str:
    stream = with_retries_stream(
        _flaky_stream(failures, errors), policy=_TWO_PASSES, label="x", surface="test"
    )
    return "".join([chunk async for chunk in stream])


_Runner = Callable[[int, list[_Transient]], Awaitable[str]]
_LOOPS = pytest.mark.parametrize(
    "run", [_run_buffered, _run_stream], ids=["with_retries", "with_retries_stream"]
)
_OCCURRENCES = pytest.mark.parametrize("occurrences", [1, 2], ids=["first", "repeated"])


def _raising_callback(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
    del label, attempt, max_attempts, error
    raise _CallbackBroke("callback broke")


# --- the loops, driven directly ---------------------------------------------


@_LOOPS
@_OCCURRENCES
async def test_transient_then_success_returns_the_result(run: _Runner, occurrences: int) -> None:
    before = lost_diagnostics()
    results: list[str] = []
    with raising_handler(_RETRY) as handler:
        for _ in range(occurrences):
            results.append(await run(1, []))

    assert results == ["hello" if run is _run_stream else "ok"] * occurrences
    assert lost_diagnostics() - before == occurrences
    assert _shape(handler) == [(logging.WARNING, _ATTEMPT_FAILED)] * occurrences


@_LOOPS
@_OCCURRENCES
async def test_exhaustion_raises_the_original_provider_error(
    run: _Runner, occurrences: int
) -> None:
    before = lost_diagnostics()
    with raising_handler(_RETRY) as handler:
        for _ in range(occurrences):
            errors: list[_Transient] = []
            with pytest.raises(_Transient) as excinfo:
                _ = await run(2, errors)
            # The bare ``raise`` re-raises the provider's last error itself.
            assert excinfo.value is errors[-1]

    assert lost_diagnostics() - before == 2 * occurrences
    assert (
        _shape(handler)
        == [(logging.WARNING, _ATTEMPT_FAILED), (logging.ERROR, _STOPPED)] * occurrences
    )


@_LOOPS
@pytest.mark.parametrize("failures", [1, 2], ids=["success", "exhausted"])
async def test_raising_progress_callback_and_handler_keep_the_outcome(
    run: _Runner, failures: int
) -> None:
    before = lost_diagnostics()
    errors: list[_Transient] = []
    with raising_handler(_RETRY) as handler, retry_progress_callback(_raising_callback):
        if failures == 1:
            assert await run(failures, errors) in {"ok", "hello"}
        else:
            with pytest.raises(_Transient) as excinfo:
                _ = await run(failures, errors)
            assert excinfo.value is errors[-1]

    expected = [(logging.WARNING, _ATTEMPT_FAILED), (logging.ERROR, _CALLBACK_RAISED)]
    if failures == 2:
        expected.append((logging.ERROR, _STOPPED))
    assert _shape(handler) == expected
    assert lost_diagnostics() - before == len(expected)


class _Recording(logging.Handler):
    """A non-raising handler that keeps every record."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@_LOOPS
async def test_progress_callback_record_carries_the_callback_exception(
    run: _Runner,
) -> None:
    logger = logging.getLogger(_RETRY)
    recorder = _Recording()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(recorder)
    before = lost_diagnostics()
    raised: list[_CallbackBroke] = []

    def callback(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
        del label, attempt, max_attempts, error
        broke = _CallbackBroke("callback broke")
        raised.append(broke)
        raise broke

    try:
        with retry_progress_callback(callback):
            _ = await run(1, [])
    finally:
        logger.removeHandler(recorder)
        logger.setLevel(previous)

    callback_records = [r for r in recorder.records if r.msg == _CALLBACK_RAISED]
    assert len(callback_records) == 1
    record = callback_records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert len(raised) == 1
    assert record.exc_info[1] is raised[0]
    assert record.funcName == "handle_retry_failure"
    assert lost_diagnostics() == before


@_LOOPS
@pytest.mark.parametrize(
    "error", [asyncio.CancelledError, KeyboardInterrupt], ids=["cancelled", "interrupt"]
)
async def test_cancellation_from_the_handler_propagates(
    run: _Runner, error: type[BaseException]
) -> None:
    before = lost_diagnostics()
    with raising_handler(_RETRY, error=error) as handler, pytest.raises(error):
        _ = await run(1, [])

    assert _shape(handler) == [(logging.WARNING, _ATTEMPT_FAILED)]
    assert lost_diagnostics() == before


# --- the public text calls, over the fake provider ---------------------------


def _times_out(failures: int) -> Fake:
    def fake(request: TransportRequest) -> Reply:
        if request.attempt is None or request.attempt <= failures:
            raise timeout_error()
        return Reply(chunks=["he", "llo"])

    return fake


async def _text(failures: int) -> str:
    with install(_times_out(failures)):
        return await text_llm_call("q", feature="f", retry=_TEXT_TWO_PASSES)


async def _text_stream(failures: int) -> str:
    with install(_times_out(failures)):
        stream = text_llm_call_stream("q", feature="f", retry=_TEXT_TWO_PASSES)
        return "".join([chunk async for chunk in stream])


_CALLS = pytest.mark.parametrize(
    "call", [_text, _text_stream], ids=["text_llm_call", "text_llm_call_stream"]
)


@_CALLS
async def test_public_call_transient_then_success_returns_the_answer(
    call: Callable[[int], Awaitable[str]],
) -> None:
    before = lost_diagnostics()
    with raising_handler(_RETRY) as handler:
        answer = await call(1)

    assert answer == "hello"
    assert [r.levelno for r in handler.records] == [logging.WARNING]
    assert lost_diagnostics() - before == 1


@_CALLS
async def test_public_call_exhaustion_raises_the_provider_error(
    call: Callable[[int], Awaitable[str]],
) -> None:
    before = lost_diagnostics()
    with raising_handler(_RETRY) as handler, pytest.raises(openai.APITimeoutError):
        _ = await call(2)

    assert [r.levelno for r in handler.records] == [logging.WARNING, logging.ERROR]
    assert lost_diagnostics() - before == 2
