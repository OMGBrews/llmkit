"""The shared diagnostics seam contains ordinary handler failures and counts them.

:func:`llmkit._diagnostics.report` is the one place every in-flight llmkit
diagnostic goes through. These tests drive it directly on a private logger
(``propagate=False``, so nothing reaches the root logger or pytest's capture)
and check that:

* a normal call emits one record and returns ``True`` without counting a loss;
* a handler whose ``emit`` raises an :class:`Exception` is contained, counted
  exactly once in :func:`llmkit.get_diagnostics_health`, and never re-logged;
* the record still names the real reporting site (``funcName``/``lineno``),
  honouring an explicit ``stacklevel``;
* ``exc_info`` reaches the record unchanged, and a bare ``raise`` after a
  swallowed failure still re-raises the caller's original exception;
* ``asyncio.CancelledError``, ``KeyboardInterrupt`` and ``SystemExit`` from a
  handler propagate and are not counted;
* an exception whose ``__str__`` raises is described by its type name alone;
* concurrent losses are counted exactly; and
* a loss never touches :func:`llmkit.get_logging_health`'s counters.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, timedelta
from typing import override

import pytest

import llmkit
from llmkit import DiagnosticLoss, DiagnosticsHealth, get_diagnostics_health, get_logging_health
from llmkit._diagnostics import report
from tests._support.diagnostics import lost_diagnostics, raising_handler

LOGGER_NAME = "tests.reliability.diagnostics_seam"


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    """A private logger that never propagates to the root logger."""
    log = logging.getLogger(LOGGER_NAME)
    previous = log.propagate
    log.propagate = False
    try:
        yield log
    finally:
        log.propagate = previous


class _ListHandler(logging.Handler):
    """Keep every record it is handed; never raise."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def recorder(logger: logging.Logger) -> Iterator[_ListHandler]:
    """Install a non-raising handler on the private logger at DEBUG."""
    handler = _ListHandler()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _next_line() -> int:
    """The line number of the caller's next line."""
    frame = inspect.currentframe()
    assert frame is not None
    assert frame.f_back is not None
    return frame.f_back.f_lineno + 1


def test_success_returns_true_and_counts_nothing(
    logger: logging.Logger, recorder: _ListHandler
) -> None:
    before = get_diagnostics_health()

    delivered = report(logger, logging.WARNING, "cache %s failed", "redis")

    assert delivered is True
    assert [(r.levelno, r.getMessage()) for r in recorder.records] == [
        (logging.WARNING, "cache redis failed")
    ]
    assert get_diagnostics_health() == before


def test_raising_handler_is_contained_counted_once_and_not_relogged(
    logger: logging.Logger,
) -> None:
    lost_before = lost_diagnostics()

    with raising_handler(LOGGER_NAME) as handler:
        delivered = report(logger, logging.WARNING, "sink %s failed", "yaml")

    assert delivered is False
    assert lost_diagnostics() == lost_before + 1
    assert [(r.levelno, r.getMessage()) for r in handler.records] == [
        (logging.WARNING, "sink yaml failed")
    ]
    loss = get_diagnostics_health().last_loss
    assert loss is not None
    assert loss.logger == LOGGER_NAME
    assert loss.level == "WARNING"
    assert loss.error == "DiagnosticHandlerError: diagnostic handler failed"
    assert loss.occurred_at.tzinfo is not None
    assert loss.occurred_at.utcoffset() == timedelta(0)
    assert loss.occurred_at.tzinfo is UTC


def test_record_names_the_calling_function_and_line(
    logger: logging.Logger, recorder: _ListHandler
) -> None:
    line = _next_line()
    _ = report(logger, logging.WARNING, "here")

    (record,) = recorder.records
    assert record.funcName == "test_record_names_the_calling_function_and_line"
    assert record.lineno == line
    assert record.pathname == __file__


def _helper_reporting_for_its_caller(logger: logging.Logger) -> None:
    _ = report(logger, logging.WARNING, "via helper", stacklevel=2)


def test_stacklevel_two_names_the_helpers_caller(
    logger: logging.Logger, recorder: _ListHandler
) -> None:
    line = _next_line()
    _helper_reporting_for_its_caller(logger)

    (record,) = recorder.records
    assert record.funcName == "test_stacklevel_two_names_the_helpers_caller"
    assert record.lineno == line


def test_record_fields_survive_a_raising_handler(logger: logging.Logger) -> None:
    with raising_handler(LOGGER_NAME) as handler:
        line = _next_line()
        _ = report(logger, logging.WARNING, "lost")

    (record,) = handler.records
    assert record.funcName == "test_record_fields_survive_a_raising_handler"
    assert record.lineno == line


def test_explicit_exception_object_is_forwarded_unchanged(
    logger: logging.Logger, recorder: _ListHandler
) -> None:
    cause = ConnectionError("redis down")

    _ = report(logger, logging.WARNING, "decode failed", exc_info=cause)

    (record,) = recorder.records
    assert record.exc_info is not None
    assert record.exc_info[1] is cause
    assert record.exc_info[0] is ConnectionError


def test_exc_info_true_captures_the_callers_exception(
    logger: logging.Logger, recorder: _ListHandler
) -> None:
    original = ValueError("callback broke")
    try:
        raise original
    except ValueError:
        _ = report(logger, logging.ERROR, "callback raised", exc_info=True)

    (record,) = recorder.records
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.exc_info[1] is original


def test_exc_info_true_is_captured_even_when_the_handler_raises(
    logger: logging.Logger,
) -> None:
    original = ValueError("callback broke")
    with raising_handler(LOGGER_NAME) as handler:
        try:
            raise original
        except ValueError:
            _ = report(logger, logging.ERROR, "callback raised", exc_info=True)

    (record,) = handler.records
    assert record.exc_info is not None
    assert record.exc_info[1] is original


def test_bare_raise_after_a_swallowed_failure_reraises_the_original(
    logger: logging.Logger,
) -> None:
    original = TimeoutError("provider timed out")
    lost_before = lost_diagnostics()

    def exhaust() -> None:
        try:
            raise original
        except TimeoutError:
            _ = report(logger, logging.ERROR, "stopped after %d attempts", 3)
            raise

    with raising_handler(LOGGER_NAME) as handler, pytest.raises(TimeoutError) as caught:
        exhaust()

    assert caught.value is original
    assert lost_diagnostics() == lost_before + 1
    assert len(handler.records) == 1


@pytest.mark.parametrize(
    "error",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
    ids=["cancelled", "keyboard-interrupt", "system-exit"],
)
def test_base_exceptions_from_the_handler_propagate_uncounted(
    logger: logging.Logger, error: type[BaseException]
) -> None:
    health_before = get_diagnostics_health()

    with raising_handler(LOGGER_NAME, error=error) as handler, pytest.raises(error):
        _ = report(logger, logging.WARNING, "interrupted")

    assert get_diagnostics_health() == health_before
    assert len(handler.records) == 1


class _UnprintableError(RuntimeError):
    @override
    def __str__(self) -> str:
        raise ValueError("cannot describe")


def test_unprintable_handler_exception_is_described_by_type_name(
    logger: logging.Logger,
) -> None:
    lost_before = lost_diagnostics()

    with raising_handler(LOGGER_NAME, error=_UnprintableError):
        delivered = report(logger, logging.DEBUG, "repeated failure")

    assert delivered is False
    assert lost_diagnostics() == lost_before + 1
    loss = get_diagnostics_health().last_loss
    assert loss is not None
    assert loss.error == "_UnprintableError"
    assert loss.level == "DEBUG"


def test_concurrent_losses_are_counted_exactly(logger: logging.Logger) -> None:
    threads, per_thread = 8, 200
    lost_before = lost_diagnostics()
    start = threading.Barrier(threads)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        _ = start.wait()
        outcomes = [report(logger, logging.WARNING, "n=%d", i) for i in range(per_thread)]
        with results_lock:
            results.extend(outcomes)

    with raising_handler(LOGGER_NAME) as handler:
        pool = [threading.Thread(target=worker) for _ in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

    assert lost_diagnostics() == lost_before + threads * per_thread
    assert results == [False] * (threads * per_thread)
    assert len(handler.records) == threads * per_thread


def test_diagnostics_health_is_frozen() -> None:
    health = get_diagnostics_health()
    with pytest.raises(dataclasses.FrozenInstanceError):
        health.lost = 0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("name", "obj"),
    [
        ("get_diagnostics_health", get_diagnostics_health),
        ("DiagnosticsHealth", DiagnosticsHealth),
        ("DiagnosticLoss", DiagnosticLoss),
    ],
)
def test_diagnostics_health_api_is_exported(name: str, obj: Callable[..., object]) -> None:
    assert name in llmkit.__all__
    assert getattr(llmkit, name) is obj


def test_a_loss_does_not_touch_logging_health(logger: logging.Logger) -> None:
    before = get_logging_health()
    lost_before = lost_diagnostics()

    with raising_handler(LOGGER_NAME):
        delivered = report(logger, logging.WARNING, "sink failed")

    assert delivered is False
    assert lost_diagnostics() == lost_before + 1
    after = get_logging_health()
    assert after.failed == before.failed
    assert after.last_failure == before.last_failure
