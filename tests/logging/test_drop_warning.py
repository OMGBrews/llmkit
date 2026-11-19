"""The first record an installation drops announces itself once.

Delivery drops a record when its queue is full, when it is closed or its worker
is gone, and when the exit drain runs out of time. The counters always moved;
these tests pin the one WARNING that now goes with the first drop: once per
installation, sink-agnostic, naming ``get_logging_health()``, without the
record's content, and unable to escape through a host handler that raises.

Each test drives its own :class:`_Delivery`, so the process-wide worker other
tests share is never closed or filled.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Iterator
from typing import override

import pytest

from llmkit import (
    LLMCallRecord,
    configure_llm_logging,
    get_diagnostics_health,
    get_logging_health,
)
from llmkit.logging import _delivery
from llmkit.logging._policy import get_delivery_policy
from tests._support import make_record

_SECRET = "do-not-print-this-prompt"


class _ListSink:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    def write(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class _BlockingSink:
    """Holds the worker in its first write until released."""

    def __init__(self, *, then_raise: BaseException | None = None) -> None:
        self.entered: threading.Event = threading.Event()
        self.release: threading.Event = threading.Event()
        self.then_raise: BaseException | None = then_raise

    def write(self, record: LLMCallRecord) -> None:
        del record
        self.entered.set()
        _ = self.release.wait(5)
        if self.then_raise is not None:
            raise self.then_raise


class _Stop(BaseException):
    """A sink failure fatal enough to stop the delivery worker."""


def _submit(delivery: _delivery._Delivery) -> _delivery.Receipt | None:
    record = make_record(label=_SECRET, prompt=[{"role": "user", "content": _SECRET}])
    return delivery.submit(record, contextvars.copy_context())


def _drop_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "dropped an LLM call record" in r.getMessage()
    ]


@pytest.fixture
def caplog_warnings(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
        yield caplog


def test_first_drop_warns_once_per_installation(
    caplog_warnings: pytest.LogCaptureFixture,
) -> None:
    """A closed delivery drops every record: the first drop warns, later ones
    only count, and a reconfigure re-arms the warning."""
    delivery = _delivery._Delivery()
    delivery.closed = True
    configure_llm_logging(_ListSink())
    assert _submit(delivery) is None
    assert _submit(delivery) is None
    assert get_logging_health().dropped == 2
    [warning] = _drop_warnings(caplog_warnings)
    message = warning.getMessage()
    assert "get_logging_health()" in message
    assert "closed or unavailable" in message
    assert _SECRET not in message
    assert warning.exc_info is None

    configure_llm_logging(_ListSink())
    assert _submit(delivery) is None
    assert len(_drop_warnings(caplog_warnings)) == 2


def test_queue_full_drop_warns(caplog_warnings: pytest.LogCaptureFixture) -> None:
    """A record refused for want of queue space warns the same way."""
    delivery = _delivery._Delivery()
    delivery.reservations = get_delivery_policy().queue_capacity
    configure_llm_logging(_ListSink())
    assert _submit(delivery) is None
    assert get_logging_health().dropped == 1
    [warning] = _drop_warnings(caplog_warnings)
    assert "queue full" in warning.getMessage()


def test_exit_drain_timeout_drop_warns(caplog_warnings: pytest.LogCaptureFixture) -> None:
    """Queued records the terminal drain abandons are a drop, and warn."""
    sink = _BlockingSink()
    delivery = _delivery._Delivery()
    configure_llm_logging(sink)
    try:
        assert _submit(delivery) is not None
        assert sink.entered.wait(5)
        assert _submit(delivery) is not None
        assert _drop_warnings(caplog_warnings) == []
        assert delivery.flush(0.01, terminal=True) is False
    finally:
        sink.release.set()
    assert get_logging_health().dropped == 1
    [warning] = _drop_warnings(caplog_warnings)
    assert "exit drain timed out" in warning.getMessage()


def test_worker_stop_drop_warns(caplog_warnings: pytest.LogCaptureFixture) -> None:
    """Records queued behind a write that stopped the worker are dropped, and
    the worker warns once it has released the delivery lock."""
    sink = _BlockingSink(then_raise=_Stop())
    delivery = _delivery._Delivery()
    configure_llm_logging(sink)
    assert _submit(delivery) is not None
    assert sink.entered.wait(5)
    queued = _submit(delivery)
    assert queued is not None
    sink.release.set()
    assert queued.done.wait(5)
    thread = delivery.thread
    assert thread is not None
    thread.join(5)
    assert get_logging_health().dropped == 1
    [warning] = _drop_warnings(caplog_warnings)
    assert "worker stopped" in warning.getMessage()


def test_raising_host_handler_does_not_escape_the_drop(
    caplog_warnings: pytest.LogCaptureFixture,
) -> None:
    """A host logging handler that raises on the warning is contained: the
    caller sees an ordinary drop, not the handler's exception."""

    emitted: list[str] = []

    class _Raising(logging.Handler):
        @override
        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())
            raise RuntimeError("host handler is broken")

    handler = _Raising()
    lost_before = get_diagnostics_health().lost
    llmkit_logger = logging.getLogger("llmkit.logging")
    llmkit_logger.addHandler(handler)
    try:
        delivery = _delivery._Delivery()
        delivery.closed = True
        configure_llm_logging(_ListSink())
        assert _submit(delivery) is None
    finally:
        llmkit_logger.removeHandler(handler)
    assert get_logging_health().dropped == 1
    # The warning reached the raising handler and its loss was counted, so
    # the containment above is of a warning that was actually emitted.
    [message] = emitted
    assert "dropped an LLM call record" in message
    assert get_diagnostics_health().lost == lost_before + 1
    del caplog_warnings


def test_worker_drop_warning_contains_a_base_exception_from_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the exiting worker, even a ``BaseException`` from a host handler is
    contained: the worker ends quietly instead of through the thread
    excepthook."""

    emitted: list[str] = []

    class _Interrupting(logging.Handler):
        @override
        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())
            raise _Stop

    escaped: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)
    handler = _Interrupting(level=logging.WARNING)
    llmkit_logger = logging.getLogger("llmkit.logging")
    llmkit_logger.addHandler(handler)
    sink = _BlockingSink(then_raise=_Stop())
    delivery = _delivery._Delivery()
    try:
        configure_llm_logging(sink)
        assert _submit(delivery) is not None
        assert sink.entered.wait(5)
        assert _submit(delivery) is not None
        sink.release.set()
        thread = delivery.thread
        assert thread is not None
        thread.join(5)
        assert not thread.is_alive()
    finally:
        sink.release.set()
        llmkit_logger.removeHandler(handler)
    assert get_logging_health().dropped == 1
    assert escaped == []
    # The worker did emit the drop warning; only its handler's raise was contained.
    assert [m for m in emitted if "dropped an LLM call record" in m] != []
