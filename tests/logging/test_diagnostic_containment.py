"""A raising ``llmkit.logging`` handler never replaces a call's outcome.

A sink failure, a failed index line or retention pass, and the first-write
announcement are all reported on the ``llmkit.logging`` logger. A host
handler whose ``emit`` raises used to turn that report into the handler's
exception: out of :func:`~llmkit.logging.write_llm_log` on the direct path,
and into a dead delivery worker on the managed path. These tests install such
a handler and check that the result or the provider's error survives, that
each lost record is counted once in :func:`llmkit.get_diagnostics_health`
(and never in :func:`~llmkit.get_logging_health`), that the seam does not
re-log its own failure, and that managed delivery stays available. A handler
that raises cancellation or ``KeyboardInterrupt`` is not contained.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from collections.abc import Callable, Iterator
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llmkit import (
    NO_RETRY,
    LLMCallRecord,
    LocalYamlLogSink,
    capture_llm_records,
    configure_llm_logging,
    flush_llm_logs_async,
    get_diagnostics_health,
    get_logging_health,
    text_llm_call,
)
from llmkit.capture import record_call
from llmkit.logging import INDEX_FILENAME, local_yaml, write_llm_log
from llmkit.testing import Reply, TransportRequest, install
from tests._support import make_record
from tests._support.diagnostics import RaisingHandler, lost_diagnostics, raising_handler

if TYPE_CHECKING:
    from _typeshed import OpenTextMode

_LOGGING = "llmkit.logging"
_ROOT = Path(__file__).resolve().parents[2]


class _SinkDown(Exception):
    """The sink's own failure, carried as the diagnostic's ``exc_info``."""


class _ProviderDown(Exception):
    """The provider's own error, which must reach the caller unchanged."""


class _SelectiveSink:
    """Raises for records labelled ``broken``; keeps every other record."""

    def __init__(self) -> None:
        self.labels: list[str | None] = []

    def write(self, record: LLMCallRecord) -> None:
        if record.label == "broken":
            raise _SinkDown(f"collector down for {record.feature}")
        self.labels.append(record.label)


def _answers(request: TransportRequest) -> Reply:
    del request
    return Reply.text("answer")


def _fails(request: TransportRequest) -> Reply:
    del request
    raise _ProviderDown("provider down")


def _levels(handler: RaisingHandler) -> list[int]:
    return [r.levelno for r in handler.records]


def _broken_log_dir(tmp_path: Path) -> Path:
    blocker = tmp_path / "not-a-directory"
    _ = blocker.write_text("occupied", encoding="utf-8")
    return blocker


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    yield
    configure_llm_logging(None)


# --- direct path: write_llm_log and capture.record_call ---------------------


@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
@pytest.mark.parametrize("entry", ["write_llm_log", "record_call"])
def test_direct_sink_failure_with_raising_handler_returns_and_counts_once(
    calls: int, entry: str
) -> None:
    configure_llm_logging(_SelectiveSink())
    write: Callable[[LLMCallRecord], Path | None] = (
        write_llm_log if entry == "write_llm_log" else record_call
    )
    before = lost_diagnostics()
    with raising_handler(_LOGGING) as handler, capture_llm_records() as records:
        results = [write(make_record(label="broken")) for _ in range(calls)]

    assert results == [None] * calls
    # Only the capture seam feeds capture_llm_records; the record survives there.
    assert len(records) == (calls if entry == "record_call" else 0)
    health = get_logging_health()
    assert (health.failed, health.handed_off) == (calls, 0)
    assert health.last_failure is not None
    assert health.last_failure.stage == "sink_raised"
    assert lost_diagnostics() - before == calls
    # One record per write: WARNING first, DEBUG for the latched repeat, and
    # nothing further from the seam about its own failure.
    assert _levels(handler) == [logging.WARNING, logging.DEBUG][:calls]
    assert all(r.exc_info is not None and r.exc_info[0] is _SinkDown for r in handler.records)


def test_direct_path_keeps_working_after_a_lost_warning() -> None:
    sink = _SelectiveSink()
    configure_llm_logging(sink)
    with raising_handler(_LOGGING):
        assert write_llm_log(make_record(label="broken")) is None
        assert write_llm_log(make_record(label="ok")) is None

    assert sink.labels == ["ok"]
    health = get_logging_health()
    assert (health.failed, health.handed_off) == (1, 1)


def test_lost_sink_warning_is_described_in_diagnostics_health() -> None:
    configure_llm_logging(_SelectiveSink())
    with raising_handler(_LOGGING):
        _ = write_llm_log(make_record(label="broken"))

    loss = get_diagnostics_health().last_loss
    assert loss is not None
    assert (loss.logger, loss.level) == (_LOGGING, "WARNING")
    assert loss.error == "DiagnosticHandlerError: diagnostic handler failed"


def test_sink_warning_record_names_the_real_call_site() -> None:
    configure_llm_logging(_SelectiveSink())
    with raising_handler(_LOGGING) as handler:
        _ = write_llm_log(make_record(label="broken"))

    (record,) = handler.records
    assert record.funcName == "write_installed"
    assert Path(record.pathname).name == "registry.py"


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt, asyncio.CancelledError],
    ids=["keyboard-interrupt", "cancelled"],
)
def test_direct_path_handler_interrupt_propagates(error: type[BaseException]) -> None:
    configure_llm_logging(_SelectiveSink())
    before = lost_diagnostics()
    with raising_handler(_LOGGING, error=error) as handler, pytest.raises(error):
        _ = write_llm_log(make_record(label="broken"))

    assert lost_diagnostics() == before
    assert _levels(handler) == [logging.WARNING]
    # The sink failure itself was counted before the warning was attempted.
    assert get_logging_health().failed == 1


def test_no_llmkit_record_reaches_other_handlers_after_the_swallowed_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The test_logging_health convention: caplog sees no llmkit.logging record.

    caplog's handler sits on the root logger, below the raising handler, so a
    record reaches it only if the seam logged something further (on this or
    any other llmkit logger) after containing the failure.
    """
    caplog.set_level(logging.DEBUG)
    configure_llm_logging(_SelectiveSink())
    with raising_handler(_LOGGING) as handler:
        assert write_llm_log(make_record(label="broken")) is None
        assert write_llm_log(make_record(label="broken")) is None

    assert len(handler.records) == 2
    assert [r for r in caplog.records if r.name.startswith("llmkit")] == []


# --- built-in sink: latched write/index/housekeeping and the announcement ---


@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
def test_builtin_write_failure_with_raising_handler_counts_once(tmp_path: Path, calls: int) -> None:
    configure_llm_logging(LocalYamlLogSink(_broken_log_dir(tmp_path)))
    before = lost_diagnostics()
    with raising_handler(_LOGGING) as handler:
        results = [write_llm_log(make_record()) for _ in range(calls)]

    assert results == [None] * calls
    health = get_logging_health()
    assert (health.written, health.failed) == (0, calls)
    assert health.last_failure is not None
    assert health.last_failure.stage == "write"
    assert lost_diagnostics() - before == calls
    assert _levels(handler) == [logging.WARNING, logging.DEBUG][:calls]
    assert {r.funcName for r in handler.records} == {"write_with_outcome"}


def _not_info(record: logging.LogRecord) -> bool:
    return record.levelno != logging.INFO


@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
def test_index_and_housekeeping_failures_with_raising_handler_keep_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: int
) -> None:
    (tmp_path / INDEX_FILENAME).mkdir()
    sink = LocalYamlLogSink(tmp_path, retention_days=30)

    def _boom() -> None:
        raise OSError("prune exploded")

    monkeypatch.setattr(sink, "_prune", _boom)
    configure_llm_logging(sink)
    before = lost_diagnostics()
    paths: list[Path | None] = []
    with raising_handler(_LOGGING, raise_on=_not_info) as handler:
        for _ in range(calls):
            # Re-arm the hourly prune throttle so every write reaches it.
            sink._last_prune = None  # test reaches the throttle
            paths.append(write_llm_log(make_record()))

    assert all(p is not None and p.is_file() for p in paths)
    health = get_logging_health()
    assert (health.written, health.degraded, health.failed) == (calls, calls, 0)
    # Index then housekeeping per write; the announcement (INFO) is not
    # raised on here, so only the two failure reports per write are lost.
    assert lost_diagnostics() - before == 2 * calls
    failures = [r for r in handler.records if r.levelno != logging.INFO]
    expected = (
        [logging.WARNING, logging.WARNING]
        if calls == 1
        else [logging.WARNING] * 2 + [logging.DEBUG] * 2
    )
    assert [r.levelno for r in failures] == expected
    assert [r.funcName for r in failures[:2]] == ["_append_index", "_maybe_prune"]


def _only_info(record: logging.LogRecord) -> bool:
    return record.levelno == logging.INFO


def test_raising_announcement_handler_does_not_fail_a_written_record(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    before = lost_diagnostics()
    with raising_handler(_LOGGING, raise_on=_only_info) as handler:
        first = write_llm_log(make_record(label="first"))
        second = write_llm_log(make_record(label="second"))

    assert first is not None and first.is_file()
    assert second is not None and second.is_file()
    health = get_logging_health()
    assert (health.written, health.failed, health.degraded) == (2, 0, 0)
    assert health.last_failure is None
    assert lost_diagnostics() - before == 1
    # The announcement fires once, from its own function, and is not retried.
    (announcement,) = handler.records
    assert announcement.levelno == logging.INFO
    assert announcement.funcName == "_announce_once"


def test_announcement_handler_interrupt_propagates(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    before = lost_diagnostics()
    with (
        raising_handler(_LOGGING, error=KeyboardInterrupt, raise_on=_only_info),
        pytest.raises(KeyboardInterrupt),
    ):
        _ = write_llm_log(make_record())

    assert lost_diagnostics() == before


def test_unseedable_gitignore_with_raising_handler_keeps_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "logs"
    # The project-root default is the one location that owes a .gitignore.
    monkeypatch.setattr(local_yaml, "resolve_default_log_dir", lambda: (log_dir, True))
    real_open = open

    def _open(
        file: Path,
        mode: OpenTextMode,
        *,
        encoding: str,
        opener: Callable[[str, int], int] | None = None,
    ) -> TextIOWrapper:
        """Fail only the seed; every other file the sink opens is real."""
        if file.name == ".gitignore":
            raise PermissionError("read-only checkout")
        return real_open(file, mode, encoding=encoding, opener=opener)

    monkeypatch.setattr(local_yaml, "open", _open, raising=False)
    configure_llm_logging(LocalYamlLogSink())
    before = lost_diagnostics()
    # Raise on the seed's DEBUG only; the announcement (INFO) is just recorded.
    with raising_handler(_LOGGING, raise_on=_not_info) as handler:
        path = write_llm_log(make_record())

    assert path is not None and path.is_file()
    assert path.parent == log_dir
    assert not (log_dir / ".gitignore").exists()
    health = get_logging_health()
    assert (health.written, health.failed) == (1, 0)
    assert lost_diagnostics() - before == 1
    seed = handler.records[0]
    assert seed.levelno == logging.DEBUG
    assert seed.funcName == "_ensure_log_dir"
    assert seed.exc_info is not None and seed.exc_info[0] is PermissionError


# --- managed path: provider calls deliver on the worker thread --------------


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", [1, 2], ids=["first-warning", "repeated-debug"])
@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_managed_sink_failure_with_raising_handler_keeps_outcome_and_delivery(
    calls: int, outcome: str
) -> None:
    sink = _SelectiveSink()
    configure_llm_logging(sink)
    before = lost_diagnostics()
    with raising_handler(_LOGGING) as handler:
        if outcome == "success":
            with install(_answers) as installed:
                results = [
                    await text_llm_call("q", feature="f", label="broken") for _ in range(calls)
                ]
            assert results == ["answer"] * calls
        else:
            with install(_fails) as installed:
                for _ in range(calls):
                    with pytest.raises(_ProviderDown):
                        _ = await text_llm_call("q", feature="f", label="broken", retry=NO_RETRY)
        assert len(installed.requests) == calls
        assert await flush_llm_logs_async(timeout=2)
        lost = lost_diagnostics() - before
        levels = _levels(handler)

        # Delivery survived the handler: a later record still reaches the sink.
        with install(_answers):
            assert await text_llm_call("q", feature="f", label="after") == "answer"
        assert await flush_llm_logs_async(timeout=2)

    assert lost == calls
    assert levels == [logging.WARNING, logging.DEBUG][:calls]
    assert sink.labels == ["after"]
    health = get_logging_health()
    assert (health.failed, health.handed_off) == (calls, 1)
    assert health.unavailable == 0
    assert health.dropped == 0


_MANAGED_INTERRUPT = r"""
import asyncio
import logging
import sys
from llmkit import (
    configure_llm_logging, flush_llm_logs_async, get_diagnostics_health,
    get_logging_health,
)
from llmkit.capture import record_call_async, submit_record_call
from tests._support import make_record

error = {"keyboard-interrupt": KeyboardInterrupt, "cancelled": asyncio.CancelledError}[sys.argv[1]]

class Handler(logging.Handler):
    def emit(self, record):
        raise error()

class Sink:
    def write(self, record):
        raise ValueError("sink failed")

logger = logging.getLogger("llmkit.logging")
logger.handlers[:] = [Handler()]
logger.propagate = False
logger.setLevel(logging.WARNING)
configure_llm_logging(Sink())

async def main():
    lost = get_diagnostics_health().lost
    # The handler runs on the delivery worker, so its interrupt cannot reach
    # this caller; the call itself survives.
    assert await record_call_async(make_record()) is None
    assert await flush_llm_logs_async(timeout=2)
    assert get_diagnostics_health().lost == lost, "an interrupt was contained"
    health = get_logging_health()
    assert health.failed == 1, health
    # The worker took its existing fatal path: delivery is now unavailable.
    submit_record_call(make_record(label="after-interrupt"))
    assert get_logging_health().unavailable == 1, get_logging_health()

asyncio.run(main())
print("done", flush=True)
"""


@pytest.mark.parametrize("error", ["keyboard-interrupt", "cancelled"])
def test_managed_handler_interrupt_takes_the_worker_fatal_path(error: str) -> None:
    # A fresh process: a dead delivery worker stays dead for its process.
    completed = subprocess.run(
        [sys.executable, "-c", _MANAGED_INTERRUPT, error],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "done" in completed.stdout, output
