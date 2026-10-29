"""``get_logging_health()``: detected logging failures are inspectable state.

Every test filters the ``llmkit.logging`` logger to ERROR, so the warnings
these failures normally produce are never emitted: the evidence must come from
the health snapshot alone, as it would for a host whose logging configuration
hides llmkit's diagnostics.

Failures are injected structurally (a regular file where the log directory
should be, a directory where ``index.jsonl`` should be) rather than through
permissions, which a root test runner ignores.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import override

import pytest

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
    get_logging_health,
    text_llm_call_sync,
)
from llmkit.logging import INDEX_FILENAME, write_llm_log
from llmkit.testing import Reply, install
from tests._support import make_record as _record


@pytest.fixture(autouse=True)
def _hidden_diagnostics(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Hide llmkit's logging warnings, then prove none were emitted."""
    caplog.set_level(logging.ERROR, logger="llmkit.logging")
    try:
        yield
    finally:
        configure_llm_logging(None)
    assert [r for r in caplog.records if r.name == "llmkit.logging"] == []


class _RaisingSink:
    def write(self, record: LLMCallRecord) -> None:
        raise RuntimeError(f"collector down for {record.feature}")


class _ListSink:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    def write(self, record: LLMCallRecord) -> None:
        self.records.append(record)


def _broken_log_dir(tmp_path: Path) -> Path:
    blocker = tmp_path / "not-a-directory"
    _ = blocker.write_text("occupied", encoding="utf-8")
    return blocker


def test_builtin_write_failure_is_counted_with_its_stage(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(_broken_log_dir(tmp_path)))

    assert write_llm_log(_record(feature="extract", label="doc-1")) is None

    health = get_logging_health()
    assert (health.sink, health.written, health.failed) == ("builtin", 0, 1)
    failure = health.last_failure
    assert failure is not None
    assert (failure.stage, failure.feature, failure.label) == ("write", "extract", "doc-1")
    # The underlying exception, not a generic stand-in.
    assert failure.error.split(":")[0] in {
        "NotADirectoryError",
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "OSError",
    }
    assert "wrote no file" not in failure.error
    assert failure.occurred_at.tzinfo is not None
    assert health.last_success_at is None


def test_index_failure_after_a_write_is_written_and_degraded(tmp_path: Path) -> None:
    (tmp_path / INDEX_FILENAME).mkdir()
    configure_llm_logging(LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=None))

    path = write_llm_log(_record(label="kept"))

    assert path is not None and path.exists()
    health = get_logging_health()
    assert (health.written, health.degraded, health.failed) == (1, 1, 0)
    assert health.last_failure is not None
    assert health.last_failure.stage == "index"
    assert health.last_success_at is not None


def test_housekeeping_failure_reports_the_later_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record that loses both its index line and housekeeping is degraded
    once, and ``last_failure`` names the stage that failed last."""
    (tmp_path / INDEX_FILENAME).mkdir()
    sink = LocalYamlLogSink(tmp_path, retention_days=30)

    def _boom() -> None:
        raise OSError("prune exploded")

    monkeypatch.setattr(sink, "_prune", _boom)
    configure_llm_logging(sink)

    assert write_llm_log(_record()) is not None

    health = get_logging_health()
    assert (health.written, health.degraded) == (1, 1)
    assert health.last_failure is not None
    assert health.last_failure.stage == "housekeeping"
    assert health.last_failure.error == "OSError: prune exploded"


def test_raising_custom_sink_is_a_failure() -> None:
    configure_llm_logging(_RaisingSink())

    assert write_llm_log(_record(feature="summarise")) is None

    health = get_logging_health()
    assert (health.sink, health.failed, health.handed_off) == ("custom", 1, 0)
    assert health.last_failure is not None
    assert health.last_failure.stage == "sink_raised"
    assert health.last_failure.error == "RuntimeError: collector down for summarise"


def test_custom_sink_returning_normally_is_a_handoff_not_a_write() -> None:
    sink = _ListSink()
    configure_llm_logging(sink)

    _ = write_llm_log(_record())

    health = get_logging_health()
    assert (health.sink, health.handed_off, health.written, health.failed) == ("custom", 1, 0, 0)
    assert health.last_success_at is not None
    assert len(sink.records) == 1


def test_third_party_path_returning_sink_is_still_a_handoff(tmp_path: Path) -> None:
    class _PathSink:
        def write(self, record: LLMCallRecord) -> None:
            _ = self.write_returning_path(record)

        def write_returning_path(self, record: LLMCallRecord) -> Path | None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            return tmp_path / "claimed.yaml"

    class _JunkSink(_PathSink):
        @override
        def write_returning_path(self, record: LLMCallRecord) -> Path | None:
            return "not a path"  # pyright: ignore[reportReturnType]  # deliberate contract breach

    configure_llm_logging(_PathSink())
    _ = write_llm_log(_record())
    health = get_logging_health()
    assert (health.sink, health.handed_off, health.written) == ("custom", 1, 0)

    configure_llm_logging(_JunkSink())
    _ = write_llm_log(_record())
    health = get_logging_health()
    assert health.failed == 1
    assert health.last_failure is not None
    assert health.last_failure.stage == "contract"


def test_disabled_logging_reports_disabled_and_counts_nothing() -> None:
    configure_llm_logging(None)

    _ = write_llm_log(_record())

    health = get_logging_health()
    assert health.sink == "disabled"
    assert (health.written, health.handed_off, health.failed, health.degraded) == (0, 0, 0, 0)
    assert health.last_failure is None


def test_repeated_failures_then_recovery(tmp_path: Path) -> None:
    """Every failure counts, even after the warning latch drops repeats to
    DEBUG, and a later success records ``last_success_at`` while keeping the
    earlier failure inspectable."""
    blocker = _broken_log_dir(tmp_path)
    configure_llm_logging(LocalYamlLogSink(blocker))
    for _ in range(3):
        _ = write_llm_log(_record())
    blocker.unlink()

    assert write_llm_log(_record()) is not None

    health = get_logging_health()
    assert (health.failed, health.written) == (3, 1)
    assert health.last_failure is not None
    assert health.last_success_at is not None
    assert health.last_success_at >= health.last_failure.occurred_at


def test_reconfiguring_starts_fresh_counts() -> None:
    sink = _RaisingSink()
    configure_llm_logging(sink)
    _ = write_llm_log(_record())
    assert get_logging_health().failed == 1

    configure_llm_logging(sink)

    health = get_logging_health()
    assert (health.failed, health.last_failure) == (0, None)


def test_an_in_flight_write_counts_against_the_installation_it_started_under() -> None:
    started = threading.Event()
    release = threading.Event()

    class _SlowSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            started.set()
            _ = release.wait(timeout=5)

    configure_llm_logging(_SlowSink())
    writer = threading.Thread(target=write_llm_log, args=(_record(),))
    writer.start()
    assert started.wait(timeout=5)
    replacement = _ListSink()
    configure_llm_logging(replacement)
    release.set()
    writer.join(timeout=5)

    health = get_logging_health()
    assert (health.handed_off, health.failed) == (0, 0)


def test_concurrent_writes_sum_exactly() -> None:
    class _AlternatingSink:
        def write(self, record: LLMCallRecord) -> None:
            if record.label == "fail":
                raise ValueError("odd one out")

    configure_llm_logging(_AlternatingSink())
    per_thread = 300
    barrier = threading.Barrier(4)

    def _hammer(label: str) -> None:
        _ = barrier.wait(timeout=5)
        for _ in range(per_thread):
            _ = write_llm_log(_record(label=label))

    threads = [threading.Thread(target=_hammer, args=(lbl,)) for lbl in ("ok", "fail") * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    health = get_logging_health()
    assert (health.handed_off, health.failed) == (2 * per_thread, 2 * per_thread)


def test_failure_snapshot_holds_no_record_payload() -> None:
    configure_llm_logging(_RaisingSink())

    _ = write_llm_log(_record(prompt="SECRET-PROMPT", response={"answer": "SECRET-RESPONSE"}))

    failure = get_logging_health().last_failure
    assert failure is not None
    assert set(vars(failure)) == {"stage", "error", "feature", "label", "occurred_at"}
    assert "SECRET" not in repr(failure)
    assert isinstance(failure.occurred_at, datetime)


def test_a_logging_failure_keeps_the_result_and_sends_one_request(tmp_path: Path) -> None:
    """Through the public call path: a broken sink, hidden warnings, the
    caller still gets its answer, the provider was asked exactly once, and
    the loss shows in the snapshot."""
    requests: list[object] = []

    def _fake(request: object) -> Reply:
        requests.append(request)
        return Reply.text("still here")

    configure_llm_logging(LocalYamlLogSink(_broken_log_dir(tmp_path)))
    with install(_fake):
        answer = text_llm_call_sync("say hi", feature="demo", label="x")

    assert answer == "still here"
    assert len(requests) == 1
    health = get_logging_health()
    assert health.failed == 1
    assert health.last_failure is not None
    assert (health.last_failure.feature, health.last_failure.label) == ("demo", "x")
