"""Logging-health state: what happened to the records handed to the sink.

A leaf module (no imports from the rest of the package) so both the registry,
which counts, and the shipped sink, which reports its outcome, can depend on
it without a cycle.

Diagnostic warnings can be hidden by the host's logging configuration; this
state cannot. The registry owns one :class:`HealthState` per installed sink
and is the only writer, so a sink driven directly (outside
:func:`~llmkit.logging.configure_llm_logging`) or wrapped by another sink is
never counted twice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

#: Where a detected logging failure happened.
#:
#: ``write`` — the built-in sink could not write the per-call file;
#: ``index`` — the file was written but its ``index.jsonl`` line was not;
#: ``housekeeping`` — the file was written but retention pruning failed;
#: ``sink_raised`` — a configured sink raised out of its write method;
#: ``contract`` — a sink's ``write_returning_path`` returned neither a path nor
#: ``None``.
FailureStage = Literal["write", "index", "housekeeping", "sink_raised", "contract"]

#: What kind of sink the health snapshot describes.
SinkKind = Literal["disabled", "builtin", "custom"]


@dataclass(frozen=True)
class LoggingFailure:
    """The most recent detected logging failure.

    Carries identifying metadata only, never the record's prompt or response,
    so holding it costs nothing sensitive.

    Attributes:
        stage: Where the failure happened (see the ``FailureStage`` values).
        error: ``"ExceptionType: message"`` text of the underlying exception.
        feature: The failed record's ``feature``.
        label: The failed record's ``label``.
        occurred_at: Timezone-aware UTC time the failure was recorded.
    """

    stage: FailureStage
    error: str
    feature: str
    label: str | None
    occurred_at: datetime


@dataclass(frozen=True)
class LoggingHealth:
    """A point-in-time snapshot of the installed sink's outcomes.

    Counts cover the current installation only: every
    :func:`~llmkit.logging.configure_llm_logging` call starts a fresh set, and
    a write still in flight when the sink is replaced is counted against the
    installation it started under.

    Attributes:
        sink: ``"disabled"`` when logging is off, ``"builtin"`` for the shipped
            :class:`~llmkit.logging.LocalYamlLogSink` (whose writes llmkit
            verifies), ``"custom"`` for any other sink.
        written: Records the built-in sink wrote to disk.
        handed_off: Records a custom sink accepted by returning normally.
            llmkit cannot see past that return, so this is not evidence that
            the record was persisted.
        failed: Records lost to a detected failure: the built-in write failed,
            a sink raised, or a sink broke its return contract.
        degraded: Written records whose index line or retention housekeeping
            failed afterwards. Each is also counted in ``written``.
        queued: Waiting records, including snapshot preparation reservations.
        in_flight: Records currently inside a sink write.
        dropped: Records rejected or discarded before a sink write.
        unavailable: Subset of drops caused by shutdown or a dead worker.
        wait_timeouts: Caller waits that expired; delivery may still succeed.
        preparation_failed: Snapshot/startup failures, also counted in failed.
        last_failure: The most recent sink failure, or ``None``. Admission and
            preparation counters do not retain payloads or exception text.
        last_success_at: When a record was last written or handed off.
    """

    sink: SinkKind
    written: int = 0
    handed_off: int = 0
    failed: int = 0
    degraded: int = 0
    last_failure: LoggingFailure | None = None
    last_success_at: datetime | None = None
    queued: int = 0
    in_flight: int = 0
    dropped: int = 0
    unavailable: int = 0
    wait_timeouts: int = 0
    preparation_failed: int = 0


@dataclass(frozen=True)
class WriteOutcome:
    """What one built-in write produced, returned instead of swallowed.

    ``path`` is the written file, or ``None`` when the write failed with
    ``write_error``. ``post_write_failures`` lists the index and housekeeping
    failures that followed a successful write, in the order they occurred.
    """

    path: Path | None
    write_error: Exception | None = None
    post_write_failures: tuple[tuple[FailureStage, Exception], ...] = ()


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


@dataclass
class HealthState:
    """Lock-guarded counters for one sink installation.

    Unlike the warn-once latches, which tolerate a duplicate warning and so
    stay unlocked, counters must sum exactly across the worker threads that
    perform concurrent writes.
    """

    sink: SinkKind
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _written: int = 0
    _handed_off: int = 0
    _failed: int = 0
    _degraded: int = 0
    _queued: int = 0
    _in_flight: int = 0
    _dropped: int = 0
    _unavailable: int = 0
    _wait_timeouts: int = 0
    _preparation_failed: int = 0
    _last_failure: LoggingFailure | None = None
    _last_success_at: datetime | None = None

    @staticmethod
    def _failure(
        stage: FailureStage, exc: BaseException, feature: str, label: str | None
    ) -> LoggingFailure:
        # Exception formatting may run arbitrary sink code: never hold _lock.
        return LoggingFailure(
            stage=stage,
            error=_describe(exc),
            feature=feature,
            label=label,
            occurred_at=datetime.now(UTC),
        )

    def record_written(
        self,
        feature: str,
        label: str | None,
        post_write_failures: tuple[tuple[FailureStage, Exception], ...] = (),
    ) -> None:
        failures = [self._failure(stage, exc, feature, label) for stage, exc in post_write_failures]
        with self._lock:
            self._written += 1
            self._last_success_at = datetime.now(UTC)
            if post_write_failures:
                self._degraded += 1
                self._last_failure = failures[-1]

    def record_handed_off(self) -> None:
        with self._lock:
            self._handed_off += 1
            self._last_success_at = datetime.now(UTC)

    def record_failed(
        self, stage: FailureStage, exc: BaseException, feature: str, label: str | None
    ) -> None:
        failure = self._failure(stage, exc, feature, label)
        with self._lock:
            self._failed += 1
            self._last_failure = failure

    def delivery(
        self,
        *,
        queued: int = 0,
        in_flight: int = 0,
        dropped: int = 0,
        unavailable: int = 0,
        wait_timeouts: int = 0,
        preparation_failed: int = 0,
    ) -> None:
        """Update delivery counters without diagnostics or user callbacks.

        Queued includes preparation reservations. Drops never reached the sink;
        expired waits may still succeed later. Preparation failures count as
        failed without formatting arbitrary payloads or exceptions on the caller.
        """
        with self._lock:
            self._queued += queued
            self._in_flight += in_flight
            self._dropped += dropped
            self._unavailable += unavailable
            self._wait_timeouts += wait_timeouts
            self._preparation_failed += preparation_failed
            self._failed += preparation_failed

    def record_worker_failure(self) -> None:
        """Count a fatal worker exception without invoking its formatting hooks."""
        with self._lock:
            self._failed += 1

    def snapshot(self) -> LoggingHealth:
        with self._lock:
            return LoggingHealth(
                sink=self.sink,
                written=self._written,
                handed_off=self._handed_off,
                failed=self._failed,
                degraded=self._degraded,
                queued=self._queued,
                in_flight=self._in_flight,
                dropped=self._dropped,
                unavailable=self._unavailable,
                wait_timeouts=self._wait_timeouts,
                preparation_failed=self._preparation_failed,
                last_failure=self._last_failure,
                last_success_at=self._last_success_at,
            )
