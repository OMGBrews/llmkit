"""Opting out writes nothing, and every redirection receives every record.

For each of the five record-writing lanes:

* ``configure_llm_logging(None)`` writes nothing to disk. This is asserted on
  the directories the default sink would have used (``LLMKIT_LOG_DIR`` and the
  project-root default), after a flush, never through
  ``capture_llm_records``, which sees a record before delivery admits it.
* A custom :class:`~llmkit.LogSink` receives every record the default sink
  writes, compared record for record against the same calls under a
  :class:`~llmkit.LocalYamlLogSink`.
* ``LLMKIT_LOG_DIR`` (default sink resolution) and an explicit ``log_dir``
  each receive every record, complete.

Each scenario is two attempts of one call (a failed attempt, then a
successful retry) so "every record" covers an error record and an ``ok`` one.
The disk sequence is the matrix's (``tests/logging/_record_disk.py``).

Before this module, ``configure_llm_logging(None)`` was exercised through a
call lane only for ``structured_llm_call`` (``tests/calls/test_capture_records.py``,
asserting no captured *path*, not an empty directory) and
``text_llm_call_stream`` (``tests/reliability/test_default_retries.py``, which
asserts retry behaviour, not the disk); the rest drive ``write_llm_log``
directly. ``LLMKIT_LOG_DIR`` was exercised through a lane only for
``structured_llm_call_sync`` (``tests/logging/test_run_scope.py``); the sink
hardening tests drive the sink directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llmkit import LLMCallRecord, LocalYamlLogSink, RetryPolicy
from llmkit.logging import LOG_DIR_ENV_VAR
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install, server_error
from tests.logging._record_disk import (
    FAMILIES,
    DiskRecord,
    Family,
    assert_delivery_available,
    call,
    check_contract,
    fake_answer,
    read_disk,
    settle,
    sink_installed,
)

_LABEL = "redirect"
_RETRY = RetryPolicy(max_attempts=2, backoff_base_seconds=0)


def _fails_then_answers(request: TransportRequest) -> Reply:
    if request.attempt == 1:
        raise server_error()
    if request.kind.startswith("tool"):
        return Reply(content="ok", tool_calls=[FakeToolCall("lookup", {"q": "a"})])
    return fake_answer(request)


async def _two_attempts(family: Family) -> int:
    """One call that fails once and then answers; returns the round trips made."""
    with install(_fails_then_answers) as installed:
        _ = await call(family, label=_LABEL, retry=_RETRY)
    return len(installed.requests)


type Summary = tuple[object, ...]


def _summary(
    feature: object, label: object, schema: object, attempt: object, error: object
) -> Summary:
    return (feature, label, schema, attempt, error)


def _disk_summary(records: list[DiskRecord]) -> list[Summary]:
    summaries = [
        _summary(
            r.body["feature"], r.body["label"], r.body["schema"], r.body["attempt"], r.body["error"]
        )
        for r in records
    ]
    return sorted(summaries, key=repr)


def _record_summary(records: list[LLMCallRecord]) -> list[Summary]:
    return sorted(
        (_summary(r.feature, r.label, r.schema, r.attempt, r.error) for r in records), key=repr
    )


class _ListSink:
    """A custom sink: not the built-in, so health counts it as ``custom``."""

    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    def write(self, record: LLMCallRecord) -> None:
        self.records.append(record)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run inside a fresh project root with ``LLMKIT_LOG_DIR`` pointing beside it.

    Both places the default sink could resolve to are under ``tmp_path``, so a
    write that should not happen is visible, and none can reach the
    repository's own ``data/llm-logs``. The third, the per-user folder a
    default sink falls back to when its project folder cannot hold files, is
    out of reach: with ``LLMKIT_LOG_DIR`` set, the default never falls back.
    """
    assert_delivery_available()
    root = tmp_path / "project"
    root.mkdir()
    _ = (root / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv(LOG_DIR_ENV_VAR, str(tmp_path / "env-logs"))
    yield root


def _env_dir(project: Path) -> Path:
    return project.parent / "env-logs"


def _project_default_dir(project: Path) -> Path:
    return project / "data" / "llm-logs"


@pytest.mark.parametrize("family", FAMILIES)
async def test_opting_out_writes_nothing_to_disk(family: Family, project: Path) -> None:
    with sink_installed(None):
        round_trips = await _two_attempts(family)
        health = await settle()

    assert round_trips == 2, "the call did not make both attempts"
    assert list(project.parent.rglob("*.yaml")) == []
    assert list(project.parent.rglob("*.jsonl")) == []
    assert not _env_dir(project).exists()
    assert not _project_default_dir(project).exists()
    assert health.sink == "disabled"
    assert (health.written, health.handed_off) == (0, 0), health


@pytest.mark.parametrize("family", FAMILIES)
async def test_a_custom_sink_receives_every_record_the_default_writes(
    family: Family, project: Path
) -> None:
    default_dir = project.parent / "default-sink"
    with sink_installed(LocalYamlLogSink(default_dir)):
        _ = await _two_attempts(family)
        default_health = await settle()
    on_disk = read_disk(default_dir)

    sink = _ListSink()
    with sink_installed(sink):
        _ = await _two_attempts(family)
        custom_health = await settle()

    assert default_health.written == len(on_disk) == 2
    assert custom_health.sink == "custom"
    assert custom_health.handed_off == len(sink.records) == 2
    assert _record_summary(sink.records) == _disk_summary(on_disk)
    assert [r.error is None for r in sorted(sink.records, key=lambda r: r.attempt or 0)] == [
        False,
        True,
    ]


@pytest.mark.parametrize("family", FAMILIES)
async def test_llmkit_log_dir_receives_every_record(family: Family, project: Path) -> None:
    sink = LocalYamlLogSink()  # default resolution, at first write
    with sink_installed(sink):
        _ = await _two_attempts(family)
        health = await settle()

    records = read_disk(_env_dir(project))
    assert sink.log_dir == _env_dir(project)
    assert health.written == len(records) == 2
    for record in records:
        check_contract(record)
    assert [r.body["attempt"] for r in records] == [1, 2]
    assert not _project_default_dir(project).exists()


@pytest.mark.parametrize("family", FAMILIES)
async def test_an_explicit_log_dir_receives_every_record(family: Family, project: Path) -> None:
    explicit = project.parent / "explicit"
    with sink_installed(LocalYamlLogSink(explicit)):
        _ = await _two_attempts(family)
        health = await settle()

    records = read_disk(explicit)
    assert health.written == len(records) == 2
    for record in records:
        check_contract(record)
    assert [r.body["attempt"] for r in records] == [1, 2]
    # The explicit directory wins over the environment and the project root.
    assert not _env_dir(project).exists()
    assert not _project_default_dir(project).exists()
