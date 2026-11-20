"""The default sink's fallback from an unusable project folder to the per-user one.

When nobody chose a log directory and the project-root default
(``<root>/data/llm-logs``) cannot hold files, a :class:`LocalYamlLogSink`
moves to the per-user state directory once, warns once, and reports the move
in :func:`get_logging_health` as a ``degraded`` write at the ``location``
stage. Pinned here:

* **Which failures move it** — only an error from
  :data:`~llmkit.logging._paths.FALLBACK_ERRNOS`, raised while creating the
  folder or opening the first record's file: a file named ``data``, a file at
  ``data/llm-logs``, a read-only root, or a deleted current directory. An
  unlisted error (a full disk) or a listed one raised later in the write
  keeps today's visible failure.
* **When** — only until the sink's first successful write, and only for a
  directory the default resolution chose: ``LLMKIT_LOG_DIR`` and an explicit
  ``log_dir=`` keep failing visibly.
* **What it says** — one WARNING per sink naming both folders (or, for a
  deleted current directory, the reason and the per-user folder), and no
  ``.gitignore`` in the per-user folder.

Every test points ``HOME``, ``XDG_STATE_HOME`` and ``LOCALAPPDATA`` under
``tmp_path`` (the conftest clears neither), so a fallback can never reach the
real per-user folder.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llmkit.logging import (
    LocalYamlLogSink,
    configure_llm_logging,
    default_log_dir,
    get_logging_health,
    local_yaml,
    write_llm_log,
)
from llmkit.logging._paths import FALLBACK_ERRNOS, LOG_DIR_ENV_VAR, per_user_log_dir
from llmkit.logging.record import INDEX_FILENAME
from tests._support import make_record

if TYPE_CHECKING:
    from _typeshed import OpenTextMode

_LOGGER = "llmkit.logging"
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX filesystem semantics")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root as the current directory; per-user folders under ``tmp_path``."""
    monkeypatch.delenv(LOG_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    root = tmp_path / "proj"
    root.mkdir()
    _ = (root / "pyproject.toml").write_text("[project]\nname = 'p'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def _state_dir() -> Path:
    return per_user_log_dir()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


def _block_data(project: Path) -> Path:
    """A regular file where the ``data`` folder must go; mkdir fails."""
    _ = (project / "data").write_text("occupied", encoding="utf-8")
    return project / "data" / "llm-logs"


def _block_leaf(project: Path) -> Path:
    """A regular file at ``data/llm-logs``; mkdir passes, the record open fails."""
    (project / "data").mkdir()
    leaf = project / "data" / "llm-logs"
    _ = leaf.write_text("occupied", encoding="utf-8")
    return leaf


def _assert_fell_back(
    sink: LocalYamlLogSink, project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two writes land in the per-user folder, after exactly one WARNING."""
    state = _state_dir()
    first = sink.write_returning_path(make_record(label="first"))
    second = sink.write_returning_path(make_record(label="second"))
    assert first is not None and first.parent == state
    assert second is not None and second.parent == state
    assert sink.log_dir == state
    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    assert str(project_dir) in warnings[0]
    assert str(state) in warnings[0]
    assert LOG_DIR_ENV_VAR in warnings[0]
    # The per-user folder is never inside a repository: nothing is seeded.
    assert not (state / ".gitignore").exists()


@pytest.mark.parametrize("block", [_block_data, _block_leaf], ids=["file-at-data", "file-at-leaf"])
def test_blocked_project_folder_falls_back_once(
    project: Path,
    caplog: pytest.LogCaptureFixture,
    block: Callable[[Path], Path],
) -> None:
    """A file where the folder must go — caught at ``mkdir`` or only at the
    record's open — moves the sink, and a read made before the first write
    still showed the project folder."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    project_dir = block(project)
    sink = LocalYamlLogSink()
    assert sink.log_dir == project_dir
    _assert_fell_back(sink, project_dir, caplog)
    # The ordinary first-write notice names the folder actually in use.
    notices = [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER and r.levelno == logging.INFO and "is logging" in r.getMessage()
    ]
    assert len(notices) == 1 and str(_state_dir()) in notices[0]


def _require_enforced_permissions(tmp_path: Path) -> None:
    """Skip unless a read-only directory really refuses a new entry here.

    Windows ignores the mode bits, and root (or any process holding
    ``CAP_DAC_OVERRIDE``) writes through them.
    """
    if os.name == "nt":
        pytest.skip("POSIX permissions")
    probe = tmp_path / "permission-probe"
    probe.mkdir()
    probe.chmod(0o555)
    try:
        (probe / "entry").mkdir()
    except PermissionError:
        return
    finally:
        probe.chmod(0o755)
    pytest.skip("read-only directories are writable here (root or CAP_DAC_OVERRIDE)")


def test_read_only_project_root_falls_back(project: Path, caplog: pytest.LogCaptureFixture) -> None:
    """``mkdir`` of ``data`` under a read-only root raises ``PermissionError``."""
    _require_enforced_permissions(project.parent)
    caplog.set_level(logging.INFO, logger=_LOGGER)
    project.chmod(0o555)
    try:
        _assert_fell_back(LocalYamlLogSink(), project / "data" / "llm-logs", caplog)
    finally:
        project.chmod(0o755)


def test_read_only_existing_project_folder_falls_back(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An existing read-only ``data/llm-logs`` fails at the record's open."""
    _require_enforced_permissions(project.parent)
    caplog.set_level(logging.INFO, logger=_LOGGER)
    leaf = project / "data" / "llm-logs"
    leaf.mkdir(parents=True)
    leaf.chmod(0o555)
    try:
        _assert_fell_back(LocalYamlLogSink(), leaf, caplog)
    finally:
        leaf.chmod(0o755)


@_POSIX_ONLY
def test_deleted_current_directory_falls_back(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No current directory means no project root to name: the sink and
    :func:`default_log_dir` use the per-user folder, and the one WARNING says
    why instead of naming a project path."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    gone = project / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    state = _state_dir()

    assert default_log_dir() == state
    sink = LocalYamlLogSink()
    assert sink.log_dir == state
    assert _warnings(caplog) == []  # owed until the first write attempt
    first = sink.write_returning_path(make_record(label="first"))
    second = sink.write_returning_path(make_record(label="second"))
    assert first is not None and first.parent == state
    assert second is not None and second.parent == state
    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "current directory" in warnings[0]
    assert str(state) in warnings[0]
    assert LOG_DIR_ENV_VAR in warnings[0]
    assert not (state / ".gitignore").exists()


@_POSIX_ONLY
def test_deleted_current_directory_with_relative_state_home_still_fails(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``XDG_STATE_HOME`` cannot be made absolute without a
    current directory either, so the failure stays visible."""
    if not str(_state_dir()).startswith(str(project.parent / "state")):
        pytest.skip("XDG_STATE_HOME is not this platform's per-user folder")
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    gone = project / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()

    with pytest.raises(FileNotFoundError):
        _ = default_log_dir()
    configure_llm_logging(LocalYamlLogSink())
    assert write_llm_log(make_record()) is None
    health = get_logging_health()
    assert (health.written, health.failed) == (0, 1)


def test_unusable_env_log_dir_does_not_fall_back(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``LLMKIT_LOG_DIR`` is the caller's choice: it fails where it points."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    blocker = project / "blocker"
    _ = blocker.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv(LOG_DIR_ENV_VAR, str(blocker / "logs"))
    sink = LocalYamlLogSink()

    assert sink.write_returning_path(make_record()) is None
    assert sink.log_dir == blocker / "logs"
    assert not _state_dir().exists()
    assert not any("instead" in w for w in _warnings(caplog))


def test_unusable_explicit_log_dir_does_not_fall_back(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicit ``log_dir=`` naming the very path the default would use is
    still the caller's choice."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    project_dir = _block_data(project)
    sink = LocalYamlLogSink(project_dir)

    assert sink.write_returning_path(make_record()) is None
    assert sink.log_dir == project_dir
    assert not _state_dir().exists()
    assert not any("instead" in w for w in _warnings(caplog))


def test_failure_after_first_successful_write_does_not_fall_back(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Once a record is in the project folder, that folder is kept: a later
    failure is reported as it always was, and nothing moves."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    configure_llm_logging(LocalYamlLogSink(retention_days=None, max_index_bytes=None))
    project_dir = project / "data" / "llm-logs"
    assert write_llm_log(make_record(label="kept")) is not None

    for child in project_dir.iterdir():
        child.unlink()
    project_dir.rmdir()
    _ = project_dir.write_text("occupied", encoding="utf-8")

    assert write_llm_log(make_record(label="lost")) is None
    assert not _state_dir().exists()
    health = get_logging_health()
    assert (health.written, health.failed, health.degraded) == (1, 1, 0)
    assert health.last_failure is not None and health.last_failure.stage == "write"
    assert not any("instead" in w for w in _warnings(caplog))


def _failing_part_open(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    """Fail every record-file open with *error*; other opens are real."""
    real_open = open

    def _open(
        file: Path,
        mode: OpenTextMode,
        *,
        encoding: str,
        opener: Callable[[str, int], int] | None = None,
    ) -> TextIOWrapper:
        if file.name.endswith(".part"):
            raise error
        return real_open(file, mode, encoding=encoding, opener=opener)

    monkeypatch.setattr(local_yaml, "open", _open, raising=False)


def _dangling_symlink(data: Path) -> None:
    data.symlink_to(data.parent / "missing")


def _looping_symlink(data: Path) -> None:
    data.symlink_to(data)


@_POSIX_ONLY
@pytest.mark.parametrize(
    ("make_link", "code"),
    [
        (_dangling_symlink, errno.ENOENT),
        (_looping_symlink, errno.ELOOP),
    ],
    ids=["dangling-symlink-ENOENT", "symlink-loop-ELOOP"],
)
def test_broken_symlink_at_data_falls_back(
    project: Path,
    caplog: pytest.LogCaptureFixture,
    make_link: Callable[[Path], None],
    code: int,
) -> None:
    """A ``data`` symlink that leads nowhere fails only at the record's open,
    and the error that moved the sink is the one the folder really raised."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    make_link(project / "data")
    sink = LocalYamlLogSink()

    outcome = sink.write_with_outcome(make_record(label="first"))
    assert outcome.path is not None and outcome.path.parent == _state_dir()
    stage, error = outcome.post_write_failures[0]
    assert stage == "location"
    assert isinstance(error, OSError) and error.errno == code
    _assert_fell_back(sink, project / "data" / "llm-logs", caplog)


# Written out rather than read from FALLBACK_ERRNOS, so dropping an entry
# from the constant fails here instead of silently removing its case.
_DECIDED_ERRNOS = (
    errno.EACCES,
    errno.EPERM,
    errno.EROFS,
    errno.ENOTDIR,
    errno.ENOENT,
    errno.ELOOP,
)


def test_the_trigger_list_is_the_decided_one() -> None:
    assert frozenset(_DECIDED_ERRNOS) == FALLBACK_ERRNOS


@pytest.mark.parametrize(
    "code", _DECIDED_ERRNOS, ids=[errno.errorcode[code] for code in _DECIDED_ERRNOS]
)
def test_every_listed_errno_at_the_open_falls_back(
    project: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """Each listed error, raised by the project folder's record open, moves
    the sink; the per-user folder's open is left real, so the retry lands."""
    project_dir = project / "data" / "llm-logs"
    real_open = open

    def _open(
        file: Path,
        mode: OpenTextMode,
        *,
        encoding: str,
        opener: Callable[[str, int], int] | None = None,
    ) -> TextIOWrapper:
        if file.parent == project_dir and file.name.endswith(".part"):
            raise OSError(code, os.strerror(code))
        return real_open(file, mode, encoding=encoding, opener=opener)

    monkeypatch.setattr(local_yaml, "open", _open, raising=False)
    sink = LocalYamlLogSink()

    path = sink.write_returning_path(make_record())
    assert path is not None and path.parent == _state_dir()


def test_unlisted_error_does_not_fall_back(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full disk is not "this location can never hold my files"."""
    _failing_part_open(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
    sink = LocalYamlLogSink()

    assert sink.write_returning_path(make_record()) is None
    assert sink.log_dir == project / "data" / "llm-logs"
    assert not _state_dir().exists()


def test_listed_error_after_the_open_does_not_fall_back(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed errno from publishing the record is not a directory failure."""

    def _refuse(_part: Path, _final: Path) -> bool:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(local_yaml, "_publish", _refuse)
    sink = LocalYamlLogSink()

    assert sink.write_returning_path(make_record()) is None
    assert sink.log_dir == project / "data" / "llm-logs"
    assert not _state_dir().exists()


def test_fallback_is_counted_once_as_degraded_location(project: Path) -> None:
    configure_llm_logging(LocalYamlLogSink())
    _ = _block_data(project)

    assert write_llm_log(make_record(label="first")) is not None
    health = get_logging_health()
    assert (health.written, health.degraded, health.failed) == (1, 1, 0)
    failure = health.last_failure
    assert failure is not None
    assert (failure.stage, failure.label) == ("location", "first")
    # POSIX reports a file in the folder's path as ENOTDIR; Windows as ENOENT.
    assert failure.error.split(":")[0] in {"NotADirectoryError", "FileNotFoundError"}

    assert write_llm_log(make_record(label="second")) is not None
    health = get_logging_health()
    assert (health.written, health.degraded, health.failed) == (2, 1, 0)


@_POSIX_ONLY
def test_deleted_current_directory_is_counted_as_location(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gone = project / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    configure_llm_logging(LocalYamlLogSink())

    assert write_llm_log(make_record()) is not None
    health = get_logging_health()
    assert (health.written, health.degraded, health.failed) == (1, 1, 0)
    assert health.last_failure is not None
    assert health.last_failure.stage == "location"
    assert health.last_failure.error.startswith("FileNotFoundError")


def test_a_later_failure_on_the_same_write_stays_last(project: Path) -> None:
    """The location entry goes first, so an index failure on the same write is
    still the one ``last_failure`` reports, and the write counts once."""
    configure_llm_logging(LocalYamlLogSink())
    _ = _block_data(project)
    (_state_dir() / INDEX_FILENAME).mkdir(parents=True)

    assert write_llm_log(make_record()) is not None
    health = get_logging_health()
    assert (health.written, health.degraded) == (1, 1)
    assert health.last_failure is not None and health.last_failure.stage == "index"


def test_racing_first_writes_both_land_after_one_warning(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two first writes that both fail in the project folder: one moves the
    sink, the other sees it moved and retries, and neither record is lost."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    project_dir = _block_leaf(project)
    both_failing = threading.Barrier(2, timeout=10)
    real_open = open

    def _open(
        file: Path,
        mode: OpenTextMode,
        *,
        encoding: str,
        opener: Callable[[str, int], int] | None = None,
    ) -> TextIOWrapper:
        if file.parent == project_dir and file.name.endswith(".part"):
            # Hold both writers at the failing open so neither has switched.
            _ = both_failing.wait()
        return real_open(file, mode, encoding=encoding, opener=opener)

    monkeypatch.setattr(local_yaml, "open", _open, raising=False)
    sink = LocalYamlLogSink()
    results: list[Path | None] = []

    def _write(label: str) -> None:
        results.append(sink.write_returning_path(make_record(label=label)))

    threads = [threading.Thread(target=_write, args=(f"t{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    state = _state_dir()
    assert len(results) == 2
    assert all(path is not None and path.parent == state for path in results), results
    assert len(_warnings(caplog)) == 1


def test_a_failure_after_a_racing_first_record_landed_does_not_move_the_sink(
    project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A record published in the project folder closes the fallback window at
    once: a second first write failing there a moment later is reported as a
    failed write, not turned into a move away from the folder that took it."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    project_dir = project / "data" / "llm-logs"
    published = threading.Event()
    real_publish = local_yaml._publish
    real_open = open
    failing_thread: list[threading.Thread] = []

    def _slow_publish(part: Path, final: Path) -> bool:
        result = real_publish(part, final)
        published.set()
        # Give the other writer time to fail before this one returns.
        _ = threading.Event().wait(0.3)
        return result

    def _open(
        file: Path,
        mode: OpenTextMode,
        *,
        encoding: str,
        opener: Callable[[str, int], int] | None = None,
    ) -> TextIOWrapper:
        if (
            threading.current_thread() in failing_thread
            and file.parent == project_dir
            and file.name.endswith(".part")
        ):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(file, mode, encoding=encoding, opener=opener)

    monkeypatch.setattr(local_yaml, "_publish", _slow_publish)
    monkeypatch.setattr(local_yaml, "open", _open, raising=False)
    sink = LocalYamlLogSink(retention_days=None, max_index_bytes=None)
    results: dict[str, Path | None] = {}

    def _write(label: str) -> None:
        results[label] = sink.write_returning_path(make_record(label=label))

    first = threading.Thread(target=_write, args=("first",))
    second = threading.Thread(target=_write, args=("second",))
    failing_thread.append(second)
    first.start()
    assert published.wait(10)
    second.start()
    first.join(timeout=20)
    second.join(timeout=20)

    landed = results["first"]
    assert landed is not None and landed.parent == project_dir
    assert results["second"] is None
    assert sink.log_dir == project_dir
    assert not _state_dir().exists()
    assert not any("instead" in w for w in _warnings(caplog))
