"""Tests for the logging-sink hardening pass.

Pins the July-2026 audit fixes to :class:`LocalYamlLogSink` /
:func:`write_llm_log`:

* **Default directory** — :func:`default_log_dir` resolves lazily
  (``LLMKIT_LOG_DIR`` > nearest project root > per-user state dir), the sink
  freezes the answer at first use so a mid-run ``chdir`` can't split logs,
  and the project-root default seeds a ``.gitignore`` (whoever created the
  directory) so prompt logs never land in the repo's history.
* **Permissions** — files ``0o600`` and a sink-created directory ``0o700``
  (POSIX), with a pre-existing directory never re-chmodded.
* **Warn-once latch** — a persistently broken sink warns once (per failure
  signature) and drops to DEBUG, re-arming on success/reconfigure.
* **Retention** — aged records and rotated index generations are pruned,
  only by the names llmkit generates (a co-located ``docker-compose.yaml`` or
  ``index-notes.jsonl`` survives), and an oversized ``index.jsonl`` rotates
  losslessly, with a rotation another process already made counted as done,
  throttled to hourly; the first successful write announces the directory and
  policy at INFO.
* **Whole files only** — a record is written under a hidden ``.part`` name
  and published without clobbering, so a write cut short never leaves a
  ``*.yaml``; retention sweeps stale ``.part`` files, and lone surrogates in a
  record never lose its file or index line.
* **Eager config validation** — the growth bounds take a positive int or
  ``None`` (``0`` raises rather than being read as either "keep nothing" or
  "unlimited"), a relative log directory is absolutised before it is frozen,
  a non-sink is rejected by :func:`configure_llm_logging` instead of on every
  later call, and a third-party ``write_returning_path`` that breaks its
  ``Path | None`` contract never reaches ``capture_llm_log_paths``.

Records are built directly via :func:`tests._support.make_record`; no real
LLM call is made.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import cast

import pytest
import yaml

import llmkit.logging.local_yaml as local_yaml
from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    capture_llm_log_paths,
    configure_llm_logging,
    default_log_dir,
    get_log_sink,
)
from llmkit.capture import record_call
from llmkit.logging import INDEX_FILENAME, write_llm_log
from llmkit.logging._paths import is_prunable_name
from tests._support import make_record as _record

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-only")


def _age(path: Path, *, days: float) -> None:
    """Backdate *path*'s mtime by *days* so retention sees it as stale."""
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def _unthrottle(sink: LocalYamlLogSink) -> None:
    """Clear the hourly prune throttle so the next write prunes again."""
    # Private access on purpose: simulating the hour elapsing beats sleeping.
    sink._last_prune = None


# 1. Default-directory resolution.


def test_env_var_overrides_default_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLMKIT_LOG_DIR`` wins over every other resolution source."""
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path / "envlogs"))
    assert default_log_dir() == tmp_path / "envlogs"


def test_env_var_is_read_lazily_not_at_import_or_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env var is honored even when set *after* the sink is constructed —
    resolution happens at first write, never at import time."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    sink = LocalYamlLogSink()
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path / "late-env"))
    path = sink.write_returning_path(_record())
    assert path is not None
    assert path.parent == tmp_path / "late-env"


def test_default_dir_is_frozen_at_first_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once resolved, the default directory sticks for the sink's lifetime:
    a later env change (or chdir) cannot split one process's logs."""
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path / "first"))
    sink = LocalYamlLogSink()
    p1 = sink.write_returning_path(_record())
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path / "second"))
    p2 = sink.write_returning_path(_record())
    assert p1 is not None and p2 is not None
    assert p1.parent == tmp_path / "first"
    assert p2.parent == tmp_path / "first"
    assert not (tmp_path / "second").exists()


def test_relative_log_dir_is_frozen_absolutely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit *relative* ``log_dir`` names one directory for the sink's
    lifetime: it is made absolute at construction, so a later ``chdir`` cannot
    re-anchor it and split one process's logs across two directories."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    sink = LocalYamlLogSink(Path("logs"))
    p1 = sink.write_returning_path(_record())
    monkeypatch.chdir(second)
    p2 = sink.write_returning_path(_record())
    assert p1 is not None and p2 is not None
    assert p1.parent.is_absolute()
    assert p1.parent == p2.parent
    assert p1.parent.resolve() == (first / "logs").resolve()
    assert not (second / "logs").exists()


def test_relative_env_log_dir_is_frozen_absolutely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same for a *relative* ``LLMKIT_LOG_DIR``: resolution absolutises it,
    so the default sink's frozen answer survives a ``chdir`` too."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("LLMKIT_LOG_DIR", "relative-logs")
    monkeypatch.chdir(first)
    assert default_log_dir().is_absolute()
    sink = LocalYamlLogSink()
    p1 = sink.write_returning_path(_record())
    monkeypatch.chdir(second)
    p2 = sink.write_returning_path(_record())
    assert p1 is not None and p2 is not None
    assert p1.parent == p2.parent
    assert p1.parent.resolve() == (first / "relative-logs").resolve()
    assert not (second / "relative-logs").exists()


def test_str_log_dir_is_rejected() -> None:
    """A ``str`` log directory is the one config mistake at this boundary that
    used to survive construction and then lose every log inside the write
    path's best-effort swallow."""
    with pytest.raises(TypeError, match="log_dir"):
        _ = LocalYamlLogSink("logs")  # pyright: ignore[reportArgumentType]  # the mistake under test


def test_project_root_walk_finds_nearest_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the env var, the default lands in ``data/llm-logs`` under the
    nearest ancestor with a ``pyproject.toml``/``.git`` marker."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    project = tmp_path / "proj"
    deep = project / "src" / "pkg"
    deep.mkdir(parents=True)
    _ = (project / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(deep)
    assert default_log_dir() == project / "data" / "llm-logs"


def test_nested_project_wins_over_enclosing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nearest marker wins: a nested project logs under its own root, not the
    enclosing monorepo's."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    outer = tmp_path / "mono"
    inner = outer / "libs" / "child"
    workdir = inner / "src"
    workdir.mkdir(parents=True)
    _ = (outer / "pyproject.toml").write_text("[project]\n")
    (inner / ".git").mkdir()
    monkeypatch.chdir(workdir)
    assert default_log_dir() == inner / "data" / "llm-logs"


@pytest.mark.skipif(
    sys.platform == "darwin" or os.name == "nt", reason="XDG state path is Linux-specific"
)
def test_no_marker_falls_back_to_xdg_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process running outside any project logs to the per-user state dir —
    never to a CWD-relative path."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    bare = tmp_path / "no-project-here"
    bare.mkdir()
    monkeypatch.chdir(bare)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert default_log_dir() == tmp_path / "state" / "llmkit" / "llm-logs"


@pytest.mark.skipif(
    sys.platform == "darwin" or os.name == "nt", reason="XDG state path is Linux-specific"
)
def test_relative_state_dir_env_is_absolutised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state-dir fallback reads ``XDG_STATE_HOME`` verbatim, so a relative
    value there is the same chdir-splitting hazard as a relative
    ``LLMKIT_LOG_DIR`` — the docstring's "never a CWD-relative path" has to
    hold for the *whole* resolution, not just the branches with an absolute
    env value."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    bare = tmp_path / "no-project-here"
    bare.mkdir()
    monkeypatch.chdir(bare)
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    resolved = default_log_dir()
    assert resolved.is_absolute()
    assert resolved.resolve() == (bare / "relative-state" / "llmkit" / "llm-logs").resolve()


# 2. .gitignore seeding.


def test_project_root_default_seeds_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the *default* resolution lands inside a project root and the sink
    creates the directory, a ``.gitignore`` keeps prompt logs out of the repo."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    project = tmp_path / "proj"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(project)
    path = LocalYamlLogSink().write_returning_path(_record())
    assert path is not None
    gitignore = project / "data" / "llm-logs" / ".gitignore"
    assert gitignore.read_text() == "*\n"


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root as the current directory, with no ``LLMKIT_LOG_DIR``."""
    monkeypatch.delenv("LLMKIT_LOG_DIR", raising=False)
    project = tmp_path / "proj"
    project.mkdir()
    _ = (project / "pyproject.toml").write_text("[project]\n")
    monkeypatch.chdir(project)
    return project


def test_pre_existing_project_root_default_is_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project-root default directory that already existed (a checkout, a
    ``mkdir``, an older llmkit) is seeded too, and is still never re-chmodded."""
    log_dir = _project(tmp_path, monkeypatch) / "data" / "llm-logs"
    log_dir.mkdir(parents=True)
    os.chmod(log_dir, 0o755)
    sink = LocalYamlLogSink()
    assert sink.write_returning_path(_record()) is not None
    assert (log_dir / ".gitignore").read_text() == "*\n"
    if os.name == "posix":
        assert log_dir.stat().st_mode & 0o777 == 0o755


def test_existing_gitignore_is_left_alone_and_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The seed is exclusive-create: the user's own ``.gitignore`` survives,
    and finding one is not a "Could not seed" diagnostic."""
    log_dir = _project(tmp_path, monkeypatch) / "data" / "llm-logs"
    log_dir.mkdir(parents=True)
    _ = (log_dir / ".gitignore").write_text("*.yaml\n")
    with caplog.at_level(logging.DEBUG, logger="llmkit.logging"):
        assert LocalYamlLogSink().write_returning_path(_record()) is not None
    assert (log_dir / ".gitignore").read_text() == "*.yaml\n"
    assert not [r for r in caplog.records if "Could not seed" in r.message]


def test_explicit_log_dir_is_not_seeded(tmp_path: Path) -> None:
    """An explicit ``log_dir`` is the caller's choice — no ``.gitignore``."""
    path = LocalYamlLogSink(tmp_path / "logs").write_returning_path(_record())
    assert path is not None
    assert not (tmp_path / "logs" / ".gitignore").exists()


def test_env_var_log_dir_is_not_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory chosen via ``LLMKIT_LOG_DIR`` is likewise the caller's
    choice — no ``.gitignore``."""
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path / "envlogs"))
    path = LocalYamlLogSink().write_returning_path(_record())
    assert path is not None
    assert not (tmp_path / "envlogs" / ".gitignore").exists()


# 3. Permissions.


@_POSIX_ONLY
def test_sink_created_dir_and_files_are_owner_only(tmp_path: Path) -> None:
    """A sink-created directory is ``0o700`` and both file kinds ``0o600`` —
    no group/other access to prompt data on a multi-user host."""
    log_dir = tmp_path / "logs"
    path = LocalYamlLogSink(log_dir).write_returning_path(_record())
    assert path is not None
    dir_mode = log_dir.stat().st_mode & 0o777
    assert dir_mode & 0o077 == 0, f"log dir is group/other-accessible: {oct(dir_mode)}"
    for created in (path, log_dir / INDEX_FILENAME):
        mode = created.stat().st_mode & 0o777
        assert mode & 0o077 == 0, f"{created.name} is group/other-readable: {oct(mode)}"
        assert mode & 0o600 == 0o600, f"{created.name} lost owner rw: {oct(mode)}"


@_POSIX_ONLY
def test_pre_existing_dir_permissions_are_left_alone(tmp_path: Path) -> None:
    """A directory the *user* created is never re-chmodded — pre-creating the
    dir is the documented escape hatch for multi-reader deployments."""
    log_dir = tmp_path / "shared-logs"
    log_dir.mkdir()
    os.chmod(log_dir, 0o755)
    path = LocalYamlLogSink(log_dir).write_returning_path(_record())
    assert path is not None
    assert log_dir.stat().st_mode & 0o777 == 0o755


# 4. Warn-once latch.


def test_repeated_write_failure_warns_once_then_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A persistently failing write emits one WARNING; repeats of the same
    failure signature drop to DEBUG (with the traceback preserved there)."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("llmkit.logging.local_yaml.yaml.dump", _boom)
    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="llmkit.logging"):
        for _ in range(3):
            assert sink.write_returning_path(_record()) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(warnings) == 1
    assert "Failed to write LLM invocation log" in warnings[0].message
    assert len(debugs) == 2
    assert all(r.exc_info for r in debugs)


def test_recovery_rearms_the_latch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A success between failures re-arms the latch: the next failure warns
    loudly again instead of hiding at DEBUG."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("transient")

    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("llmkit.logging.local_yaml.yaml.dump", _boom)
            assert sink.write_returning_path(_record()) is None
        assert sink.write_returning_path(_record()) is not None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("llmkit.logging.local_yaml.yaml.dump", _boom)
            assert sink.write_returning_path(_record()) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_new_failure_signature_warns_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A *different* failure (new exception type) breaks through the latch —
    only identical repeats are demoted."""
    failures: list[BaseException] = [RuntimeError("first kind"), OSError(28, "No space left")]

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise failures[0]

    monkeypatch.setattr("llmkit.logging.local_yaml.yaml.dump", _boom)
    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is None
        _ = failures.pop(0)
        assert sink.write_returning_path(_record()) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_wrapper_latch_covers_custom_sinks_and_resets_on_configure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising third-party sink warns once (then DEBUG), and
    ``configure_llm_logging`` re-arms the latch for the newly installed sink."""

    class _BoomSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            raise RuntimeError("sink exploded")

    configure_llm_logging(_BoomSink())
    try:
        with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
            assert write_llm_log(_record()) is None
            assert write_llm_log(_record()) is None
            configure_llm_logging(_BoomSink())
            assert write_llm_log(_record()) is None
    finally:
        configure_llm_logging(LocalYamlLogSink())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("LLM log sink raised" in r.message for r in warnings)


def test_get_log_sink_reads_back_exactly_what_was_configured() -> None:
    """The read half of the configuration pair, asserted as an identity — a
    getter that returned an equivalent-but-different sink would be useless for
    the save/restore idiom it exists to serve, and ``None`` (logging off) has
    to survive the round trip as a value rather than degrade to a default."""

    class _MinimalSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            ...

    installed = _MinimalSink()
    try:
        configure_llm_logging(installed)
        assert get_log_sink() is installed
        configure_llm_logging(None)
        assert get_log_sink() is None
        file_sink = LocalYamlLogSink()
        configure_llm_logging(file_sink)
        assert get_log_sink() is file_sink
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_get_log_sink_restores_the_previous_sink_across_a_temporary_swap() -> None:
    """The documented save/restore idiom, end to end. This is the failure the
    accessor exists to prevent: a host reading a *private* module attribute
    (``llmkit.logging._sink`` before this package was split out) silently got
    ``None`` after the move and restored logging to *off* — no error anywhere.
    Asserted through the public pair only."""

    class _CapturingSink:
        def __init__(self) -> None:
            self.records: list[LLMCallRecord] = []

        def write(self, record: LLMCallRecord) -> None:
            self.records.append(record)

    outer = LocalYamlLogSink()
    inner = _CapturingSink()
    try:
        configure_llm_logging(outer)

        previous = get_log_sink()
        configure_llm_logging(inner)
        try:
            _ = write_llm_log(_record())
        finally:
            configure_llm_logging(previous)

        assert len(inner.records) == 1
        assert get_log_sink() is outer
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_configure_llm_logging_rejects_a_non_sink() -> None:
    """A configuration mistake that is knowable at configure time fails there,
    instead of turning every later call's log into a latched warning far from
    the mistake. A minimal object that *does* implement ``write`` still gets
    in, so the check can't be satisfied by rejecting everything."""

    class _NotASink:
        pass

    class _MinimalSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            ...

    try:
        with pytest.raises(TypeError, match="LogSink"):
            configure_llm_logging(_NotASink())  # pyright: ignore[reportArgumentType]  # the mistake under test
        configure_llm_logging(_MinimalSink())
        assert write_llm_log(_record()) is None  # accepted; just has no path to give
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_configure_llm_logging_rejects_the_class_instead_of_an_instance() -> None:
    """Passing the sink *class* — the missing-parens slip — would satisfy a
    purely structural check (a class carries ``write`` as an attribute) and
    then fail every write on the absent ``self``. That is the deferred failure
    the guard exists to prevent, so it is rejected by name."""
    try:
        with pytest.raises(TypeError, match="not the class itself"):
            configure_llm_logging(LocalYamlLogSink)  # pyright: ignore[reportArgumentType]  # the mistake under test
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_non_path_return_from_custom_sink_is_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A third-party sink that breaks ``write_returning_path``'s ``Path | None``
    contract is treated as a failed write — one latched warning, no path — so
    nothing but a real path reaches ``capture_llm_log_paths``' ``list[Path]``."""

    class _StringReturningSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            ...

        def write_returning_path(self, record: LLMCallRecord) -> Path | None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            return "not-a-path.yaml"  # pyright: ignore[reportReturnType]  # the contract breach under test

    configure_llm_logging(_StringReturningSink())
    try:
        with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
            with capture_llm_log_paths() as captured:
                assert record_call(_record()) is None
                # A permanently broken sink must not warn per call: the breach
                # goes through the same warn-once latch as a raising sink,
                # which only holds while the failure path skips its
                # ``succeeded()`` re-arm.
                assert record_call(_record()) is None
        assert captured == []
    finally:
        configure_llm_logging(LocalYamlLogSink())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "LLM log sink raised" in warnings[0].message


# 5. Retention.


def test_retention_days_zero_is_rejected(tmp_path: Path) -> None:
    """``retention_days=0`` is a plausible spelling of "no age limit" whose
    prune cutoff is *now* — it would delete every log the moment it is written,
    including the file whose path the write hands back. It raises instead, and
    the message names the real keep-forever opt-out."""
    with pytest.raises(ValueError, match="retention_days") as excinfo:
        _ = LocalYamlLogSink(tmp_path, retention_days=0)
    assert "None" in str(excinfo.value)


def test_retention_days_negative_is_rejected(tmp_path: Path) -> None:
    """A negative retention is the same defect with no ambiguity at all."""
    with pytest.raises(ValueError, match="retention_days") as excinfo:
        _ = LocalYamlLogSink(tmp_path, retention_days=-5)
    assert "None" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, -1])
def test_max_index_bytes_zero_and_negative_are_rejected(tmp_path: Path, value: int) -> None:
    """A non-positive ``max_index_bytes`` rotates on every housekeeping pass,
    so the ``index.jsonl`` the docs tell operators to scan never exists."""
    with pytest.raises(ValueError, match="max_index_bytes") as excinfo:
        _ = LocalYamlLogSink(tmp_path, max_index_bytes=value)
    assert "None" in str(excinfo.value)


def test_smallest_accepted_bounds_still_construct(tmp_path: Path) -> None:
    """The rule is "positive", not "sensible": no arbitrary floor is invented,
    so a later minimum has to argue against this test."""
    sink = LocalYamlLogSink(tmp_path, retention_days=1, max_index_bytes=1)
    assert sink.write_returning_path(_record()) is not None


def test_none_opt_outs_survive_validation(tmp_path: Path) -> None:
    """``None`` is the opt-out for both bounds and is not collateral damage of
    the non-positive check."""
    sink = LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=None)
    assert sink.retention_days is None
    assert sink.max_index_bytes is None
    assert sink.write_returning_path(_record()) is not None


@pytest.mark.parametrize("retention_days", [1, 30, None])
def test_returned_path_always_exists_for_every_accepted_retention(
    tmp_path: Path, retention_days: int | None
) -> None:
    """The invariant the defect violated: a path handed to a caller names a
    file that is actually there. Guaranteed once the smallest accepted
    retention is a full day, which no just-written file can be older than."""
    sink = LocalYamlLogSink(tmp_path / f"r-{retention_days}", retention_days=retention_days)
    path = sink.write_returning_path(_record())
    assert path is not None
    assert path.exists()


def test_aged_yaml_files_are_pruned_on_write(tmp_path: Path) -> None:
    """Files older than ``retention_days`` are pruned by the write path;
    recent files survive."""
    stale = tmp_path / "2026-06-01T00-00-00-000000_old_call_deadbeef.yaml"
    _ = stale.write_text("# old\n")
    _age(stale, days=40)
    fresh = tmp_path / "2026-07-19T00-00-00-000000_new_call_cafebabe.yaml"
    _ = fresh.write_text("# new\n")
    path = LocalYamlLogSink(tmp_path, retention_days=30).write_returning_path(_record())
    assert path is not None
    assert not stale.exists()
    assert fresh.exists()
    assert path.exists()


def test_retention_none_keeps_everything(tmp_path: Path) -> None:
    """``retention_days=None`` is the documented keep-forever opt-out."""
    stale = tmp_path / "2020-01-01T00-00-00-000000_ancient_call_deadbeef.yaml"
    _ = stale.write_text("# ancient\n")
    _age(stale, days=2000)
    sink = LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=None)
    assert sink.write_returning_path(_record()) is not None
    assert stale.exists()


def test_stale_part_files_are_swept_and_fresh_ones_kept(tmp_path: Path) -> None:
    """A ``.part`` file a cut-short write left behind ages out under the same
    cutoff; a fresh one may be a write in progress and is kept."""
    stale = tmp_path / ".2026-06-01T00-00-00-000000_old_call_deadbeef.yaml.part"
    _ = stale.write_text("# ok | cut\n")
    _age(stale, days=40)
    fresh = tmp_path / ".2026-07-19T00-00-00-000000_new_call_cafebabe.yaml.part"
    _ = fresh.write_text("# ok | in progress\n")
    assert LocalYamlLogSink(tmp_path, retention_days=30).write_returning_path(_record())
    assert not stale.exists()
    assert fresh.exists()


def test_retention_none_keeps_part_files(tmp_path: Path) -> None:
    """With retention opted out, nothing is aged out — ``.part`` files included."""
    stale = tmp_path / ".2020-01-01T00-00-00-000000_ancient_call_deadbeef.yaml.part"
    _ = stale.write_text("# ok | cut\n")
    _age(stale, days=2000)
    sink = LocalYamlLogSink(tmp_path, retention_days=None)
    assert sink.write_returning_path(_record()) is not None
    assert stale.exists()


def test_prune_is_throttled_to_hourly(tmp_path: Path) -> None:
    """After one prune, the next writes skip housekeeping until the interval
    elapses — the steady-state write cost is one clock read."""
    sink = LocalYamlLogSink(tmp_path, retention_days=30)
    assert sink.write_returning_path(_record()) is not None  # first write prunes
    stale = tmp_path / "2026-06-01T00-00-00-000000_old_call_deadbeef.yaml"
    _ = stale.write_text("# old\n")
    _age(stale, days=40)
    assert sink.write_returning_path(_record()) is not None
    assert stale.exists(), "pruned again within the throttle window"
    _unthrottle(sink)
    assert sink.write_returning_path(_record()) is not None
    assert not stale.exists()


def test_oversized_index_rotates_without_losing_lines(tmp_path: Path) -> None:
    """An index past ``max_index_bytes`` is renamed to a date-stamped sibling;
    every line remains readable across the generations."""
    sink = LocalYamlLogSink(tmp_path, max_index_bytes=10)
    assert sink.write_returning_path(_record(label="one")) is not None
    # The first write's prune saw an index over the 10-byte cap and rotated it.
    assert not (tmp_path / INDEX_FILENAME).exists()
    _unthrottle(sink)
    assert sink.write_returning_path(_record(label="two")) is not None
    all_lines: list[str] = []
    for index_file in tmp_path.glob("index*.jsonl"):
        all_lines.extend(index_file.read_text().strip().splitlines())
    assert len(all_lines) == 2
    assert any('"one"' in line for line in all_lines)
    assert any('"two"' in line for line in all_lines)


def test_rotated_index_generations_age_out(tmp_path: Path) -> None:
    """Rotated ``index-*.jsonl`` files fall under the same age policy as the
    per-call YAMLs — rotation bounds the active file, retention bounds the rest."""
    rotated = tmp_path / "index-2026-06-01T00-00-00-000000.jsonl"
    _ = rotated.write_text('{"file": "old.yaml"}\n')
    _age(rotated, days=40)
    assert LocalYamlLogSink(tmp_path, retention_days=30).write_returning_path(_record()) is not None
    assert not rotated.exists()


def test_prune_leaves_files_llmkit_did_not_name(tmp_path: Path) -> None:
    """Retention deletes only the names llmkit generates: a co-located
    project file of a ``*.yaml`` or ``index-*.jsonl`` shape outlives the
    cutoff, while llmkit's own expired record, rotated index and ``.part``
    file in the same directory are still removed."""
    foreign = [
        "docker-compose.yaml",
        "mkdocs.yaml",
        "index-notes.jsonl",
        "trace_2026-06-01_lookup.yaml",
        ".settings.yaml.part",
        # Pre-v0.2.0 records carried no uuid suffix; they are not matched.
        "2026-06-01T00-00-00-000000_old_call.yaml",
        INDEX_FILENAME,
    ]
    own = [
        "2026-06-01T00-00-00-000000_old_call_deadbeef.yaml",
        "index-2026-06-01T00-00-00-000000.jsonl",
        ".2026-06-01T00-00-00-000000_old_call_deadbeef.yaml.part",
    ]
    for name in (*foreign, *own):
        path = tmp_path / name
        _ = path.write_text("# kept or not\n")
        _age(path, days=40)
    sink = LocalYamlLogSink(tmp_path, retention_days=30, max_index_bytes=None)
    # Housekeeping alone, not a write: a write appends to ``index.jsonl``
    # first and would refresh the aged active index's mtime before the prune.
    assert sink._maybe_prune() is None  # private: the housekeeping verdict under test
    assert [name for name in foreign if not (tmp_path / name).exists()] == []
    assert [name for name in own if (tmp_path / name).exists()] == []


@pytest.mark.parametrize(
    "name",
    [
        "2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml",
        "2026-06-01T00-00-00-000000_my_feature_with_underscores_unlabeled_0123abcd.yaml",
        "2026-06-01T00-00-00-000000_résumé_étape_0123abcd.yaml",
        ".2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml.part",
        "index-2026-06-01T00-00-00-000000.jsonl",
    ],
)
def test_generated_shapes_are_prunable(name: str) -> None:
    assert is_prunable_name(name)


@pytest.mark.parametrize(
    "name",
    [
        INDEX_FILENAME,
        ".gitignore",
        "docker-compose.yaml",
        "index-notes.jsonl",
        "index-2026-06-01.jsonl",
        "2026-06-01T00-00-00-000000_old_call.yaml",
        "2026-06-01T00-00-00-000000_extraction_summary_0123ABCD.yaml",
        "2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml.bak",
        "notes-2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml",
        "2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml.part",
        ".2026-06-01T00-00-00-000000_extraction_summary_0123abcd.yaml",
    ],
)
def test_other_shapes_are_not_prunable(name: str) -> None:
    assert not is_prunable_name(name)


def test_every_name_the_sink_generates_is_prunable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write path, the ``.part`` file and the rotation all produce names
    the prune patterns recognise, so a change to the generated shape cannot
    silently strand files that retention would no longer delete."""
    sink = LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=10)
    record_path = sink.write_returning_path(
        _record(feature="my_feature/v2", label="a label with spaces")
    )
    assert record_path is not None
    [rotated] = list(tmp_path.glob("index-*.jsonl"))  # the first write's prune rotated it
    real_dump = yaml.dump

    def _cut_short(_data: object, stream: object, **kwargs: object) -> None:
        real_dump({"timestamp": "half"}, stream, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]  # forwarding the sink's own kwargs
        raise _Interrupted

    monkeypatch.setattr("llmkit.logging.local_yaml.yaml.dump", _cut_short)
    with pytest.raises(_Interrupted):
        _ = sink.write_returning_path(_record())
    [part] = list(tmp_path.glob(".*.part"))
    generated = {record_path.name, rotated.name, part.name}
    assert {name for name in generated if not is_prunable_name(name)} == set()


def test_rotation_already_done_by_another_process_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two processes sharing a directory can both see an oversized index;
    when the other one renames it first, this rotation's rename finds no
    source, and that is success — the bound is already enforced."""
    _ = (tmp_path / INDEX_FILENAME).write_text('{"file": "a.yaml"}\n' * 4)
    real_replace = os.replace

    def _other_process_wins(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src).name == INDEX_FILENAME:
            # The other process renames the index first ...
            real_replace(src, tmp_path / "index-2026-06-01T00-00-00-000001.jsonl")
        real_replace(src, dst)  # ... so this rename genuinely finds nothing.

    monkeypatch.setattr("llmkit.logging.local_yaml.os.replace", _other_process_wins)
    sink = LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=10)
    with caplog.at_level(logging.DEBUG, logger="llmkit"):
        assert sink._maybe_prune() is None  # private: the housekeeping verdict under test
    assert [r for r in caplog.records if "prune" in r.getMessage()] == []
    assert not (tmp_path / INDEX_FILENAME).exists()
    assert len(list(tmp_path.glob("index-*.jsonl"))) == 1


def test_other_rotation_errors_still_count_as_housekeeping_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a vanished source is treated as another process's rotation; any
    other failure to rename keeps surfacing."""
    _ = (tmp_path / INDEX_FILENAME).write_text('{"file": "a.yaml"}\n' * 4)

    def _denied(_src: object, _dst: object) -> None:
        raise PermissionError("rename denied")

    monkeypatch.setattr("llmkit.logging.local_yaml.os.replace", _denied)
    sink = LocalYamlLogSink(tmp_path, retention_days=None, max_index_bytes=10)
    assert isinstance(
        sink._maybe_prune(), PermissionError
    )  # private: the housekeeping verdict under test


def test_max_index_bytes_none_never_rotates(tmp_path: Path) -> None:
    """``max_index_bytes=None`` disables rotation entirely."""
    sink = LocalYamlLogSink(tmp_path, max_index_bytes=None)
    for label in ("one", "two", "three"):
        _unthrottle(sink)
        assert sink.write_returning_path(_record(label=label)) is not None
    assert (tmp_path / INDEX_FILENAME).exists()
    assert list(tmp_path.glob("index-*.jsonl")) == []


# 6. First-write announcement.


def test_first_successful_write_announces_location_and_policy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The first successful write emits exactly one INFO naming the absolute
    directory and the retention policy — before anything could be pruned."""
    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
        assert sink.write_returning_path(_record()) is not None
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert str(tmp_path) in infos[0].message
    assert "30-day retention" in infos[0].message


def test_announcement_names_the_frozen_path_not_the_symlink_target(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The notice names the directory as the sink froze it (absolute, symlinks
    intact), the same path ``default_log_dir()`` and ``log_dir`` report."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert LocalYamlLogSink(link / "logs").write_returning_path(_record()) is not None
    [info] = [r for r in caplog.records if r.levelno == logging.INFO]
    assert str(link / "logs") in info.message
    assert str(real) not in info.message


def test_announcement_states_keep_forever_when_retention_off(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With retention opted out, the announcement says so."""
    sink = LocalYamlLogSink(tmp_path, retention_days=None)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "kept forever" in infos[0].message


def test_failed_write_does_not_announce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only a *successful* write announces — a broken sink stays quiet at INFO."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("nope")

    monkeypatch.setattr("llmkit.logging.local_yaml.yaml.dump", _boom)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert LocalYamlLogSink(tmp_path).write_returning_path(_record()) is None
    assert [r for r in caplog.records if r.levelno == logging.INFO] == []


# 7. Whole files only.


class _Interrupted(BaseException):
    """Stands in for SIGKILL / KeyboardInterrupt / exit cutting a write short."""


def test_interrupted_write_leaves_no_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write cut off mid-body (a ``BaseException`` skips the cleanup) leaves
    only a hidden ``.part`` file, never a truncated ``*.yaml`` that
    ``head -1 *.yaml`` would read as ``# ok``."""
    real_dump = yaml.dump

    def _cut_short(_data: object, stream: object, **kwargs: object) -> None:
        real_dump({"timestamp": "half"}, stream, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]  # forwarding the sink's own kwargs
        raise _Interrupted

    monkeypatch.setattr("llmkit.logging.local_yaml.yaml.dump", _cut_short)
    with pytest.raises(_Interrupted):
        _ = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert list(tmp_path.glob("*.yaml")) == []
    [part] = list(tmp_path.glob(".*.yaml.part"))
    assert part.read_text().startswith("# ok |")


def test_published_record_leaves_no_part_file(tmp_path: Path) -> None:
    """A completed write publishes the ``*.yaml`` and removes its ``.part``."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    assert yaml.safe_load(path.read_text())["feature"] == "extraction"
    assert {p.name for p in tmp_path.iterdir()} == {INDEX_FILENAME, path.name}


def test_publish_without_hard_links_still_writes_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a filesystem without hard links the record is still published under
    its own exclusively reserved name, with no ``.part`` left behind."""

    def _no_links(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "hard links unsupported")

    monkeypatch.setattr("llmkit.logging.local_yaml.os.link", _no_links)
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    assert yaml.safe_load(path.read_text())["feature"] == "extraction"
    assert {p.name for p in tmp_path.iterdir()} == {INDEX_FILENAME, path.name}
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_publish_never_clobbers_an_existing_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the final name is taken between the ``.part`` create and the
    publish, the existing file is untouched and the record lands elsewhere."""
    real_link = os.link
    squatted: list[Path] = []

    def _squat_then_link(src: str, dst: str) -> None:
        if not squatted:
            squatted.append(Path(dst))
            _ = Path(dst).write_text("someone else's\n")
        real_link(src, dst)

    monkeypatch.setattr("llmkit.logging.local_yaml.os.link", _squat_then_link)
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    assert path != squatted[0]
    assert squatted[0].read_text() == "someone else's\n"
    assert list(tmp_path.glob(".*.part")) == []


def _no_links(*_args: object, **_kwargs: object) -> None:
    raise PermissionError(1, "hard links unsupported")


def test_publish_without_hard_links_shows_no_yaml_until_it_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without hard links nothing appears under a ``*.yaml`` name before the
    body is moved there, so a process killed mid-publish leaves no empty
    record; a name already taken is left untouched."""
    real_replace = os.replace
    seen: list[list[Path]] = []

    def _observe_then_replace(src: str, dst: str) -> None:
        seen.append(sorted(tmp_path.glob("*.yaml")))
        real_replace(src, dst)

    monkeypatch.setattr("llmkit.logging.local_yaml.os.link", _no_links)
    monkeypatch.setattr("llmkit.logging.local_yaml.os.replace", _observe_then_replace)
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    assert seen == [[]]
    assert yaml.safe_load(path.read_text())["feature"] == "extraction"

    taken = tmp_path / "taken.yaml"
    _ = taken.write_text("someone else's\n")
    part = tmp_path / ".taken.yaml.part"
    _ = part.write_text("ours\n")
    assert not local_yaml._publish(part, taken)
    assert taken.read_text() == "someone else's\n"
    assert part.read_text() == "ours\n"


@pytest.mark.parametrize("field", ["error", "label", "feature", "prompt"])
def test_lone_surrogate_keeps_record_and_index_line(tmp_path: Path, field: str) -> None:
    """A lone surrogate (``surrogateescape`` text from a filename or OSError)
    in any record field loses neither the YAML nor the index line."""
    value = "bad \udcff byte"
    override: object = [{"role": "user", "content": value}] if field == "prompt" else value
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(**{field: override}))
    assert path is not None
    body = cast("dict[str, object]", yaml.safe_load(path.read_text(encoding="utf-8")))
    assert body[field] == override
    [line] = (tmp_path / INDEX_FILENAME).read_text(encoding="utf-8").splitlines()
    entry = cast("dict[str, object]", json.loads(line))
    assert entry["file"] == path.name
    if field != "prompt":
        assert entry[field] == value
