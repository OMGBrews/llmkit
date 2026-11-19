"""The logging environment switches: ``LLMKIT_LOG`` and ``LLMKIT_LOG_RETENTION_DAYS``.

``LLMKIT_LOG`` turns the import-time default sink off without a code change.
It is read once, at the first registry read or write, so the end-to-end cases
run in a fresh interpreter where the import-time default is still implicit;
the in-process cases put the registry back into that state with
``monkeypatch``. ``LLMKIT_LOG_RETENTION_DAYS`` sets the retention of every
``LocalYamlLogSink`` built without ``retention_days``, read at its first write.
Explicit configuration wins over both. The logging ``conftest`` clears both
variables around every test.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import cast

import pytest

import llmkit.logging.registry as registry
from llmkit._latch import OnceLatch
from llmkit.logging import (
    LOG_DIR_ENV_VAR,
    LOG_ENV_VAR,
    LOG_RETENTION_ENV_VAR,
    LocalYamlLogSink,
    LogSink,
    configure_llm_logging,
    get_log_sink,
    get_logging_health,
    write_llm_log,
)
from llmkit.logging._paths import blank_is_unset, resolve_default_log_dir
from llmkit.logging.local_yaml import implicit_default_sink
from tests._support import make_record as _record

_CHILD = r"""
import json
from llmkit import (
    LLMClientConfig, Provider, configure_llm_client, configure_llm_logging,
    flush_llm_logs, get_logging_health, text_llm_call_sync,
)
from llmkit.capture import submit_record_call
from llmkit.logging import LocalYamlLogSink, write_llm_log
from llmkit.testing import Reply, install
from pathlib import Path
import os, sys

if sys.argv[1] == "explicit":
    configure_llm_logging(LocalYamlLogSink(Path(os.environ["LLMKIT_LOG_DIR"])))
configure_llm_client(lambda: LLMClientConfig(provider=Provider.OLLAMA, model="fake-model"))
with install(lambda request: Reply.text("still answered")):
    answer = text_llm_call_sync("say hi", feature="switch", label="call")
assert answer == "still answered", answer
from datetime import UTC, datetime
from llmkit import LLMCallRecord
record = LLMCallRecord(
    started_at=datetime.now(UTC), feature="switch", label="direct", model="m",
    provider="p", temperature=None, duration_ms=1.0, schema="text", prompt="hi",
    response=None, error=None,
)
write_llm_log(record)
submit_record_call(record)
assert flush_llm_logs(timeout=5)
health = get_logging_health()
print(json.dumps({
    "sink": health.sink, "written": health.written, "failed": health.failed,
    "dropped": health.dropped, "degraded": health.degraded,
}))
"""


def _run_child(
    tmp_path: Path, log_value: str | None, mode: str = "implicit"
) -> tuple[dict[str, object], Path]:
    """Run a fresh interpreter that makes a call and two direct writes, then flushes."""
    log_dir = tmp_path / "logs"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {LOG_ENV_VAR, LOG_RETENTION_ENV_VAR}
    }
    env[LOG_DIR_ENV_VAR] = str(log_dir)
    if log_value is not None:
        env[LOG_ENV_VAR] = log_value
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, mode],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = cast("dict[str, object]", json.loads(result.stdout.strip().splitlines()[-1]))
    return summary, log_dir


def test_log_off_disables_the_default_sink_in_a_fresh_process(tmp_path: Path) -> None:
    """A call, a direct write and a managed submission leave no directory after
    a flush; health reports the sink disabled with nothing failed or dropped."""
    summary, log_dir = _run_child(tmp_path, " Off ")
    assert summary == {"sink": "disabled", "written": 0, "failed": 0, "dropped": 0, "degraded": 0}
    assert not log_dir.exists()


@pytest.mark.parametrize("value", [None, "  "])
def test_log_unset_or_blank_keeps_the_default_sink_in_a_fresh_process(
    tmp_path: Path, value: str | None
) -> None:
    """With no value or a whitespace-only one, the same three records reach disk."""
    summary, log_dir = _run_child(tmp_path, value)
    assert summary == {"sink": "builtin", "written": 3, "failed": 0, "dropped": 0, "degraded": 0}
    assert len(list(log_dir.glob("*.yaml"))) == 3


def test_explicit_sink_beats_log_off_in_a_fresh_process(tmp_path: Path) -> None:
    """``configure_llm_logging`` before the first read wins over ``LLMKIT_LOG=off``."""
    summary, log_dir = _run_child(tmp_path, "off", mode="explicit")
    assert summary["sink"] == "builtin"
    assert summary["written"] == 3
    assert len(list(log_dir.glob("*.yaml"))) == 3


@pytest.fixture
def implicit_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalYamlLogSink:
    """Put the registry back in its import-time state, with the default sink in tmp_path."""
    sink = implicit_default_sink(tmp_path / "default")
    monkeypatch.setattr(registry, "_installed", registry._install(sink))
    monkeypatch.setattr(registry, "_implicit", True)
    return sink


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_log_off_is_read_once_and_frozen(
    implicit_default: LocalYamlLogSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first read decides; changing the variable afterwards changes nothing."""
    monkeypatch.setenv(LOG_ENV_VAR, "on")
    assert get_log_sink() is implicit_default
    monkeypatch.setenv(LOG_ENV_VAR, "off")
    assert get_log_sink() is implicit_default
    assert write_llm_log(_record()) is not None
    assert get_logging_health().written == 1


def test_explicit_configuration_beats_log_off(
    implicit_default: LocalYamlLogSink, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit sink installed while the default is still implicit is never
    replaced by the environment, and neither is the explicit restore of the
    default sink object itself."""
    monkeypatch.setenv(LOG_ENV_VAR, "off")
    explicit = LocalYamlLogSink(tmp_path / "explicit")
    configure_llm_logging(explicit)
    assert get_log_sink() is explicit
    assert write_llm_log(_record()) is not None
    assert get_logging_health().written == 1

    monkeypatch.setattr(registry, "_implicit", True)
    configure_llm_logging(implicit_default)
    assert get_log_sink() is implicit_default


@pytest.mark.usefixtures("implicit_default")
@pytest.mark.parametrize("value", ["off", " OFF ", "0", "False", "no\n"])
def test_every_off_spelling_disables(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Off spellings are stripped and case-folded."""
    monkeypatch.setenv(LOG_ENV_VAR, value)
    assert get_log_sink() is None
    assert get_logging_health().sink == "disabled"


@pytest.mark.parametrize("value", ["on", " ON ", "1", "True", "yes"])
def test_every_on_spelling_keeps_the_default(
    implicit_default: LocalYamlLogSink, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(LOG_ENV_VAR, value)
    assert get_log_sink() is implicit_default


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_log_value_is_unset(
    implicit_default: LocalYamlLogSink, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(LOG_ENV_VAR, value)
    assert get_log_sink() is implicit_default
    assert get_logging_health().sink == "builtin"


@pytest.mark.parametrize("value", ["maybe", "of", "disable", "2"])
def test_invalid_log_value_fails_closed_with_one_warning(
    implicit_default: LocalYamlLogSink,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    """An unrecognised value disables logging rather than writing prompts the
    user tried to stop, warns exactly once, and never raises."""
    monkeypatch.setenv(LOG_ENV_VAR, value)
    with caplog.at_level(logging.DEBUG, logger="llmkit.logging"):
        assert write_llm_log(_record()) is None
        assert get_log_sink() is None
        health = get_logging_health()
    assert (health.sink, health.failed, health.dropped) == ("disabled", 0, 0)
    assert not implicit_default.log_dir.exists()
    [warning] = _warnings(caplog)
    assert LOG_ENV_VAR in warning.getMessage()
    assert repr(value) in warning.getMessage()


@pytest.mark.usefixtures("implicit_default")
def test_a_concurrent_first_read_never_sees_the_default_sink_under_log_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second thread that reads the registry while the first is still
    applying ``LLMKIT_LOG=off`` gets the disabled installation, not the default
    sink the switch is replacing (which would write its prompt to disk)."""
    monkeypatch.setenv(LOG_ENV_VAR, "off")
    concurrent: list[LogSink | None] = []

    def _read() -> None:
        concurrent.append(registry.get_installed().sink)

    reader = threading.Thread(target=_read)

    def _slow_blank_is_unset(value: str | None) -> str | None:
        # The switch is mid-read: let the second reader run now. With the
        # flag cleared before the swap it returns at once with the default
        # sink; done right it blocks on the switch lock until this returns.
        reader.start()
        reader.join(timeout=0.5)
        return blank_is_unset(value)

    monkeypatch.setattr("llmkit.logging.registry.blank_is_unset", _slow_blank_is_unset)
    assert registry.get_installed().sink is None
    reader.join(timeout=5)
    assert not reader.is_alive()
    assert concurrent == [None]


def _backdated(log_dir: Path, name: str, days: float) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    _ = path.write_text("old: true\n", encoding="utf-8")
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_retention_is_read_at_first_write_not_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sink built before the variable is set still honours it, and the first
    notice says the bound came from the environment."""
    sink = LocalYamlLogSink(tmp_path)
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, " 7 ")
    older = _backdated(tmp_path, "older.yaml", 8)
    newer = _backdated(tmp_path, "newer.yaml", 6)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
    assert sink.retention_days == 7
    assert not older.exists()
    assert newer.exists()
    [info] = [r for r in caplog.records if r.levelno == logging.INFO]
    assert f"7-day retention from {LOG_RETENTION_ENV_VAR}" in info.getMessage()
    assert f"so {LOG_ENV_VAR}=off does not disable it" in info.getMessage()


def test_only_the_default_sinks_notice_names_the_off_switch(
    implicit_default: LocalYamlLogSink,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``LLMKIT_LOG=off`` disables only the import-time default, so only its
    first-write notice offers it; a sink set up in code says the variable does
    not disable it, which stays true while the variable is set to off."""
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert write_llm_log(_record()) is not None
        monkeypatch.setenv(LOG_ENV_VAR, "off")
        explicit = LocalYamlLogSink(tmp_path / "explicit")
        configure_llm_logging(explicit)
        assert write_llm_log(_record()) is not None
    default_notice, explicit_notice = [
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    ]
    assert str(implicit_default.log_dir) in default_notice
    assert f"{LOG_ENV_VAR}=off or configure_llm_logging(None) disables" in default_notice
    assert str(explicit.log_dir) in explicit_notice
    assert f"set up in code, so {LOG_ENV_VAR}=off does not disable it" in explicit_notice
    assert f"{LOG_ENV_VAR}=off or" not in explicit_notice


def test_retention_is_frozen_after_the_first_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = LocalYamlLogSink(tmp_path)
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, "7")
    assert sink.retention_days == 7
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, "1")
    assert sink.retention_days == 7


@pytest.mark.parametrize("value", ["forever", " FOREVER "])
def test_forever_keeps_a_back_dated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, value)
    ancient = _backdated(tmp_path, "ancient.yaml", 400)
    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
    assert sink.retention_days is None
    assert ancient.exists()
    [info] = [r for r in caplog.records if r.levelno == logging.INFO]
    assert f"kept forever) from {LOG_RETENTION_ENV_VAR}" in info.getMessage()


@pytest.mark.parametrize("value", ["", "  "])
def test_blank_retention_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    """A blank value falls back to the 30-day default, and the notice names the
    variable that would change it."""
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, value)
    stale = _backdated(tmp_path, "stale.yaml", 31)
    sink = LocalYamlLogSink(tmp_path)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
    assert sink.retention_days == 30
    assert not stale.exists()
    [info] = [r for r in caplog.records if r.levelno == logging.INFO]
    assert f"30-day retention, {LOG_RETENTION_ENV_VAR} changes it" in info.getMessage()


@pytest.mark.parametrize("value", ["0", "-3", "1.5", "week", "+5", "1_0"])
def test_invalid_retention_keeps_files_warns_once_and_degrades_housekeeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    """An unusable value never raises: age pruning is off for the sink (files
    are kept), one WARNING covers every sink that reads it, and the first
    successful write of each sink counts one degraded housekeeping outcome."""
    monkeypatch.setattr("llmkit.logging.local_yaml._retention_env_latch", OnceLatch())
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, value)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    kept = _backdated(first_dir, "ancient.yaml", 400)
    first = LocalYamlLogSink(first_dir)
    with caplog.at_level(logging.DEBUG, logger="llmkit.logging"):
        configure_llm_logging(first)
        assert write_llm_log(_record()) is not None
        assert write_llm_log(_record()) is not None
        health = get_logging_health()
        assert LocalYamlLogSink(second_dir).write_returning_path(_record()) is not None
    assert first.retention_days is None
    assert kept.exists()
    assert (health.written, health.degraded, health.failed) == (2, 1, 0)
    assert health.last_failure is not None
    assert health.last_failure.stage == "housekeeping"
    assert LOG_RETENTION_ENV_VAR in health.last_failure.error
    [warning] = _warnings(caplog)
    assert LOG_RETENTION_ENV_VAR in warning.getMessage()
    assert repr(value) in warning.getMessage()
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(f"{LOG_RETENTION_ENV_VAR} is invalid" in message for message in infos)


@pytest.mark.parametrize(("retention_days", "age"), [(None, 400), (30, 10)])
def test_explicit_retention_beats_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    retention_days: int | None,
    age: int,
) -> None:
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, "1")
    kept = _backdated(tmp_path, "kept.yaml", age)
    sink = LocalYamlLogSink(tmp_path, retention_days=retention_days)
    with caplog.at_level(logging.INFO, logger="llmkit.logging"):
        assert sink.write_returning_path(_record()) is not None
    assert sink.retention_days == retention_days
    assert kept.exists()
    [info] = [r for r in caplog.records if r.levelno == logging.INFO]
    assert f"set in code, so {LOG_RETENTION_ENV_VAR} is ignored" in info.getMessage()


def test_assigned_retention_beats_the_environment_and_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attribute stays assignable; an assignment before the first write
    replaces the environment's answer, and the constructor's bounds apply."""
    monkeypatch.setenv(LOG_RETENTION_ENV_VAR, "1")
    kept = _backdated(tmp_path, "kept.yaml", 10)
    sink = LocalYamlLogSink(tmp_path)
    sink.retention_days = None
    assert sink.write_returning_path(_record()) is not None
    assert kept.exists()
    with pytest.raises(ValueError, match="retention_days"):
        sink.retention_days = 0
    assert sink.retention_days is None


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_blank_log_dir_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A blank or whitespace-only ``LLMKIT_LOG_DIR`` falls through to the
    default location instead of naming a directory called ``"   "``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(LOG_DIR_ENV_VAR, raising=False)
    expected = resolve_default_log_dir()
    monkeypatch.setenv(LOG_DIR_ENV_VAR, value)
    assert resolve_default_log_dir() == expected
