"""Tests for the hardened LLM-log sink (:mod:`llmkit.logging`).

These pin the recent hardening of :class:`LocalYamlLogSink` /
:func:`write_llm_log`: the verdict-first YAML layout and ``index.jsonl``
summary, plus three robustness guards — filename collisions never
overwrite, ``feature``/``label`` can't escape ``log_dir`` via path
traversal, and a newline in a caller field can't forge a second verdict
header line — and the best-effort swallowing that keeps a logging failure
from ever breaking the LLM call.

Records are built directly via :func:`_record`; no real LLM call is made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
)
from llmkit.logging import write_llm_log


def _record(**overrides: object) -> LLMCallRecord:
    base: dict[str, object] = {
        "started_at": datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC),
        "feature": "extraction",
        "label": "summary",
        "model": "gemini-2.5-flash-lite",
        "provider": "Google AI Studio",
        "temperature": 0.0,
        "duration_ms": 12.3,
        "schema": "Schema",
        "prompt": "hi",
        "response": None,
        "error": None,
    }
    base.update(overrides)
    return LLMCallRecord(**base)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat


# 1. Verdict-first header + metadata-before-blobs body.


def test_header_is_verdict_first_for_success(tmp_path: Path) -> None:
    """The first line is the single-glance ``# ok | ...`` verdict with the
    feature/label, model, schema, duration, and cost."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(approximate_cost=0.0123))
    assert path is not None
    first_line = path.read_text().splitlines()[0]
    assert first_line.startswith("# ok | extraction/summary | gemini-2.5-flash-lite | Schema |")
    assert "ms" in first_line
    assert "$0.0123" in first_line


def test_header_leads_with_error_when_record_has_error(tmp_path: Path) -> None:
    """An errored record leads with ``ERROR``, never ``ok``; cost is ``$?``
    when unknown."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(error="APIError: boom"))
    assert path is not None
    first_line = path.read_text().splitlines()[0]
    assert first_line.startswith("# ERROR |")
    assert not first_line.startswith("# ok")
    assert "$?" in first_line


def test_metadata_keys_precede_blobs_in_body(tmp_path: Path) -> None:
    """The cheap metadata fields sort ahead of the large response/prompt
    blobs (response before prompt)."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(
        _record(prompt="PROMPT_TEXT", response="RESP_TEXT")
    )
    assert path is not None
    body = path.read_text()
    assert body.index("feature:") < body.index("response:")
    assert body.index("error:") < body.index("response:")
    assert body.index("approximate_cost:") < body.index("response:")
    assert body.index("response:") < body.index("prompt:")


# 2. index.jsonl append.


def test_index_jsonl_appends_one_documented_line_per_write(tmp_path: Path) -> None:
    """Each write appends exactly one JSON line carrying the documented
    fields; two writes -> two lines."""
    sink = LocalYamlLogSink(tmp_path)
    p1 = sink.write_returning_path(_record(label="first", approximate_cost=1e-06))
    p2 = sink.write_returning_path(_record(label="second", error="Timeout: slow"))
    assert p1 is not None and p2 is not None

    lines = (tmp_path / "index.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    first = cast("dict[str, object]", json.loads(lines[0]))
    expected_keys = {
        "file",
        "timestamp",
        "feature",
        "label",
        "model",
        "provider",
        "schema",
        "duration_ms",
        "approximate_cost",
        "error",
    }
    assert set(first.keys()) == expected_keys
    assert first["file"] == p1.name
    assert first["feature"] == "extraction"
    assert first["label"] == "first"
    assert first["model"] == "gemini-2.5-flash-lite"
    assert first["provider"] == "Google AI Studio"
    assert first["schema"] == "Schema"
    assert first["approximate_cost"] == 1e-06
    assert first["error"] is None
    # The big blobs are deliberately absent from the index.
    assert "prompt" not in first and "response" not in first

    second = cast("dict[str, object]", json.loads(lines[1]))
    assert second["file"] == p2.name
    assert second["label"] == "second"
    assert second["error"] == "Timeout: slow"


# 3. Filename collision — the headline guard.


def test_identical_records_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """Two records with identical started_at, feature, and label produce two
    distinct files that both exist with their own content — neither clobbers
    the other (would fail under the old ``{ts}_{feature}_{label}.yaml``)."""
    sink = LocalYamlLogSink(tmp_path)
    common = {
        "started_at": datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC),
        "feature": "extraction",
        "label": "summary",
    }
    p1 = sink.write_returning_path(_record(response="FIRST_RESPONSE", **common))
    p2 = sink.write_returning_path(_record(response="SECOND_RESPONSE", **common))
    assert p1 is not None and p2 is not None

    assert p1 != p2
    assert p1.exists() and p2.exists()
    yaml_files = sorted(tmp_path.glob("*.yaml"))
    assert len(yaml_files) == 2
    assert "FIRST_RESPONSE" in p1.read_text()
    assert "SECOND_RESPONSE" in p2.read_text()


# 4. feature/label path-traversal guard.


@pytest.mark.parametrize("hostile", ["a/b/../c", "../escape", "..", "/etc/passwd"])
def test_feature_path_traversal_stays_inside_log_dir(tmp_path: Path, hostile: str) -> None:
    """A feature with path separators / traversal segments still writes a
    file whose parent resolves to ``log_dir`` — nothing escapes."""
    log_dir = tmp_path / "logs"
    before = set(tmp_path.rglob("*"))
    path = LocalYamlLogSink(log_dir).write_returning_path(_record(feature=hostile))
    assert path is not None
    assert path.exists()
    # The written file's parent is log_dir itself — no climbing out.
    assert path.resolve().parent == log_dir.resolve()
    # No file landed outside log_dir (e.g. a sibling of tmp_path or in tmp_path).
    created_outside = {
        p for p in set(tmp_path.rglob("*")) - before if p.is_file() and log_dir not in p.parents
    }
    assert created_outside == set()


def test_label_path_traversal_stays_inside_log_dir(tmp_path: Path) -> None:
    """The same containment holds for a hostile ``label``."""
    log_dir = tmp_path / "logs"
    path = LocalYamlLogSink(log_dir).write_returning_path(_record(label="../../escape"))
    assert path is not None
    assert path.resolve().parent == log_dir.resolve()


# 5. Header injection guard.


def test_feature_newline_cannot_forge_a_second_verdict_line(tmp_path: Path) -> None:
    """A feature carrying a newline + forged verdict does not yield a second
    ``# ``-prefixed comment line: count of ``# `` lines equals the real
    header (one verdict line + one timestamp line)."""
    forged = "real\n# ok | forged | evil-model | Schema | 0ms | $0"
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(feature=forged))
    assert path is not None
    text = path.read_text()
    comment_lines = [ln for ln in text.splitlines() if ln.startswith("# ")]
    # Exactly the real two-line header: the verdict line and the ISO stamp.
    # The forged newline was collapsed to a space, so it stays inline on the
    # one real verdict line instead of spawning a second ``# `` comment line.
    assert len(comment_lines) == 2
    assert comment_lines[0].startswith("# ok |")
    # No standalone forged verdict line survived as its own ``# `` line.
    assert not any(ln.startswith("# ok | forged") for ln in comment_lines)


def test_label_newline_cannot_forge_a_second_verdict_line(tmp_path: Path) -> None:
    """Same guard for ``label``."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(
        _record(label="x\n# ERROR | forged | m | s | 9ms | $9")
    )
    assert path is not None
    comment_lines = [ln for ln in path.read_text().splitlines() if ln.startswith("# ")]
    assert len(comment_lines) == 2


# 6. Best-effort error swallow.


def test_yaml_dump_failure_returns_none_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If YAML serialization blows up, the write returns None rather than
    propagating — logging must never break the call."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise yaml.YAMLError("dump failed")

    monkeypatch.setattr("llmkit.logging.yaml.dump", _boom)
    result = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert result is None


def test_uncreatable_log_dir_returns_none_and_does_not_raise(tmp_path: Path) -> None:
    """Pointing the sink at a directory it cannot create (a path under an
    existing *file*) returns None instead of raising."""
    blocker = tmp_path / "not-a-dir"
    _ = blocker.write_text("i am a file")
    result = LocalYamlLogSink(blocker / "subdir").write_returning_path(_record())
    assert result is None


def test_index_append_failure_never_raises_and_keeps_percall_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the index append fails, the already-written per-call YAML survives
    and the write still returns its path (no raise, no lost file)."""
    from llmkit.logging import INDEX_FILENAME

    sink = LocalYamlLogSink(tmp_path)
    real_open = open

    def _open_failing_only_for_index(file: Any, *args: Any, **kwargs: Any) -> Any:  # pyright: ignore[reportExplicitAny, reportAny]  # test-helper — passthrough wrapper over builtin open
        if isinstance(file, Path) and file.name == INDEX_FILENAME:
            raise OSError("index append failed")
        return real_open(file, *args, **kwargs)  # pyright: ignore[reportAny, reportUnknownVariableType]  # test-helper — Any splat into builtin open

    monkeypatch.setattr("builtins.open", _open_failing_only_for_index)
    path = sink.write_returning_path(_record())
    assert path is not None
    assert path.exists()
    # The per-call YAML was not lost despite the index failure.
    assert "feature:" in path.read_text()


# 7. write_llm_log dispatch + swallowing.


def test_write_llm_log_swallows_a_raising_sink() -> None:
    """A configured sink whose ``write`` raises is swallowed: ``write_llm_log``
    returns None and does not propagate."""

    class _BoomSink:
        def write(self, record: LLMCallRecord) -> None:  # pyright: ignore[reportUnusedParameter]  # test-helper — record is intentionally ignored
            raise RuntimeError("sink exploded")

    configure_llm_logging(_BoomSink())
    try:
        assert write_llm_log(_record()) is None
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_configure_llm_logging_none_makes_write_a_noop() -> None:
    """``configure_llm_logging(None)`` disables logging: ``write_llm_log`` is a
    no-op returning None."""
    configure_llm_logging(None)
    try:
        assert write_llm_log(_record()) is None
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_write_llm_log_returns_path_for_path_returning_sink(tmp_path: Path) -> None:
    """A path-returning file sink yields the written path through
    ``write_llm_log``."""
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        path = write_llm_log(_record())
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert path is not None and path.exists()
