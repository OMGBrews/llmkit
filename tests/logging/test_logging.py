"""Tests for the hardened LLM-log sink (:mod:`llmkit.logging`).

These pin the recent hardening of :class:`LocalYamlLogSink` /
:func:`write_llm_log`: the verdict-first YAML layout and ``index.jsonl``
summary, plus the robustness guards — filename collisions never
overwrite, ``feature``/``label`` can't escape ``log_dir`` via path
traversal or blow the 255-byte filename component limit, and a newline in
a caller field can't forge a second verdict header line — and the
best-effort swallowing that keeps *any* logging failure (not just the
common enumerated ones) from ever breaking the LLM call, with no orphaned
partial file and no discarded YAML path when only the index append fails. They also pin the serialization contract:
``max_tokens``/``reasoning_effort`` land in the per-call YAML (but stay out
of the compact index), and the body is always ``yaml.safe_load``-able even
when the response carries arbitrary Python objects.

Records are built directly via :func:`_record`; no real LLM call is made.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
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
from tests._support import make_record as _record

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
        "run_id",
        "call_id",
        "attempt",
        "duration_ms",
        "queue_wait_ms",
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


# 3. max_tokens / reasoning_effort serialization + safe-load-able YAML.


def test_max_tokens_and_reasoning_effort_round_trip_into_yaml(tmp_path: Path) -> None:
    """``max_tokens`` and ``reasoning_effort`` land in the per-call YAML body
    and round-trip through ``yaml.safe_load`` with their original values."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(
        _record(max_tokens=512, reasoning_effort="high")
    )
    assert path is not None
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert doc["max_tokens"] == 512
    assert doc["reasoning_effort"] == "high"


def test_unset_max_tokens_and_reasoning_effort_serialize_as_null(tmp_path: Path) -> None:
    """The defaults (both ``None``) still appear as explicit keys, so the log
    shape is stable whether or not the caller set them."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert "max_tokens" in doc and doc["max_tokens"] is None
    assert "reasoning_effort" in doc and doc["reasoning_effort"] is None


def test_max_tokens_and_reasoning_effort_stay_out_of_the_index(tmp_path: Path) -> None:
    """The index is deliberately compact (cross-call triage fields only):
    request-shaping knobs live in the per-call YAML, not ``index.jsonl``."""
    sink = LocalYamlLogSink(tmp_path)
    assert sink.write_returning_path(_record(max_tokens=512, reasoning_effort="high")) is not None
    line = cast(
        "dict[str, object]", json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    )
    assert "max_tokens" not in line
    assert "reasoning_effort" not in line


class _Color(Enum):
    RED = "red"


def test_response_with_python_objects_stays_safe_loadable(tmp_path: Path) -> None:
    """A response carrying an Enum member, a Decimal, and a set (as a python-mode
    ``model_dump()`` can produce) yields a file with no ``!!python`` tags that
    ``yaml.safe_load`` parses into sensible plain values."""
    response = {
        "status": _Color.RED,
        "price": Decimal("1.25"),
        "tags": {"alpha", "beta"},
        "plain": "text",
        "count": 3,
    }
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(response=response))
    assert path is not None
    text = path.read_text()
    assert "!!python" not in text

    doc = cast("dict[str, object]", yaml.safe_load(text))
    parsed = cast("dict[str, object]", doc["response"])
    # Enum member -> its payload value; Decimal -> its string form; set stays a set.
    assert parsed["status"] == "red"
    assert parsed["price"] == "1.25"
    assert parsed["tags"] == {"alpha", "beta"}
    # Standard types are untouched.
    assert parsed["plain"] == "text"
    assert parsed["count"] == 3


def test_standard_record_round_trips_through_safe_load(tmp_path: Path) -> None:
    """The safe dumper preserves the existing format for ordinary records: the
    full document safe-loads with the standard scalar types intact."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(
        _record(
            prompt="line one\nline two",
            response={"answer": 42, "ok": True, "ratio": 0.5, "items": ["a", "b"]},
            approximate_cost=0.001,
        )
    )
    assert path is not None
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert doc["feature"] == "extraction"
    assert doc["temperature"] == 0.0
    assert doc["approximate_cost"] == 0.001
    assert doc["prompt"] == "line one\nline two"
    assert doc["response"] == {"answer": 42, "ok": True, "ratio": 0.5, "items": ["a", "b"]}


# 4. Filename collision — the headline guard.


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


# 5. feature/label path-traversal guard.


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


# 6. Header injection guard.


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


# 7. Best-effort error swallow.


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


def test_cyclic_prompt_is_dumped_via_yaml_aliases_not_an_error(tmp_path: Path) -> None:
    """A self-referential structure smuggled into ``prompt`` does not blow the
    dump: PyYAML renders the cycle with anchors/aliases, so the write succeeds
    (this is why the non-enumerated-failure test below needs a synthetic
    trigger)."""
    cyclic: dict[str, object] = {"role": "user"}
    cyclic["self"] = cyclic
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(prompt=[cyclic]))
    assert path is not None
    assert path.exists()
    assert "&id" in path.read_text()  # the cycle became an anchor, not a crash


def test_non_enumerated_dump_failure_warns_and_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure *outside* the historically enumerated set (here a synthetic
    ``RuntimeError`` from the dumper, standing in for RecursionError /
    representer oddities) is still swallowed by the best-effort handler — no
    exception, a warning with context — and the truncated exclusive-create
    file is unlinked, so the log dir holds no orphan."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("exotic dumper failure")

    monkeypatch.setattr("llmkit.logging.yaml.dump", _boom)
    with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
        result = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert result is None
    assert any("Failed to write LLM invocation log" in r.message for r in caplog.records)
    # The exclusive-create file opened before yaml.dump raised was cleaned up:
    # nothing (YAML, index, or otherwise) is left in the log dir.
    assert list(tmp_path.iterdir()) == []


def test_index_serialization_failure_still_returns_yaml_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A record whose field violates the declared types (the dataclass doesn't
    enforce them) makes ``json.dumps`` raise ``TypeError`` in the index append
    — the YAML write itself survives (the dumper's fallback stringifies the
    object), and the already-written per-call path is still returned instead
    of being discarded."""
    # ``provider`` is the one indexed field that is neither part of the
    # filename nor the header, so only the index's json.dumps chokes on it.
    record = _record(provider=object())
    with caplog.at_level(logging.WARNING, logger="llmkit.logging"):
        path = LocalYamlLogSink(tmp_path).write_returning_path(record)
    assert path is not None
    assert path.exists()
    assert "feature:" in path.read_text()
    assert any("Failed to append LLM log index" in r.message for r in caplog.records)
    # The index line was the casualty, never the per-call file or the path.
    assert not (tmp_path / "index.jsonl").exists()


# 8. write_llm_log dispatch + swallowing.


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


# 9. Filename length clamp.


def test_overlong_feature_is_clamped_and_still_written(tmp_path: Path) -> None:
    """A 300-char feature no longer drives the filename past the 255-byte
    component limit (which would make every write fail ENAMETOOLONG): the
    component is clamped to 80 bytes and the file is written successfully."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(feature="f" * 300))
    assert path is not None
    assert path.exists()
    assert len(path.name.encode("utf-8")) <= 255
    # The sanitized component was clamped to exactly the 80-byte budget.
    assert "f" * 80 in path.name
    assert "f" * 81 not in path.name
    # The full, unclamped feature still lives inside the YAML body.
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert doc["feature"] == "f" * 300


def test_overlong_label_is_clamped_too(tmp_path: Path) -> None:
    """The same clamp applies to ``label``."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(label="l" * 300))
    assert path is not None
    assert path.exists()
    assert len(path.name.encode("utf-8")) <= 255


def test_multibyte_feature_clamps_on_a_codepoint_boundary(tmp_path: Path) -> None:
    """The clamp counts bytes (the filesystem limit is bytes, not chars) and
    never cuts mid-codepoint: 300 two-byte chars become exactly 40 of them
    (80 bytes), with no mojibake in the name."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(feature="é" * 300))
    assert path is not None
    assert path.exists()
    assert "é" * 40 in path.name
    assert "é" * 41 not in path.name
