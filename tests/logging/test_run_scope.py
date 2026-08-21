"""Tests for run scoping — the ``run_id`` on records, YAML bodies and index lines.

A shared log directory is only greppable by run if something writes the run
down. ``call_id`` cannot: it joins the retry attempts of one *logical call*, not
the calls of one run. These pin the field that can:

* **Resolution** — context scope, then process default, then ``LLMKIT_RUN_ID``;
  first one set wins, blank environment counts as unset, blank programmatic
  value raises. The two programmatic layers exist because they fail in opposite
  directions (a ContextVar crosses the sync bridge but not a
  :class:`threading.Thread`; a process global does the reverse), so both
  behaviours are pinned here rather than left to the docstring.
* **Propagation** — the stamp reaches every record the call layer builds:
  structured and text surfaces, the ``run_sync`` bridge, and *every* attempt of
  a retried call, not just the first.
* **Rendering** — the same ``run_id`` lands in the per-call YAML and in
  ``index.jsonl``, so index → YAML cross-referencing needs no new join key; and
  with no scope set both carry an explicit ``null``, leaving every pre-existing
  key and its order untouched.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Awaitable, Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import yaml

from llmkit import (
    LocalYamlLogSink,
    RetryPolicy,
    capture_llm_records,
    configure_llm_logging,
    get_run_id,
    run_scope,
    set_run_id,
    structured_output,
)
from llmkit.run_scope import RUN_ID_ENV_VAR
from tests._support import OkSchema, provider_mock
from tests._support import make_record as _record

_NO_BACKOFF = RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)

# The index keys as of 0.8.0, before ``run_id`` existed — the shape every
# current consumer parses. Pinned so a future edit that reorders or drops one
# of them fails here rather than in someone else's audit tool.
_LEGACY_INDEX_KEYS = [
    "file",
    "timestamp",
    "feature",
    "label",
    "model",
    "provider",
    "schema",
    "call_id",
    "attempt",
    "duration_ms",
    "queue_wait_ms",
    "approximate_cost",
    "error",
]


@pytest.fixture(autouse=True)
def _reset_run_scope() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    """Clear the process-wide run id around every test in this module.

    ``set_run_id`` writes a module global, so without this a test that sets one
    would leak it into every later test in the session — including tests in
    other files that assert ``run_id is None``.
    """
    set_run_id(None)
    try:
        yield
    finally:
        set_run_id(None)


@pytest.fixture(autouse=True)
def _no_ambient_run_id(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    """Drop any real ``LLMKIT_RUN_ID`` so the suite behaves the same everywhere."""
    monkeypatch.delenv(RUN_ID_ENV_VAR, raising=False)


async def _ok_transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
    return OkSchema(ok=True), None


_StructuredTransport = Callable[..., Awaitable[tuple[OkSchema, float | None]]]
_TextTransport = Callable[..., Awaitable[tuple[str, float | None]]]


@contextmanager
def patch_transport(transport: _StructuredTransport = _ok_transport) -> Generator[None]:
    """Run a structured call fully offline: fake transport, fake provider."""
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):
        yield


@contextmanager
def patch_text_transport(transport: _TextTransport) -> Generator[None]:
    """The plain-text counterpart of :func:`patch_transport`."""
    with (
        patch("llmkit._litellm.acompletion_text", side_effect=transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):
        yield


def _index_lines(log_dir: Path) -> list[dict[str, object]]:
    """Parse every ``index.jsonl`` line under *log_dir*, in write order."""
    return [
        cast("dict[str, object]", json.loads(line))
        for line in (log_dir / "index.jsonl").read_text().splitlines()
    ]


# 1. Resolution order.


def test_env_var_supplies_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-code entry point: a shell export is picked up as-is."""
    monkeypatch.setenv(RUN_ID_ENV_VAR, "from-env")
    assert get_run_id() == "from-env"


def test_env_var_is_read_fresh_not_cached_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution reads ``os.environ`` per call, so a mid-process change lands."""
    assert get_run_id() is None
    monkeypatch.setenv(RUN_ID_ENV_VAR, "late")
    assert get_run_id() == "late"


def test_blank_env_var_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLMKIT_RUN_ID=`` from a shell is "not set", not a run id — the same
    reading ``LLMKIT_LOG_DIR`` gives an empty value."""
    monkeypatch.setenv(RUN_ID_ENV_VAR, "")
    assert get_run_id() is None
    monkeypatch.setenv(RUN_ID_ENV_VAR, "   ")
    assert get_run_id() is None


def test_process_default_overrides_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit configuration beats the ambient environment."""
    monkeypatch.setenv(RUN_ID_ENV_VAR, "from-env")
    set_run_id("from-code")
    assert get_run_id() == "from-code"


def test_clearing_the_process_default_falls_back_to_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``set_run_id(None)`` clears the *programmatic* layer only."""
    monkeypatch.setenv(RUN_ID_ENV_VAR, "from-env")
    set_run_id("from-code")
    set_run_id(None)
    assert get_run_id() == "from-env"


def test_scope_overrides_the_process_default() -> None:
    """The context scope is the most specific layer, and restores on exit."""
    set_run_id("process")
    with run_scope("scoped"):
        assert get_run_id() == "scoped"
    assert get_run_id() == "process"


def test_scope_restores_the_previous_value_on_an_exception() -> None:
    """An error inside the block must not leave the scope latched on."""
    with pytest.raises(ValueError, match="boom"), run_scope("scoped"):
        raise ValueError("boom")
    assert get_run_id() is None


def test_nested_scopes_unwind_in_order() -> None:
    """Nesting is a stack, so an inner run cannot cross-tag the outer one."""
    with run_scope("outer"):
        with run_scope("inner"):
            assert get_run_id() == "inner"
        assert get_run_id() == "outer"
    assert get_run_id() is None


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_programmatic_run_id_raises(blank: str) -> None:
    """Configuration that would silently group nothing fails loudly instead."""
    with pytest.raises(ValueError, match="non-blank"):
        set_run_id(blank)
    with pytest.raises(ValueError, match="non-blank"):
        with run_scope(blank):
            pass  # pragma: no cover - the context manager raises on entry


def test_process_default_reaches_a_worker_thread() -> None:
    """Why the process layer exists: a new thread starts with a *fresh* context,
    so a ContextVar-only design would be invisible to a thread-pool fan-out."""
    set_run_id("process")
    seen: list[str | None] = []

    def _worker() -> None:
        seen.append(get_run_id())

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert seen == ["process"]


def test_context_scope_does_not_leak_into_a_worker_thread() -> None:
    """The documented limitation, pinned: ``run_scope`` is context-scoped, and a
    thread you start yourself does not inherit the caller's context."""
    seen: list[str | None] = []

    def _worker() -> None:
        seen.append(get_run_id())

    with run_scope("scoped"):
        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
    assert seen == [None]


# 2. Propagation onto records.


@pytest.mark.asyncio
async def test_structured_call_stamps_the_active_run_id() -> None:
    """The structured surface carries the scope onto its record."""
    with (
        patch_transport(),
        run_scope("eval-sweep"),
        capture_llm_records() as records,
    ):
        _ = await structured_output.structured_llm_call("hi", OkSchema, feature="test")

    assert [r.run_id for r in records] == ["eval-sweep"]


@pytest.mark.asyncio
async def test_text_call_stamps_the_active_run_id() -> None:
    """So does the buffered plain-text surface, which builds its record through
    a different helper (``_build_text_record``)."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "hello", None

    with (
        patch_text_transport(_transport),
        run_scope("eval-sweep"),
        capture_llm_records() as records,
    ):
        _ = await structured_output.text_llm_call("hi", feature="test")

    assert [r.run_id for r in records] == ["eval-sweep"]


def test_sequential_process_scopes_never_cross_tag() -> None:
    """Two runs in one process, tagged with their own ids and nothing else's."""
    with patch_transport(), capture_llm_records() as records:
        set_run_id("run-a")
        _ = structured_output.structured_llm_call_sync("hi", OkSchema, feature="test")
        set_run_id("run-b")
        _ = structured_output.structured_llm_call_sync("hi", OkSchema, feature="test")
        set_run_id(None)
        _ = structured_output.structured_llm_call_sync("hi", OkSchema, feature="test")

    assert [r.run_id for r in records] == ["run-a", "run-b", None]


def test_scope_crosses_the_sync_bridge() -> None:
    """``run_sync`` copies the calling thread's context onto the persistent
    loop, so a ContextVar-held scope reaches records built there."""
    with patch_transport(), run_scope("scoped"), capture_llm_records() as records:
        _ = structured_output.structured_llm_call_sync("hi", OkSchema, feature="test")

    assert [r.run_id for r in records] == ["scoped"]


# 3. Rendering in the sink.


def test_env_scoped_sync_call_tags_every_index_line_including_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requester's headline case, end to end: export the variable, make a
    ``*_sync`` call that retries twice, and every line in the shared index —
    attempt 1, 2 and 3 alike — carries the run id."""
    monkeypatch.setenv(RUN_ID_ENV_VAR, "abc")
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    calls = [0]

    async def _flaky(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] < 3:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    with patch_transport(_flaky):
        result = structured_output.structured_llm_call_sync(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    lines = _index_lines(tmp_path)
    assert [line["attempt"] for line in lines] == [1, 2, 3]
    assert [line["run_id"] for line in lines] == ["abc", "abc", "abc"]


def test_yaml_and_index_agree_on_run_id(tmp_path: Path) -> None:
    """One call is cross-referenceable index → YAML on the existing join key,
    with the same ``run_id`` on both sides."""
    record = _record(run_id="eval-42", call_id="feedface" + "0" * 24, attempt=1)
    path = LocalYamlLogSink(tmp_path).write_returning_path(record)
    assert path is not None

    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    (line,) = _index_lines(tmp_path)
    assert doc["run_id"] == "eval-42"
    assert line["run_id"] == "eval-42"
    assert line["file"] == path.name
    assert doc["call_id"] == line["call_id"]


def test_unscoped_record_keeps_the_pre_run_id_shape(tmp_path: Path) -> None:
    """With no scope set, the only change to the index line is an explicit
    ``run_id: null`` — every pre-existing key survives, in its original order,
    so a current consumer parses it unchanged."""
    assert LocalYamlLogSink(tmp_path).write_returning_path(_record()) is not None
    (line,) = _index_lines(tmp_path)

    assert line["run_id"] is None
    assert [key for key in line if key != "run_id"] == _LEGACY_INDEX_KEYS


def test_unscoped_yaml_carries_an_explicit_null(tmp_path: Path) -> None:
    """The YAML body spells the absence out too, rather than omitting the key —
    matching how every other optional field renders."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert "run_id" in doc and doc["run_id"] is None


def test_run_id_does_not_touch_the_pinned_header_line(tmp_path: Path) -> None:
    """``head -1`` triage tooling greps line 1; adding a run id must not
    reshape it."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record(run_id="eval-42"))
    assert path is not None
    first_line = path.read_text().splitlines()[0]
    assert first_line.startswith("# ok | extraction/summary |")
    assert "eval-42" not in first_line


def test_run_id_composes_with_the_log_dir_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two mechanisms are independent: redirecting the directory and
    tagging the lines can be used together, and neither disables the other."""
    monkeypatch.setenv("LLMKIT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv(RUN_ID_ENV_VAR, "both-set")
    configure_llm_logging(LocalYamlLogSink())

    with patch_transport():
        _ = structured_output.structured_llm_call_sync("hi", OkSchema, feature="test")

    (line,) = _index_lines(tmp_path)
    assert line["run_id"] == "both-set"
    assert (tmp_path / cast("str", line["file"])).is_file()
