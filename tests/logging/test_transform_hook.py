"""``LocalYamlLogSink(transform=...)``: redact or skip records, keep the built-in accounting.

The hook exists so redaction no longer needs a wrapping sink, which the
registry counts as ``custom``: no ``written`` count, no captured paths. These
tests pin that a transformed write still counts as built-in, that a skip and
a failed hook are counted on their own terms, and that a failing hook never
lets the untransformed record, or its exception message, reach disk or the
logs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
import yaml

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    capture_llm_log_paths,
    capture_llm_records,
    configure_llm_logging,
    get_logging_health,
    text_llm_call_sync,
)
from llmkit.capture import record_call
from llmkit.logging import INDEX_FILENAME, write_llm_log
from llmkit.testing import Reply, install
from tests._support import make_record

SECRET = "sk-SECRET-4242"


@pytest.fixture(autouse=True)
def _reset_sink() -> Iterator[None]:
    try:
        yield
    finally:
        configure_llm_logging(None)


def _redact(record: LLMCallRecord) -> LLMCallRecord:
    # Every field that can carry conversation content: the prompt and the
    # per-attempt request snapshots both hold the messages.
    return dataclasses.replace(
        record,
        prompt="[redacted]",
        response="[redacted]",
        partial_text=None,
        requests=[],
        tool_calls=None,
        label="redacted",
    )


def _skip(_record: LLMCallRecord) -> None:
    return None


def _raise_key_error(record: LLMCallRecord) -> LLMCallRecord:
    raise KeyError(f"no rule for {record.prompt}")


def _raise_value_error(_record: LLMCallRecord) -> LLMCallRecord:
    raise ValueError(SECRET)


def _yaml_files(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*.yaml"))


def _index_lines(log_dir: Path) -> list[dict[str, object]]:
    index = log_dir / INDEX_FILENAME
    if not index.exists():
        return []
    return [
        cast("dict[str, object]", json.loads(line))
        for line in index.read_text(encoding="utf-8").splitlines()
    ]


def _all_disk_text(log_dir: Path) -> str:
    return "".join(
        p.read_text(encoding="utf-8", errors="replace") for p in log_dir.rglob("*") if p.is_file()
    )


def _llmkit_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "llmkit.logging"]


# --- the transformed record is what is written --------------------------------


def test_written_file_header_and_index_carry_the_transformed_record(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_redact))

    path = write_llm_log(make_record(prompt=SECRET, response=SECRET, label="doc-1"))

    assert path is not None
    assert path.name.endswith(".yaml") and "_redacted_" in path.name
    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert "/redacted |" in header
    body = cast("dict[str, object]", yaml.safe_load(text))
    assert (body["prompt"], body["response"], body["label"]) == (
        "[redacted]",
        "[redacted]",
        "redacted",
    )
    [line] = _index_lines(tmp_path)
    assert (line["file"], line["label"]) == (path.name, "redacted")
    assert SECRET not in _all_disk_text(tmp_path)


def test_a_managed_call_keeps_builtin_health_and_captured_paths(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_redact))

    with install(lambda _: Reply.text(SECRET)), capture_llm_log_paths() as paths:
        answer = text_llm_call_sync(SECRET, feature="demo", label="x")

    assert answer == SECRET
    health = get_logging_health()
    assert (health.sink, health.written, health.handed_off, health.failed) == (
        "builtin",
        1,
        0,
        0,
    )
    assert health.skipped == 0
    assert paths == _yaml_files(tmp_path)
    assert len(paths) == 1
    assert SECRET not in _all_disk_text(tmp_path)


@pytest.mark.parametrize("entry_point", ["write", "write_returning_path", "write_llm_log"])
def test_every_entry_point_runs_the_transform(tmp_path: Path, entry_point: str) -> None:
    sink = LocalYamlLogSink(tmp_path, transform=_redact)
    record = make_record(prompt=SECRET)
    if entry_point == "write":
        sink.write(record)
    elif entry_point == "write_returning_path":
        assert sink.write_returning_path(record) is not None
    else:
        configure_llm_logging(sink)
        assert write_llm_log(record) is not None

    assert len(_yaml_files(tmp_path)) == 1
    assert SECRET not in _all_disk_text(tmp_path)


def test_captured_records_stay_original_on_both_paths(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_redact))

    with install(lambda _: Reply.text("answer")), capture_llm_records() as managed:
        _ = text_llm_call_sync(SECRET, feature="demo", label="managed")
    with capture_llm_records() as direct:
        assert record_call(make_record(prompt=SECRET, label="direct")) is not None

    [managed_record] = managed
    [direct_record] = direct
    assert SECRET in str(managed_record.prompt)
    assert managed_record.label == "managed"
    assert (direct_record.prompt, direct_record.label) == (SECRET, "direct")
    assert get_logging_health().written == 2
    assert SECRET not in _all_disk_text(tmp_path)


# --- a skip ------------------------------------------------------------------------


def test_a_skip_writes_nothing_and_counts_only_as_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="llmkit.logging")
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_skip))

    with capture_llm_log_paths() as paths:
        assert write_llm_log(make_record()) is None

    health = get_logging_health()
    assert (health.sink, health.skipped, health.written, health.failed, health.dropped) == (
        "builtin",
        1,
        0,
        0,
        0,
    )
    assert health.last_success_at is None
    assert health.last_failure is None
    assert paths == []
    assert list(tmp_path.iterdir()) == []
    assert _llmkit_logs(caplog) == []


def test_a_skip_leaves_earlier_success_and_failure_untouched(tmp_path: Path) -> None:
    decisions = iter(["write", "raise", "skip"])

    def _transform(record: LLMCallRecord) -> LLMCallRecord | None:
        decision = next(decisions)
        if decision == "raise":
            raise RuntimeError("boom")
        return record if decision == "write" else None

    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_transform))
    _ = write_llm_log(make_record())
    _ = write_llm_log(make_record())
    before = get_logging_health()
    _ = write_llm_log(make_record())
    after = get_logging_health()

    assert (after.written, after.failed, after.skipped) == (1, 1, 1)
    assert after.last_success_at == before.last_success_at
    assert after.last_failure == before.last_failure


# --- a failing transform ----------------------------------------------------------


def test_a_raising_transform_writes_nothing_and_hides_its_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="llmkit.logging")
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_raise_key_error))

    assert write_llm_log(make_record(prompt=SECRET, feature="extract", label="doc-1")) is None

    assert list(tmp_path.iterdir()) == []
    health = get_logging_health()
    assert (health.sink, health.written, health.failed, health.skipped) == ("builtin", 0, 1, 0)
    failure = health.last_failure
    assert failure is not None
    assert (failure.stage, failure.error) == ("transform", "KeyError")
    # The original record's identifiers, as passed.
    assert (failure.feature, failure.label) == ("extract", "doc-1")
    assert SECRET not in repr(health)

    [warning] = _llmkit_logs(caplog)
    assert warning.levelno == logging.WARNING
    assert warning.exc_info is None and warning.exc_text is None
    message = warning.getMessage()
    assert SECRET not in message
    assert "KeyError" in message
    # Where it was raised: the hook's own function, file and line.
    raise_line = _raise_key_error.__code__.co_firstlineno + 1
    assert f"_raise_key_error ({__file__}:{raise_line})" in message
    assert "extract/doc-1" in message
    assert "not written" in message


def test_transform_warnings_are_latched_per_type_and_rearmed_by_success(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="llmkit.logging")
    behaviours: list[Callable[[LLMCallRecord], LLMCallRecord]] = [
        _raise_key_error,
        _raise_key_error,
        _raise_value_error,
        lambda record: record,
        _raise_value_error,
    ]
    calls = iter(behaviours)
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=lambda r: next(calls)(r)))

    for _ in behaviours:
        _ = write_llm_log(make_record(prompt=SECRET))

    levels = [
        (r.levelno, r.getMessage().split(" at ")[0])
        for r in _llmkit_logs(caplog)
        if r.getMessage().startswith("transform")
    ]
    assert levels == [
        (logging.WARNING, "transform raised KeyError"),
        (logging.DEBUG, "transform raised KeyError"),
        (logging.WARNING, "transform raised ValueError"),
        (logging.WARNING, "transform raised ValueError"),
    ]
    for record in _llmkit_logs(caplog):
        assert SECRET not in record.getMessage()
        assert record.exc_info is None and record.exc_text is None
    health = get_logging_health()
    assert (health.written, health.failed) == (1, 4)
    assert health.last_failure is not None
    assert health.last_failure.error == "ValueError"


def test_a_raising_transform_on_a_managed_call_keeps_the_answer(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=_raise_value_error))

    with install(lambda _: Reply.text("still here")), capture_llm_log_paths() as paths:
        answer = text_llm_call_sync(SECRET, feature="demo", label="x")

    assert answer == "still here"
    assert paths == []
    assert list(tmp_path.iterdir()) == []
    health = get_logging_health()
    assert (health.written, health.failed) == (0, 1)
    assert health.last_failure is not None
    assert health.last_failure.stage == "transform"


def test_a_wrong_typed_return_is_a_transform_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="llmkit.logging")

    def _to_dict(record: LLMCallRecord) -> object:
        return {"prompt": record.prompt}

    transform = cast("Callable[[LLMCallRecord], LLMCallRecord | None]", _to_dict)
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=transform))

    assert write_llm_log(make_record(prompt=SECRET)) is None

    assert list(tmp_path.iterdir()) == []
    health = get_logging_health()
    assert (health.written, health.failed, health.skipped) == (0, 1, 0)
    assert health.last_failure is not None
    assert (health.last_failure.stage, health.last_failure.error) == (
        "transform",
        "_TransformContractError",
    )
    [warning] = _llmkit_logs(caplog)
    assert warning.levelno == logging.WARNING
    message = warning.getMessage()
    assert "returned dict" in message
    assert _to_dict.__qualname__ in message
    assert SECRET not in message


def test_a_wrong_return_and_a_real_type_error_warn_separately(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="llmkit.logging")
    results: list[object] = ["not a record", TypeError("real")]

    def _transform(_record: LLMCallRecord) -> object:
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    transform = cast("Callable[[LLMCallRecord], LLMCallRecord | None]", _transform)
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=transform))
    _ = write_llm_log(make_record())
    _ = write_llm_log(make_record())

    assert [r.levelno for r in _llmkit_logs(caplog)] == [logging.WARNING, logging.WARNING]


def test_base_exception_from_the_transform_propagates(tmp_path: Path) -> None:
    def _interrupt(_record: LLMCallRecord) -> LLMCallRecord:
        raise KeyboardInterrupt

    sink = LocalYamlLogSink(tmp_path, transform=_interrupt)

    with pytest.raises(KeyboardInterrupt):
        sink.write(make_record())
    assert list(tmp_path.iterdir()) == []


# --- construction -----------------------------------------------------------------


async def _async_transform(record: LLMCallRecord) -> LLMCallRecord:
    return record


@pytest.mark.parametrize(
    ("transform", "match"),
    [
        ("not callable", "transform must be a callable or None, got str"),
        (_async_transform, "transform must be a synchronous function"),
    ],
)
def test_a_bad_transform_is_rejected_before_any_state(
    tmp_path: Path, transform: object, match: str
) -> None:
    sink = LocalYamlLogSink.__new__(LocalYamlLogSink)
    directory = tmp_path / "absent"
    initialize = cast("Callable[..., None]", sink.__init__)
    with pytest.raises(TypeError, match=match):
        initialize(directory, transform=transform)
    assert vars(sink) == {}
    assert not directory.exists()


def test_without_a_transform_nothing_changes(tmp_path: Path) -> None:
    configure_llm_logging(LocalYamlLogSink(tmp_path))

    path = write_llm_log(make_record(prompt=SECRET))

    assert path is not None
    assert SECRET in path.read_text(encoding="utf-8")
    health = get_logging_health()
    assert (health.written, health.skipped) == (1, 0)


# --- the README example -----------------------------------------------------------


def _readme_redact() -> Callable[[LLMCallRecord], LLMCallRecord | None]:
    """The ``redact`` function from the README's transform example, run as published."""
    readme = Path(__file__).parents[2] / "README.md"
    blocks = cast(
        "list[str]",
        re.findall(r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.DOTALL),
    )
    [block] = [b for b in blocks if "LocalYamlLogSink(transform=" in b]
    namespace: dict[str, object] = {}
    exec(block, namespace)  # the published example, from this repository
    return cast("Callable[[LLMCallRecord], LLMCallRecord | None]", namespace["redact"])


def test_the_readme_example_redacts_every_copy_and_skips_its_feature(tmp_path: Path) -> None:
    redact = _readme_redact()
    configure_llm_logging(LocalYamlLogSink(tmp_path, transform=redact))

    with install(lambda _: Reply.text(SECRET)):
        _ = text_llm_call_sync(SECRET, feature="demo", label="x")
        _ = text_llm_call_sync(SECRET, feature="medical-intake", label="y")

    health = get_logging_health()
    assert (health.written, health.skipped, health.failed) == (1, 1, 0)
    assert len(_yaml_files(tmp_path)) == 1
    assert SECRET not in _all_disk_text(tmp_path)
