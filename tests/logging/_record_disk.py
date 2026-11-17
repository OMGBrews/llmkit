"""Shared scaffolding for the tests that drive a call lane and read the disk.

``test_record_matrix.py`` and ``test_record_redirection.py`` both drive the
five record-writing call lanes through an :mod:`llmkit.testing` fake and then
read what reached the configured sink. Records are delivered by the logging
worker (``llmkit/logging/_delivery.py``), so nothing here reads a directory
before :func:`settle` has flushed delivery and checked that no record was
dropped or failed on the way.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from llmkit import (
    NO_RETRY,
    RetryPolicy,
    ToolDefinition,
    configure_llm_logging,
    flush_llm_logs_async,
    get_logging_health,
)
from llmkit import calls as llm_calls
from llmkit.logging import INDEX_FILENAME, LoggingHealth, LogSink, get_log_sink
from llmkit.logging import _delivery as delivery
from llmkit.testing import Reply, TransportRequest
from tests._support import OkSchema

#: The five lanes that build and hand off a call record. The sync forms run
#: through the same lanes and get no entries of their own.
type Family = Literal["structured", "text", "text_stream", "tool", "tool_stream"]
FAMILIES: tuple[Family, ...] = ("structured", "text", "text_stream", "tool", "tool_stream")

#: The schema each family writes in the header and the ``schema`` key.
SCHEMA: dict[Family, str] = {
    "structured": "OkSchema",
    "text": "text",
    "text_stream": "stream",
    "tool": "tools",
    "tool_stream": "tools-stream",
}

#: The YAML body keys, in ``LocalYamlLogSink.write_with_outcome`` order
#: (``library/src/llmkit/logging/local_yaml.py``). One constant: a key added to
#: or removed from the record changes it here, once. If the companion audit
#: ``audit-logs-written-for-agents`` lands ``format_version`` and
#: ``llmkit_version``, they join this tuple.
YAML_KEYS: tuple[str, ...] = (
    "timestamp",
    "feature",
    "label",
    "model",
    "provider",
    "schema",
    "run_id",
    "call_id",
    "attempt",
    "cache_hit",
    "source_call_id",
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "tools",
    "tool_calls",
    "usage",
    "output_limit",
    "native_finish_reason",
    "requested_service_tier",
    "served_service_tier",
    "duration_ms",
    "queue_wait_ms",
    "first_chunk_ms",
    "approximate_cost",
    "error",
    "partial_text",
    "response",
    "prompt",
    "requests",
)

#: The ``index.jsonl`` line keys, in ``LocalYamlLogSink._append_index`` order.
INDEX_KEYS: tuple[str, ...] = (
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
    "cache_hit",
    "duration_ms",
    "queue_wait_ms",
    "first_chunk_ms",
    "approximate_cost",
    "error",
    "requested_service_tier",
    "served_service_tier",
)

#: Header line 1, as ``LocalYamlLogSink._summary_header`` builds it.
HEADER_LINE_1 = re.compile(
    r"^# (?P<status>ok|ERROR) \| (?P<feature>[^|/]+)/(?P<label>[^|]+) \| "
    + r"(?P<model>[^|]+) \| (?P<schema>[^|]+) \| (?P<ms>\d+)ms \| "
    + r"\$(?P<cost>\?|[0-9.e+-]+)$"
)

FEATURE = "matrix"
TOOLS = [
    ToolDefinition(
        "lookup",
        "Look a term up.",
        {"type": "object", "properties": {"q": {"type": "string"}}},
    )
]


@dataclass(frozen=True)
class DiskRecord:
    """One record as it reached disk: the YAML file and its index line."""

    path: Path
    header: list[str]
    body: dict[str, object]
    index: dict[str, object]

    @property
    def status(self) -> str:
        match = HEADER_LINE_1.match(self.header[0])
        assert match is not None, self.header[0]
        return match["status"]


def assert_delivery_available() -> None:
    """The delivery worker is not closed or dead.

    A ``BaseException`` from any earlier sink write in this process leaves
    delivery unavailable for good; every later record would then be dropped
    and a cell would fail for a reason that is not its own.
    """
    worker = delivery._delivery
    assert not worker.closed, "log delivery was shut down earlier in this process"
    assert not worker.unavailable, "log delivery died earlier in this process"


@contextmanager
def sink_installed(sink: LogSink | None) -> Generator[None]:
    """Install *sink* (fresh health counters), restoring the previous sink."""
    previous = get_log_sink()
    configure_llm_logging(sink)
    try:
        yield
    finally:
        configure_llm_logging(previous)


def assert_nothing_lost(health: LoggingHealth) -> None:
    """No record was dropped, failed, or left in the queue."""
    assert health.dropped == 0, health
    assert health.unavailable == 0, health
    assert health.failed == 0, health
    assert health.degraded == 0, health
    assert health.preparation_failed == 0, health
    assert health.queued == 0 and health.in_flight == 0, health


async def settle() -> LoggingHealth:
    """Flush delivery, then prove nothing was lost; return the health snapshot."""
    assert await flush_llm_logs_async(timeout=5.0), "log delivery did not settle"
    health = get_logging_health()
    assert_nothing_lost(health)
    return health


def read_disk(log_dir: Path) -> list[DiskRecord]:
    """Every YAML record in *log_dir* paired with its ``index.jsonl`` line.

    Asserts the pairing is one to one: every file has exactly one index line
    and every index line names a file that exists.
    """
    files = sorted(log_dir.glob("*.yaml"))
    index_path = log_dir / INDEX_FILENAME
    lines = (
        [
            cast("dict[str, object]", json.loads(line))
            for line in index_path.read_text(encoding="utf-8").splitlines()
        ]
        if index_path.exists()
        else []
    )
    by_file = {cast("str", line["file"]): line for line in lines}
    assert len(by_file) == len(lines), "two index lines name one file"
    assert sorted(by_file) == [path.name for path in files], (sorted(by_file), files)
    records: list[DiskRecord] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        body = cast("object", yaml.safe_load(text))
        assert isinstance(body, dict), path
        records.append(
            DiskRecord(
                path=path,
                header=text.splitlines()[:2],
                body=cast("dict[str, object]", body),
                index=by_file[path.name],
            )
        )
    return records


def check_contract(record: DiskRecord) -> None:
    """What every record on disk must satisfy, whatever its cell."""
    body = record.body
    assert tuple(body) == YAML_KEYS, (record.path.name, set(body) ^ set(YAML_KEYS))
    assert isinstance(body["requests"], list), body["requests"]

    match = HEADER_LINE_1.match(record.header[0])
    assert match is not None, record.header[0]
    assert match["status"] == ("ERROR" if body["error"] else "ok")
    assert (match["feature"], match["label"]) == (body["feature"], body["label"] or "unlabeled")
    assert (match["model"], match["schema"]) == (body["model"], body["schema"])
    assert abs(int(match["ms"]) - cast("float", body["duration_ms"])) <= 1
    cost = body["approximate_cost"]
    assert match["cost"] == ("?" if cost is None else f"{cast('float', cost):.3g}")

    line_2 = (
        f"# {body['timestamp']} | call={cast('str', body['call_id'])[:8]} attempt={body['attempt']}"
    )
    if body["cache_hit"]:
        line_2 += " cache=hit"
    assert record.header[1] == line_2

    assert tuple(record.index) == INDEX_KEYS, set(record.index) ^ set(INDEX_KEYS)
    assert record.index["file"] == record.path.name
    for key in INDEX_KEYS[1:]:
        assert record.index[key] == body[key], (key, record.index[key], body[key])


def fake_answer(request: TransportRequest) -> Reply:
    """A clean answer for whichever family asks."""
    if request.kind == "structured":
        return Reply.json(OkSchema(ok=True))
    return Reply.text("ok")


async def call(
    family: Family,
    *,
    label: str,
    retry: RetryPolicy = NO_RETRY,
    on_result: Callable[[object], Awaitable[None] | None] | None = None,
    after_first: Callable[[], Awaitable[None]] | None = None,
) -> object:
    """Run one call on *family*; a stream is drained and its items returned.

    ``after_first`` runs after the first streamed item and then the stream is
    closed with ``aclose()``: the abandoned-stream shape.
    """
    if family == "structured":
        return await llm_calls.structured_llm_call(
            "hi", OkSchema, feature=FEATURE, label=label, retry=retry, on_result=on_result
        )
    if family == "text":
        return await llm_calls.text_llm_call(
            "hi", feature=FEATURE, label=label, retry=retry, on_result=on_result
        )
    assert on_result is None, f"{family} has no on_result hook"
    if family == "tool":
        return await llm_calls.tool_llm_call("hi", TOOLS, feature=FEATURE, label=label, retry=retry)
    stream = (
        llm_calls.text_llm_call_stream("hi", feature=FEATURE, label=label, retry=retry)
        if family == "text_stream"
        else llm_calls.tool_llm_call_stream("hi", TOOLS, feature=FEATURE, label=label, retry=retry)
    )
    items: list[object] = []
    if after_first is not None:
        items.append(await anext(stream))
        await after_first()
        await stream.aclose()
        return items
    items.extend([item async for item in stream])
    return items
