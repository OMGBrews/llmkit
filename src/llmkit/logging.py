"""Per-call LLM invocation logging via a pluggable sink.

Every LLM round-trip is recorded as an :class:`LLMCallRecord` and handed
to the configured :class:`LogSink`. The default sink writes one YAML file
per call to a directory (``data/llm-logs/`` by default), preserving the
historical log shape so existing analysis tooling keeps working.

Logging is unconditional and best-effort — a sink that raises is swallowed
so the LLM call itself never breaks because logging did. The host
application points the sink at its chosen directory once at startup via
:func:`configure_llm_logging`, mirroring the ``configure_rate_limit``
module-level pattern.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("data/llm-logs")

# Compact append-only summary sibling to the per-call YAML files: one JSON
# line per call, so cross-call scans don't have to glob + parse every YAML.
INDEX_FILENAME = "index.jsonl"


@dataclass(frozen=True)
class LLMCallRecord:
    """A single LLM round-trip, as written to the log sink.

    ``model`` is the *resolved effective* model (the provider default
    substituted when the caller passed ``None``) and ``provider`` names
    the active provider, so cost attribution is a ``grep`` over the logs
    rather than a code trace. ``schema`` is the output schema name, or the
    literal ``"stream"`` for streamed plain-text calls. ``response`` is the
    Pydantic-dumped result, the accumulated stream text, or ``None``.

    ``approximate_cost`` is a best-effort USD estimate for budget
    visibility — NOT a billing figure. It is sourced from LiteLLM's
    per-response cost (no local price table) and is ``None`` when the
    provider does not report it (e.g. streamed calls).
    """

    started_at: datetime
    feature: str
    label: str | None
    model: str | None
    provider: str | None
    temperature: float
    duration_ms: float
    schema: str
    prompt: str | list[dict[str, str]]
    response: Any  # pyright: ignore[reportExplicitAny]  # raw-llm — Pydantic dump or accumulated text
    error: str | None
    approximate_cost: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None


class LogSink(Protocol):
    """Destination for :class:`LLMCallRecord`s.

    ``write`` consumes a record and returns nothing. A sink that persists
    to a file (e.g. :class:`LocalYamlLogSink`) exposes the written path via
    its own method/attribute — the shared contract stays file-agnostic so a
    third-party sink (a database, an in-memory buffer, a metrics pipe) is a
    one-method object. To capture per-call *records* (cost/metadata) or
    *file paths* without authoring a sink, use
    :func:`~llmkit.structured_output.capture_llm_records` or
    :func:`~llmkit.structured_output.capture_llm_log_paths`.
    """

    def write(self, record: LLMCallRecord) -> None: ...


@runtime_checkable
class _PathReturningLogSink(Protocol):
    """A :class:`LogSink` that also exposes the file path it wrote.

    Internal capability protocol: a file-backed sink (e.g.
    :class:`LocalYamlLogSink`) advertises ``write_returning_path`` so
    :func:`write_llm_log` can hand the path to the path-capture primitive
    without that file detail leaking into the public :class:`LogSink`
    contract. Third-party sinks implement only :class:`LogSink` and never
    match this check.
    """

    def write_returning_path(self, record: LLMCallRecord) -> Path | None: ...


class LocalYamlLogSink:
    """Default sink: one YAML file per call under ``log_dir``, plus a
    compact append-only ``index.jsonl`` summarising every call.

    The per-call YAML is laid out **verdict-first** — a one-line summary
    comment header (status / feature / model / schema / duration / cost),
    then the small metadata fields, with the large ``response`` and
    ``prompt`` blobs last — so a reader (a human, but in practice mostly a
    coding agent) learns what happened from the head of the file without
    paying to scan the whole prompt. ``index.jsonl`` carries one short
    line per call (file, timestamp, feature, label, model, schema,
    duration, cost, error) so cross-call questions — "which calls errored
    / were slowest / most expensive / the last call for feature X" — are a
    single small scan instead of globbing and parsing every YAML.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        self.log_dir: Path = log_dir

    def write(self, record: LLMCallRecord) -> None:
        """Persist *record* as a YAML file (the :class:`LogSink` contract).

        Returns nothing, per the shared sink contract. Callers that need
        the written path use :meth:`write_returning_path` (or the
        :func:`~llmkit.structured_output.capture_llm_log_paths` context
        manager, which calls it for the configured file sink).
        """
        _ = self.write_returning_path(record)

    def write_returning_path(self, record: LLMCallRecord) -> Path | None:
        """Persist *record* and return the file path it wrote (or ``None``).

        The file-specific counterpart to :meth:`write`: the per-call YAML
        path is returned so the path-capture primitive
        (:func:`~llmkit.structured_output.capture_llm_log_paths`) can
        cross-reference it, without that file detail leaking into the shared
        :class:`LogSink` contract. ``None`` is returned when the write
        failed (best-effort: logging must never break the LLM call).
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts = record.started_at.strftime("%Y-%m-%dT%H-%M-%S-%f")
            safe_label = (record.label or "unlabeled").replace(".", "_").replace("/", "_")
            filepath = self.log_dir / f"{ts}_{record.feature}_{safe_label}.yaml"

            # Verdict-first order: cheap, high-signal metadata up top; the
            # large ``response``/``prompt`` blobs last (``response`` first —
            # it's what a debugger usually wants), so the head of the file
            # is the whole story for most reads.
            doc: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]  # raw-llm — YAML log body dict
                "timestamp": record.started_at.isoformat(),
                "feature": record.feature,
                "label": record.label,
                "model": record.model,
                "provider": record.provider,
                "schema": record.schema,
                "temperature": record.temperature,
                "duration_ms": round(record.duration_ms, 1),
                "approximate_cost": record.approximate_cost,
                "error": record.error,
                "response": cast("object", record.response),
                "prompt": record.prompt,
            }

            with open(filepath, "w") as f:
                _ = f.write(self._summary_header(record))
                yaml.dump(
                    doc,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=120,
                )
        except (OSError, yaml.YAMLError):
            logger.warning(
                "Failed to write LLM invocation log for %s/%s",
                record.feature,
                record.label,
                exc_info=True,
            )
            return None

        # Best-effort index append, kept separate so an index failure can
        # never lose the per-call record that was just written successfully.
        self._append_index(record, filepath)
        return filepath

    @staticmethod
    def _summary_header(record: LLMCallRecord) -> str:
        """Build the two-line ``#`` comment that opens each per-call YAML.

        The first line is a single-glance verdict — ``ok``/``ERROR``,
        feature/label, resolved model, schema, duration, approximate cost —
        so ``head -1`` across the directory triages a whole run.
        """
        status = "ERROR" if record.error else "ok"
        cost = f"${record.approximate_cost:.3g}" if record.approximate_cost is not None else "$?"
        return (
            f"# {status} | {record.feature}/{record.label or 'unlabeled'} | "
            f"{record.model or '?'} | {record.schema} | "
            f"{round(record.duration_ms)}ms | {cost}\n"
            f"# {record.started_at.isoformat()}\n\n"
        )

    def _append_index(self, record: LLMCallRecord, filepath: Path) -> None:
        """Append one compact JSON line for *record* to ``index.jsonl``.

        Best-effort and swallowed on failure (logging must never break the
        call). A single ``write`` of a sub-4KB line under ``O_APPEND`` is
        atomic on POSIX, so concurrent calls don't interleave lines.
        """
        line: dict[str, str | float | None] = {
            "file": filepath.name,
            "timestamp": record.started_at.isoformat(),
            "feature": record.feature,
            "label": record.label,
            "model": record.model,
            "provider": record.provider,
            "schema": record.schema,
            "duration_ms": round(record.duration_ms, 1),
            "approximate_cost": record.approximate_cost,
            "error": record.error,
        }
        try:
            with open(self.log_dir / INDEX_FILENAME, "a", encoding="utf-8") as f:
                _ = f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning(
                "Failed to append LLM log index for %s/%s",
                record.feature,
                record.label,
                exc_info=True,
            )


# Module-level configured sink, defaulting to the local-YAML sink at the
# default directory. The host overrides it once at startup; tests typically
# point it at a tmp directory.
_sink: LogSink | None = LocalYamlLogSink()


def configure_llm_logging(sink: LogSink | None) -> None:
    """Set the sink that receives every :class:`LLMCallRecord`.

    Pass ``None`` to disable logging entirely (writes become no-ops).
    """
    global _sink
    _sink = sink


def write_llm_log(record: LLMCallRecord) -> Path | None:
    """Hand ``record`` to the configured sink, swallowing any failure.

    Logging must never break the LLM call, so a sink that raises is
    caught here in addition to the sink's own best-effort handling.

    Returns the written file path when the configured sink is a
    file sink that exposes one (it advertises a ``write_returning_path``
    method, as :class:`LocalYamlLogSink` does), so the
    :func:`~llmkit.structured_output.capture_llm_log_paths` primitive can
    cross-reference it. For a third-party sink that only implements the
    file-agnostic :class:`LogSink` contract (``write(record) -> None``),
    there is no path to return, so this returns ``None`` — path-capture is
    simply empty for such sinks, while
    :func:`~llmkit.structured_output.capture_llm_records` still captures
    the record itself.
    """
    if _sink is None:
        return None
    try:
        if isinstance(_sink, _PathReturningLogSink):
            return _sink.write_returning_path(record)
        _sink.write(record)
        return None
    except Exception:
        logger.warning("LLM log sink raised for %s/%s", record.feature, record.label, exc_info=True)
        return None
