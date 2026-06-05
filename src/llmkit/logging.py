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
from typing import Any, Protocol

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


class LogSink(Protocol):
    """Destination for :class:`LLMCallRecord`s.

    ``write`` returns the path it wrote (so callers tracking log paths can
    cross-reference), or ``None`` if nothing was persisted.
    """

    def write(self, record: LLMCallRecord) -> Path | None: ...


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
        self.log_dir = log_dir

    def write(self, record: LLMCallRecord) -> Path | None:
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
                "response": record.response,
                "prompt": record.prompt,
            }

            with open(filepath, "w") as f:
                f.write(self._summary_header(record))
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
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
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
    Returns the written path, or ``None`` when nothing was persisted.
    """
    if _sink is None:
        return None
    try:
        return _sink.write(record)
    except Exception:
        logger.warning("LLM log sink raised for %s/%s", record.feature, record.label, exc_info=True)
        return None
