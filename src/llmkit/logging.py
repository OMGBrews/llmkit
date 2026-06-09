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

import enum
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("data/llm-logs")

# Bounded retry budget for the exclusive-create filename loop: a uuid4 suffix
# collision is astronomically unlikely, so a handful of attempts is plenty
# while still guaranteeing the loop terminates.
_MAX_FILENAME_ATTEMPTS = 8

# Matches a run of characters that are unsafe in a path component: path
# separators, control characters (incl. CR/LF), and other filesystem-hostile
# punctuation. Collapsed to a single "_" so names stay readable.
_UNSAFE_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f/\\<>:\"|?*]+")
# Strips the leading/trailing "_" and collapses any "." runs that remain so a
# value can never resolve to "." / ".." or a hidden traversal segment.
_DOT_RUN = re.compile(r"\.+")

# Matches control characters (CR, LF, tabs, NUL, etc.) for one-lining values
# interpolated into the "#" header comment, so a value can't forge a second
# comment line.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")


def _safe_path_component(value: str) -> str:
    """Neutralize *value* so it is safe as a single filename component.

    Path separators (``/``, ``\\``, :data:`os.sep`), control characters, and
    other filesystem-hostile punctuation are replaced with ``_``; ``.`` runs
    are collapsed to a single ``_`` so the result can never be ``.``/``..`` or
    a hidden traversal segment. The output stays human-readable and is
    guaranteed to contain no directory separators, so composing it into a path
    cannot escape ``log_dir``. Empty/all-unsafe input degrades to ``"_"``.
    """
    cleaned = _UNSAFE_PATH_CHARS.sub("_", value)
    cleaned = cleaned.replace(os.sep, "_")
    if os.altsep:
        cleaned = cleaned.replace(os.altsep, "_")
    cleaned = _DOT_RUN.sub("_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "_"


def _oneline(value: str) -> str:
    """Collapse CR/LF and other control characters in *value* to single spaces.

    Used on every caller-derived field interpolated into the ``#`` header
    comment so a value containing a newline cannot forge a second
    ``# ok | ...`` verdict line and corrupt the ``head -1`` triage.
    """
    return _CONTROL_CHARS.sub(" ", value).strip()


class _LogSafeDumper(yaml.SafeDumper):
    """:class:`yaml.SafeDumper` that degrades unknown objects to plain scalars.

    ``record.response`` is whatever the caller produced — typically a Pydantic
    ``model_dump()`` in python mode, which can carry Enum members, ``Decimal``,
    ``set``, datetime subclasses, and other arbitrary objects. The stock
    (unsafe) ``Dumper`` serializes those as ``!!python/object`` tags, which
    ``yaml.safe_load`` refuses to parse (breaking the documented
    safe-load-able analysis-tooling contract) and which are an
    arbitrary-code-execution hazard for anyone using full ``yaml.load``.

    This dumper keeps the standard types (str/int/float/bool/None/dict/list,
    dates, sets, …) exactly as ``SafeDumper`` renders them, and registers two
    fallbacks for everything else: an :class:`enum.Enum` member is rendered as
    its ``.value`` (the payload, not the Python identity), and any other
    unrepresentable object is rendered as ``str(obj)``. The log therefore
    always contains plain, safe-load-able YAML regardless of what the sink is
    fed.
    """


def _represent_enum(dumper: yaml.SafeDumper, data: enum.Enum) -> yaml.Node:
    """Render an Enum member as its underlying ``.value``."""
    return dumper.represent_data(data.value)  # pyright: ignore[reportAny, reportUnknownMemberType]  # raw-llm — Enum payload is arbitrary; yaml stubs leave represent_data untyped


def _represent_fallback(dumper: yaml.SafeDumper, data: object) -> yaml.Node:
    """Render any otherwise-unrepresentable object as a plain string scalar."""
    try:
        text = str(data)
    except Exception:
        # A hostile/broken __str__ must not break logging; object.__repr__
        # never raises.
        text = object.__repr__(data)
    return dumper.represent_str(text)


# Enum first: multi-representers match in registration order, and an Enum
# member is also an ``object``, so the generic fallback would shadow it.
_LogSafeDumper.add_multi_representer(enum.Enum, _represent_enum)
_LogSafeDumper.add_multi_representer(object, _represent_fallback)


# Compact append-only summary sibling to the per-call YAML files: one JSON
# line per call, so cross-call scans don't have to glob + parse every YAML.
INDEX_FILENAME = "index.jsonl"


@dataclass(frozen=True)
class LLMCallRecord:
    """A single LLM round-trip, as written to the log sink.

    ``model`` is the *resolved effective* model (the provider default
    substituted when the caller passed ``None``) and ``provider`` names
    the active provider, so cost attribution is a ``grep`` over the logs
    rather than a code trace. ``schema`` is the output schema name for a
    structured call, or the literal ``"text"`` / ``"stream"`` for a buffered
    or streamed plain-text call. ``response`` is the Pydantic-dumped result,
    the accumulated text, or ``None``.

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
    contract. Being ``@runtime_checkable``, the :func:`isinstance` test matches
    structurally — any sink exposing a ``write_returning_path`` method counts,
    so a third-party sink opts into path-capture purely by defining one (it
    *must* then return ``Path | None``). A sink that implements only
    :meth:`LogSink.write` does not match and is path-capture-invisible.
    """

    def write_returning_path(self, record: LLMCallRecord) -> Path | None: ...


class LocalYamlLogSink:
    """Default sink: one YAML file per call under ``log_dir``, plus a
    compact append-only ``index.jsonl`` summarising every call.

    The per-call YAML is laid out **verdict-first** — a one-line summary
    comment header (status / feature / model / schema / duration / cost),
    then the small metadata fields (including the request-shaping knobs:
    temperature, max_tokens, reasoning_effort), with the large ``response``
    and ``prompt`` blobs last — so a reader (a human, but in practice mostly
    a coding agent) learns what happened from the head of the file without
    paying to scan the whole prompt. The body is dumped with
    :class:`_LogSafeDumper`, so the file is always ``yaml.safe_load``-able —
    plain tags only, never ``!!python/object``. ``index.jsonl`` carries one
    short line per call (file, timestamp, feature, label, model, provider,
    schema, duration, cost, error) so cross-call questions — "which calls
    errored / were slowest / most expensive / the last call for feature X" —
    are a single small scan instead of globbing and parsing every YAML. The
    index is deliberately compact: per-call request knobs (temperature,
    max_tokens, reasoning_effort) live only in the per-call YAML.
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

        The filename is ``{timestamp}_{feature}_{label}_{uniquifier}.yaml``:
        the microsecond ``started_at`` stamp keeps the directory naturally
        sortable, ``feature``/``label`` are sanitized with
        :func:`_safe_path_component` so neither can escape ``log_dir``, and a
        short ``uuid4`` suffix plus exclusive-create (``mode="x"``) retry loop
        guarantees two records can never share a path or silently overwrite
        each other.
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts = record.started_at.strftime("%Y-%m-%dT%H-%M-%S-%f")
            safe_feature = _safe_path_component(record.feature)
            safe_label = _safe_path_component(record.label or "unlabeled")

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
                "max_tokens": record.max_tokens,
                "reasoning_effort": record.reasoning_effort,
                "duration_ms": round(record.duration_ms, 1),
                "approximate_cost": record.approximate_cost,
                "error": record.error,
                "response": cast("object", record.response),
                "prompt": record.prompt,
            }
            header = self._summary_header(record)

            # Exclusive-create (``mode="x"``) so a path collision raises
            # FileExistsError instead of truncating an existing log. The
            # uuid4 suffix makes a collision astronomically unlikely; the
            # bounded loop regenerates it on the off chance one happens, so
            # even then no record is ever overwritten.
            filepath: Path | None = None
            for _attempt in range(_MAX_FILENAME_ATTEMPTS):
                candidate = (
                    self.log_dir / f"{ts}_{safe_feature}_{safe_label}_{uuid.uuid4().hex[:8]}.yaml"
                )
                try:
                    with open(candidate, "x", encoding="utf-8") as f:
                        _ = f.write(header)
                        # _LogSafeDumper keeps the file safe_load-able: only
                        # plain YAML tags, never ``!!python/object`` (Enum
                        # members become their .value, anything else SafeDumper
                        # can't represent degrades to str(obj)).
                        yaml.dump(
                            doc,
                            f,
                            Dumper=_LogSafeDumper,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True,
                            width=120,
                        )
                except FileExistsError:
                    # Suffix collision — regenerate and retry; never overwrite.
                    continue
                except (OSError, yaml.YAMLError, UnicodeError):
                    # A mid-write failure (disk full, or a residual encode error
                    # such as un-encodable surrogates even on the utf-8 stream;
                    # representer errors are designed out by _LogSafeDumper's
                    # str() fallback, but YAMLError stays guarded for emitter
                    # edge cases) leaves a truncated file behind under the
                    # exclusive-create name. Remove the orphan before degrading so the log dir
                    # never accumulates empty/partial YAML, then re-raise to the
                    # best-effort handler below.
                    candidate.unlink(missing_ok=True)
                    raise
                filepath = candidate
                break
            if filepath is None:
                # Exhausted the retry budget without a free name (should be
                # unreachable in practice). Best-effort: skip the write.
                raise OSError(
                    f"could not allocate a unique log filename after {_MAX_FILENAME_ATTEMPTS} attempts"
                )
        except (OSError, yaml.YAMLError, UnicodeError):
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

        Every caller-derived field (feature, label, model, schema) is passed
        through :func:`_oneline` so a value containing a newline cannot forge
        a second ``# ok | ...`` verdict line and corrupt that triage. The
        second line is the ISO ``started_at`` stamp, which is machine-built
        and contains no newlines.
        """
        status = "ERROR" if record.error else "ok"
        cost = f"${record.approximate_cost:.3g}" if record.approximate_cost is not None else "$?"
        feature = _oneline(record.feature)
        label = _oneline(record.label or "unlabeled")
        model = _oneline(record.model or "?")
        schema = _oneline(record.schema)
        return (
            f"# {status} | {feature}/{label} | "
            f"{model} | {schema} | "
            f"{round(record.duration_ms)}ms | {cost}\n"
            f"# {record.started_at.isoformat()}\n\n"
        )

    def _append_index(self, record: LLMCallRecord, filepath: Path) -> None:
        """Append one compact JSON line for *record* to ``index.jsonl``.

        Best-effort and swallowed on failure (logging must never break the
        call). A single ``write`` of a sub-4KB line under ``O_APPEND`` is
        atomic on POSIX, so concurrent calls don't interleave lines.

        The line carries only the cross-call triage fields; request-shaping
        knobs (temperature, max_tokens, reasoning_effort) are deliberately
        omitted to keep the index compact — they live in the per-call YAML.
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
