"""The shipped default sink: one YAML file per call, plus an index line.

Stays one class. Its ten methods share the frozen-once log directory, the
announce latch, the prune throttle and three warn-once latches, so carving
write / index / retention apart would relocate the coupling rather than reduce
it.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from llmkit.logging._latch import OnceLatch
from llmkit.logging._paths import (
    MAX_FILENAME_ATTEMPTS,
    oneline,
    open_private,
    resolve_default_log_dir,
    safe_path_component,
)
from llmkit.logging._yaml import LogSafeDumper
from llmkit.logging.record import INDEX_FILENAME, LLMCallRecord

# Named explicitly rather than via ``__name__`` so every module in this package
# keeps emitting under the one logger name operators already filter on.
logger = logging.getLogger("llmkit.logging")


#: Default age bound for the per-call YAML files (and rotated index
#: generations): files older than this many days are pruned. ``None`` on the
#: sink keeps everything forever.
DEFAULT_RETENTION_DAYS = 30

#: Default size bound for the active ``index.jsonl``: past this many bytes it
#: is rotated to a date-stamped sibling (which then ages out under the same
#: retention policy). ``None`` on the sink disables rotation.
DEFAULT_MAX_INDEX_BYTES = 50 * 2**20

# Retention housekeeping runs at most this often per sink instance, so the
# per-write cost of a bounded log dir stays one monotonic-clock read.
_PRUNE_INTERVAL_SECONDS = 3600.0


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
    :class:`LogSafeDumper`, so the file is always ``yaml.safe_load``-able —
    plain tags only, never ``!!python/object``. ``index.jsonl`` carries one
    short line per call (file, timestamp, feature, label, model, provider,
    schema, duration, cost, error) so cross-call questions — "which calls
    errored / were slowest / most expensive / the last call for feature X" —
    are a single small scan instead of globbing and parsing every YAML. The
    index is deliberately compact: per-call request knobs (temperature,
    max_tokens, reasoning_effort) live only in the per-call YAML.

    ``log_dir=None`` (the default) resolves via :func:`default_log_dir` at
    the first write and freezes the answer for the sink's lifetime, so a
    mid-run ``chdir`` cannot split logs across directories; an explicit path
    is frozen the same way, made absolute at construction (symlinks intact)
    so a *relative* one names one directory forever rather than following
    the process around. The first successful write emits one INFO naming the
    absolute directory and the retention policy — before anything is ever
    pruned. When the sink itself creates the directory it is ``0o700`` and
    files are ``0o600`` (a pre-existing directory is never re-chmodded —
    pre-create it to share logs with other readers), and the project-root
    default location is seeded with a ``.gitignore`` so prompt logs cannot
    land in the enclosing repository's history.

    Growth is bounded by default: per-call YAML files older than
    ``retention_days`` (default 30; ``None`` keeps forever) are pruned, and
    an ``index.jsonl`` past ``max_index_bytes`` (default 50 MiB; ``None``
    never rotates) is rotated to a date-stamped sibling that ages out under
    the same policy. Housekeeping runs at most hourly, on the write path.
    **Pruning is a glob over the directory, not a list of files the sink
    remembers writing**: the sink owns ``log_dir``'s ``*.yaml`` and
    ``index-*.jsonl`` namespace, so a co-located file of either shape ages
    out under the same policy whoever wrote it — deliberate (a
    cross-reference file and the logs it references rot on one clock), but
    it makes ``log_dir`` llmkit's directory rather than shared storage.
    Give llmkit a directory of its own if you need co-located files to
    outlive the policy.

    Both bounds take a *positive* value or ``None``; ``0`` and negatives are
    rejected at construction rather than given one of their two plausible
    meanings (see :meth:`__init__`).
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        retention_days: int | None = DEFAULT_RETENTION_DAYS,
        max_index_bytes: int | None = DEFAULT_MAX_INDEX_BYTES,
    ) -> None:
        """Configure the sink, validating the growth bounds eagerly.

        An explicit ``log_dir`` is made absolute here so the directory the
        sink freezes cannot move under a later ``chdir``.

        Both bounds stay plain public attributes afterwards: this validates
        *construction*, so assigning ``sink.retention_days = 0`` later reopens
        the hole. Configure the sink once, at startup, as the module docstring
        describes.

        Raises:
            ValueError: if ``retention_days`` or ``max_index_bytes`` is set to
                a non-positive value. ``0`` is rejected rather than read as
                "unlimited": the prune cutoff would be *now*, so the first
                write deletes every log in the directory — including the file
                that write just produced, whose path it still returns.
                ``None`` is the opt-out for both (keep forever / never
                rotate). Validation lives in the constructor, not the write
                path, because a host's configuration mistake should be a loud
                one-line fix at startup — the write path is best-effort by
                contract and would degrade this to a warning.
            TypeError: if ``log_dir`` is neither a :class:`~pathlib.Path` nor
                ``None``. A ``str`` would otherwise fail deep in the first
                write, where the best-effort swallow turns "every log is
                lost" into one warning.
        """
        if retention_days is not None and retention_days < 1:
            raise ValueError(
                "retention_days must be an integer >= 1 or None "
                + f"(None keeps every log forever), got {retention_days!r}"
            )
        if max_index_bytes is not None and max_index_bytes < 1:
            raise ValueError(
                "max_index_bytes must be an integer >= 1 or None "
                + f"(None never rotates index.jsonl), got {max_index_bytes!r}"
            )
        # Deliberate runtime guard at a public boundary, like the one in
        # configure_llm_logging: the annotation says Path, but an untyped
        # caller's "logs" string would only surface at the first write.
        if log_dir is not None and not isinstance(log_dir, Path):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard at public boundary
            raise TypeError(  # pyright: ignore[reportUnreachable]  # reachable from untyped callers
                f"log_dir must be a Path or None, got {type(log_dir).__name__}"
            )
        self._log_dir: Path | None = None if log_dir is None else log_dir.absolute()
        self.retention_days: int | None = retention_days
        self.max_index_bytes: int | None = max_index_bytes
        # True once the default resolution chose the project-root location —
        # the one case where the sink seeds a .gitignore on dir creation.
        self._seed_gitignore: bool = False
        self._announced: bool = False
        self._last_prune: float | None = None
        self._yaml_latch: OnceLatch = OnceLatch()
        self._index_latch: OnceLatch = OnceLatch()
        self._prune_latch: OnceLatch = OnceLatch()

    @property
    def log_dir(self) -> Path:
        """The directory this sink writes to.

        A sink constructed without an explicit ``log_dir`` resolves
        :func:`default_log_dir` on first access and freezes the result, so
        every write (and every reader of this property) sees one stable
        directory regardless of later ``chdir`` or environment changes.
        """
        if self._log_dir is None:
            self._log_dir, self._seed_gitignore = resolve_default_log_dir()
        return self._log_dir

    def _ensure_log_dir(self) -> None:
        """Create ``log_dir`` (``0o700``) if absent; seed ``.gitignore`` when owed.

        Creation is detected via the ``FileExistsError`` branch rather than
        ``exist_ok=True`` so a pre-existing (possibly user-shared) directory
        is never re-chmodded and never seeded. Only the leaf gets ``0o700``;
        the ``.gitignore`` seed applies only to the project-root *default*
        location — an explicit path or env override is the caller's choice,
        and the state-dir fallback is never inside a repository.
        """
        try:
            self.log_dir.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            return
        if self._seed_gitignore:
            try:
                with open(self.log_dir / ".gitignore", "x", encoding="utf-8") as f:
                    _ = f.write("*\n")
            except Exception:
                # Best-effort: an unseedable .gitignore must not break the
                # write that triggered the mkdir.
                logger.debug("Could not seed .gitignore in %s", self.log_dir, exc_info=True)

    def _announce_once(self) -> None:
        """One INFO after the first successful write: where the logs are.

        Persistence is on by default, so the location (and the retention
        policy, *before* the first prune could ever delete anything) must be
        discoverable without reading the docs.
        """
        if self._announced:
            return
        self._announced = True
        retention = (
            f"{self.retention_days}-day retention"
            if self.retention_days is not None
            else "no retention (files kept forever)"
        )
        logger.info(
            "llmkit is logging LLM calls to %s (%s; configure_llm_logging(None) disables)",
            self.log_dir.resolve(),
            retention,
        )

    def write(self, record: LLMCallRecord) -> None:
        """Persist *record* as a YAML file (the :class:`LogSink` contract).

        Returns nothing, per the shared sink contract. Callers that need
        the written path use :meth:`write_returning_path` (or the
        :func:`~llmkit.capture.capture_llm_log_paths` context
        manager, which calls it for the configured file sink).
        """
        _ = self.write_returning_path(record)

    def write_returning_path(self, record: LLMCallRecord) -> Path | None:
        """Persist *record* and return the file path it wrote (or ``None``).

        The file-specific counterpart to :meth:`write`: the per-call YAML
        path is returned so the path-capture primitive
        (:func:`~llmkit.capture.capture_llm_log_paths`) can
        cross-reference it, without that file detail leaking into the shared
        :class:`LogSink` contract. ``None`` is returned when the write
        failed (best-effort: logging must never break the LLM call).

        The filename is ``{timestamp}_{feature}_{label}_{uniquifier}.yaml``:
        the microsecond ``started_at`` stamp keeps the directory naturally
        sortable, ``feature``/``label`` are sanitized and length-clamped with
        :func:`safe_path_component` so neither can escape ``log_dir`` nor
        push the name past the filesystem's component limit, and a short
        ``uuid4`` suffix plus exclusive-create (``mode="x"``) retry loop
        guarantees two records can never share a path or silently overwrite
        each other.

        Best-effort means *any* :class:`Exception` is swallowed (with a
        warning); a failure after the file was exclusively created also
        removes the truncated orphan, so the log dir never accumulates
        partial YAML.
        """
        try:
            self._ensure_log_dir()
            ts = record.started_at.strftime("%Y-%m-%dT%H-%M-%S-%f")
            safe_feature = safe_path_component(record.feature)
            safe_label = safe_path_component(record.label or "unlabeled")

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
                "run_id": record.run_id,
                "call_id": record.call_id,
                "attempt": record.attempt,
                "temperature": record.temperature,
                "max_tokens": record.max_tokens,
                "reasoning_effort": record.reasoning_effort,
                "tools": record.tools,
                "tool_calls": record.tool_calls,
                "usage": record.usage,
                "duration_ms": round(record.duration_ms, 1),
                "queue_wait_ms": (
                    round(record.queue_wait_ms, 1) if record.queue_wait_ms is not None else None
                ),
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
            for _attempt in range(MAX_FILENAME_ATTEMPTS):
                candidate = (
                    self.log_dir / f"{ts}_{safe_feature}_{safe_label}_{uuid.uuid4().hex[:8]}.yaml"
                )
                try:
                    with open(candidate, "x", encoding="utf-8", opener=open_private) as f:
                        _ = f.write(header)
                        # LogSafeDumper keeps the file safe_load-able: only
                        # plain YAML tags, never ``!!python/object`` (Enum
                        # members become their .value, anything else SafeDumper
                        # can't represent degrades to str(obj)).
                        yaml.dump(
                            doc,
                            f,
                            Dumper=LogSafeDumper,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True,
                            width=120,
                        )
                except FileExistsError:
                    # Suffix collision — regenerate and retry; never overwrite.
                    continue
                except Exception:
                    # Any mid-write failure (disk full, un-encodable surrogates
                    # even on the utf-8 stream, a RecursionError from a deeply
                    # nested/cyclic payload, an exotic representer/emitter
                    # error) leaves a truncated file behind under the
                    # exclusive-create name. Remove the orphan before degrading
                    # so the log dir never accumulates empty/partial YAML, then
                    # re-raise to the best-effort handler below. Deliberately
                    # broad — this sink is best-effort by contract — but never
                    # BaseException, so KeyboardInterrupt/SystemExit still
                    # propagate (the orphan cleanup is forfeited for those).
                    candidate.unlink(missing_ok=True)
                    raise
                filepath = candidate
                break
            if filepath is None:
                # Exhausted the retry budget without a free name (should be
                # unreachable in practice). Best-effort: skip the write.
                raise OSError(
                    f"could not allocate a unique log filename after {MAX_FILENAME_ATTEMPTS} attempts"
                )
        except Exception as exc:
            # Best-effort by contract: *any* failure (not just the common
            # OSError/YAMLError/UnicodeError cases) degrades to a warning so
            # logging can never break the LLM call. Never BaseException —
            # KeyboardInterrupt/SystemExit must propagate. The latch keeps a
            # *persistently* broken sink from flooding stderr with one
            # traceback per call: the first failure (and any new failure
            # signature) warns, repeats drop to DEBUG.
            if self._yaml_latch.should_warn(exc):
                logger.warning(
                    "Failed to write LLM invocation log for %s/%s (repeats logged at DEBUG)",
                    record.feature,
                    record.label,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "Failed to write LLM invocation log for %s/%s",
                    record.feature,
                    record.label,
                    exc_info=True,
                )
            return None

        self._yaml_latch.succeeded()
        self._announce_once()
        # Best-effort index append, kept separate so an index failure can
        # never lose the per-call record that was just written successfully.
        self._append_index(record, filepath)
        self._maybe_prune()
        return filepath

    @staticmethod
    def _summary_header(record: LLMCallRecord) -> str:
        """Build the two-line ``#`` comment that opens each per-call YAML.

        The first line is a single-glance verdict — ``ok``/``ERROR``,
        feature/label, resolved model, schema, duration, approximate cost —
        so ``head -1`` across the directory triages a whole run.

        Every caller-derived field (feature, label, model, schema) is passed
        through :func:`oneline` so a value containing a newline cannot forge
        a second ``# ok | ...`` verdict line and corrupt that triage. The
        second line is the ISO ``started_at`` stamp — machine-built, no
        newlines — plus, when the record carries correlation fields, a
        ``call=<id[:8]> attempt=<n>`` suffix so retries of one logical call
        are joinable from the file heads alone. The first line's shape is
        pinned (``head -1`` tooling greps it); only line 2 gains the suffix.
        """
        status = "ERROR" if record.error else "ok"
        cost = f"${record.approximate_cost:.3g}" if record.approximate_cost is not None else "$?"
        feature = oneline(record.feature)
        label = oneline(record.label or "unlabeled")
        model = oneline(record.model or "?")
        schema = oneline(record.schema)
        correlation = ""
        if record.call_id is not None:
            correlation = f" | call={oneline(record.call_id)[:8]}"
            if record.attempt is not None:
                correlation += f" attempt={record.attempt}"
        return (
            f"# {status} | {feature}/{label} | "
            f"{model} | {schema} | "
            f"{round(record.duration_ms)}ms | {cost}\n"
            f"# {record.started_at.isoformat()}{correlation}\n\n"
        )

    def _append_index(self, record: LLMCallRecord, filepath: Path) -> None:
        """Append one compact JSON line for *record* to ``index.jsonl``.

        Best-effort: *any* :class:`Exception` is swallowed with a warning
        (logging must never break the call). The dataclass doesn't enforce
        its field types, so a directly-constructed record can make
        ``json.dumps`` raise ``TypeError``/``ValueError`` (or ``round`` raise
        on a non-numeric ``duration_ms``) — those must not escape here, where
        they would discard the per-call YAML path that was already written.
        A single ``write`` of a sub-4KB line under ``O_APPEND`` is atomic on
        POSIX, so concurrent calls don't interleave lines.

        The line carries only the cross-call triage fields; request-shaping
        knobs (temperature, max_tokens, reasoning_effort) are deliberately
        omitted to keep the index compact — they live in the per-call YAML.
        """
        try:
            line: dict[str, str | float | None] = {
                "file": filepath.name,
                "timestamp": record.started_at.isoformat(),
                "feature": record.feature,
                "label": record.label,
                "model": record.model,
                "provider": record.provider,
                "schema": record.schema,
                "run_id": record.run_id,
                "call_id": record.call_id,
                "attempt": record.attempt,
                "duration_ms": round(record.duration_ms, 1),
                "queue_wait_ms": (
                    round(record.queue_wait_ms, 1) if record.queue_wait_ms is not None else None
                ),
                "approximate_cost": record.approximate_cost,
                "error": record.error,
            }
            # Serialize before opening so a serialization failure can't even
            # create/touch the index file.
            payload = json.dumps(line, ensure_ascii=False) + "\n"
            with open(
                self.log_dir / INDEX_FILENAME, "a", encoding="utf-8", opener=open_private
            ) as f:
                _ = f.write(payload)
        except Exception as exc:
            if self._index_latch.should_warn(exc):
                logger.warning(
                    "Failed to append LLM log index for %s/%s (repeats logged at DEBUG)",
                    record.feature,
                    record.label,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "Failed to append LLM log index for %s/%s",
                    record.feature,
                    record.label,
                    exc_info=True,
                )
        else:
            self._index_latch.succeeded()

    def _maybe_prune(self) -> None:
        """Run retention housekeeping, throttled to once per hour per sink.

        Called from the write path (so a long-running service is bounded, not
        just a restarting one) but rate-limited to a monotonic-clock check so
        the steady-state per-write cost is nil. Best-effort like everything
        else here: a prune failure warns (latched) and never breaks the call.
        """
        if self.retention_days is None and self.max_index_bytes is None:
            return
        now = time.monotonic()
        if self._last_prune is not None and now - self._last_prune < _PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        try:
            self._prune()
        except Exception as exc:
            if self._prune_latch.should_warn(exc):
                logger.warning(
                    "Failed to prune LLM log dir %s (further identical failures logged at DEBUG)",
                    self.log_dir,
                    exc_info=True,
                )
            else:
                logger.debug("Failed to prune LLM log dir %s", self.log_dir, exc_info=True)
        else:
            self._prune_latch.succeeded()

    def _prune(self) -> None:
        """Apply the retention policy: age out YAMLs, rotate an oversized index.

        Age pruning covers the per-call ``*.yaml`` files and previously
        rotated ``index-*.jsonl`` generations, keyed on file mtime so tests
        (and operators) can reason about it with ``touch``. The patterns are
        globs over the directory, so this owns the whole ``*.yaml`` /
        ``index-*.jsonl`` namespace in ``log_dir`` and sweeps a co-located
        foreign file of either shape too — the documented contract, not an
        oversight (the active ``index.jsonl`` is excluded: the pattern
        requires the rotation hyphen). Rotation renames the active index with
        :func:`os.replace` — atomic on POSIX, and a concurrent writer holding
        the old file descriptor simply finishes its ``O_APPEND`` line into the
        rotated file, so no index line is ever torn or lost. A file vanishing
        mid-scan (a concurrent prune, a manual cleanup) is skipped, not an
        error.
        """
        if self.retention_days is not None:
            cutoff = time.time() - self.retention_days * 86400
            for pattern in ("*.yaml", "index-*.jsonl"):
                for stale in self.log_dir.glob(pattern):
                    try:
                        if stale.stat().st_mtime < cutoff:
                            stale.unlink(missing_ok=True)
                    except OSError:
                        continue
        if self.max_index_bytes is not None:
            index = self.log_dir / INDEX_FILENAME
            try:
                size = index.stat().st_size
            except OSError:
                return
            if size > self.max_index_bytes:
                stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S-%f")
                os.replace(index, self.log_dir / f"index-{stamp}.jsonl")
