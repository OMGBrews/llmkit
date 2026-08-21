"""Per-call LLM invocation logging via a pluggable sink.

Every LLM round-trip is recorded as an :class:`LLMCallRecord` and handed
to the configured :class:`LogSink`. The default sink writes one YAML file
per call to a directory resolved lazily at first write (see
:func:`default_log_dir`: ``LLMKIT_LOG_DIR``, else ``data/llm-logs/`` under
the enclosing project root, else a per-user state directory), preserving
the historical log shape so existing analysis tooling keeps working.

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
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import yaml

from llmkit._types import Message, ReasoningEffort

logger = logging.getLogger(__name__)

#: Environment variable overriding the default log directory. Read lazily at
#: the sink's first write (never at import), so setting it after ``import
#: llmkit`` still takes effect.
LOG_DIR_ENV_VAR = "LLMKIT_LOG_DIR"

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

#: Marker files whose presence makes a directory a "project root" for
#: :func:`default_log_dir`'s upward walk.
_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


def _find_project_root(start: Path) -> Path | None:
    """The nearest ancestor of *start* (inclusive) carrying a project marker.

    Walks upward from *start* and returns the first directory containing one
    of :data:`_PROJECT_ROOT_MARKERS` — nearest wins, so a nested project logs
    under its own root, not the enclosing monorepo's. ``None`` when no marker
    exists anywhere up the tree (the process is not running inside a project).
    """
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            return candidate
    return None


def _user_state_log_dir() -> Path:
    """The per-user state directory for llmkit logs on this platform.

    Logs are *state* (regenerable, machine-local), so the Linux bucket is
    ``$XDG_STATE_HOME`` (default ``~/.local/state``), not the data dir. macOS
    and Windows use their platform log locations.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "llmkit"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "llmkit" / "logs"
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return base / "llmkit" / "llm-logs"


def _resolve_default_log_dir() -> tuple[Path, bool]:
    """Resolve the default log directory; the bool marks the project-root case.

    Resolution order: :data:`LOG_DIR_ENV_VAR` if set, else ``data/llm-logs``
    under the nearest project root above the current directory, else the
    per-user state directory. Only the project-root case (the one where a
    repository could accidentally swallow prompt logs) returns ``True``, which
    makes the sink seed a ``.gitignore`` when it creates that directory.

    Every branch returns an absolute path — a *relative* ``LLMKIT_LOG_DIR``
    (or ``XDG_STATE_HOME`` / ``LOCALAPPDATA``, which the state-dir fallback
    reads verbatim) would otherwise be re-anchored to the process's current
    directory on each use, which is exactly the chdir-splits-the-logs failure
    the sink freezes its answer to prevent. Absolute, not resolved: symlinks
    are left intact, since the frozen answer only has to be stable, not
    canonical.
    """
    env_dir = os.environ.get(LOG_DIR_ENV_VAR)
    if env_dir:
        return Path(env_dir).absolute(), False
    root = _find_project_root(Path.cwd())
    if root is not None:
        # Already absolute: rooted at Path.cwd().
        return root / "data" / "llm-logs", True
    return _user_state_log_dir().absolute(), False


def default_log_dir() -> Path:
    """Compute where :class:`LocalYamlLogSink` writes when no ``log_dir`` is given.

    ``LLMKIT_LOG_DIR`` wins when set; otherwise ``data/llm-logs/`` under the
    nearest ancestor directory carrying a ``pyproject.toml`` or ``.git``
    (nearest wins); otherwise a per-user state directory
    (``$XDG_STATE_HOME/llmkit/llm-logs`` on Linux, ``~/Library/Logs/llmkit``
    on macOS, ``%LOCALAPPDATA%\\llmkit\\logs`` on Windows). Computed from the
    *current* environment and working directory on every call; the default
    sink calls it once at first write and freezes the answer, so a later
    ``chdir`` cannot split one process's logs across directories.
    """
    return _resolve_default_log_dir()[0]


def _open_private(path: str, flags: int) -> int:
    """``opener=`` hook: create files ``0o600`` (owner-only) instead of umask-default.

    The per-call YAML and the index carry full prompts and responses, which
    ``SECURITY.md`` promises are not world-readable on a multi-user host. The
    mode only applies at creation — POSIX ``open`` ignores it for an existing
    file — and is inert on Windows.
    """
    return os.open(path, flags, 0o600)


class _OnceLatch:
    """Warn-once state for one failure site: WARNING on a new failure
    signature, DEBUG on repeats, re-armed by success.

    A permanently broken sink (unwritable directory, full disk) would
    otherwise emit a warning **with traceback** on every call — flooding
    stderr at exactly the moment the application is busiest. The signature is
    ``(type, errno)``, so a *different* failure (disk full after permission
    denied) still warns loudly instead of hiding behind the first one.
    Instances are per-site and unlocked: a race between two writers costs at
    most one duplicate warning, which is not worth a lock on the hot path.
    """

    def __init__(self) -> None:
        self._signature: tuple[type[BaseException], int | None] | None = None

    def should_warn(self, exc: BaseException) -> bool:
        """Record a failure; ``True`` when it deserves a full WARNING."""
        errno_value: object = getattr(exc, "errno", None)
        signature = (type(exc), errno_value if isinstance(errno_value, int) else None)
        if signature == self._signature:
            return False
        self._signature = signature
        return True

    def succeeded(self) -> None:
        """Re-arm: the site recovered, so the next failure warns again."""
        self._signature = None


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

# Byte budget for a single sanitized filename component. Most filesystems cap
# a path component at 255 *bytes*; the composed name carries a ~26-byte
# timestamp, two sanitized components, an 8-char uuid suffix, separators, and
# ".yaml", so 80 bytes per component keeps the whole filename comfortably
# under the limit even for fully multi-byte input.
_MAX_COMPONENT_BYTES = 80


def _safe_path_component(value: str) -> str:
    """Neutralize *value* so it is safe as a single filename component.

    Path separators (``/``, ``\\``, :data:`os.sep`), control characters, and
    other filesystem-hostile punctuation are replaced with ``_``; ``.`` runs
    are collapsed to a single ``_`` so the result can never be ``.``/``..`` or
    a hidden traversal segment. The result is also clamped to
    :data:`_MAX_COMPONENT_BYTES` UTF-8 bytes (cutting on a codepoint boundary,
    never mid-character) so an oversized ``feature``/``label`` cannot push the
    composed filename past the filesystem's 255-byte component limit and turn
    every write into ``ENAMETOOLONG``. The output stays human-readable and is
    guaranteed to contain no directory separators, so composing it into a path
    cannot escape ``log_dir``. Empty/all-unsafe input degrades to ``"_"``.
    """
    cleaned = _UNSAFE_PATH_CHARS.sub("_", value)
    cleaned = cleaned.replace(os.sep, "_")
    if os.altsep:
        cleaned = cleaned.replace(os.altsep, "_")
    cleaned = _DOT_RUN.sub("_", cleaned)
    cleaned = cleaned.strip("_")
    encoded = cleaned.encode("utf-8")
    if len(encoded) > _MAX_COMPONENT_BYTES:
        # Hard byte clamp; errors="ignore" drops a trailing partial codepoint
        # so the cut never lands mid-character. Safe to apply after the regex
        # passes above because each of them substitutes a single "_" (no
        # multi-char escape sequences exist to split).
        cleaned = encoded[:_MAX_COMPONENT_BYTES].decode("utf-8", errors="ignore")
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

    ``call_id`` is one ``uuid4`` hex per *logical* call and ``attempt`` the
    1-based attempt number within it, so the N records a retried call
    produces join on ``call_id`` instead of feature + timestamp proximity
    (which breaks under concurrent same-feature fan-out). ``queue_wait_ms``
    is the time this attempt spent queued behind llmkit's own rate limiter
    — ``duration_ms`` includes it, so provider latency is approximately
    ``duration_ms - queue_wait_ms``.

    ``run_id`` is the *outer* scope ``call_id`` does not provide: the run —
    an eval sweep, a rehearsal, an incident replay — that this call belonged
    to, so a shared log directory can be filtered by run instead of by
    timestamp window (which breaks whenever two runs overlap). The call layer
    stamps it from :func:`~llmkit.run_scope.get_run_id`, which resolves an
    active :func:`~llmkit.run_scope.run_scope`, then a process-wide
    :func:`~llmkit.run_scope.set_run_id`, then ``LLMKIT_RUN_ID``.

    All four default ``None`` for directly-constructed records — and a
    ``None`` ``run_id`` is the pre-``run_id`` shape of every record, YAML
    body and index line.
    """

    started_at: datetime
    feature: str
    label: str | None
    model: str | None
    provider: str | None
    temperature: float
    duration_ms: float
    schema: str
    prompt: str | list[Message]
    response: Any  # pyright: ignore[reportExplicitAny]  # raw-llm — Pydantic dump or accumulated text
    error: str | None
    approximate_cost: float | None = None
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    call_id: str | None = None
    attempt: int | None = None
    queue_wait_ms: float | None = None
    run_id: str | None = None


@runtime_checkable
class LogSink(Protocol):
    """Destination for :class:`LLMCallRecord`s.

    ``write`` consumes a record and returns nothing. A sink that persists
    to a file (e.g. :class:`LocalYamlLogSink`) exposes the written path via
    its own method/attribute — the shared contract stays file-agnostic so a
    third-party sink (a database, an in-memory buffer, a metrics pipe) is a
    one-method object. To capture per-call *records* (cost/metadata) or
    *file paths* without authoring a sink, use
    :func:`~llmkit.capture.capture_llm_records` or
    :func:`~llmkit.capture.capture_llm_log_paths`.

    ``@runtime_checkable`` so :func:`configure_llm_logging` can reject a
    non-sink at configuration time instead of letting every subsequent write
    fail into the best-effort swallow. The check is structural and tests
    attribute *presence* only — a ``write`` with the wrong signature still
    passes, the same limitation :class:`_PathReturningLogSink` documents.
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
    *must* then return ``Path | None``; :func:`write_llm_log` enforces that
    much, treating anything else as a failed write rather than letting it
    reach the path-capture buffer). A sink that implements only
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
        self._yaml_latch: _OnceLatch = _OnceLatch()
        self._index_latch: _OnceLatch = _OnceLatch()
        self._prune_latch: _OnceLatch = _OnceLatch()

    @property
    def log_dir(self) -> Path:
        """The directory this sink writes to.

        A sink constructed without an explicit ``log_dir`` resolves
        :func:`default_log_dir` on first access and freezes the result, so
        every write (and every reader of this property) sees one stable
        directory regardless of later ``chdir`` or environment changes.
        """
        if self._log_dir is None:
            self._log_dir, self._seed_gitignore = _resolve_default_log_dir()
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
        :func:`_safe_path_component` so neither can escape ``log_dir`` nor
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
                "run_id": record.run_id,
                "call_id": record.call_id,
                "attempt": record.attempt,
                "temperature": record.temperature,
                "max_tokens": record.max_tokens,
                "reasoning_effort": record.reasoning_effort,
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
            for _attempt in range(_MAX_FILENAME_ATTEMPTS):
                candidate = (
                    self.log_dir / f"{ts}_{safe_feature}_{safe_label}_{uuid.uuid4().hex[:8]}.yaml"
                )
                try:
                    with open(candidate, "x", encoding="utf-8", opener=_open_private) as f:
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
                    f"could not allocate a unique log filename after {_MAX_FILENAME_ATTEMPTS} attempts"
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
        through :func:`_oneline` so a value containing a newline cannot forge
        a second ``# ok | ...`` verdict line and corrupt that triage. The
        second line is the ISO ``started_at`` stamp — machine-built, no
        newlines — plus, when the record carries correlation fields, a
        ``call=<id[:8]> attempt=<n>`` suffix so retries of one logical call
        are joinable from the file heads alone. The first line's shape is
        pinned (``head -1`` tooling greps it); only line 2 gains the suffix.
        """
        status = "ERROR" if record.error else "ok"
        cost = f"${record.approximate_cost:.3g}" if record.approximate_cost is not None else "$?"
        feature = _oneline(record.feature)
        label = _oneline(record.label or "unlabeled")
        model = _oneline(record.model or "?")
        schema = _oneline(record.schema)
        correlation = ""
        if record.call_id is not None:
            correlation = f" | call={_oneline(record.call_id)[:8]}"
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
                self.log_dir / INDEX_FILENAME, "a", encoding="utf-8", opener=_open_private
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


# Module-level configured sink, defaulting to the local-YAML sink at the
# default directory (resolved lazily at first write). The host overrides it
# once at startup; tests typically point it at a tmp directory.
_sink: LogSink | None = LocalYamlLogSink()

# Warn-once latch for a *configured* sink that raises out of ``write`` — the
# custom-sink counterpart of LocalYamlLogSink's internal latches, so a
# permanently broken third-party sink can't flood stderr either.
_sink_latch = _OnceLatch()


class _SinkContractError(TypeError):
    """A sink matched a capability protocol structurally but broke its contract.

    Its own type — rather than a bare ``TypeError`` — keeps the warn-once
    latch's ``(type, errno)`` signature distinct, so a sink that returns junk
    from ``write_returning_path`` and *also* raises a genuine ``TypeError``
    later still gets a second loud warning instead of being folded into the
    first.
    """


def configure_llm_logging(sink: LogSink | None) -> None:
    """Set the sink that receives every :class:`LLMCallRecord`.

    Pass ``None`` to disable logging entirely (writes become no-ops).
    Re-arms the warn-once latch on sink failures, so a newly configured sink
    gets a fresh loud first warning if it too turns out to be broken.

    Raises:
        TypeError: if *sink* is neither ``None`` nor a :class:`LogSink`
            instance — including the easy slip of passing the sink *class*
            rather than an instance of it. Configuration is checkable now, so
            it is checked now: an object with no ``write`` would otherwise be
            installed silently and turn every subsequent call's log into a
            latched warning at write time, far from the mistake. The
            structural check tests attribute *presence* only — a ``write``
            with the wrong signature still gets through, the limitation
            :class:`_PathReturningLogSink` already documents and a type
            checker catches at the call site. One consequence worth knowing
            in tests: a bare ``MagicMock()`` does **not** match (structural
            checks use static attribute lookup, which sees through no
            ``__getattr__``), so stub with ``Mock(spec=LogSink)``, a small
            fake class, or ``None``.
    """
    global _sink
    # Deliberate runtime guards at the library's public boundary: the annotation
    # says ``LogSink | None``, but an untyped caller passing something else
    # would otherwise install it silently and lose every subsequent log.
    if isinstance(sink, type):
        # A class object carries ``write`` as an attribute, so it satisfies the
        # structural check below while failing every write on the missing
        # ``self`` — the exact deferred failure this guard exists to prevent.
        raise TypeError(
            f"sink must be a LogSink instance, not the class itself — did you mean {sink.__name__}()?"
        )
    if sink is not None and not isinstance(sink, LogSink):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard at public boundary
        raise TypeError(  # pyright: ignore[reportUnreachable]  # reachable from untyped callers
            "sink must implement LogSink (a write(record) method) or be None, "
            + f"got {type(sink).__name__}"
        )
    _sink = sink
    _sink_latch.succeeded()


def write_llm_log(record: LLMCallRecord) -> Path | None:
    """Hand ``record`` to the configured sink, swallowing any failure.

    Logging must never break the LLM call, so a sink that raises is
    caught here in addition to the sink's own best-effort handling.

    Returns the written file path when the configured sink is a
    file sink that exposes one (it advertises a ``write_returning_path``
    method, as :class:`LocalYamlLogSink` does), so the
    :func:`~llmkit.capture.capture_llm_log_paths` primitive can
    cross-reference it. For a third-party sink that only implements the
    file-agnostic :class:`LogSink` contract (``write(record) -> None``),
    there is no path to return, so this returns ``None`` — path-capture is
    simply empty for such sinks, while
    :func:`~llmkit.capture.capture_llm_records` still captures
    the record itself.

    A third-party ``write_returning_path`` that breaks its ``Path | None``
    contract is treated as a sink failure — latched warning, ``None``
    returned — rather than passed through, so nothing but a real path ever
    reaches :func:`~llmkit.capture.capture_llm_log_paths`'s ``list[Path]``.
    """
    if _sink is None:
        return None
    try:
        if isinstance(_sink, _PathReturningLogSink):
            path = _sink.write_returning_path(record)
            # Same deliberate guard as the configure-time one: a structural
            # protocol match promises the method exists, never that it honors
            # its return annotation. Raising here routes the contract breach
            # through the latched warning below, so a broken third-party sink
            # degrades like any other write failure instead of smuggling a
            # non-Path into capture_llm_log_paths()'s list[Path].
            if path is not None and not isinstance(path, Path):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard on a third-party return
                raise _SinkContractError(
                    f"write_returning_path must return Path | None, got {type(path).__name__}"
                )
        else:
            _sink.write(record)
            path = None
    except Exception as exc:
        if _sink_latch.should_warn(exc):
            logger.warning(
                "LLM log sink raised for %s/%s (further identical failures logged at DEBUG)",
                record.feature,
                record.label,
                exc_info=True,
            )
        else:
            logger.debug(
                "LLM log sink raised for %s/%s", record.feature, record.label, exc_info=True
            )
        return None
    _sink_latch.succeeded()
    return path
