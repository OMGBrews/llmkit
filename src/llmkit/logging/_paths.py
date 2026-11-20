"""Where the log directory is, and what a filename component may contain.

Two filesystem-only concerns with no llmkit dependencies: resolving the default
log directory (``LLMKIT_LOG_DIR``, else the enclosing project root, else a
per-user state directory, which is also where the sink falls back to when the
project-root folder cannot hold files), and reducing caller-supplied ``feature`` / ``label``
strings to something safe to put in a path. The private opener that creates
files 0600 lives here too, so the security-relevant filesystem policy is in one
place rather than spread through the sink. The names of the logging
environment variables live here as well, with the blank-is-unset rule they
share, and so do the timestamp format of generated file names and the
full-name patterns retention uses to recognise them.

Symbols the sink imports carry public names; the helpers used only within this
module stay underscored.
"""

from __future__ import annotations

import errno
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

#: Environment variable overriding the default log directory. Read lazily at
#: the sink's first write (never at import), so setting it after ``import
#: llmkit`` still takes effect.
LOG_DIR_ENV_VAR = "LLMKIT_LOG_DIR"

#: Environment variable switching the import-time default sink off (``off``,
#: ``0``, ``false``, ``no``) or explicitly on (``on``, ``1``, ``true``,
#: ``yes``), stripped and case-insensitive. Read once, at the first registry
#: read or write after import, never at import; an explicit
#: ``configure_llm_logging(...)`` always wins. Any other non-blank value turns
#: logging off with one warning (fail closed: a prompt written to disk cannot
#: be taken back).
LOG_ENV_VAR = "LLMKIT_LOG"

#: Environment variable setting the age bound of every ``LocalYamlLogSink``
#: constructed without ``retention_days``: a decimal integer ``>= 1`` (days),
#: or ``forever`` to keep every file. Read at each such sink's first write; an
#: explicit ``retention_days=`` (including ``None``) always wins. Any other
#: non-blank value disables age pruning for that sink, with one warning and a
#: degraded ``housekeeping`` entry in ``get_logging_health()``.
LOG_RETENTION_ENV_VAR = "LLMKIT_LOG_RETENTION_DAYS"

#: Marker files whose presence makes a directory a "project root" for
#: :func:`default_log_dir`'s upward walk.
_PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")

#: ``errno`` values that move a default sink from its project-root folder to
#: the per-user state directory, when raised while creating that folder or
#: opening the first record's file in it (see
#: :class:`~llmkit.logging.LocalYamlLogSink`).
#:
#: An error belongs here only when it means *this location can never hold
#: this process's files*: permission denied, a read-only filesystem, a file
#: where a folder must be, a path that does not exist, a symlink loop.
#: Never add a transient or record-specific error — a full disk
#: (``ENOSPC``), a quota (``EDQUOT``), file-handle exhaustion (``EMFILE``,
#: ``ENFILE``), an over-long name (``ENAMETOOLONG``) — because moving the
#: logs would hide a problem the user has to fix, and the per-user folder
#: usually shares it. Those keep failing visibly.
#:
#: To add one: (1) reproduce the failure and note the ``errno`` it raises
#: and where (folder creation or the record open); (2) add the ``errno``
#: constant here; (3) add it to ``_DECIDED_ERRNOS`` in
#: ``tests/logging/test_default_dir_fallback.py``, which runs every listed
#: error through the fallback, and add a case that produces the error on a
#: real filesystem where one can. CONTRIBUTING.md carries the same steps.
FALLBACK_ERRNOS: frozenset[int] = frozenset(
    {errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOTDIR, errno.ENOENT, errno.ELOOP}
)


class DefaultLogDir(NamedTuple):
    """What default resolution chose, and why.

    Attributes:
        path: The absolute directory.
        project_root: ``True`` only for ``data/llm-logs`` under a project
            root: the case that seeds a ``.gitignore`` and the only one a
            sink may later fall back from.
        cwd_error: The error from reading the current directory when that
            failed and ``path`` is the per-user state directory because of
            it; ``None`` otherwise.
    """

    path: Path
    project_root: bool
    cwd_error: OSError | None = None


def blank_is_unset(value: str | None) -> str | None:
    """*value* as read from the environment, or ``None`` when blank.

    The one rule every llmkit logging variable shares: an empty or
    whitespace-only value means "not set", so ``LLMKIT_LOG_DIR=" "`` falls
    through to the default directory instead of naming a directory called
    ``"   "``. A non-blank value is returned unchanged. It takes the value the
    caller already read, not the variable's name, so each read stays a plain
    ``os.environ.get(<constant>)`` at its own site.
    """
    if value is None or not value.strip():
        return None
    return value


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


def per_user_log_dir() -> Path:
    """The absolute per-user state directory for llmkit logs.

    Where default resolution lands outside a project, and where a default
    sink falls back to when its project-root folder cannot hold files. Made
    absolute because ``XDG_STATE_HOME`` and ``LOCALAPPDATA`` are read
    verbatim; a relative value with no current directory raises.
    """
    return _user_state_log_dir().absolute()


def resolve_default_log_dir_detail() -> DefaultLogDir:
    """Resolve the default log directory, saying which case chose it.

    Resolution order: :data:`LOG_DIR_ENV_VAR` if set and not blank, else ``data/llm-logs``
    under the nearest project root above the current directory, else the
    per-user state directory. Only the project-root case (the one where a
    repository could accidentally swallow prompt logs) sets
    ``project_root``, which makes the sink seed a ``.gitignore`` there when
    none exists, whoever created the directory.

    When the current directory cannot be read (it was deleted under the
    process), no project root can be found, so the per-user state directory
    is returned with the error in ``cwd_error`` rather than raised; the sink
    reports that once as a fallback. ``LLMKIT_LOG_DIR`` is read first, so it
    still decides when set.

    Every branch returns an absolute path — a *relative* ``LLMKIT_LOG_DIR``
    (or ``XDG_STATE_HOME`` / ``LOCALAPPDATA``, which the state-dir fallback
    reads verbatim) would otherwise be re-anchored to the process's current
    directory on each use, which is exactly the chdir-splits-the-logs failure
    the sink freezes its answer to prevent. Absolute, not resolved: symlinks
    are left intact, since the frozen answer only has to be stable, not
    canonical. A relative value with no current directory cannot be made
    absolute, so that one still raises.
    """
    env_dir = blank_is_unset(os.environ.get(LOG_DIR_ENV_VAR))
    if env_dir is not None:
        return DefaultLogDir(Path(env_dir).absolute(), project_root=False)
    try:
        cwd = Path.cwd()
    except OSError as exc:
        return DefaultLogDir(per_user_log_dir(), project_root=False, cwd_error=exc)
    root = _find_project_root(cwd)
    if root is not None:
        # Already absolute: rooted at Path.cwd().
        return DefaultLogDir(root / "data" / "llm-logs", project_root=True)
    return DefaultLogDir(per_user_log_dir(), project_root=False)


def resolve_default_log_dir() -> tuple[Path, bool]:
    """The default log directory and whether it is the project-root case.

    :func:`resolve_default_log_dir_detail` without the current-directory
    error.
    """
    detail = resolve_default_log_dir_detail()
    return detail.path, detail.project_root


def default_log_dir() -> Path:
    """Compute where :class:`LocalYamlLogSink` writes when no ``log_dir`` is given.

    ``LLMKIT_LOG_DIR`` wins when set (a blank or whitespace-only value counts
    as unset); otherwise ``data/llm-logs/`` under the
    nearest ancestor directory carrying a ``pyproject.toml`` or ``.git``
    (nearest wins); otherwise a per-user state directory
    (``$XDG_STATE_HOME/llmkit/llm-logs`` on Linux, ``~/Library/Logs/llmkit``
    on macOS, ``%LOCALAPPDATA%\\llmkit\\logs`` on Windows), which is also
    the answer when the current directory has been deleted. Computed from the
    *current* environment and working directory on every call; the default
    sink resolves the same way once, at its first write or first read of
    :attr:`~llmkit.logging.LocalYamlLogSink.log_dir`, and freezes the answer,
    so a later ``chdir`` cannot split one process's logs across directories.

    This reports where the default *would* go, not where a sink is writing:
    a default sink whose project-root folder cannot hold files moves to the
    per-user state directory at its first write, while this keeps naming the
    project folder. Read ``get_log_sink().log_dir`` after a write for that.
    """
    return resolve_default_log_dir()[0]


def open_private(path: str, flags: int) -> int:
    """``opener=`` hook: create files ``0o600`` (owner-only) instead of umask-default.

    The per-call YAML and the index carry full prompts and responses, which
    ``SECURITY.md`` promises are not world-readable on a multi-user host. The
    mode only applies at creation — POSIX ``open`` ignores it for an existing
    file — and is inert on Windows.
    """
    return os.open(path, flags, 0o600)


#: ``strftime`` format of the timestamp in every file name llmkit generates in
#: a log directory: the leading stamp of a per-call record (and so of its
#: temporary ``.part`` file) and the stamp of a rotated index generation. The
#: prune patterns below are written against this format, so the two live side
#: by side; changing one without the other strands files that pruning would
#: no longer recognise.
FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S-%f"

# The timestamp FILENAME_TIMESTAMP_FORMAT produces: date, "T", then hour,
# minute, second and six-digit microseconds, all hyphen-separated.
_TIMESTAMP_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}"

# A per-call record: ``{timestamp}_{feature}_{label}_{uuid8}.yaml``. The
# sanitized feature and label may themselves contain "_", so the middle is
# matched as one non-empty run rather than split further.
_RECORD_PATTERN = rf"{_TIMESTAMP_PATTERN}_.+_[0-9a-f]{{8}}\.yaml"

#: Full-name patterns (applied with ``fullmatch``) for the files llmkit itself
#: names in a log directory and that retention may delete: a per-call record,
#: the hidden ``.{record}.part`` temporary file a cut-short write leaves
#: behind, and a rotated ``index-{timestamp}.jsonl`` generation. Anything else
#: in the directory, whatever its extension, is not llmkit's to prune. The
#: active ``index.jsonl`` matches none of them: it is never age-pruned.
PRUNABLE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(_RECORD_PATTERN),
    re.compile(rf"\.{_RECORD_PATTERN}\.part"),
    re.compile(rf"index-{_TIMESTAMP_PATTERN}\.jsonl"),
)


def is_prunable_name(name: str) -> bool:
    """Whether *name* is a file name llmkit generates and retention may delete.

    True only when the *whole* name matches one of
    :data:`PRUNABLE_NAME_PATTERNS`, so ``docker-compose.yaml`` or
    ``index-notes.jsonl`` beside the logs is never touched. A foreign file
    whose entire name happens to have the generated shape (a 26-character
    timestamp, then eight hex digits before ``.yaml``) is indistinguishable
    and is still treated as llmkit's.
    """
    return any(pattern.fullmatch(name) for pattern in PRUNABLE_NAME_PATTERNS)


# Bounded retry budget for the exclusive-create filename loop: a uuid4 suffix
# collision is astronomically unlikely, so a handful of attempts is plenty
# while still guaranteeing the loop terminates.
MAX_FILENAME_ATTEMPTS = 8

# Matches a run of characters that are unsafe in a path component: path
# separators, control characters (incl. CR/LF), and other filesystem-hostile
# punctuation, and lone surrogates (``surrogateescape`` text from a filename or
# OSError), which no filesystem encoding accepts. Collapsed to a single "_" so
# names stay readable.
_UNSAFE_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f/\\<>:\"|?*\ud800-\udfff]+")

# Strips the leading/trailing "_" and collapses any "." runs that remain so a
# value can never resolve to "." / ".." or a hidden traversal segment.
_DOT_RUN = re.compile(r"\.+")

# Matches control characters (CR, LF, tabs, NUL, etc.) for one-lining values
# interpolated into the "#" header comment, so a value can't forge a second
# comment line.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")

# Matches one lone surrogate, which the UTF-8 log file cannot encode.
_SURROGATE = re.compile(r"[\ud800-\udfff]")

# Byte budget for a single sanitized filename component. Most filesystems cap
# a path component at 255 *bytes*; the composed name carries a ~26-byte
# timestamp, two sanitized components, an 8-char uuid suffix, separators, and
# ".yaml", so 80 bytes per component keeps the whole filename comfortably
# under the limit even for fully multi-byte input.
_MAX_COMPONENT_BYTES = 80


def safe_path_component(value: str) -> str:
    """Neutralize *value* so it is safe as a single filename component.

    Path separators (``/``, ``\\``, :data:`os.sep`), control characters, lone
    surrogates, and other filesystem-hostile punctuation are replaced with
    ``_``; ``.`` runs are collapsed to a single ``_`` so the result can never
    be ``.``/``..`` or a hidden traversal segment. The result is also clamped to
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


def oneline(value: str) -> str:
    """Collapse CR/LF and other control characters in *value* to single spaces.

    Used on every caller-derived field interpolated into the ``#`` header
    comment so a value containing a newline cannot forge a second
    ``# ok | ...`` verdict line and corrupt the ``head -1`` triage. A lone
    surrogate is written as its ``\\udXXX`` escape, so it cannot make the
    header unencodable and lose the whole record.
    """
    collapsed = _CONTROL_CHARS.sub(" ", value).strip()
    return _SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", collapsed)
