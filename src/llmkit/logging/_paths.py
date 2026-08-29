"""Where the log directory is, and what a filename component may contain.

Two filesystem-only concerns with no llmkit dependencies: resolving the default
log directory (``LLMKIT_LOG_DIR``, else the enclosing project root, else a
per-user state directory), and reducing caller-supplied ``feature`` / ``label``
strings to something safe to put in a path. The private opener that creates
files 0600 lives here too, so the security-relevant filesystem policy is in one
place rather than spread through the sink.

Symbols the sink imports carry public names; the helpers used only within this
module stay underscored.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

#: Environment variable overriding the default log directory. Read lazily at
#: the sink's first write (never at import), so setting it after ``import
#: llmkit`` still takes effect.
LOG_DIR_ENV_VAR = "LLMKIT_LOG_DIR"

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


def resolve_default_log_dir() -> tuple[Path, bool]:
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
    return resolve_default_log_dir()[0]


def open_private(path: str, flags: int) -> int:
    """``opener=`` hook: create files ``0o600`` (owner-only) instead of umask-default.

    The per-call YAML and the index carry full prompts and responses, which
    ``SECURITY.md`` promises are not world-readable on a multi-user host. The
    mode only applies at creation — POSIX ``open`` ignores it for an existing
    file — and is inert on Windows.
    """
    return os.open(path, flags, 0o600)


# Bounded retry budget for the exclusive-create filename loop: a uuid4 suffix
# collision is astronomically unlikely, so a handful of attempts is plenty
# while still guaranteeing the loop terminates.
MAX_FILENAME_ATTEMPTS = 8

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


def safe_path_component(value: str) -> str:
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


def oneline(value: str) -> str:
    """Collapse CR/LF and other control characters in *value* to single spaces.

    Used on every caller-derived field interpolated into the ``#`` header
    comment so a value containing a newline cannot forge a second
    ``# ok | ...`` verdict line and corrupt the ``head -1`` triage.
    """
    return _CONTROL_CHARS.sub(" ", value).strip()
