"""Run scoping: tag every :class:`~llmkit.logging.LLMCallRecord` with a run id.

A consumer that logs many calls into one shared log directory eventually wants
to ask "which of these belonged to *that* run?" — an eval sweep, a rehearsal, an
incident reconstruction. Neither ``call_id`` nor the timestamp answers it:
``call_id`` is per *logical call* (it joins retry attempts, not runs), and a time
window breaks the moment two things run at once. This module supplies the
missing key. Set a run id once and every record — and every ``index.jsonl``
line — carries it, so ``jq 'select(.run_id == "…")'`` over the shared index
replaces guessing at a time range.

**Three layers, most specific wins.** :func:`get_run_id` resolves them on every
record build:

1. **A context scope** — :func:`run_scope`, held in a :class:`~contextvars.ContextVar`.
2. **A process default** — :func:`set_run_id`, a plain module global.
3. **The environment** — :data:`RUN_ID_ENV_VAR` (``LLMKIT_RUN_ID``), read fresh
   each time so a shell-launched process needs no code change at all.

Anything unset falls through; all three unset means ``run_id=None`` and records
identical to what llmkit wrote before this existed.

**Why two programmatic layers and not just the ContextVar.** They fail in
opposite directions, and llmkit's own concurrency shapes hit both:

* A ContextVar *does* survive the sync bridge. :func:`llmkit.sync.run_sync`
  captures ``contextvars.copy_context()`` in the calling thread and creates the
  task on the persistent loop inside that copy, precisely so context-scoped
  state (the capture lists, the retry progress callback) crosses the thread
  boundary — see that module's docstring. So :func:`run_scope` reaches records
  produced on the shared loop.
* A ContextVar does **not** survive :class:`threading.Thread`. A new thread
  starts with a fresh empty context, so every ContextVar reads its default
  there: a run scope entered on the main thread is invisible to a worker pool
  fanning out sync calls.

The process default has the mirror-image profile — visible from every thread and
every loop, but a single value, so it cannot express two runs overlapping in one
process. Together they cover both: :func:`set_run_id` for "this process is one
run" (the common case, and the one a thread pool needs), :func:`run_scope` for a
host driving several runs concurrently in one process.

**Blank is not a run id.** :func:`set_run_id` and :func:`run_scope` reject an
empty or whitespace-only string rather than accepting configuration that
silently does nothing — pass ``None`` to mean "no run id". A blank *environment*
variable is instead ignored, matching ``LLMKIT_LOG_DIR``: a shell exporting
``LLMKIT_RUN_ID=`` is an ordinary "not set", not a programming error.

``run_id`` is independent of ``LLMKIT_LOG_DIR``. Redirecting the directory and
tagging the lines are separate mechanisms and compose freely — the point of
tagging is that a *shared* directory stays greppable, but nothing stops a caller
doing both.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from contextvars import ContextVar
from typing import Final

__all__ = ["RUN_ID_ENV_VAR", "get_run_id", "run_scope", "set_run_id"]

#: Environment variable naming the run every record from this process belongs
#: to. Read at record-build time (not cached at import), so a test or a host
#: that mutates ``os.environ`` takes effect immediately. Lowest precedence: an
#: explicit :func:`set_run_id` or :func:`run_scope` overrides it.
RUN_ID_ENV_VAR: Final = "LLMKIT_RUN_ID"

# The context-scoped run id (:func:`run_scope`). ``None`` means "no scope
# entered here" and falls through to the process default — deliberately *not* a
# distinct sentinel: an inner scope suppressing an outer run id is not a use
# case worth a second unset state in the resolution table.
_scoped_run_id: ContextVar[str | None] = ContextVar("llmkit_run_id", default=None)

# The process-wide default (:func:`set_run_id`). A plain global, not a
# ContextVar, so it is visible from worker threads and from llmkit's persistent
# event loop alike. Assignment is atomic under the GIL and this is a
# configure-once knob, so no lock guards it.
_process_run_id: str | None = None


def _validate(run_id: str | None) -> str | None:
    """Return *run_id*, rejecting a blank string.

    ``""`` and ``"   "`` are almost always a bug — an unpopulated f-string, a
    stripped CLI argument — and storing one would tag every record with a value
    that reads as "set" to a consumer while grouping nothing. Fail loudly
    instead; ``None`` is how you say "no run id".
    """
    if run_id is not None and not run_id.strip():
        raise ValueError(f"run_id must be a non-blank string or None; got {run_id!r}")
    return run_id


def set_run_id(run_id: str | None) -> None:
    """Set the process-wide default run id, or clear it with ``None``.

    Every record built afterwards — on any thread, on llmkit's persistent event
    loop, from a ``*_sync`` helper or an ``async`` one — carries *run_id*, unless
    an enclosing :func:`run_scope` overrides it. This is the setter for the
    common case: one process, one run.

    Clearing with ``None`` removes the *programmatic* default only; a
    :data:`RUN_ID_ENV_VAR` in the environment then applies again, since it sits
    one layer below. To end up with no run id at all, make sure the variable is
    unset too.

    Raises :exc:`ValueError` on a blank string.
    """
    global _process_run_id
    _process_run_id = _validate(run_id)


@contextlib.contextmanager
def run_scope(run_id: str) -> Generator[None]:
    """Tag every record built inside the ``with`` block with *run_id*.

    The context-scoped layer, for a host driving several runs in one process::

        with llmkit.run_scope("eval-sweep-2026-08-21"):
            results = [structured_llm_call_sync(p, output_schema=S) for p in prompts]

    Restores the previous scope on exit, exceptions included, so nested and
    sequential scopes never cross-tag. It reaches records produced by
    ``*_sync`` calls: the sync bridge copies the calling thread's context onto
    the persistent loop. It does **not** reach a :class:`threading.Thread` you
    start yourself — a new thread gets a fresh context — so for a thread-pool
    fan-out use :func:`set_run_id`, or re-enter the scope inside each worker.

    Takes a required non-blank ``str``: a ``with`` block that tags nothing is a
    mistake worth catching, so unlike :func:`set_run_id` this does not accept
    ``None``.
    """
    token = _scoped_run_id.set(_validate(run_id))
    try:
        yield
    finally:
        _scoped_run_id.reset(token)


def get_run_id() -> str | None:
    """Resolve the run id in force right now, or ``None``.

    Context scope, then process default, then :data:`RUN_ID_ENV_VAR`; the first
    one set wins and a blank environment variable counts as unset. Called at
    record-build time by the call layer, so the answer tracks the *current*
    scope and environment rather than whatever held when the module was
    imported.
    """
    scoped = _scoped_run_id.get()
    if scoped is not None:
        return scoped
    if _process_run_id is not None:
        return _process_run_id
    env_run_id = os.environ.get(RUN_ID_ENV_VAR)
    if env_run_id and env_run_id.strip():
        return env_run_id
    return None
