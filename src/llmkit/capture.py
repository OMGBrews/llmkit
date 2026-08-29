"""Per-call record/log-path capture, and the record sink seam.

The call functions in :mod:`llmkit.calls` funnel every built
:class:`~llmkit.logging.LLMCallRecord` through :func:`record_call` — the
single seam that writes the configured sink and feeds the two opt-in capture
context managers — usually via :func:`record_call_async`, which runs that
same seam on a worker thread so sink I/O never blocks the event loop:

* :func:`capture_llm_records` — yields the full per-call records
  (``approximate_cost``, resolved ``model``/``provider``, ``duration_ms``,
  ``error`` …) without authoring a :class:`~llmkit.logging.LogSink`;
* :func:`capture_llm_log_paths` — yields the per-call log-file paths the file
  sink wrote.

This module owns capture + the record seam only: with the record's
model/provider resolution moved to the call layer that builds records
(:mod:`llmkit.calls`), its whole dependency set is :mod:`llmkit.logging`. The
option merge lives in :mod:`llmkit.options`, and the call surface itself in
:mod:`llmkit.calls`.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from llmkit.logging import LLMCallRecord, write_llm_log

logger = logging.getLogger(__name__)


_captured_log_paths: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "_llm_captured_log_paths", default=None
)

_captured_records: contextvars.ContextVar[list[LLMCallRecord] | None] = contextvars.ContextVar(
    "_llm_captured_records", default=None
)


def record_call(record: LLMCallRecord) -> Path | None:
    """Write *record* to the configured sink and feed the active captures.

    The single seam every call path funnels its built
    :class:`~llmkit.logging.LLMCallRecord` through, so the two capture
    primitives stay in lock-step: the record itself is appended to any
    active :func:`capture_llm_records` buffer, and the returned file path
    (when a file sink wrote one) is appended to any active
    :func:`capture_llm_log_paths` buffer. Returns the path so the existing
    path-capture call sites keep working unchanged.
    """
    captured_records = _captured_records.get()
    if captured_records is not None:
        captured_records.append(record)
    path = write_llm_log(record)
    captured_paths = _captured_log_paths.get()
    if captured_paths is not None and path is not None:
        captured_paths.append(path)
    return path


async def record_call_async(record: LLMCallRecord) -> Path | None:
    """:func:`record_call`, offloaded so sink I/O never blocks the event loop.

    The default sink's write is real blocking work — ``mkdir``, a
    full-payload ``yaml.dump``, two file writes, retention housekeeping —
    which :func:`asyncio.to_thread` moves onto a worker thread.
    ``to_thread`` runs the seam inside a *copy* of the caller's context, and
    the capture ContextVars hold shared list objects, so capture appends made
    in the worker are visible to the caller exactly as on the sync path; the
    thread completes even if the awaiting task is cancelled mid-write, so
    the record (and its captured path) still lands.

    Falls back to the synchronous seam when the offload itself is impossible
    (``RuntimeError``: no running loop, or the loop's default executor was
    already shut down by a late teardown write) — logging must never break
    the call, and a blocking write in a dying process beats a lost record.

    Deliberately additive: the sync :func:`record_call` stays, both because
    it is an importable seam and because the abandoned-stream path *must*
    write synchronously (suspending while ``GeneratorExit`` unwinds risks
    losing the record — see ``_stream_once``).
    """
    try:
        return await asyncio.to_thread(record_call, record)
    except RuntimeError:
        logger.debug("Log-write offload unavailable; writing synchronously", exc_info=True)
        return record_call(record)


@contextmanager
def capture_llm_records() -> Generator[list[LLMCallRecord]]:
    """Capture the :class:`~llmkit.logging.LLMCallRecord` for each call here.

    The records-oriented counterpart to :func:`capture_llm_log_paths`:
    yields a list that is appended to once per LLM call inside the ``with``
    block, giving the host the full record — ``approximate_cost``, resolved
    ``model``/``provider``, ``duration_ms``, ``error`` and the rest —
    without authoring a :class:`~llmkit.logging.LogSink`. Captures every
    call function (:func:`structured_llm_call`, :func:`text_llm_call`,
    :func:`text_llm_call_stream`, :func:`tool_llm_call`,
    :func:`tool_llm_call_stream`) and works across the ``run_sync`` bridge
    (e.g. :func:`structured_llm_call_sync`), since the record is appended
    inside the async call path the bridge drives.

    Like path-capture, one record is appended *per attempt* — retries each
    produce their own logged record — and capture is independent of the
    configured sink, so it works even when logging is disabled
    (``configure_llm_logging(None)``).
    """
    records: list[LLMCallRecord] = []
    token = _captured_records.set(records)
    try:
        yield records
    finally:
        _captured_records.reset(token)


@contextmanager
def capture_llm_log_paths() -> Generator[list[Path]]:
    """Capture log paths written by the call functions in this scope.

    The returned list is appended to once per LLM call inside the
    ``with`` block — including retries, since ``with_retries`` lives
    outside the call functions and each attempt is its own call. Only the
    configured file sink (:class:`~llmkit.LocalYamlLogSink`) yields a path;
    with a third-party :class:`~llmkit.logging.LogSink` (or logging
    disabled) the list stays empty — use :func:`capture_llm_records` to
    capture cost/metadata regardless of the sink.
    """
    paths: list[Path] = []
    token = _captured_log_paths.set(paths)
    try:
        yield paths
    finally:
        _captured_log_paths.reset(token)
