"""Caller-owned record/path capture and bounded managed log submission.

Call functions capture records before submitting private snapshots to the log
worker. Only the caller may append a path, when completion is observed within
its delivery wait. Late writes never mutate capture buffers. The explicit
low-level :func:`record_call` remains synchronous.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from llmkit.logging import LLMCallRecord, write_llm_log
from llmkit.logging._delivery import Receipt, submit, wait_for_receipt, worker_origin

_captured_log_paths: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "_llm_captured_log_paths", default=None
)

_captured_records: contextvars.ContextVar[list[LLMCallRecord] | None] = contextvars.ContextVar(
    "_llm_captured_records", default=None
)


def record_call(record: LLMCallRecord) -> Path | None:
    """Synchronously write *record* and feed the active captures.

    This low-level seam waits for sink completion without a delivery bound.
    Managed call functions use :func:`record_call_async` or
    :func:`submit_record_call` instead.
    """
    captured_records = _captured_records.get()
    if captured_records is not None:
        captured_records.append(record)
    path = write_llm_log(record)
    captured_paths = _captured_log_paths.get()
    if captured_paths is not None and path is not None:
        captured_paths.append(path)
    return path


def _capture_and_submit(record: LLMCallRecord) -> Receipt | None:
    captured_records = _captured_records.get()
    if captured_records is not None:
        captured_records.append(record)
    context = contextvars.copy_context()
    # Preserve application context for custom sinks, but never give the worker
    # ownership of the caller's mutable capture buffers, even through reentry.
    _ = context.run(_captured_records.set, None)
    _ = context.run(_captured_log_paths.set, None)
    return submit(record, context)


def submit_record_call(record: LLMCallRecord) -> None:
    """Capture and submit without waiting; safe during stream abandonment.

    Sink I/O stays on the delivery worker. Rejection preserves record capture,
    and this nonwaiting path never promises or later appends a captured path.
    """
    _ = _capture_and_submit(record)


async def record_call_async(record: LLMCallRecord) -> Path | None:
    """Capture, submit and wait within the configured delivery deadline.

    Record capture precedes admission, including when disabled or rejected.
    Ordinary built-in containers are snapshotted before publication; preparation
    is outside the delivery wait. A timeout or cancellation leaves accepted work
    available for later delivery but never permits later capture appends.
    Cancellation already unwinding through a call finalizer skips waiting.
    """
    receipt = _capture_and_submit(record)
    task = asyncio.current_task()
    if (
        receipt is None
        or worker_origin()
        or (task is not None and task.cancelling())
        or isinstance(sys.exception(), (asyncio.CancelledError, GeneratorExit))
    ):
        return None
    path = await wait_for_receipt(receipt)
    captured_paths = _captured_log_paths.get()
    if captured_paths is not None and path is not None:
        captured_paths.append(path)
    return path


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
    :func:`tool_llm_call_stream`) and works across the sync bridges (e.g.
    :func:`structured_llm_call_sync`, :func:`text_llm_call_stream_sync`), since
    the record is appended inside the async call path the bridge drives. A
    sync stream captures the context when it is created, so enter this block
    before calling the ``*_stream_sync`` function.

    One record is appended *per attempt*, before delivery admission and before
    the call returns or propagates its error. Retries each produce their own
    record. Capture works when logging is disabled, full or unavailable and
    proves record construction, not persistence. Delivery workers never append
    to this list after the call returns. The captured record retains its original
    nested objects; managed delivery receives a separate container snapshot.
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

    Each attempt may append a path if sink completion is observed within its
    configured delivery wait. The default file sink and compatible custom sinks
    exposing ``write_returning_path`` can supply paths. A timeout, cancellation,
    zero wait or abandoned-stream submission promises no path; later completion
    never appends one. Disabled logging and sinks without path support also leave
    the list empty. Use :func:`capture_llm_records` for sink-independent metadata.
    """
    paths: list[Path] = []
    token = _captured_log_paths.set(paths)
    try:
        yield paths
    finally:
        _captured_log_paths.reset(token)
