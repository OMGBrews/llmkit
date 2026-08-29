"""What a log sink must be.

One method, deliberately: the shared contract stays file-agnostic so a
third-party sink (a database, an in-memory buffer, a metrics pipe) is a
one-method object. The optional path-returning capability a file sink may also
advertise is an internal protocol and lives with the writer that consumes it,
in :mod:`llmkit.logging.registry`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llmkit.logging.record import LLMCallRecord


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
