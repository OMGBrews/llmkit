"""Contained diagnostics: llmkit's own warnings never replace a call's outcome.

llmkit reports problems it has already contained (a cache that failed, a sink
that raised, a retry it is about to make) through the standard ``logging``
module. A host's :class:`logging.Handler` runs inline inside that call, and a
handler whose ``emit`` raises would otherwise turn a successful answer, or the
provider error the caller needs, into the handler's exception. Every
diagnostic llmkit emits while a call is in flight goes through :func:`report`,
which contains ordinary exceptions from the logging machinery and counts each
loss where :func:`get_diagnostics_health` can read it.

Only :class:`Exception` is contained. ``asyncio.CancelledError``,
``KeyboardInterrupt`` and ``SystemExit`` derive from :class:`BaseException`
and propagate, so containment never swallows cancellation or shutdown. A
handler that blocks is not contained: diagnostics run on the calling thread,
or on the log-delivery worker, and wait for it.

A leaf module (no imports from the rest of the package) beside the warn-once
latch, so every reporting site can depend on it without a cycle.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType

_SysExcInfo = (
    tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None]
)
#: What :meth:`logging.Logger.log` accepts as ``exc_info``.
ExcInfo = bool | BaseException | _SysExcInfo | None


@dataclass(frozen=True)
class DiagnosticLoss:
    """The most recent diagnostic llmkit could not deliver.

    Attributes:
        logger: Name of the logger the diagnostic was emitted on, such as
            ``"llmkit.cache"``.
        level: Level name of the lost record, such as ``"WARNING"``.
        error: ``"ExceptionType: message"`` text of the exception the logging
            machinery raised, or the type name alone when the message itself
            could not be read.
        occurred_at: Timezone-aware UTC time the loss was recorded.
    """

    logger: str
    level: str
    error: str
    occurred_at: datetime


@dataclass(frozen=True)
class DiagnosticsHealth:
    """A point-in-time snapshot of llmkit's own lost diagnostics.

    Attributes:
        lost: Diagnostics whose logging call raised and was contained, since
            the process started. Never reset.
        last_loss: The most recent loss, or ``None`` when nothing was lost.
    """

    lost: int = 0
    last_loss: DiagnosticLoss | None = None


_lock = threading.Lock()
_lost = 0
_last_loss: DiagnosticLoss | None = None


def _describe(exc: Exception) -> str:
    # The exception came from host logging code, so its __str__ is host code
    # too: a failure to read the message must not escape the containment.
    name = type(exc).__name__
    try:
        text = str(exc)
    except Exception:
        return name
    return f"{name}: {text}"


def _record_loss(logger: logging.Logger, level: int, exc: Exception) -> None:
    global _lost, _last_loss
    # Format outside the lock: _describe may run arbitrary host code.
    loss = DiagnosticLoss(
        logger=logger.name,
        level=logging.getLevelName(level),
        error=_describe(exc),
        occurred_at=datetime.now(UTC),
    )
    with _lock:
        _lost += 1
        _last_loss = loss


def report(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args: object,
    exc_info: ExcInfo = None,
    stacklevel: int = 1,
) -> bool:
    """Emit one diagnostic, containing an ordinary exception from the handlers.

    Behaves like ``logger.log(level, msg, *args, exc_info=exc_info)`` called
    from the caller's own frame: *stacklevel* is passed on one level deeper,
    so the record's ``funcName``, ``lineno`` and ``pathname`` name the real
    reporting site rather than this function. *exc_info* is forwarded
    unchanged; ``True`` reads :func:`sys.exc_info` here, which inside the
    caller's ``except`` block is still the caller's exception.

    Returns ``True`` when the logging call returned normally and ``False``
    when it raised an :class:`Exception`, which is counted in
    :func:`get_diagnostics_health` and otherwise discarded. A failure is
    never reported through logging again: the broken handler would only
    raise again. ``BaseException`` subclasses propagate.
    """
    try:
        logger.log(level, msg, *args, exc_info=exc_info, stacklevel=stacklevel + 1)
    except Exception as exc:
        _record_loss(logger, level, exc)
        return False
    return True


def get_diagnostics_health() -> DiagnosticsHealth:
    """Return a snapshot of llmkit's own diagnostics that could not be logged.

    llmkit reports contained problems (cache failures, sink failures, retries,
    backpressure-callback errors) through the standard ``llmkit.*`` loggers.
    When a host :class:`logging.Handler` raises while handling one of those
    records, llmkit contains the exception so the call still returns its
    answer or raises the provider's error, and counts the lost record here.

    * ``lost`` counts lost diagnostics since the process started; it is never
      reset, because it is not tied to a sink installation.
    * ``last_loss`` names the most recent loss's logger, level, error text
      and time.

    :func:`~llmkit.logging.get_logging_health` answers whether LLM call
    records were lost; this answers whether llmkit's own warnings were. A
    sink failure whose warning is also lost counts once in each, for those
    two different reasons. A handler that blocks rather than raises is not
    detected here.
    """
    with _lock:
        return DiagnosticsHealth(lost=_lost, last_loss=_last_loss)
