"""A logging handler that raises, for the diagnostic-containment tests.

llmkit reports contained problems through ``llmkit.*`` loggers, and a host's
handler runs inline inside that call. :func:`raising_handler` installs a
:class:`RaisingHandler` on one named logger so a test can check that the
handler's exception never replaces the call's result or the provider's error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from llmkit import get_diagnostics_health


class DiagnosticHandlerError(RuntimeError):
    """The ordinary exception a :class:`RaisingHandler` raises by default."""


def _default_error() -> BaseException:
    return DiagnosticHandlerError("diagnostic handler failed")


class RaisingHandler(logging.Handler):
    """Record every record it is handed, then raise from ``emit``.

    ``records`` lists what reached the handler, so a test can assert that the
    seam emitted exactly one record per diagnostic and never re-logged its own
    failure. ``raise_on`` selects which records raise; the rest are only
    recorded.
    """

    def __init__(
        self,
        error: Callable[[], BaseException] = _default_error,
        raise_on: Callable[[logging.LogRecord], bool] | None = None,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self._error = error
        self._raise_on = raise_on

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self._raise_on is None or self._raise_on(record):
            raise self._error()


@contextmanager
def raising_handler(
    name: str,
    *,
    error: Callable[[], BaseException] = _default_error,
    raise_on: Callable[[logging.LogRecord], bool] | None = None,
) -> Iterator[RaisingHandler]:
    """Install a :class:`RaisingHandler` on logger *name* at DEBUG, then restore it.

    The logger's level is lowered to DEBUG for the duration so both the
    first-failure WARNING and the repeated-failure DEBUG paths reach the handler.
    """
    logger = logging.getLogger(name)
    handler = RaisingHandler(error, raise_on)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def lost_diagnostics() -> int:
    """The process-wide count of contained diagnostic failures (never reset)."""
    return get_diagnostics_health().lost
