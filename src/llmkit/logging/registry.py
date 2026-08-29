"""The configured sink, and the one function that writes through it.

The process-global sink and the two functions that mutate and read it are one
indivisible unit: a ``from ... import _sink`` in a second module would bind a
snapshot, and every :func:`configure_llm_logging` call would silently no-op
while records kept going to the old sink — a failure with no error anywhere.

The default is constructed eagerly at import, so logging is on with zero setup;
its log directory is still resolved lazily at the first write.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from llmkit.logging._latch import OnceLatch
from llmkit.logging.local_yaml import LocalYamlLogSink
from llmkit.logging.record import LLMCallRecord
from llmkit.logging.sink import LogSink

# Named explicitly rather than via ``__name__`` so every module in this package
# keeps emitting under the one logger name operators already filter on.
logger = logging.getLogger("llmkit.logging")


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


# Module-level configured sink, defaulting to the local-YAML sink at the
# default directory (resolved lazily at first write). The host overrides it
# once at startup; tests typically point it at a tmp directory.
_sink: LogSink | None = LocalYamlLogSink()

# Warn-once latch for a *configured* sink that raises out of ``write`` — the
# custom-sink counterpart of LocalYamlLogSink's internal latches, so a
# permanently broken third-party sink can't flood stderr either.
_sink_latch = OnceLatch()


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


def get_log_sink() -> LogSink | None:
    """Return the sink currently installed, or ``None`` when logging is off.

    The read half of the pair :func:`configure_llm_logging` writes, following
    the library's ``get_*`` reads / ``configure_*`` sets convention (the
    :func:`~llmkit.get_rate_limit_config` precedent). It exists so a host that
    installs a sink *temporarily* — a test harness, an eval run that captures
    records into its own buffer — can restore what was there before instead of
    guessing:

    ```python
    previous = get_log_sink()
    configure_llm_logging(MySink())
    try:
        ...
    finally:
        configure_llm_logging(previous)
    ```

    Restoring the return value is what makes that correct in every starting
    state, including the two a guess gets wrong: logging deliberately disabled
    (``None``), and the eagerly-constructed default sink, which is a *specific
    instance* holding its own resolved log directory and housekeeping clock —
    ``configure_llm_logging(LocalYamlLogSink())`` puts back an equivalent sink,
    not the same one.

    Reading the module attribute instead (``llmkit.logging.registry._sink``, or
    the pre-split ``llmkit.logging._sink``) is what this replaces, and is the
    reason it is worth a public name: a private attribute that moves — as it
    did when this package was split out of one module — turns a save/restore
    helper into a permanent ``configure_llm_logging(None)`` with no error
    anywhere, caught only by a test that reads the private name too.
    """
    return _sink


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
