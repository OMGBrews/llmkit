"""The configured sink, and the one function that writes through it.

The process-global sink and the two functions that mutate and read it are one
indivisible unit: a ``from ... import _installed`` in a second module would bind
a snapshot, and every :func:`configure_llm_logging` call would silently no-op
while records kept going to the old sink — a failure with no error anywhere.

The default is constructed eagerly at import, so logging is on with zero setup;
its log directory is still resolved lazily at the first write. ``LLMKIT_LOG``
can switch that import-time default off: it is read once, at the first read or
write of the installation, never at import, and an explicit
:func:`configure_llm_logging` always wins over it.

The sink and its health counters are installed together as one object, so a
write reads both with a single attribute load and a reconfigure during an
in-flight write cannot land that write's outcome in the new counters.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from llmkit._diagnostics import report
from llmkit._latch import OnceLatch
from llmkit.logging._health import HealthState, LoggingHealth, SinkKind, WriteOutcome
from llmkit.logging._paths import LOG_ENV_VAR, blank_is_unset
from llmkit.logging._policy import LogDeliveryPolicy, configure_delivery
from llmkit.logging.local_yaml import LocalYamlLogSink, implicit_default_sink
from llmkit.logging.local_yaml import fork_locks as _sink_fork_locks
from llmkit.logging.local_yaml import reset_after_fork as _reset_sink_after_fork
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

    A third-party sink matching only this protocol is still a *custom* sink to
    :func:`get_logging_health`: a returned path or ``None`` counts as a
    handoff, because llmkit verifies only its own writes.
    """

    def write_returning_path(self, record: LLMCallRecord) -> Path | None: ...


@runtime_checkable
class _OutcomeReportingLogSink(Protocol):
    """A sink whose writes llmkit verifies, because it returns what happened.

    Internal capability protocol, matched structurally like
    :class:`_PathReturningLogSink`. :class:`LocalYamlLogSink` exposes it, so
    the failures it handles internally still reach the health counters even
    when its latched warnings are filtered out.
    """

    def write_with_outcome(self, record: LLMCallRecord) -> WriteOutcome: ...


def _classify(sink: LogSink | None) -> SinkKind:
    if sink is None:
        return "disabled"
    if isinstance(sink, _OutcomeReportingLogSink):
        return "builtin"
    return "custom"


@dataclass(frozen=True)
class Installed:
    """The configured sink and the health state counting its outcomes."""

    sink: LogSink | None
    health: HealthState


def _install(sink: LogSink | None) -> Installed:
    return Installed(sink=sink, health=HealthState(sink=_classify(sink)))


# Module-level installation, defaulting to the local-YAML sink at the default
# directory (resolved lazily at first write). The host overrides it once at
# startup; tests typically point it at a tmp directory. Replaced whole, never
# mutated, so every reader sees a consistent sink/health pair.
_installed: Installed = _install(implicit_default_sink())

# True while ``_installed`` is still the import-time default and ``LLMKIT_LOG``
# has not been read. The first reader clears it under the lock, swapping in the
# disabled installation when the variable says off; configure_llm_logging
# clears it too, so an explicit choice is never overridden by the environment.
_implicit: bool = True
_switch_lock = threading.Lock()

# Warn-once latch for a *configured* sink that raises out of ``write`` — the
# custom-sink counterpart of LocalYamlLogSink's internal latches, so a
# permanently broken third-party sink can't flood stderr either.
_sink_latch = OnceLatch()

# Locks :func:`before_fork` acquired, released again by
# :func:`after_fork_in_parent`.
_fork_held: list[threading.Lock] = []


class _SinkContractError(TypeError):
    """A sink matched a capability protocol structurally but broke its contract.

    Its own type — rather than a bare ``TypeError`` — keeps the warn-once
    latch's ``(type, errno)`` signature distinct, so a sink that returns junk
    from ``write_returning_path`` and *also* raises a genuine ``TypeError``
    later still gets a second loud warning instead of being folded into the
    first.
    """


def configure_llm_logging(
    sink: LogSink | None, *, delivery_policy: LogDeliveryPolicy | None = None
) -> None:
    """Set the sink that receives every :class:`LLMCallRecord`.

    Pass ``None`` to disable logging entirely (writes become no-ops).
    Without a call, the import-time default :class:`LocalYamlLogSink` is
    installed unless ``LLMKIT_LOG`` says off (``off``, ``0``, ``false`` or
    ``no``; any unrecognised value also turns it off, with one warning). An
    explicit call always wins: once this has run, ``LLMKIT_LOG`` is never
    read, whatever it says.
    Optional ``delivery_policy`` configures managed queue capacity and wait/drain
    budgets before the first managed admission. Later explicit policy changes
    raise RuntimeError; omitting it preserves policy during sink swaps.
    Re-arms the warn-once latch on sink failures, so a newly configured sink
    gets a fresh loud first warning if it too turns out to be broken, and
    starts fresh :func:`get_logging_health` counts — including when the call
    restores a previously installed sink. A forked child also starts fresh
    counts for the sink it inherited.

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
    global _installed, _implicit
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
    if delivery_policy is not None:
        configure_delivery(delivery_policy)
    with _switch_lock:
        _installed = _install(sink)
        _implicit = False
    _sink_latch.succeeded()


def before_fork() -> None:
    """Hold the switch lock and the built-in sink's settings locks across ``fork``.

    Each guards a few assignments that must be seen together, such as
    ``_installed`` and ``_implicit`` in :func:`configure_llm_logging`, and
    none is held while host code runs, so waiting for them is brief. A
    child forked in the middle of one would otherwise inherit half-applied
    state: an explicitly configured sink with ``_implicit`` still set, which
    ``LLMKIT_LOG=off`` would then replace.
    """
    _ = _switch_lock.acquire()
    _fork_held.append(_switch_lock)
    sink = _installed.sink
    if isinstance(sink, LocalYamlLogSink):
        for lock in _sink_fork_locks(sink):
            _ = lock.acquire()
            _fork_held.append(lock)


def after_fork_in_parent() -> None:
    """Release what :func:`before_fork` acquired."""
    while _fork_held:
        _fork_held.pop().release()


def reset_after_fork() -> None:
    """Re-install the current sink in a forked child, with fresh locks and counts.

    Runs in the child between ``fork`` and its first line of code. The
    child's copies of the locks :func:`before_fork` held, and of the health
    lock a parent thread may have held, would otherwise stay held forever,
    and the inherited counters would describe the parent's records.
    ``_implicit`` is kept, so an unread ``LLMKIT_LOG`` still applies at the
    child's first use. A custom sink's own locks are the host's to reset.
    Takes no lock and cannot raise.
    """
    global _installed, _switch_lock
    _fork_held.clear()
    _switch_lock = threading.Lock()
    sink = _installed.sink
    if isinstance(sink, LocalYamlLogSink):
        _reset_sink_after_fork(sink)
    _installed = _install(sink)


def _apply_log_switch() -> None:
    """Read ``LLMKIT_LOG`` once and, when it says off, disable the default sink.

    Idempotent and locked against :func:`configure_llm_logging`: whichever
    takes the lock first decides, and an explicit configuration that got
    there first is left alone. An unrecognised value fails closed, because a
    prompt written to disk cannot be taken back; it warns once (the read
    happens once per process) and never raises.

    ``_implicit`` is cleared only after ``_installed`` holds the answer:
    :func:`get_installed` reads the flag without the lock, so a reader that
    sees it cleared must also see the disabled installation, never the
    default sink the switch is about to replace.
    """
    global _installed, _implicit
    invalid: str | None = None
    with _switch_lock:
        if not _implicit:
            return
        raw = blank_is_unset(os.environ.get(LOG_ENV_VAR))
        value = None if raw is None else raw.strip().casefold()
        if value is not None and value not in {"on", "1", "true", "yes"}:
            if value not in {"off", "0", "false", "no"}:
                invalid = raw
            _installed = _install(None)
        _implicit = False
    if invalid is not None:
        # Outside the lock: a host handler reading the health back must not
        # deadlock on it.
        _ = report(
            logger,
            logging.WARNING,
            "%s=%r is not one of on/off, 1/0, true/false or yes/no; LLM call logging is "
            + "off (an unrecognised value fails closed)",
            LOG_ENV_VAR,
            invalid,
        )


def get_installed() -> Installed:
    """Bind the sink and counters together before managed preparation.

    The first call (or the first :func:`get_log_sink`,
    :func:`get_logging_health` or :func:`write_llm_log`) applies
    ``LLMKIT_LOG`` to the import-time default sink.
    """
    if _implicit:
        _apply_log_switch()
    return _installed


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
    not the same one. The restore does start fresh :func:`get_logging_health`
    counts.

    Reading a module attribute instead (``llmkit.logging.registry._sink`` before
    the sink and its health were installed together, or the pre-split
    ``llmkit.logging._sink``) is what this replaces, and is the reason it is
    worth a public name: a private attribute that moves — as it did when this
    package was split out of one module — turns a save/restore helper into a
    permanent ``configure_llm_logging(None)`` with no error anywhere, caught
    only by a test that reads the private name too.
    """
    return get_installed().sink


def get_logging_health() -> LoggingHealth:
    """Return a snapshot of what happened to the records handed to the sink.

    Diagnostic warnings about logging failures go through the standard
    ``llmkit.logging`` logger, which the host's logging configuration may
    hide. A host handler that raises while handling one is contained and
    counted in :func:`~llmkit.get_diagnostics_health`, not here. This
    snapshot does not depend on either: an application can read it after a
    batch, in a health endpoint, or before exiting, and decide for itself
    whether a lost record matters.

    * ``sink`` is ``"disabled"``, ``"builtin"`` (the shipped
      :class:`LocalYamlLogSink`, whose writes llmkit verifies) or
      ``"custom"``.
    * ``written`` counts built-in writes that reached disk; ``degraded``
      counts those whose index line or retention housekeeping then failed,
      and the first written after the default folder fell back to the
      per-user one (``last_failure.stage == "location"``).
    * ``handed_off`` counts records a custom sink accepted by returning
      normally. llmkit cannot see past that return, so it is not evidence
      the record was stored.
    * ``failed`` counts records lost to a detected failure.
    * ``last_failure`` names the most recent failure's stage, error text,
      feature, label and time, never the prompt or response.

    Counts belong to the current installation and reset on every
    :func:`configure_llm_logging` call, so a temporary sink swap — even one
    restored with :func:`get_log_sink` — clears them. A forked child starts
    from zero too; the parent's counts stay with the parent. Writes run on worker
    threads; counts are lock-guarded and exact. A write still running when
    this is read is visible in ``in_flight``, but has no outcome yet.
    ``queued`` includes preparation reservations; ``dropped`` counts records
    never written, ``unavailable`` the shutdown/dead-worker subset, and
    ``wait_timeouts`` expired waits that may still deliver later.
    ``preparation_failed`` also contributes to ``failed``.
    """
    return get_installed().health.snapshot()


def write_llm_log(record: LLMCallRecord) -> Path | None:
    """Hand ``record`` to the configured sink, swallowing any failure.

    Logging must not break the LLM call, so a sink that raises an
    :class:`Exception` is caught here in addition to the sink's own
    best-effort handling. The warning about that failure is reported through
    :func:`llmkit._diagnostics.report`, so a host logging handler that raises
    while handling it is contained too and counted in
    :func:`~llmkit.get_diagnostics_health`. A handler that blocks is not
    contained: it runs inline on the calling thread and the write waits for
    it. ``BaseException`` (cancellation, ``KeyboardInterrupt``,
    ``SystemExit``) propagates.

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

    Every outcome is also counted in :func:`get_logging_health`, against the
    installation read here at entry.
    """
    return write_installed(record, get_installed())


@dataclass
class WriteAttempt:
    """Per-write accounting witness, including when diagnostics later raise."""

    accounted: bool = False


def write_installed(
    record: LLMCallRecord, installed: Installed, attempt: WriteAttempt | None = None
) -> Path | None:
    """Write against the installation bound at managed admission or direct entry."""
    sink = installed.sink
    health = installed.health
    if sink is None:
        return None
    try:
        if isinstance(sink, _OutcomeReportingLogSink):
            # The built-in sink warns through its own latches and returns
            # what happened instead of raising; only the counting happens here.
            outcome = sink.write_with_outcome(record)
            # A skip is checked first: it also carries no path, and must not
            # be counted as a failed write. Failures keep the original
            # record's feature and label, the identifiers the caller passed.
            if outcome.skipped:
                health.record_skipped()
            elif outcome.path is None:
                error = outcome.write_error or OSError("built-in sink wrote no file")
                health.record_failed(outcome.failure_stage, error, record.feature, record.label)
            else:
                health.record_written(record.feature, record.label, outcome.post_write_failures)
            if attempt is not None:
                attempt.accounted = True
            _sink_latch.succeeded()
            return outcome.path
        if isinstance(sink, _PathReturningLogSink):
            path = sink.write_returning_path(record)
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
            sink.write(record)
            path = None
    except Exception as exc:
        health.record_failed(
            "contract" if isinstance(exc, _SinkContractError) else "sink_raised",
            exc,
            record.feature,
            record.label,
        )
        if attempt is not None:
            attempt.accounted = True
        if _sink_latch.should_warn(exc):
            _ = report(
                logger,
                logging.WARNING,
                "LLM log sink raised for %s/%s (further identical failures logged at DEBUG)",
                record.feature,
                record.label,
                exc_info=True,
            )
        else:
            _ = report(
                logger,
                logging.DEBUG,
                "LLM log sink raised for %s/%s",
                record.feature,
                record.label,
                exc_info=True,
            )
        return None
    health.record_handed_off()
    if attempt is not None:
        attempt.accounted = True
    _sink_latch.succeeded()
    return path
