"""Backpressure events and the per-attempt queue-wait stamp.

The limiter's two observability channels, kept together because both are
context-scoped state that must exist exactly once in the process:

* :func:`backpressure_callback` installs a callback that
  :func:`emit_backpressure` fires whenever the adaptive limit moves or the
  circuit breaker changes state;
* the queue-wait stamp records how long one attempt waited behind the RPM /
  TPM / concurrency gates, so a log record can separate queueing inside llmkit
  from real provider latency.

Both ride context variables, which propagate across
``run_sync`` and ``to_thread`` boundaries without threading a parameter through
every call. Their **object identity is load-bearing**: a second definition of
either would not raise, it would silently disconnect installation from emission
(callbacks that never fire) or stamping from reading (``queue_wait_ms`` forever
``None`` in every record). Define each here once; never re-declare it in a
facade.

The queue-wait protocol has three parties and is named rather than open-coded:
the call layer calls :func:`begin_queue_wait` before handing an attempt to the
transport, :meth:`~llmkit.rate_limiting.GlobalRateLimiter.acquire_async` calls
:func:`stamp_queue_wait` the moment a slot is held, and the call layer reads
:func:`current_queue_wait_ms` when it builds the attempt's record. The reset is
what keeps a stale stamp from a previous attempt off an attempt that failed
before ever acquiring.
"""

import contextlib
import logging
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, Protocol

# Named explicitly rather than via ``__name__`` so every module in this package
# keeps emitting under the one logger name operators already filter on.
logger = logging.getLogger("llmkit.rate_limiting")


@dataclass(frozen=True)
class BackpressureEvent:
    """A per-provider backpressure transition, for observability.

    Emitted to the callback installed by :func:`backpressure_callback` whenever
    the AIMD limit actually moves *or* the opt-in circuit breaker changes state,
    so a host can *see* backpressure in real time (and wire its own metrics)
    instead of reconstructing it from call logs.

    Attributes:
        provider: The normalized (casefolded) provider key the budget is keyed by
            — the same identity the limiter accounts every dimension under.
        old_limit: The effective per-provider concurrency limit before the change.
        new_limit: The effective limit after the change.
        reason: which transition this is —

            * ``"throttle"`` — a provider overload signal lowered the AIMD limit;
            * ``"recover"`` — sustained no-throttle time raised the AIMD limit
              back toward the configured ceiling;
            * ``"breaker_open"`` — the breaker tripped and is now fast-failing;
              ``new_limit`` is ``0`` (it admits nothing) and ``old_limit`` is the
              ceiling it collapsed from;
            * ``"breaker_half_open"`` — the cooldown elapsed and the breaker
              admitted a single probe (``old_limit`` ``0`` → ``new_limit`` ``1``);
            * ``"breaker_closed"`` — the probe succeeded; the breaker is back to
              full capacity (``old_limit`` ``1`` → ``new_limit`` the ceiling).

            The breaker reasons report the breaker's *own* effective ceiling
            (ceiling → 0 → 1 → ceiling), which is independent of — and not to be
            confused with — the AIMD limit the ``"throttle"`` / ``"recover"``
            reasons carry.
    """

    provider: str
    old_limit: int
    new_limit: int
    reason: Literal[
        "throttle",
        "recover",
        "breaker_open",
        "breaker_half_open",
        "breaker_closed",
    ]


class BackpressureCallback(Protocol):
    """Receives :class:`BackpressureEvent`s from the adaptive limiter.

    Installed via :func:`backpressure_callback`. Invoked on the loop that observed
    the limit change; implementations must not raise (a raise is caught and
    logged, never allowed to break a call) and should not block. The event is
    passed positionally, so a plain ``def f(event): ...``, a lambda, or even
    ``list.append`` all satisfy this protocol.
    """

    def __call__(self, event: BackpressureEvent, /) -> None: ...


_backpressure_callback: ContextVar[BackpressureCallback | None] = ContextVar(
    "_backpressure_callback", default=None
)

# The most recent completed acquire's queue wait in *this context*, in
# milliseconds. Stamped by ``GlobalRateLimiter.acquire_async`` (0.0 when the
# limiter is disabled), reset to None by the call layer before each attempt
# and copied onto that attempt's LLMCallRecord — so the log can separate time
# queued behind llmkit's own limiter from true provider latency. ``None``
# means the attempt never completed an acquire (it failed before or during
# the gate phase).
_queue_wait_ms: ContextVar[float | None] = ContextVar("_llmkit_queue_wait_ms", default=None)


def begin_queue_wait() -> None:
    """Clear the queue-wait stamp before an attempt reaches the transport.

    The first of the three steps described in this module's docstring, and the
    one that is easy to forget: without it an attempt that failed *before* the
    limiter ever admitted it would inherit the previous attempt's wait — or an
    unrelated earlier call's — and the log would attribute queueing to a call
    that never queued. ``None`` is the honest reading of "this attempt never
    completed an acquire".
    """
    _ = _queue_wait_ms.set(None)


def stamp_queue_wait(milliseconds: float) -> None:
    """Record how long the caller waited before its slot was granted.

    Called by :meth:`~llmkit.rate_limiting.GlobalRateLimiter.acquire_async` at
    the instant a slot is held — and with ``0.0`` when the limiter is disabled,
    which is a real measurement (no wait), not a missing one.
    """
    _ = _queue_wait_ms.set(milliseconds)


def current_queue_wait_ms() -> float | None:
    """The stamp for the attempt running in this context, or ``None``.

    ``None`` means the attempt never completed an acquire: it failed before or
    during the gate phase, or it ran a transport that does not acquire at all.
    Read by the call layer when it builds the attempt's
    :class:`~llmkit.LLMCallRecord`.
    """
    return _queue_wait_ms.get()


def emit_backpressure(event: BackpressureEvent | None) -> None:
    """Fire the installed backpressure callback for *event* (a no-op if ``None``).

    Read from a context variable so it propagates across the ``run_sync`` /
    ``to_thread`` boundaries (the persistent-loop bridge copies the caller's
    context) without threading a parameter through every call. A callback that
    raises is swallowed and logged — observability must never break a call.
    """
    if event is None:
        return
    callback = _backpressure_callback.get()
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        logger.exception("backpressure callback raised for %s", event.provider)


@contextlib.contextmanager
def backpressure_callback(callback: BackpressureCallback | None) -> Generator[None]:
    """Install a backpressure callback for the current dynamic scope.

    The callback receives a :class:`BackpressureEvent` each time the adaptive
    limiter changes a provider's concurrency limit (a throttle-driven decrease or
    a time-driven recovery) or the opt-in circuit breaker changes state
    (``breaker_open`` / ``breaker_half_open`` / ``breaker_closed``). Read from a
    context variable, so it crosses the ``run_sync`` thread boundary like the
    retry progress callback. Pass ``None`` to disable callbacks within an inner
    scope; the previous value is restored on exit.

    Usage::

        def on_event(event: BackpressureEvent) -> None:
            metrics.gauge(f"llm.concurrency.{event.provider}", event.new_limit)


        with backpressure_callback(on_event):
            ...  # calls here report limit changes to ``on_event``
    """
    token = _backpressure_callback.set(callback)
    try:
        yield
    finally:
        _backpressure_callback.reset(token)
