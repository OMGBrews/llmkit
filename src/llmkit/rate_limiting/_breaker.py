"""The opt-in circuit breaker: the "limit is effectively 0" case.

AIMD's floor of one cannot express a provider that is simply *down*, so this
state machine stops doomed work outright for a cooldown instead of letting every
call burn its retry budget into the storm. Off by default, because it flips
"eventually succeeds" into "fails fast".

Self-contained: the ring of recent outcomes, the three states, and the
single-probe claim that decides whether a cooldown ends in recovery or another
open period.
"""

import collections
import enum
import threading

import llmkit.rate_limiting._tuning as _tuning
from llmkit.rate_limiting._observability import BackpressureEvent
from llmkit.rate_limiting._tuning import (
    BREAKER_COOLDOWN,
    BREAKER_MIN_SAMPLES,
    BREAKER_THRESHOLD,
    BREAKER_WINDOW,
)


class Admit(enum.Enum):
    """A circuit breaker's verdict on one admission request.

    * ``NORMAL`` — admit and account the outcome into the rolling window (CLOSED).
    * ``PROBE`` — admit as the single HALF_OPEN probe; its outcome alone decides
      whether the breaker closes or re-opens.
    * ``REJECT`` — fast-fail with :class:`~llmkit.exceptions.CircuitOpenError`
      (OPEN within its cooldown, or HALF_OPEN with a probe already in flight).
    """

    NORMAL = "normal"
    PROBE = "probe"
    REJECT = "reject"


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker over the shared throttle/success stream.

    The aggregate guard that complements AIMD: AIMD drives the per-provider limit
    *toward* 1 under sustained pushback; the breaker is the "limit is effectively
    0 for a cooldown" case — once a provider's throttle rate over a rolling window
    crosses the trip threshold it **opens**, and while open the limiter fast-fails
    every call with :class:`~llmkit.exceptions.CircuitOpenError` instead of letting
    each one burn its retry budget into the storm.

    One instance per provider *name* (like :class:`AdaptiveState` and the
    RPM/TPM buckets — not per loop), shared across every event loop and sync
    thread. State is a ``CLOSED / OPEN / HALF_OPEN`` machine plus a count-based
    ring of the last :data:`BREAKER_WINDOW` real outcomes:

    * **CLOSED** — admit normally; each real outcome (a throttle or a success;
      an ambiguous neutral error, like AIMD, records nothing) is appended to the
      ring. Once the ring is full and its throttled fraction reaches
      :data:`BREAKER_THRESHOLD`, **open** (clearing the ring) and arm the
      cooldown.
    * **OPEN** — reject every admission for :data:`BREAKER_COOLDOWN` seconds.
      The fast rejections feed nothing back (no real call ran).
    * **HALF_OPEN** — the first admission after the cooldown becomes a single,
      process-wide, lock-guarded **probe**; everyone else is rejected until it
      resolves. The probe's outcome alone decides: a clean success **closes** the
      breaker (and clears the ring); any failure — a throttle, a neutral error,
      or cancellation before/while it ran — **re-opens** it and re-arms the
      cooldown, so a probe can never wedge the breaker HALF_OPEN.

    State is guarded by a plain :class:`threading.Lock` (shared safely across
    loops and threads, exactly like :class:`RateBucket` / :class:`AdaptiveState`),
    held only for O(1) work and **never across an await**. Each mutator returns
    the :class:`BackpressureEvent` to emit (or ``None``) so the caller fires the
    observability callback *outside* the lock. The clock is read through
    :func:`_tuning.now`, so cooldown is deterministic under a monkeypatched clock.
    """

    def __init__(self, provider: str, ceiling: int) -> None:
        self._provider: str = provider
        # The breaker's own effective ceiling, reported on transitions so an
        # event reads ceiling → 0 (open) → 1 (one probe) → ceiling (closed).
        self._ceiling: int = ceiling
        self._state: CircuitState = CircuitState.CLOSED
        self._window: collections.deque[bool] = collections.deque(maxlen=BREAKER_WINDOW)
        # When the breaker last opened, for the cooldown comparison. ``-inf`` is
        # unused while CLOSED (the state guard gates every read of it).
        self._opened_at: float = float("-inf")
        # Whether the single HALF_OPEN probe is currently out — the lock-guarded
        # flag that keeps a second concurrent call (on any loop) from probing too.
        self._probe_in_flight: bool = False
        self._lock: threading.Lock = threading.Lock()

    def admit(self) -> tuple[Admit, BackpressureEvent | None]:
        """Decide whether to admit one call, transitioning OPEN → HALF_OPEN if the
        cooldown has elapsed. Returns the verdict and any transition event to emit.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return Admit.NORMAL, None
            if self._state is CircuitState.OPEN:
                if _tuning.now() - self._opened_at < BREAKER_COOLDOWN:
                    return Admit.REJECT, None
                # Cooldown elapsed: admit exactly one probe and announce HALF_OPEN.
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return Admit.PROBE, BackpressureEvent(self._provider, 0, 1, "breaker_half_open")
            # HALF_OPEN: the probe is in flight (or, defensively, just resolved on
            # another loop). Admit a probe only if none is out; otherwise reject.
            if self._probe_in_flight:
                return Admit.REJECT, None
            self._probe_in_flight = True
            return Admit.PROBE, None

    def on_record(self, *, throttled: bool) -> BackpressureEvent | None:
        """Account one *normal* (non-probe) call outcome; open if the ring trips.

        Only the CLOSED path accumulates the ring: a normal call that was admitted
        CLOSED but completes after the breaker has since opened (or a probe is
        underway) does not disturb the decision — the probe owns the transition.
        """
        with self._lock:
            if self._state is not CircuitState.CLOSED:
                return None
            self._window.append(throttled)
            if len(self._window) < BREAKER_MIN_SAMPLES:
                return None
            throttled_fraction = sum(self._window) / len(self._window)
            if throttled_fraction < BREAKER_THRESHOLD:
                return None
            return self._open_locked()

    def on_probe_success(self) -> BackpressureEvent | None:
        """Resolve the HALF_OPEN probe with a clean success: close and clear."""
        with self._lock:
            self._probe_in_flight = False
            if self._state is not CircuitState.HALF_OPEN:
                return None
            self._state = CircuitState.CLOSED
            self._window.clear()
            return BackpressureEvent(self._provider, 1, self._ceiling, "breaker_closed")

    def on_probe_failure(self) -> BackpressureEvent | None:
        """Resolve the HALF_OPEN probe with *any* failure (throttle, neutral error,
        or cancellation before it ran): re-open and re-arm the cooldown.

        Always releases the probe claim, so the breaker can never wedge HALF_OPEN.
        """
        with self._lock:
            self._probe_in_flight = False
            if self._state is not CircuitState.HALF_OPEN:
                return None
            return self._open_locked()

    def _open_locked(self) -> BackpressureEvent:
        # Caller holds ``self._lock``. Transition to OPEN, arm the cooldown, and
        # clear the ring so the next CLOSED period (after a probe closes it) starts
        # fresh. ``old_limit`` is the ceiling for every open — the breaker collapses
        # effective capacity to 0 regardless of whether it tripped from CLOSED or a
        # failed probe re-opened it.
        self._state = CircuitState.OPEN
        self._opened_at = _tuning.now()
        self._window.clear()
        return BackpressureEvent(self._provider, self._ceiling, 0, "breaker_open")
