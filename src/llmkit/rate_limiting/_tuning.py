"""Internal tuning constants and the shared monotonic clock.

The package's dependency-free leaf: every other module reads its knobs from
here, and nothing here reads anything back. The constants are deliberately
*internal* — the host owns the ceiling (``max_concurrent``) and the RPM/TPM
numbers; the library owns how the limit moves beneath them.

:func:`now` is the package's single clock read, and it is a **test seam**: a
test freezes or advances time with ``monkeypatch.setattr(rate_limiting._tuning,
"now", ...)``. That only works while every caller resolves it late, through this
module — ``_tuning.now()``, never ``from llmkit.rate_limiting._tuning import
now``, which would bind the function object at import and silently kill the
seam. The symbols are public-named inside a private module so siblings can
import them without tripping ``reportPrivateUsage``.
"""

import time


def now() -> float:
    """Monotonic clock read, indirected so offline tests can advance time.

    The token buckets read the clock through this one function; a test can
    monkeypatch ``llmkit.rate_limiting._tuning.now`` to drive refill deterministically
    without real sleeping.
    """
    return time.monotonic()


#: Burst depth for a TPM bucket, expressed in seconds of the sustained rate:
#: ``capacity = tpm * TPM_BURST_SECONDS / 60``. A one-second reservoir keeps the
#: worst-case per-window overshoot to ~1.7% of ``tpm`` instead of the 2x a
#: full-minute capacity (``= tpm``) would allow after an idle stretch. TPM debits
#: *after* the call and gates only while exhausted, so even this small reservoir
#: never makes a quiet process's first call wait — capacity only bounds how much
#: an idle bucket may bank. (RPM's burst is the concurrency width instead — see
#: :meth:`GlobalRateLimiter._get_rpm_bucket` — because requests have a
#: concurrency analog and tokens do not.) Decoupling capacity from the per-minute
#: number is the burst-semantics decision recorded in opinions.md §6.4.
TPM_BURST_SECONDS: float = 1.0

#: HTTP status codes that count as a provider *overload* signal for adaptive
#: concurrency (AIMD). 429 (rate limit), 503 (service unavailable / shared
#: capacity), 529 (overloaded). Deliberately **excludes** 500 (a generic server
#: error, not necessarily overload), 408, and bare network timeouts — those are
#: ambiguous transport noise, and over-reacting to them by collapsing concurrency
#: is the failure mode the saturation gate (below) exists to avoid. Widen on
#: evidence, not on speculation.
THROTTLE_STATUS_CODES: frozenset[int] = frozenset({429, 503, 529})

#: AIMD tuning. These are deliberately **internal** constants, not public knobs:
#: the host owns the *ceiling* (``max_concurrent``) and the RPM/TPM numbers; the
#: library owns *how the limit moves beneath them* — mechanism it should own, like
#: the temperature default and the burst-depth choice (see opinions.md §6.4/§8).
#:
#: * ``AIMD_DECREASE_FACTOR`` — multiplicative decrease on a throttle: the limit
#:   halves (floored at 1).
#: * ``AIMD_DECREASE_COOLDOWN`` — refractory period (seconds): at most one
#:   decrease per window, so a fan-out's correlated burst of throttles collapses
#:   to a *single* halving instead of crashing to the floor on one bad instant.
#: * ``AIMD_RECOVERY_INTERVAL`` — additive increase is **wall-clock-paced**: the
#:   limit climbs by one for each interval that has elapsed since the last change
#:   with no intervening throttle. Recovery from the floor to a ceiling of 8 is
#:   ~``7 * interval`` seconds regardless of offered load — bounded in wall-clock,
#:   unlike a per-success ("generation") rule that needs O(ceiling**2) successful
#:   calls and starves the tail of a finite batch.
AIMD_DECREASE_FACTOR: float = 0.5
AIMD_DECREASE_COOLDOWN: float = 1.0
AIMD_RECOVERY_INTERVAL: float = 3.0

#: Bounded re-check interval (seconds) for a blocked **sync** concurrency waiter
#: (:class:`SyncAdaptiveGate`). The sync gate and the async gate park on
#: different primitives over the *same* shared :class:`AdaptiveState`, so an
#: async-side success that raises the limit wakes no sync waiter — nothing on the
#: sync side signalled. A sync waiter therefore never blocks *unboundedly* on its
#: own condition: it re-reads the live limit at most this many seconds later, so a
#: cross-population limit increase can lift it in bounded time. Kept small so that
#: latency stays low; a decrease/refill is still honoured immediately via the
#: normal release path. Deliberately a module constant (not a public knob), like
#: the AIMD constants.
SYNC_GATE_POLL_INTERVAL: float = 0.05

#: Circuit-breaker tuning. Like the AIMD constants these are deliberately
#: **internal**: the host owns the *switch* (``breaker``), the library owns the
#: mechanism. The breaker is the "limit is effectively 0 for a cooldown" case
#: AIMD's floor-of-1 cannot express — it stops doomed work outright when a
#: provider is *down* (see opinions.md §6.4). Defaults are conservative because
#: it is opt-in and flips "eventually succeeds" → "fails fast".
#:
#: * ``BREAKER_WINDOW`` — size of the count-based ring of recent *real* outcomes
#:   (a throttle or a success; a fast ``CircuitOpenError`` and an ambiguous
#:   neutral error record nothing). O(N) state, no timestamps — one wrong OPEN is
#:   bounded to a single cooldown by the HALF_OPEN probe, so a count ring beats a
#:   time-bucketed window's complexity here.
#: * ``BREAKER_MIN_SAMPLES`` — never trip before the ring holds this many
#:   outcomes, so one early throttle in a near-empty ring can't open the breaker.
#:   Equal to the window, so the ring must be *full* before it can trip.
#: * ``BREAKER_THRESHOLD`` — open once the throttled fraction of a full ring
#:   reaches this (0.5 ⇒ half of the recent real outcomes were throttles).
#: * ``BREAKER_COOLDOWN`` — seconds the breaker fast-fails while OPEN before it
#:   admits a single HALF_OPEN probe to test whether the provider has recovered.
BREAKER_WINDOW: int = 20
BREAKER_MIN_SAMPLES: int = 20
BREAKER_THRESHOLD: float = 0.5
BREAKER_COOLDOWN: float = 30.0
