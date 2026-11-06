"""Classifying one call's outcome, and feeding it back to both mechanisms.

The async and sync acquire paths must react to a completed call identically —
two parallel copies of "was this a throttle?" would drift — so both call the one
:func:`record_gate_outcome` here. AIMD reacts only to overload; the breaker
also reacts to connection failures and transient server errors.

:class:`ConcurrencyGate` is satisfied **structurally**: neither gate in
``_adaptive`` inherits from it. That absence is deliberate — it is what keeps
this module a one-way dependency of the gates rather than a cycle with them.
"""

from typing import Protocol

import httpx
import openai

from llmkit.exceptions import underlying_provider_error
from llmkit.rate_limiting._breaker import CircuitBreaker
from llmkit.rate_limiting._observability import BackpressureEvent, emit_backpressure
from llmkit.rate_limiting._tuning import THROTTLE_STATUS_CODES


def is_throttle_signal(exc: BaseException) -> bool:
    """Whether *exc* is a provider overload signal worth backing off on (AIMD).

    Unwraps first (:func:`~llmkit.exceptions.underlying_provider_error`) — a
    structured call surfaces the provider error wrapped in
    ``InstructorRetryException``, so without the unwrap a 429/503 from the
    *structured* path (the incident workload) would be invisible here. Classifies
    by HTTP ``status_code`` against :data:`THROTTLE_STATUS_CODES`; LiteLLM/OpenAI
    status errors (including litellm's 503 ``ServiceUnavailableError``) carry the
    code, so a name check is unnecessary. Anything without a throttle status —
    a ``ValidationError``, a bare network timeout, a 500 — is neutral.
    """
    root = underlying_provider_error(exc)
    status = getattr(root, "status_code", None)
    return isinstance(status, int) and status in THROTTLE_STATUS_CODES


# Generic 5xx statuses such as 501/505 can signal permanent incompatibility.
# Count the transient statuses supported here, not every server-side failure.
_OUTAGE_STATUS_CODES = THROTTLE_STATUS_CODES | {500, 502, 504}


def is_outage_signal(exc: BaseException) -> bool:
    """Whether an outcome is evidence that the provider cannot serve requests.

    The breaker shares AIMD's overload signals, and additionally sees connection
    failures and transient server errors. A positive HTTP outage status takes
    precedence over the exception class: LiteLLM represents a provider's HTTP
    504 response as a ``Timeout``. Without that status, timeout classes are
    neutral: llmkit's caller deadline raises a built-in ``TimeoutError``, and
    an SDK timeout cannot establish whether the chosen wait was too short or
    the provider was down. OpenAI's timeout inherits from its connection error,
    so exclude it before matching connection classes.
    Authentication, validation, cancellation and programming errors are neutral.
    """
    root = underlying_provider_error(exc)
    status = getattr(root, "status_code", None)
    if isinstance(status, int) and status in _OUTAGE_STATUS_CODES:
        return True
    if isinstance(root, (TimeoutError, openai.APITimeoutError, httpx.TimeoutException)):
        return False
    return isinstance(root, (ConnectionError, openai.APIConnectionError, httpx.NetworkError))


class ConcurrencyGate(Protocol):
    """The outcome-feedback surface shared by the async and sync concurrency gates.

    Both :class:`AdaptiveGate` (async) and :class:`SyncAdaptiveGate` (sync)
    implement it, so :func:`record_gate_outcome` can drive either path through
    the *same* classification code — the parity is one implementation, not two
    parallel copies. Saturation is no longer part of this surface: it is judged
    inside :meth:`AdaptiveState.on_throttle` on the provider-wide aggregate.
    """

    def on_throttle(self) -> BackpressureEvent | None: ...
    def on_success(self) -> BackpressureEvent | None: ...


def record_gate_outcome(
    gate: ConcurrencyGate,
    breaker: CircuitBreaker | None,
    *,
    is_probe: bool,
    adaptive: bool,
    outcome: BaseException | None,
) -> None:
    """Feed one call's outcome to AIMD and the breaker, emitting backpressure.

    The single shared classifier at the ``yield`` for **both**
    :meth:`GlobalRateLimiter.acquire_async` and
    :meth:`GlobalRateLimiter.acquire_sync`, so their per-provider feedback is
    genuinely identical rather than a parallel copy that can drift.

    ``gate`` is the concurrency gate the call held (async or sync); ``breaker`` is
    the provider's :class:`CircuitBreaker` or ``None`` when the breaker is off;
    ``is_probe`` marks a HALF_OPEN probe (whose outcome alone resolves the
    breaker); ``adaptive`` is whether AIMD is on; ``outcome`` is the exception the
    call body raised, or ``None`` on success.

    A throttle (429/503/529, **unwrapped** so a structured call's wrapped error is
    seen) while saturated lowers the AIMD limit; a success recovers it. A
    non-throttle error leaves AIMD alone. The breaker independently classifies
    outage signals; neutral errors feed neither its window nor recovery. A
    neutral probe releases its claim so a later caller may probe again. Backpressure
    events fire outside the gate/breaker locks.
    """
    throttled = (
        is_throttle_signal(outcome)
        if (outcome is not None and (adaptive or breaker is not None))
        else False
    )
    # Resolve both mechanisms before invoking host code. A callback interrupt
    # (or a failing diagnostic logger) must not strand a HALF_OPEN probe after
    # AIMD has changed but before the breaker has recorded the same outcome.
    adaptive_event: BackpressureEvent | None = None
    breaker_event: BackpressureEvent | None = None
    if outcome is not None:
        if adaptive and throttled:
            adaptive_event = gate.on_throttle()
        if breaker is not None:
            outage = is_outage_signal(outcome)
            if is_probe:
                if outage:
                    breaker_event = breaker.on_probe_failure()
                else:
                    breaker.on_probe_neutral()
            elif outage:
                breaker_event = breaker.on_record(outage=True)
    else:
        if adaptive:
            adaptive_event = gate.on_success()
        if breaker is not None:
            if is_probe:
                breaker_event = breaker.on_probe_success()
            else:
                breaker_event = breaker.on_record(outage=False)
    emit_backpressure(adaptive_event)
    emit_backpressure(breaker_event)
