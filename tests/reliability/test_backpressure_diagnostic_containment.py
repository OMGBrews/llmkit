"""A raising ``llmkit.rate_limiting`` handler never breaks a backpressure emit.

:func:`~llmkit.rate_limiting._observability.emit_backpressure` logs a raising
backpressure callback at ERROR with the callback's exception attached. The
limiter fires it on both the admission path (a ``recover`` event while a slot
is granted) and the outcome path (a ``throttle`` event while a provider error
is being classified). A host handler whose ``emit`` raises used to escape that
log call, replacing the call's answer or the provider's error with the
handler's exception. These tests install such a handler on
``llmkit.rate_limiting`` together with a raising callback and check that:

* the emit returns normally, and the original answer or provider exception
  type still reaches the caller through the real limiter;
* each lost record is counted exactly once in
  :func:`llmkit.get_diagnostics_health`, on the first occurrence and again on a
  second occurrence in the same process (this seam has no latch);
* the handler sees exactly one ERROR record per emit, carrying the callback's
  exception as ``exc_info``, and nothing further from the seam; and
* ``asyncio.CancelledError`` and ``KeyboardInterrupt`` raised by the handler
  are not contained.

Driven with a frozen clock and an :mod:`llmkit.testing` fake, so everything is
deterministic and offline.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import BackpressureEvent, backpressure_callback, configure_rate_limit, text_llm_call
from llmkit.rate_limiting import GlobalRateLimiter, RateLimitPolicy, _tuning
from llmkit.rate_limiting._observability import emit_backpressure
from llmkit.rate_limiting._tuning import AIMD_RECOVERY_INTERVAL
from llmkit.testing import Reply, install
from tests._support.diagnostics import RaisingHandler, lost_diagnostics, raising_handler

_LOGGER = "llmkit.rate_limiting"
_MESSAGE = "backpressure callback raised for %s"
_EVENT = BackpressureEvent("google", 2, 1, "throttle")


class _CallbackBroke(Exception):
    """What the raising backpressure callback raises."""


class _BrokenCallback:
    """A backpressure callback that raises, keeping each exception it raised."""

    def __init__(self) -> None:
        self.raised: list[_CallbackBroke] = []

    def __call__(self, event: BackpressureEvent, /) -> None:
        error = _CallbackBroke(f"callback broke on {event.reason}")
        self.raised.append(error)
        raise error


def _assert_records(handler: RaisingHandler, callback: _BrokenCallback, provider: str) -> None:
    """One ERROR record per callback failure, each carrying that failure."""
    assert [(r.levelno, r.msg, r.args) for r in handler.records] == [
        (logging.ERROR, _MESSAGE, (provider,))
    ] * len(callback.raised)
    for record, error in zip(handler.records, callback.raised, strict=True):
        assert record.exc_info is not None
        assert record.exc_info[1] is error


def _wrapped_429() -> InstructorRetryException:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    inner = openai.RateLimitError("boom", response=httpx.Response(429, request=request), body=None)
    return InstructorRetryException(inner, n_attempts=1, total_usage=0)


async def _saturated_throttle(key: str) -> None:
    """Throttle one call while another holds a slot (cap 2), re-raising its error."""
    holder_in = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async(key):
            holder_in.set()
            _ = await release_holder.wait()

    holder_task = asyncio.create_task(holder())
    _ = await holder_in.wait()
    try:
        async with GlobalRateLimiter.acquire_async(key):
            raise _wrapped_429()
    finally:
        release_holder.set()
        await holder_task


def _arm_recovery(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """Make the next successful acquire on *key* emit a ``recover`` event."""
    clock = {"t": 1_000.0}
    monkeypatch.setattr(_tuning, "now", lambda: clock["t"])
    configure_rate_limit(policy=RateLimitPolicy(max_concurrent=4))
    gate = GlobalRateLimiter._get_async_gate(key)
    gate._state._limit = 1  # as if AIMD had driven it down
    gate._state._recovery_anchor = clock["t"]
    clock["t"] += AIMD_RECOVERY_INTERVAL


# --- the emit itself: first and second occurrence ---------------------------


def test_emit_contains_raising_handler_first_and_repeated() -> None:
    """Both emits return normally; each loss is counted once; no re-logging."""
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with backpressure_callback(callback), raising_handler(_LOGGER) as handler:
        emit_backpressure(_EVENT)
        assert lost_diagnostics() == before + 1
        emit_backpressure(_EVENT)
        assert lost_diagnostics() == before + 2

    assert len(callback.raised) == 2
    _assert_records(handler, callback, "google")


def test_emit_with_recording_handler_pins_exc_info() -> None:
    """With a handler that only records, the ERROR record carries the callback's error
    and nothing is counted as lost."""
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with (
        backpressure_callback(callback),
        raising_handler(_LOGGER, raise_on=lambda _record: False) as handler,
    ):
        emit_backpressure(_EVENT)

    assert lost_diagnostics() == before
    _assert_records(handler, callback, "google")
    assert handler.records[0].funcName == "emit_backpressure"


# --- through the real limiter: admission success and provider failure ------


async def test_recover_on_admission_keeps_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recover event fired while a slot is granted does not break the call's body."""
    _arm_recovery(monkeypatch, "google")
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with backpressure_callback(callback), raising_handler(_LOGGER) as handler:
        async with GlobalRateLimiter.acquire_async("google"):
            answer = "answer"

    assert answer == "answer"
    assert [e.args[0] for e in callback.raised] == ["callback broke on recover"]
    assert lost_diagnostics() == before + 1
    _assert_records(handler, callback, "google")


async def test_throttle_on_outcome_keeps_the_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throttle event fired while classifying a 429 leaves that 429 as the error,
    on the first throttle and on a second one in the same process."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)
    configure_rate_limit(policy=RateLimitPolicy(max_concurrent=2))
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with backpressure_callback(callback), raising_handler(_LOGGER) as handler:
        with pytest.raises(InstructorRetryException):
            await _saturated_throttle("google")
        assert lost_diagnostics() == before + 1
        assert GlobalRateLimiter._adaptive_states["google"].limit() == 1

        # A second provider's fresh AIMD state throttles again. Reconfiguring
        # would instead log the configure-time info, which stays uncontained.
        with pytest.raises(InstructorRetryException):
            await _saturated_throttle("anthropic")
        assert lost_diagnostics() == before + 2

    assert [e.args[0] for e in callback.raised] == ["callback broke on throttle"] * 2
    assert [r.args for r in handler.records] == [("google",), ("anthropic",)]
    assert all(r.levelno == logging.ERROR and r.msg == _MESSAGE for r in handler.records)
    for record, error in zip(handler.records, callback.raised, strict=True):
        assert record.exc_info is not None
        assert record.exc_info[1] is error


async def test_text_call_returns_its_answer_through_a_recover_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: a text call whose admission emits ``recover`` still returns."""
    _arm_recovery(monkeypatch, "ollama")
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with (
        install(lambda _request: Reply.text("hi")),
        backpressure_callback(callback),
        raising_handler(_LOGGER, raise_on=lambda r: r.msg == _MESSAGE) as handler,
    ):
        answer = await text_llm_call("hi", feature="backpressure-containment")

    assert answer == "hi"
    assert [e.args[0] for e in callback.raised] == ["callback broke on recover"]
    assert lost_diagnostics() == before + 1
    errors = [r for r in handler.records if r.levelno >= logging.ERROR]
    assert [(r.msg, r.args) for r in errors] == [(_MESSAGE, ("ollama",))]


# --- BaseException from the handler propagates ------------------------------


@pytest.mark.parametrize("error", [asyncio.CancelledError, KeyboardInterrupt])
def test_base_exception_from_handler_propagates(error: type[BaseException]) -> None:
    """Containment covers ``Exception`` only: cancellation and interrupts escape."""
    callback = _BrokenCallback()
    before = lost_diagnostics()
    with (
        backpressure_callback(callback),
        raising_handler(_LOGGER, error=error) as handler,
        pytest.raises(error),
    ):
        emit_backpressure(_EVENT)

    assert lost_diagnostics() == before
    assert len(handler.records) == 1
