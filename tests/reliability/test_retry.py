"""Tests for :mod:`llmkit.retry`.

Covers the progress-callback channel a caller can use to surface
per-attempt retries on its own progress stream. The default
`logger.warning` path is exercised implicitly elsewhere; these tests
pin the callback contract specifically.
"""

from __future__ import annotations

import asyncio
import logging
import random

import pytest

from llmkit.retry import RetryPolicy, handle_retry_failure, retry_progress_callback, with_retries


class _Callback:
    """Records progress callback invocations in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, str]] = []

    def __call__(
        self,
        *,
        label: str,
        attempt: int,
        max_attempts: int,
        error: BaseException,
    ) -> None:
        self.calls.append((label, attempt, max_attempts, str(error)))


@pytest.mark.asyncio
async def test_progress_callback_invoked_on_non_final_failure() -> None:
    """Failing once then succeeding fires exactly one callback for attempt 1/2."""
    cb = _Callback()
    attempts = [0]

    async def flaky() -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("first try fails")
        return "ok"

    with retry_progress_callback(cb):
        result = await with_retries(
            flaky,
            policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)),
            label="x",
        )

    assert result == "ok"
    assert cb.calls == [("x", 1, 2, "first try fails")]


@pytest.mark.asyncio
async def test_progress_callback_receives_max_attempts_keyword() -> None:
    """The callback is invoked with the ``max_attempts`` keyword (the unified
    name) carrying the total outer-attempt ceiling — not the removed
    ``max_retries`` alias."""
    seen: list[dict[str, object]] = []

    def record(**kwargs: object) -> None:
        seen.append(kwargs)

    attempts = [0]

    async def flaky() -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("first try fails")
        return "ok"

    with retry_progress_callback(record):
        result = await with_retries(
            flaky,
            policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)),
            label="x",
        )

    assert result == "ok"
    assert len(seen) == 1
    kwargs = seen[0]
    assert kwargs["max_attempts"] == 2
    assert "max_retries" not in kwargs
    assert kwargs["label"] == "x"
    assert kwargs["attempt"] == 1


@pytest.mark.asyncio
async def test_progress_callback_not_invoked_on_first_success() -> None:
    """If the first attempt succeeds, the callback is never called."""
    cb = _Callback()

    async def succeed() -> int:
        return 42

    with retry_progress_callback(cb):
        result = await with_retries(
            succeed,
            policy=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0, retry_on=(ValueError,)),
            label="x",
        )

    assert result == 42
    assert cb.calls == []


@pytest.mark.asyncio
async def test_progress_callback_not_invoked_for_final_failure() -> None:
    """When every attempt fails, the final attempt logs ERROR and does
    NOT fire the callback. Callers learn about exhaustion via the
    re-raised exception. The callback is reserved for retry events that
    are recoverable in principle."""
    cb = _Callback()

    async def always_fails() -> str:
        raise ValueError("nope")

    with retry_progress_callback(cb), pytest.raises(ValueError, match="nope"):
        _ = await with_retries(
            always_fails,
            policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)),
            label="x",
        )

    # Only one callback: attempt 1/2 (non-final). Attempt 2/2 is final → no callback.
    assert cb.calls == [("x", 1, 2, "nope")]


@pytest.mark.asyncio
async def test_progress_callback_default_none_no_op() -> None:
    """No installed callback ⇒ no error, retry semantics preserved."""
    attempts = [0]

    async def flaky() -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("first try fails")
        return "ok"

    # No retry_progress_callback context — relies on default ContextVar value (None).
    result = await with_retries(
        flaky,
        policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)),
        label="x",
    )
    assert result == "ok"


@pytest.mark.asyncio
async def test_progress_callback_exception_does_not_break_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A misbehaving callback must not break retry semantics — the retry
    helper is the contract surface for resilience."""

    def bad_callback(**_kwargs: object) -> None:
        raise RuntimeError("callback boom")

    attempts = [0]

    async def flaky() -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("first try fails")
        return "ok"

    with retry_progress_callback(bad_callback), caplog.at_level(logging.ERROR):
        result = await with_retries(
            flaky,
            policy=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)),
            label="x",
        )

    assert result == "ok"
    assert any("retry progress callback raised" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_explicit_zero_backoff_disables_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A policy with zero backoff performs eligible retries without sleeping."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    attempts = [0]

    async def flaky() -> str:
        attempts[0] += 1
        if attempts[0] < 3:
            raise ValueError("fail")
        return "ok"

    result = await with_retries(
        flaky,
        policy=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0, retry_on=(ValueError,)),
        label="x",
    )
    assert result == "ok"
    assert slept == []  # no backoff configured


@pytest.mark.asyncio
async def test_backoff_sleeps_between_retries_with_jittered_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With backoff configured, a sleep fires once per non-final failure,
    each bounded by exponential ceiling ``base * 2**(attempt-1)``. Pin the
    jitter to its max so the ceiling is asserted deterministically."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    # random.uniform(0, ceiling) -> ceiling, so we assert the exact ceiling.
    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    async def always_fails() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        _ = await with_retries(
            always_fails,
            policy=RetryPolicy(max_attempts=3, backoff_base_seconds=2.0, retry_on=(ValueError,)),
            label="x",
        )

    # Two non-final failures (attempts 1 and 2) → two sleeps; attempt 3 is
    # final (no sleep). Ceilings: 2*2**0=2.0, 2*2**1=4.0.
    assert slept == [2.0, 4.0]


@pytest.mark.asyncio
async def test_backoff_ceiling_is_capped_at_default_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a large attempt budget, no sleep ever exceeds the default
    ``max_backoff_seconds`` cap (30s). Uncapped, attempt 11 at base 2.0
    would permit ``2 * 2**10 = 2048`` seconds; the cap bounds every sleep.
    Jitter is pinned to its max so the asserted values are the ceilings."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    async def always_fails() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        _ = await with_retries(
            always_fails,
            policy=RetryPolicy(max_attempts=12, backoff_base_seconds=2.0, retry_on=(ValueError,)),
            label="x",
        )

    # 11 non-final failures → 11 sleeps. Early attempts keep the pure
    # exponential ceiling; once it crosses 30s the cap holds it flat.
    assert len(slept) == 11
    assert slept[:4] == [2.0, 4.0, 8.0, 16.0]
    assert all(delay <= 30.0 for delay in slept)
    assert slept[4:] == [30.0] * 7


@pytest.mark.asyncio
async def test_custom_max_backoff_seconds_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``max_backoff_seconds`` caps the jitter ceiling in place
    of the 30s default."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    async def always_fails() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        _ = await with_retries(
            always_fails,
            policy=RetryPolicy(
                max_attempts=5,
                backoff_base_seconds=2.0,
                max_backoff_seconds=5.0,
                retry_on=(ValueError,),
            ),
            label="x",
        )

    # Ceilings 2.0, 4.0 stay exponential; 8.0 and 16.0 are capped at 5.0.
    assert slept == [2.0, 4.0, 5.0, 5.0]


@pytest.mark.asyncio
async def test_backoff_not_slept_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first-attempt success never sleeps, even with backoff configured."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def succeed() -> int:
        return 7

    result = await with_retries(
        succeed,
        policy=RetryPolicy(max_attempts=3, backoff_base_seconds=5.0, retry_on=(ValueError,)),
    )
    assert result == 7
    assert slept == []


@pytest.mark.asyncio
async def test_retry_progress_callback_resets_on_exit() -> None:
    """The context manager restores the previous value on exit, including
    when the body raises."""
    cb_outer = _Callback()
    cb_inner = _Callback()

    async def flaky() -> str:
        raise ValueError("always")

    with retry_progress_callback(cb_outer):
        with retry_progress_callback(cb_inner), pytest.raises(ValueError):
            _ = await with_retries(
                flaky,
                policy=RetryPolicy(
                    max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)
                ),
                label="inner",
            )
        # Outer scope's callback should be active again.
        with pytest.raises(ValueError):
            _ = await with_retries(
                flaky,
                policy=RetryPolicy(
                    max_attempts=2, backoff_base_seconds=0.0, retry_on=(ValueError,)
                ),
                label="outer",
            )

    assert cb_inner.calls == [("inner", 1, 2, "always")]
    assert cb_outer.calls == [("outer", 1, 2, "always")]


@pytest.mark.parametrize("attempt", [1024, 1025, 1026, 10**6])
@pytest.mark.asyncio
async def test_huge_attempt_number_sleeps_the_cap_without_overflow(
    monkeypatch: pytest.MonkeyPatch, attempt: int
) -> None:
    """A late attempt under a large valid budget sleeps ``max_backoff_seconds``.

    The exponential ceiling used to be an ``int`` power of two scaled to a
    float, which raised ``OverflowError`` from attempt 1025 instead of being
    capped; the cap must apply however large the attempt number grows.
    """
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    await handle_retry_failure(
        tag="x",
        attempt=attempt,
        max_attempts=attempt + 1,
        error=ValueError("nope"),
        backoff_base_seconds=0.5,
        max_backoff_seconds=30.0,
    )

    assert slept == [30.0]


@pytest.mark.asyncio
async def test_early_attempt_ceilings_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The overflow-safe power yields the same ceilings for ordinary attempts."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    for attempt in range(1, 6):
        await handle_retry_failure(
            tag="x",
            attempt=attempt,
            max_attempts=6,
            error=ValueError("nope"),
            backoff_base_seconds=0.5,
            max_backoff_seconds=30.0,
        )

    assert slept == [0.5, 1.0, 2.0, 4.0, 8.0]
