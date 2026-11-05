"""Waiting for RPM or TPM budget must leave concurrency capacity available.

Real buckets are exhausted under a frozen clock. Only their refill sleep is
controlled, letting each test inspect and acquire the real concurrency gate
while a limiter caller is parked, then release the budget and finish normally.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Literal

import pytest

from llmkit.rate_limiting import GlobalRateLimiter, _tuning, configure_rate_limit
from llmkit.rate_limiting._buckets import RateBucket

_Dimension = Literal["rpm", "tpm"]
_KEY = "rate-wait"


def _exhaust(dimension: _Dimension) -> RateBucket:
    configure_rate_limit(
        max_concurrent=1,
        rpm=60 if dimension == "rpm" else None,
        tpm=60 if dimension == "tpm" else None,
        adaptive=False,
        breaker=False,
    )
    bucket = (
        GlobalRateLimiter._get_rpm_bucket(_KEY)
        if dimension == "rpm"
        else GlobalRateLimiter._get_tpm_bucket(_KEY)
    )
    assert bucket is not None
    assert bucket._capacity == 1.0
    if dimension == "rpm":
        assert bucket._try_acquire(1.0) == 0.0
    else:
        bucket.record(1.0)
    assert bucket._level == 0.0
    return bucket


@pytest.mark.parametrize("dimension", ["rpm", "tpm"])
async def test_async_rate_wait_holds_no_concurrency_slot(
    dimension: _Dimension, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    waiting = asyncio.Event()
    release = asyncio.Event()
    entered = False

    def now() -> float:
        return clock[0]

    async def refill_sleep(delay: float) -> None:
        assert delay == 1.0
        waiting.set()
        _ = await asyncio.wait_for(release.wait(), timeout=2.0)
        clock[0] += delay

    monkeypatch.setattr(_tuning, "now", now)
    bucket = _exhaust(dimension)
    gate = GlobalRateLimiter._get_async_gate(_KEY)
    monkeypatch.setattr("llmkit.rate_limiting._buckets.asyncio.sleep", refill_sleep)

    async def caller() -> None:
        nonlocal entered
        async with GlobalRateLimiter.acquire_async(_KEY):
            entered = True
            assert gate._in_flight == 1

    task = asyncio.create_task(caller())
    try:
        _ = await asyncio.wait_for(waiting.wait(), timeout=2.0)
        assert bucket._level == 0.0
        assert not entered
        assert gate._in_flight == 0, "a rate waiter occupied the concurrency slot"
        # Capacity is usable, not merely reported as free by a counter.
        await asyncio.wait_for(gate.acquire(), timeout=0.5)
        try:
            assert gate._in_flight == 1
            assert not entered
        finally:
            gate.release()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=2.0)
    assert entered
    assert gate._in_flight == 0
    assert bucket._level == (0.0 if dimension == "rpm" else 1.0)


@pytest.mark.parametrize("dimension", ["rpm", "tpm"])
def test_sync_rate_wait_holds_no_concurrency_slot(
    dimension: _Dimension, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    waiting = threading.Event()
    release = threading.Event()
    entered = threading.Event()
    errors: list[BaseException] = []

    def now() -> float:
        return clock[0]

    def refill_sleep(delay: float) -> None:
        assert delay == 1.0
        waiting.set()
        assert release.wait(timeout=2.0), "test did not release the parked rate waiter"
        clock[0] += delay

    monkeypatch.setattr(_tuning, "now", now)
    bucket = _exhaust(dimension)
    gate = GlobalRateLimiter._get_sync_gate(_KEY)
    monkeypatch.setattr("llmkit.rate_limiting._buckets.time.sleep", refill_sleep)

    def caller() -> None:
        try:
            with GlobalRateLimiter.acquire_sync(_KEY):
                entered.set()
                assert gate._in_flight == 1
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=caller, daemon=True)
    worker.start()
    try:
        assert waiting.wait(timeout=2.0), "caller never reached the exhausted rate bucket"
        assert bucket._level == 0.0
        assert not entered.is_set()
        assert gate._in_flight == 0, "a rate waiter occupied the concurrency slot"
        gate.acquire()
        try:
            assert gate._in_flight == 1
            assert not entered.is_set()
        finally:
            gate.release()
    finally:
        release.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive(), "rate waiter did not finish after budget release"
    assert not errors
    assert entered.is_set()
    assert gate._in_flight == 0
    assert bucket._level == (0.0 if dimension == "rpm" else 1.0)
