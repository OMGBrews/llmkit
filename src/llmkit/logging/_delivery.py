"""Bounded managed log delivery, independent of event loops and executors.

One daemon worker owns sink I/O. Callers reserve queue space before preparing
private payloads, then publish into FIFO order. Neither caller cancellation nor
flush timeouts cancel delivery or create replacement workers.
"""

from __future__ import annotations

import contextvars
import threading
import time
from asyncio import sleep
from collections import deque
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import cast

import llmkit.logging.registry as registry
from llmkit._durations import duration
from llmkit.logging._health import HealthState
from llmkit.logging._policy import (
    LogDeliveryPolicy,
    freeze_delivery_policy,
    get_delivery_policy,
)
from llmkit.logging.record import LLMCallRecord

_worker_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_llm_delivery_worker", default=False
)


def worker_origin() -> bool:
    """Includes reentrant work propagated onto the synchronous bridge's loop."""
    return _worker_context.get()


def _snapshot(value: object, memo: dict[int, object]) -> object:
    """Copy exact built-in containers without calling custom copy hooks.

    Tuple handling mirrors its immutable topology: a recursive list can finish
    a tuple first. Reuse that completed tuple to preserve list/tuple cycles.
    Opaque objects and container subclasses deliberately retain references.
    """
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if type(value) is list:
        result: list[object] = []
        memo[identity] = result
        result.extend(_snapshot(item, memo) for item in cast("list[object]", value))
        return result
    if type(value) is dict:
        mapping: dict[object, object] = {}
        memo[identity] = mapping
        for key, item in cast("dict[object, object]", value).items():
            mapping[_snapshot(key, memo)] = _snapshot(item, memo)
        return mapping
    if type(value) is tuple:
        items = [_snapshot(item, memo) for item in cast("tuple[object, ...]", value)]
        if identity in memo:
            return memo[identity]
        result_tuple = tuple(items)
        memo[identity] = result_tuple
        return result_tuple
    return value


def _snapshot_record(record: LLMCallRecord) -> LLMCallRecord:
    memo: dict[int, object] = {}
    values = {
        item.name: _snapshot(cast("object", getattr(record, item.name)), memo)
        for item in fields(record)
    }
    return replace(record, **values)


@dataclass(eq=False)
class Receipt:
    """A completion signal with no references to caller capture buffers."""

    health: HealthState
    deadline: float
    wait: float
    done: threading.Event = field(default_factory=threading.Event)
    path: Path | None = None
    completed_at: float | None = None


@dataclass
class _Job:
    sequence: int
    record: LLMCallRecord
    installed: registry.Installed
    context: contextvars.Context
    receipt: Receipt


class _Delivery:
    def __init__(self) -> None:
        self.condition: threading.Condition = threading.Condition()
        self.queue: deque[_Job] = deque()
        self.reservations: int = 0
        self.closed: bool = False
        self.unavailable: bool = False
        self.thread: threading.Thread | None = None
        self.accepted: int = 0
        self.pending: set[int] = set()

    @property
    def policy(self) -> LogDeliveryPolicy:
        return get_delivery_policy()

    def submit(self, record: LLMCallRecord, context: contextvars.Context) -> Receipt | None:
        installed = registry.get_installed()
        if installed.sink is None:
            return None
        health = installed.health
        with self.condition:
            freeze_delivery_policy()
            if self.closed or self.unavailable:
                health.delivery(dropped=1, unavailable=1)
                return None
            if len(self.queue) + self.reservations >= self.policy.queue_capacity:
                health.delivery(dropped=1)
                return None
            self.reservations += 1
            health.delivery(queued=1)
        try:
            payload = _snapshot_record(record)
            _ = context.run(_worker_context.set, True)
        except BaseException:
            with self.condition:
                self.reservations -= 1
                health.delivery(queued=-1, preparation_failed=1)
            return None
        with self.condition:
            self.reservations -= 1
            if self.closed or self.unavailable:
                health.delivery(queued=-1, dropped=1, unavailable=1)
                return None
            if self.thread is None:
                self.thread = threading.Thread(
                    target=self._work, name="llmkit-log-delivery", daemon=True
                )
                try:
                    self.thread.start()
                except BaseException:
                    self.unavailable = True
                    health.delivery(queued=-1, preparation_failed=1)
                    return None
            receipt = Receipt(
                health=health,
                deadline=time.monotonic() + self.policy.delivery_wait,
                wait=self.policy.delivery_wait,
            )
            self.accepted += 1
            self.pending.add(self.accepted)
            self.queue.append(_Job(self.accepted, payload, installed, context, receipt))
            self.condition.notify_all()
            return receipt

    def _settle(self, job: _Job, path: Path | None = None) -> None:
        # Called under the condition; event publication orders receipt reads.
        job.receipt.path = path
        job.receipt.completed_at = time.monotonic()
        job.receipt.done.set()
        self.pending.remove(job.sequence)
        self.condition.notify_all()

    def _discard(self) -> None:
        while self.queue:
            job = self.queue.popleft()
            job.installed.health.delivery(queued=-1, dropped=1, unavailable=1)
            self._settle(job)

    def _work(self) -> None:
        while True:
            with self.condition:
                while not self.queue:
                    if self.closed or self.unavailable:
                        return
                    _ = self.condition.wait()
                job = self.queue.popleft()
                job.installed.health.delivery(queued=-1, in_flight=1)
            path: Path | None = None
            terminated = False
            attempt = registry.WriteAttempt()
            try:
                path = job.context.run(registry.write_installed, job.record, job.installed, attempt)
            except BaseException:
                # No exception formatting or diagnostics here: either can be
                # arbitrary code supplied by the sink/host that just failed.
                terminated = True
                if not attempt.accounted:
                    job.installed.health.record_worker_failure()
            finally:
                with self.condition:
                    job.installed.health.delivery(in_flight=-1)
                    self._settle(job, path)
                    if terminated:
                        self.unavailable = True
                        self._discard()
            if terminated:
                return
            del job  # Do not retain the last payload/context while the worker is idle.

    def covered(self, prefix: int) -> bool:
        # Caller holds the condition.
        return not any(sequence <= prefix for sequence in self.pending)

    def flush(self, timeout: float | None, *, terminal: bool = False) -> bool:
        budget = self.policy.drain_timeout if timeout is None else duration(timeout, "timeout")
        deadline = time.monotonic() + budget
        with self.condition:
            if terminal:
                freeze_delivery_policy()
                self.closed = True
                self.condition.notify_all()
            prefix = self.accepted
            if worker_origin():
                return False
            while not self.covered(prefix):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if terminal:
                        self._discard()
                    return False
                _ = self.condition.wait(min(remaining, threading.TIMEOUT_MAX))
            return True


_delivery = _Delivery()


def submit(record: LLMCallRecord, context: contextvars.Context) -> Receipt | None:
    return _delivery.submit(record, context)


async def wait_for_receipt(receipt: Receipt) -> Path | None:
    """Wait without an executor; only the caller may capture the returned path.

    Polling keeps no event-loop callbacks or caller context in a stuck job.
    The one-millisecond cadence is capped by the single publication deadline.
    """
    if receipt.wait == 0 or worker_origin():
        return None
    while True:
        now = time.monotonic()
        if now > receipt.deadline:
            receipt.health.delivery(wait_timeouts=1)
            return None
        if receipt.done.is_set() and time.monotonic() <= receipt.deadline:
            return receipt.path
        await sleep(min(0.001, receipt.deadline - now))


def flush_llm_logs(*, timeout: float | None = None) -> bool:
    """Wait up to a finite budget for the accepted prefix to settle.

    Defaults to the configured one second. True includes failed writes: inspect
    health for persistence evidence. False leaves work pending for a later flush.
    Zero polls; worker-originated calls return False to avoid self-deadlock.
    Direct synchronous writes and preparations not yet published are excluded.
    """
    return _delivery.flush(timeout)


async def flush_llm_logs_async(*, timeout: float | None = None) -> bool:
    """Async flush with the same prefix contract; cancellation stops only the wait."""
    budget = _delivery.policy.drain_timeout if timeout is None else duration(timeout, "timeout")
    deadline = time.monotonic() + budget
    with _delivery.condition:
        prefix = _delivery.accepted
    if worker_origin():
        return False
    while True:
        with _delivery.condition:
            if _delivery.covered(prefix):
                return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await sleep(min(0.005, remaining))


def shutdown_delivery() -> bool:
    """One terminal drain, called after sync-loop cleanup by its atexit hook."""
    return _delivery.flush(None, terminal=True)
