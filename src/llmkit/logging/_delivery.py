"""Bounded managed log delivery, independent of event loops and executors.

One daemon worker per process owns sink I/O. Callers reserve queue space before
preparing private payloads, then publish into FIFO order. Neither caller
cancellation nor flush timeouts cancel delivery or create replacement workers.

A forked child inherits the parent's delivery state but not its worker, and any
lock a parent thread held at the fork stays held. At-fork hooks therefore hold
the registry's short critical sections across the fork, then give the child a
fresh delivery service, re-install the current sink with fresh counters, and
replace the other locks on the child's submit and write path. The parent's
queued records are left to the parent's worker.

The first record an installation drops (queue full, delivery closed or
unavailable, the exit drain timing out) emits one WARNING naming
:func:`~llmkit.logging.get_logging_health`; later drops are only counted. It is
emitted outside the delivery lock and carries no payload or exception text.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import threading
import time
from asyncio import sleep
from collections import deque
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import cast

import llmkit._diagnostics as _diagnostics
import llmkit.logging._policy as _policy
import llmkit.logging.registry as registry
from llmkit._diagnostics import report
from llmkit._durations import duration
from llmkit.logging._health import HealthState
from llmkit.logging._policy import (
    LogDeliveryPolicy,
    freeze_delivery_policy,
    get_delivery_policy,
)
from llmkit.logging.record import LLMCallRecord

# Named explicitly rather than via ``__name__`` so every module in this package
# keeps emitting under the one logger name operators already filter on.
logger = logging.getLogger("llmkit.logging")

_worker_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_llm_delivery_worker", default=False
)


def worker_origin() -> bool:
    """Includes reentrant work propagated onto the synchronous bridge's loop."""
    return _worker_context.get()


def _warn_dropped(healths: list[HealthState], reason: str) -> None:
    """Warn once per installation that delivery dropped a record.

    Called with the delivery lock released, so a slow or reentrant host
    handler cannot stall admission. Sink-agnostic and payload-free: only the
    drop's *reason* is formatted. :func:`~llmkit._diagnostics.report` contains
    a host handler that raises an :class:`Exception`; on the caller's path a
    ``BaseException`` from the handler propagates, as for every contained
    diagnostic.
    """
    for health in healths:
        if health.claim_drop_warning():
            _ = report(
                logger,
                logging.WARNING,
                "llmkit dropped an LLM call record before it reached the log sink (%s); "
                + "further drops are counted in get_logging_health() and not logged",
                reason,
            )


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
        dropped: str | None = None
        with self.condition:
            freeze_delivery_policy()
            if self.closed or self.unavailable:
                health.delivery(dropped=1, unavailable=1)
                dropped = "log delivery closed or unavailable"
            elif len(self.queue) + self.reservations >= self.policy.queue_capacity:
                health.delivery(dropped=1)
                dropped = "log delivery queue full"
            else:
                self.reservations += 1
                health.delivery(queued=1)
        if dropped is not None:
            _warn_dropped([health], dropped)
            return None
        try:
            payload = _snapshot_record(record)
            _ = context.run(_worker_context.set, True)
        except BaseException:
            with self.condition:
                self.reservations -= 1
                health.delivery(queued=-1, preparation_failed=1)
            return None
        late_drop = False
        receipt: Receipt | None = None
        with self.condition:
            self.reservations -= 1
            if self.closed or self.unavailable:
                health.delivery(queued=-1, dropped=1, unavailable=1)
                late_drop = True
            else:
                receipt = self._publish(payload, installed, context, health)
        if late_drop:
            _warn_dropped([health], "log delivery closed or unavailable")
            return None
        return receipt

    def _publish(
        self,
        payload: LLMCallRecord,
        installed: registry.Installed,
        context: contextvars.Context,
        health: HealthState,
    ) -> Receipt | None:
        # Caller holds the condition.
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

    def _discard(self) -> list[HealthState]:
        """Drop every queued job; return each installation that lost one.

        Caller holds the condition, and warns with :func:`_warn_dropped` once
        it is released.
        """
        losers: list[HealthState] = []
        while self.queue:
            job = self.queue.popleft()
            health = job.installed.health
            health.delivery(queued=-1, dropped=1, unavailable=1)
            if not any(health is seen for seen in losers):
                losers.append(health)
            self._settle(job)
        return losers

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
            discarded: list[HealthState] = []
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
                        discarded = self._discard()
            if terminated:
                # The worker is exiting: like the failure above, nothing a
                # host handler raises here may escape it.
                with contextlib.suppress(BaseException):
                    _warn_dropped(discarded, "log delivery worker stopped")
                return
            del job  # Do not retain the last payload/context while the worker is idle.

    def covered(self, prefix: int) -> bool:
        # Caller holds the condition.
        return not any(sequence <= prefix for sequence in self.pending)

    def flush(self, timeout: float | None, *, terminal: bool = False) -> bool:
        budget = self.policy.drain_timeout if timeout is None else duration(timeout, "timeout")
        deadline = time.monotonic() + budget
        discarded: list[HealthState] = []
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
                        discarded = self._discard()
                    break
                _ = self.condition.wait(min(remaining, threading.TIMEOUT_MAX))
            else:
                return True
        _warn_dropped(discarded, "exit drain timed out")
        return False


_delivery = _Delivery()


def _reset_after_fork() -> None:
    """Give a forked child its own delivery service and unheld locks.

    Runs in the child between ``fork`` and its first line of code, so it takes
    no lock, logs nothing and cannot raise. The inherited queue is dropped
    without settling or counting: those records are the parent's, and its
    worker still delivers them. Every consumer reads the module-level
    ``_delivery`` at call time, so rebinding it reaches them all.
    """
    global _delivery
    _delivery = _Delivery()
    _policy.reset_after_fork()
    _diagnostics.reset_after_fork()
    registry.reset_after_fork()


# Registered once, at import, never per ``_Delivery``: tests build their own
# instances, and an at-fork hook cannot be unregistered.
if sys.platform != "win32":
    os.register_at_fork(
        before=registry.before_fork,
        after_in_parent=registry.after_fork_in_parent,
        after_in_child=_reset_after_fork,
    )


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
