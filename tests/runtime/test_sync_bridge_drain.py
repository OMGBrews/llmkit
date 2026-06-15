"""The sync bridge drives coroutines on one persistent loop, draining at exit.

:func:`llmkit.sync.run_sync` routes every sync call onto a single persistent
event loop so LiteLLM's process-global logging worker binds to one loop and is
drained once, at shutdown — instead of a fresh loop per call, which races that
worker under a concurrent fan-out (see ``test_sync_concurrency.py`` for the
all-on-one-loop and no-flood regressions).

LiteLLM doesn't ``await`` its success logging inline: after a successful async
call it *eagerly constructs* the ``Logging.async_success_handler`` coroutine and
hands it to a module-global ``LoggingWorker`` queue (``GLOBAL_LOGGING_WORKER``).
That coroutine can still be queued — created but never awaited — when the
process exits, and Python then prints ``RuntimeWarning: coroutine
'Logging.async_success_handler' was never awaited`` to **stderr**: visible noise
on an otherwise successful run. The persistent loop's :func:`atexit` shutdown
drains that queue before closing the loop.

These tests pin: the leak is gone at interpreter shutdown; the call's result and
errors propagate; a timeout cancels the abandoned coroutine *on the loop*; a
reentrant call (made from inside a running loop) is served by the worker-thread
fallback without deadlock; and explicit :func:`~llmkit.sync.shutdown` is
idempotent and lets the loop restart lazily.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Coroutine
from typing import Protocol, cast

import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from llmkit import sync
from llmkit.sync import run_sync


class _LoggingWorker(Protocol):
    """Typed view of the litellm logging worker's enqueue entry point.

    The shipped worker types ``ensure_initialized_and_enqueue`` with an
    ``Unknown``-parameterised coroutine, leaking ``reportUnknownMemberType`` at
    each call site; pin the one method these tests touch.
    """

    def ensure_initialized_and_enqueue(
        self, async_coroutine: Coroutine[object, object, None]
    ) -> None: ...


_worker = cast("_LoggingWorker", GLOBAL_LOGGING_WORKER)

# A self-contained script: enqueue a logging coroutine onto LiteLLM's worker the
# way its post-call hook does, then drive it through run_sync. Run as a
# subprocess under -W error::RuntimeWarning so an un-awaited-coroutine warning
# surfaces on stderr at interpreter teardown — where the persistent loop's
# atexit shutdown must have drained it.
_LEAK_SCRIPT = textwrap.dedent(
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from llmkit.sync import run_sync

    async def call(value):
        async def async_success_handler():
            return None

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(async_success_handler())
        return value

    print("RESULT", run_sync(call(7)))
    """
)


def test_sync_call_does_not_leak_never_awaited_warning() -> None:
    """A sync call leaves stderr clean through interpreter shutdown.

    LiteLLM's queued success-logging is drained by the persistent loop's atexit
    shutdown, so the un-awaited-coroutine warning never surfaces. The subprocess
    promotes RuntimeWarning to an error and we assert the call succeeds *and*
    nothing "never awaited" reaches stderr — the exact noise FiW observed.
    """
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _LEAK_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESULT 7" in proc.stdout
    assert "never awaited" not in proc.stderr, proc.stderr


def test_returns_result_on_success() -> None:
    """The common path returns the coroutine's result."""

    async def _ok(value: int) -> int:
        return value

    assert run_sync(_ok(21)) == 21


def test_call_error_propagates() -> None:
    """A genuine error from the driven coroutine propagates to the caller.

    The bridge wraps the call's teardown but must not swallow the caller's own
    exception — nor hang on the logging the failing call had enqueued.
    """

    async def _boom() -> int:
        async def _async_success_handler() -> None:
            return None

        _worker.ensure_initialized_and_enqueue(_async_success_handler())
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _ = run_sync(_boom())


def test_timeout_cancels_the_abandoned_task_on_the_loop() -> None:
    """On timeout, ``run_sync`` cancels the coroutine on the persistent loop.

    The wait is bounded by ``timeout``; past the deadline a
    :class:`concurrent.futures.TimeoutError` (the builtin ``TimeoutError`` on
    3.11+) propagates, and — unlike the old fresh-loop worker that leaked a
    running coroutine — the task is cancelled *on the loop*, tearing down the
    in-flight work rather than leaking it. A :class:`KeyboardInterrupt` in the
    calling thread takes the same abandon-then-cancel path.
    """
    cancelled = threading.Event()

    async def _slow() -> str:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "should never return"

    with pytest.raises(TimeoutError):
        _ = run_sync(_slow(), timeout=0.1)

    assert cancelled.wait(timeout=2.0), "timed-out task was not cancelled on the loop"


def test_reentrant_call_from_running_loop_returns_result() -> None:
    """A ``run_sync`` made from inside a running loop is served, not deadlocked.

    When the calling thread already has a running loop (an async framework, or a
    coroutine that calls back into sync code), the bridge cannot block that
    thread on the persistent loop, so it offloads to a one-shot worker thread
    with its own fresh loop. The call must still complete and return its value.
    """

    async def _inner(value: int) -> int:
        return value

    async def _outer() -> int:
        # run_sync detects the running loop and uses the worker-thread fallback.
        return run_sync(_inner(11))

    assert asyncio.run(_outer()) == 11


def test_reentrant_timeout_releases_caller_promptly() -> None:
    """On the worker-thread fallback a timeout unblocks the caller at the deadline.

    The reentrant fallback drives the coroutine on another thread, which cannot
    be cancelled from here — so a timeout abandons only the *wait*. It must still
    release the caller near the deadline (the executor is torn down with
    ``wait=False``), not after the slow call finishes.
    """

    async def _slow(value: int) -> int:
        await asyncio.sleep(1.5)  # comfortably past the timeout below
        return value

    async def _outer() -> None:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            _ = run_sync(_slow(99), timeout=0.2)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"timeout took {elapsed:.2f}s — caller not released at the deadline"

    asyncio.run(_outer())


def test_shutdown_is_idempotent_and_loop_restarts() -> None:
    """``shutdown`` is a safe no-op when repeated, and the loop restarts lazily.

    A first call drives the persistent loop into existence; ``shutdown`` drains
    and closes it; a second ``shutdown`` is a no-op (globals already cleared);
    and a subsequent ``run_sync`` transparently starts a fresh persistent loop.
    """

    async def _echo(value: int) -> int:
        return value

    assert run_sync(_echo(1)) == 1
    sync.shutdown()
    sync.shutdown()  # idempotent: must not raise on an already-closed loop
    # The bridge lazily starts a new persistent loop for the next call.
    assert run_sync(_echo(2)) == 2
