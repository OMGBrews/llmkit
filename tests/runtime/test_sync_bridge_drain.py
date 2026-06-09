"""The sync bridge drains LiteLLM's pending logging before closing its loop.

LiteLLM doesn't ``await`` its success logging inline: after a successful async
call it *eagerly constructs* the ``Logging.async_success_handler`` coroutine and
hands it to a module-global ``LoggingWorker`` queue (``GLOBAL_LOGGING_WORKER``)
to run as a background task. On the synchronous bridge that coroutine can still
be sitting in the queue — created but never awaited — when
:func:`llmkit.sync.run_sync` closes the loop, and Python then prints
``RuntimeWarning: coroutine 'Logging.async_success_handler' was never awaited``
to **stderr**: visible noise on an otherwise successful sync call (FiW sees this
in their CLI and Lambda). ``run_sync`` now drains that pending logging before
teardown (see :func:`llmkit._litellm.drain_async_logging`).

Reproducing this offline needs care, because the warning is emitted by the
**GC finalizer of the un-awaited coroutine** — and that coroutine stays
reachable through the module-global worker queue until *interpreter shutdown*,
well outside any in-process ``warnings.catch_warnings`` window. So the leak
assertion runs in a **subprocess** under ``-W error::RuntimeWarning`` and
checks that no "never awaited" line reaches stderr; the worker enqueues a stray
coroutine exactly the way LiteLLM's post-call hook does
(``ensure_initialized_and_enqueue``, the entry point in ``litellm/utils.py``),
with no network and no provider credential. Confirmed to discriminate: the same
script over a bare ``asyncio.run`` (the pre-fix bridge) prints the warning.

The remaining behaviours — the worker-thread branch, the bounded drain, and
error propagation — don't depend on shutdown timing and are checked in-process.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time
from collections.abc import Coroutine
from typing import Protocol, cast

import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

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
# surfaces on stderr at interpreter teardown.
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
    """A sync call draining LiteLLM's queued logging leaves stderr clean.

    The subprocess promotes RuntimeWarning to an error and we assert the call
    succeeds *and* nothing "never awaited" reaches stderr through interpreter
    shutdown — the exact noise FiW observed.
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


def test_sync_call_in_running_loop_drains_litellm_logging_queue() -> None:
    """The worker-thread path (loop already running) drains the logging queue.

    Calling ``run_sync`` from inside a live event loop routes through the
    ThreadPoolExecutor branch, which drives a *second* loop in a worker thread.
    That branch must drain LiteLLM's queued logging too: after the call the
    worker queue is empty (the handler ran), not holding an orphaned coroutine.
    """

    async def _enqueue_and_return(value: int) -> int:
        async def _async_success_handler() -> None:
            return None

        _worker.ensure_initialized_and_enqueue(_async_success_handler())
        return value

    async def _outer() -> int:
        # run_sync detects the running loop and offloads to a worker thread.
        return run_sync(_enqueue_and_return(11))

    result = asyncio.run(_outer())
    assert result == 11
    queue = GLOBAL_LOGGING_WORKER._queue
    assert queue is None or queue.qsize() == 0


def test_drain_is_bounded_and_still_returns_result() -> None:
    """A straggler that outruns the timeout is abandoned; the result stands.

    The drain must not wedge the bridge on a hung callback: with a tiny timeout
    a slow background task is cancelled after the deadline (warning-free, because
    it was already started), and the call's own result is still returned intact.
    """

    async def _schedule_slow_stray() -> str:
        async def _slow_log() -> None:
            await asyncio.sleep(30)  # far past the drain deadline

        # Held on the coroutine frame so it isn't GC'd before the drain sees it;
        # the bridge is responsible for cancelling it at teardown.
        stray = asyncio.create_task(_slow_log())
        assert not stray.done()
        return "ok"

    result = run_sync(_schedule_slow_stray(), timeout=0.1)
    assert result == "ok"


def test_call_error_propagates_through_drain() -> None:
    """A genuine error from the driven coroutine still propagates.

    The drain wraps the call's teardown but must not swallow the caller's own
    exception — nor hang on the logging the failing call had enqueued.
    """

    async def _boom() -> int:
        async def _async_success_handler() -> None:
            return None

        _worker.ensure_initialized_and_enqueue(_async_success_handler())
        raise ValueError("boom")

    raised = False
    try:
        _ = run_sync(_boom())
    except ValueError as exc:
        raised = exc.args == ("boom",)
    assert raised


def test_running_loop_timeout_releases_caller_promptly() -> None:
    """On the worker-thread path a timeout unblocks the caller at the deadline.

    Regression: ``run_sync`` previously drove the worker through a
    ``with ThreadPoolExecutor(...)`` block, whose ``__exit__`` calls
    ``shutdown(wait=True)`` — so a ``TimeoutError`` could not leave the frame
    until the (still-running) coroutine finished, silently defeating ``timeout``.
    The executor is now torn down with ``wait=False`` so the error propagates at
    the deadline while the abandoned worker drains in the background.
    """

    async def _slow(value: int) -> int:
        await asyncio.sleep(1.5)  # comfortably past the timeout below
        return value

    async def _outer() -> None:
        # Inside a running loop -> the ThreadPoolExecutor branch.
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            _ = run_sync(_slow(99), timeout=0.2)
        elapsed = time.monotonic() - start
        # Must unblock near the 0.2s deadline, not after the ~1.5s call.
        assert elapsed < 1.0, (
            f"timeout took {elapsed:.2f}s — executor blocked on shutdown(wait=True)"
        )

    asyncio.run(_outer())
