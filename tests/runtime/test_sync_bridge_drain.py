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
reentrant call from a *foreign* loop (an async host) is routed onto the
persistent loop — no fresh loop, cancellation still works; a call from the
persistent loop itself falls back to the worker-thread fresh loop without
deadlock and still drains LiteLLM's queued logging; and explicit
:func:`~llmkit.sync.shutdown` is idempotent and lets the loop restart lazily.
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


def test_reentrant_foreign_loop_call_runs_on_the_persistent_loop() -> None:
    """A ``run_sync`` from inside a *foreign* running loop rides the persistent loop.

    An async host calling a ``*_sync`` helper used to get a fresh loop on a
    worker thread — the path that, under a concurrent reentrant fan-out,
    re-created the multi-loop regime the persistent loop exists to avoid. The
    bridge now submits the coroutine to the persistent loop instead: same
    blocking semantics for the caller, no fresh loop. The strong discriminator:
    the reentrant call observes the *persistent* loop, not the caller's and not
    a fresh one.
    """

    async def _probe() -> int:
        return id(asyncio.get_running_loop())

    persistent_id = run_sync(_probe())

    async def _outer() -> tuple[int, int]:
        # run_sync detects a running loop that is not the persistent one and
        # submits to the persistent loop, blocking this (the host's) thread.
        return id(asyncio.get_running_loop()), run_sync(_probe())

    caller_id, reentrant_id = asyncio.run(_outer())
    assert reentrant_id == persistent_id, "reentrant call did not run on the persistent loop"
    assert reentrant_id != caller_id


def test_reentrant_foreign_loop_timeout_cancels_and_releases_promptly() -> None:
    """A reentrant timeout releases the caller at the deadline *and* cancels the task.

    Because foreign-loop reentrant calls now ride the persistent loop, they gain
    the persistent-loop timeout contract: past the deadline the caller gets
    :class:`TimeoutError` promptly and the abandoned coroutine is cancelled *on
    the loop* — unlike the old worker-thread fallback, which leaked a running
    thread and request.
    """
    cancelled = threading.Event()

    async def _slow(value: int) -> int:
        try:
            await asyncio.sleep(1.5)  # comfortably past the timeout below
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return value

    async def _outer() -> None:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            _ = run_sync(_slow(99), timeout=0.2)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"timeout took {elapsed:.2f}s — caller not released at the deadline"

    asyncio.run(_outer())
    assert cancelled.wait(timeout=2.0), "timed-out reentrant task was not cancelled on the loop"


def test_loop_on_loop_call_falls_back_to_a_fresh_loop_without_deadlock() -> None:
    """A ``run_sync`` from a coroutine *on the persistent loop* is served, not deadlocked.

    This is the one reentrant case that cannot ride the persistent loop —
    blocking the loop's own thread on itself would deadlock — so it offloads to
    a one-shot worker thread with its own fresh loop. The call must complete,
    return its value, and demonstrably run on a *different* loop than the
    (blocked) persistent one.
    """

    async def _inner() -> int:
        return id(asyncio.get_running_loop())

    async def _outer() -> tuple[int, int]:
        # get_running_loop() here *is* the persistent loop → worker fallback.
        return id(asyncio.get_running_loop()), run_sync(_inner(), timeout=30)

    outer_id, inner_id = run_sync(_outer(), timeout=30)
    assert inner_id != outer_id, "loop-on-loop call did not get its own fresh loop"


# A self-contained script exercising the loop-on-loop fallback's logging drain:
# the inner call runs on the fallback's fresh loop and enqueues a LiteLLM-style
# success-logging coroutine there, which _run_and_drain must flush before that
# loop closes. Run under -W error::RuntimeWarning so a dropped drain surfaces as
# "never awaited" on stderr (mutation testing showed the drain was previously
# deletable without any test noticing).
_FALLBACK_DRAIN_SCRIPT = textwrap.dedent(
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from llmkit.sync import run_sync

    async def inner():
        async def async_success_handler():
            return None

        GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(async_success_handler())
        return 5

    async def outer():
        return run_sync(inner())

    print("RESULT", run_sync(outer()))
    """
)


def test_loop_on_loop_fallback_drains_queued_logging() -> None:
    """The fallback's fresh loop flushes LiteLLM's queued logging before closing.

    Deleting the ``drain_async_logging`` step in ``_run_and_drain`` must fail
    this test: the enqueued handler coroutine would be orphaned when the fresh
    loop closes and surface as ``coroutine '…' was never awaited``.
    """
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _FALLBACK_DRAIN_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESULT 5" in proc.stdout
    assert "never awaited" not in proc.stderr, proc.stderr


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


def test_run_sync_racing_shutdown_lands_on_a_fresh_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller that races ``shutdown()`` retries onto a fresh persistent loop.

    ``_ensure_loop`` is wrapped so its first call — from inside
    ``_submit_to_loop``, right after the loop is obtained — triggers
    ``shutdown()``, reproducing the obtain-then-torn-down race deterministically.
    With the fix the identity check under ``_lock`` fails, so the caller retries
    onto a lazily restarted loop and the call completes promptly. Pre-fix it fails
    fast with ``RuntimeError: Event loop is closed`` (the submit lands on the
    joined/closed loop); the stopped-but-not-closed 600 s flavour is unreachable
    deterministically through public seams, and the identity-check design kills
    both by one mechanism, so this plus the wall-clock bound is the right offline
    regression.
    """
    sync.shutdown()  # start from a known clean slate
    real_ensure_loop = sync._ensure_loop
    triggered = {"done": False}

    def _racing_ensure_loop() -> asyncio.AbstractEventLoop:
        loop = real_ensure_loop()
        if not triggered["done"]:
            triggered["done"] = True
            sync.shutdown()  # race: tear the just-obtained loop down before submit
        return loop

    monkeypatch.setattr(sync, "_ensure_loop", _racing_ensure_loop)

    async def _ok() -> int:
        return 42

    start = time.monotonic()
    assert run_sync(_ok(), timeout=30) == 42
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"racing caller took {elapsed:.2f}s — did not restart promptly"
    sync.shutdown()  # leave the bridge cleanly shut down (restarts lazily)


def test_shutdown_cancels_in_flight_call_and_caller_error_is_not_masked() -> None:
    """A call in flight when ``shutdown()`` runs gets a prompt ``CancelledError``,
    not a secondary ``RuntimeError`` from the cancel path masking it.

    The caller parks in ``run_sync`` on a long sleep; ``shutdown()`` cancels the
    task via its drain-and-close sweep. The caller must observe
    ``asyncio.CancelledError`` — the *type* is the assertion: without the
    ``contextlib.suppress(RuntimeError)`` hardening on the cancel path, the
    closed-loop ``call_soon_threadsafe(_cancel)`` raises and masks the real error.
    """
    sync.shutdown()  # clean slate
    started = threading.Event()
    observed: list[BaseException] = []

    async def _park() -> None:
        started.set()
        await asyncio.sleep(30)

    def _caller() -> None:
        try:
            _ = run_sync(_park(), timeout=30)
        except BaseException as exc:  # capture the exact type the caller observed
            observed.append(exc)

    caller = threading.Thread(target=_caller)
    caller.start()
    assert started.wait(timeout=2.0), "call never entered the loop"
    sync.shutdown()
    caller.join(timeout=5.0)
    assert not caller.is_alive(), "caller did not unblock after shutdown"
    assert len(observed) == 1, observed
    assert isinstance(observed[0], asyncio.CancelledError), (
        f"expected CancelledError, got {type(observed[0]).__name__}: {observed[0]}"
    )
