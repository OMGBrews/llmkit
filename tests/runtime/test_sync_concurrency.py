"""Concurrent sync fan-out is safe: one shared loop, no logging-worker flood.

The defect this guards against: a *fresh event loop per* ``run_sync`` call. Many
threads each making one sync call then spin up many short-lived loops that all
touch LiteLLM's process-global ``GLOBAL_LOGGING_WORKER`` — which rebinds its
queue / worker-task on every loop change, *without a lock*. Concurrent loops
tear that state apart, flooding stderr (``Task was destroyed but it is
pending``, ``<Queue …> is bound to a different event loop``, ``coroutine '…' was
never awaited``) and occasionally wedging a teardown ``flush()`` on an orphaned
queue until LiteLLM's ~600 s request timeout fires.

Routing every call onto one persistent loop removes the bug class. Two
regressions pin that:

* **One loop.** ≥16 parallel ``run_sync`` calls all observe the *same* running
  loop — the strong, deterministic discriminator (a fresh-loop bridge shows 16
  distinct loops; this shows one), which also exercises the thread-safe lazy
  start.
* **No flood, no hang.** ≥16 parallel calls that each enqueue a LiteLLM-style
  success-logging coroutine exit cleanly under ``-W error::RuntimeWarning`` with
  no "never awaited" / "bound to a different event loop" noise, well within a
  few seconds (no multi-second ``queue.join`` stall).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time

from llmkit.sync import run_sync

_N = 16


def test_all_sync_calls_run_on_one_persistent_loop() -> None:
    """≥16 parallel ``run_sync`` calls all execute on the same event loop.

    The core invariant of the persistent-loop bridge: a fresh-loop-per-call
    strategy would surface ``_N`` distinct loops here (and race LiteLLM's
    worker); the persistent loop surfaces exactly one. The concurrent first-call
    race also proves the lazy start is thread-safe (one loop, not several).
    """
    loop_ids: list[int] = []
    lock = threading.Lock()

    async def _probe() -> int:
        await asyncio.sleep(0.01)  # ensure genuine overlap across the threads
        return id(asyncio.get_running_loop())

    def _call() -> None:
        loop_id = run_sync(_probe())
        with lock:
            loop_ids.append(loop_id)

    threads = [threading.Thread(target=_call) for _ in range(_N)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert len(loop_ids) == _N, "not every concurrent sync call completed"
    assert len(set(loop_ids)) == 1, (
        f"expected one shared persistent loop, saw {len(set(loop_ids))} distinct loops"
    )


# A self-contained script: fan out _N threads, each driving a run_sync call that
# enqueues a LiteLLM-style success-logging coroutine the way its post-call hook
# does. Run under -W error::RuntimeWarning so any teardown flood (a never-awaited
# coroutine, a cross-loop queue) becomes a non-zero exit and surfaces on stderr.
_CONCURRENT_SCRIPT = textwrap.dedent(
    """
    import threading
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from llmkit.sync import run_sync

    def call(i):
        async def _c():
            async def handler():
                return None

            GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(handler())
            return i

        return run_sync(_c())

    results = []
    lock = threading.Lock()

    def worker(i):
        r = call(i)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("COUNT", len(results), "DISTINCT", len(set(results)))
    """
)


def test_concurrent_sync_fanout_no_flood_or_hang() -> None:
    """≥16 parallel sync calls finish cleanly: no flood, no multi-second stall.

    This is the scenario that floods and can hang for ~600 s on a fresh-loop
    bridge. Under the persistent loop the LiteLLM worker binds once and is
    drained once at exit, so the subprocess exits 0 with no teardown warnings,
    all 16 calls complete, and it finishes far inside the wall-clock budget
    (proving no orphaned-queue ``queue.join`` stall).
    """
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", _CONCURRENT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    elapsed = time.monotonic() - start

    assert proc.returncode == 0, proc.stderr
    assert "COUNT 16 DISTINCT 16" in proc.stdout, proc.stdout
    assert "never awaited" not in proc.stderr, proc.stderr
    assert "bound to a different event loop" not in proc.stderr, proc.stderr
    assert elapsed < 30.0, f"concurrent sync fan-out took {elapsed:.1f}s (possible queue.join hang)"
