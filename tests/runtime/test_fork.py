"""The sync API in a forked child, in fresh interpreters.

A child process inherits a copy of the parent's memory but only the thread that
called ``fork``. A parent that has made a ``*_sync`` call has a persistent loop
running on a daemon thread; without a reset the child inherits that loop
without the thread that runs it, so its first ``*_sync`` call waits forever and
its normal exit waits out the whole drain bound. Each scenario forks from a
fresh interpreter, gives the child a bounded time, and kills it if it does not
finish, so a hang fails its case instead of the suite. A child ends with
``os._exit`` unless the scenario is about its normal exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from llmkit import _litellm

pytestmark = pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")

_ROOT = Path(__file__).resolve().parents[2]

# What a child must never print: a parent's pending task or queued callback
# collected in the child, or LiteLLM's worker rebinding onto the child's loop.
_FORBIDDEN_OUTPUT = (
    "Task was destroyed but it is pending",
    "was never awaited",
    "is bound to a different event loop",
    "event loop changed",
)

# Runs in the forking interpreter. ``fork_child`` runs a body in the child;
# ``reap`` waits for it with a bound and reports a hang as a failure.
_COMMON = r"""
import asyncio
import gc
import os
import sys
import threading
import time
from pathlib import Path

from llmkit import text_llm_call_sync
from llmkit.testing import Reply, install
from tests._support import ollama_call_defaults

WORK_DIR = Path(sys.argv[1])

_defaults = ollama_call_defaults()
_defaults.__enter__()
_fake = install(lambda request: Reply(content="ok"))
_fake.__enter__()


def call(prompt="hi"):
    return text_llm_call_sync(prompt, feature="fork")


def fork_child(body, exit_normally=False):
    # ``exited`` receives one byte once the body is done, so the parent can
    # time the child's exit apart from its work.
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        code = 1
        try:
            body()
            # A real child allocates enough to collect garbage at any moment;
            # collect now, so whatever the parent left behind is finalized
            # while the child still reports it.
            gc.collect()
            code = 0
        except BaseException as exc:
            print(f"child failed: {exc!r}", flush=True)
        os.write(write_end, b"x")
        if exit_normally:
            sys.stdout.flush()
            # Unwinds the child's copy of this script, so the interpreter
            # exits normally and runs llmkit's atexit shutdown.
            sys.exit(code)
        os._exit(code)
    os.close(write_end)
    return pid, read_end


def reap(child, bound=5.0, exit_bound=None):
    pid, read_end = child
    deadline = time.monotonic() + bound
    os.set_blocking(read_end, False)
    finished_at = None
    while True:
        if finished_at is None:
            try:
                if os.read(read_end, 1):
                    finished_at = time.monotonic()
                    if exit_bound is not None:
                        deadline = finished_at + exit_bound
            except BlockingIOError:
                pass
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            os.close(read_end)
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            _ = os.waitpid(pid, 0)
            os.close(read_end)
            return "hung" if finished_at is None else "slow exit"
        time.sleep(0.01)


def hold(lock):
    # Holds *lock* on a parent thread until the returned event is set.
    held = threading.Event()
    release = threading.Event()

    def holder():
        with lock:
            held.set()
            release.wait()

    threading.Thread(target=holder, daemon=True).start()
    assert held.wait(5)
    return release
"""


def _run(source: str, work_dir: Path) -> None:
    """Run *source* after the common helpers; it must print ``done`` and exit 0.

    The combined output of the parent and its children must also show no sign
    of the parent's async work leaking into a child.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLMKIT_LOG")}
    env["LLMKIT_LOG_DIR"] = str(work_dir / "logs")
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", _COMMON + source, str(work_dir)],
            cwd=_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = cast("str | None", exc.output)
        pytest.fail(f"forking interpreter did not finish:\n{output}")
    assert result.returncode == 0, result.stdout
    assert "done" in result.stdout.splitlines(), result.stdout
    for text in _FORBIDDEN_OUTPUT:
        assert text not in result.stdout, result.stdout


def test_child_completes_its_own_sync_call(tmp_path: Path) -> None:
    """After the parent started the sync loop, the child's call gets its own loop."""
    _run(
        r"""
from llmkit import sync

assert call() == "ok"
parent_loop = sync._loop
assert parent_loop is not None


def child():
    assert call("child") == "ok"
    assert sync._loop is not None and sync._loop is not parent_loop


assert reap(fork_child(child)) == 0
# The parent's loop is untouched and still serves its calls.
assert sync._loop is parent_loop
assert call("parent-after") == "ok"
print("done")
""",
        tmp_path,
    )


@pytest.mark.parametrize("child_calls", [True, False], ids=["after-a-call", "without-a-call"])
def test_child_exit_does_not_wait_on_the_parents_loop(tmp_path: Path, child_calls: bool) -> None:
    """A child's normal exit runs llmkit's shutdown against its own loop, or none.

    Against the inherited loop, the exit drain waited out its whole ten-second
    bound. A normal exit of a process that has loaded LiteLLM takes about a
    second here, so the bound leaves room for a slow machine and still sits
    well under the drain bound.
    """
    _run(
        rf"""
assert call() == "ok"


def child():
    if {child_calls!r}:
        assert call("child") == "ok"


assert reap(fork_child(child, exit_normally=True), bound=10.0, exit_bound=6.0) == 0
print("done")
""",
        tmp_path,
    )


def test_litellm_logging_worker_is_detached_in_the_child(tmp_path: Path) -> None:
    """The parent's queued LiteLLM callbacks stay the parent's.

    The parent makes a real LiteLLM call, whose success path binds LiteLLM's
    logging worker to llmkit's loop, then forks while three more callbacks wait
    in the worker's queue. The child makes its own real call and exits
    normally: none of the parent's callbacks runs there, and nothing reports a
    destroyed task or an unawaited coroutine.
    """
    _run(
        r"""
import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER as worker

from llmkit import sync
from llmkit.sync import run_sync

MARKS = WORK_DIR / "marks"
REQUEST = dict(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
    mock_response="ok",
    api_key="sk-test",
)


async def mark():
    with MARKS.open("a") as file:
        file.write(f"{os.getpid()}\n")


response = run_sync(litellm.acompletion(**REQUEST))
assert response.choices[0].message.content == "ok"
loop = sync._loop
assert worker._bound_loop is loop

# Pause the loop thread with three callbacks queued and the worker task
# pending, so the fork copies that state exactly.
queued = threading.Event()
resume = threading.Event()


def pause_with_callbacks_queued():
    for _ in range(3):
        worker.ensure_initialized_and_enqueue(mark())
    queued.set()
    resume.wait()


loop.call_soon_threadsafe(pause_with_callbacks_queued)
assert queued.wait(5)


def child():
    assert worker._queue is None and worker._bound_loop is None
    response = run_sync(litellm.acompletion(**REQUEST))
    assert response.choices[0].message.content == "ok"
    assert worker._bound_loop is sync._loop


code = reap(fork_child(child, exit_normally=True), bound=10.0, exit_bound=6.0)
resume.set()
assert code == 0, code
sync.shutdown()
marks = MARKS.read_text().split()
assert marks == [str(os.getpid())] * 3, marks
print("done")
""",
        tmp_path,
    )


def test_child_forked_during_a_parent_call_exits_quietly(tmp_path: Path) -> None:
    """A parent call in flight at the fork is the parent's; the child stays silent.

    The child inherits the in-flight task on the parent's loop. It neither
    runs it nor reports it destroyed when its own normal exit collects it.
    """
    _run(
        r"""
from llmkit import sync

assert call() == "ok"
_fake.__exit__(None, None, None)
_fake = install(lambda request: Reply(content="slow", delay=1.0))
_fake.__enter__()
results = []
caller = threading.Thread(target=lambda: results.append(call("slow")), daemon=True)
caller.start()
time.sleep(0.3)  # the call is now waiting inside the fake, on the loop
assert any(not task.done() for task in asyncio.all_tasks(sync._loop))


def child():
    assert call("child") == "slow"


code = reap(fork_child(child, exit_normally=True), bound=10.0, exit_bound=6.0)
caller.join(5)
assert code == 0, code
assert results == ["slow"], results
print("done")
""",
        tmp_path,
    )


_HELD_LOCKS = {
    "sync-loop": "sync._lock",
    "sync-fallback": "sync._fallback_lock",
    "rate-limiter": "GlobalRateLimiter._lock",
    "cache-flight": "_flight._lock",
    "cache-store": "get_llm_cache()._lock",
    "transport-doubles": "_transport._doubles_lock",
    "temperature-warnings": "_litellm._temperature_warned_lock",
}


@pytest.mark.parametrize("lock", list(_HELD_LOCKS.values()), ids=list(_HELD_LOCKS))
def test_child_forked_while_a_parent_thread_holds_a_lock(tmp_path: Path, lock: str) -> None:
    """Each lock llmkit resets is unheld in the child, which completes a call.

    The parent configures the shipped in-memory cache, so the child's call
    passes the single-flight table, the store, the transport doubles, the rate
    limiter and the sync loop.
    """
    _run(
        rf"""
from llmkit import _litellm, _transport, InMemoryLLMCache, configure_llm_cache, get_llm_cache, sync
from llmkit.cache import _flight
from llmkit.rate_limiting import GlobalRateLimiter

configure_llm_cache(InMemoryLLMCache())
assert call() == "ok"
release = hold({lock})


def child():
    assert {lock}.acquire(timeout=1)
    {lock}.release()
    assert call("child") == "ok"


code = reap(fork_child(child))
release.set()
assert code == 0, code
print("done")
""",
        tmp_path,
    )


def test_child_acquires_while_a_parent_thread_waits_for_a_slot(tmp_path: Path) -> None:
    """Limiter state is the child's own; its settings are the parent's.

    With one slot per provider, a parent thread holds the slot and another
    waits for it at the fork. Neither exists in the child, whose limiter starts
    empty under the same policy.
    """
    _run(
        r"""
from llmkit import Provider, RateLimitPolicy, configure_rate_limit, rate_limit_acquire_sync
from llmkit.rate_limiting import GlobalRateLimiter

configure_rate_limit(policy=RateLimitPolicy(max_concurrent=1), adaptive=False)
holding = threading.Event()
release = threading.Event()


def holder():
    with rate_limit_acquire_sync(Provider.OLLAMA):
        holding.set()
        release.wait()


waiting_started = threading.Event()


def waiter():
    waiting_started.set()
    with rate_limit_acquire_sync(Provider.OLLAMA):
        pass


threading.Thread(target=holder, daemon=True).start()
assert holding.wait(5)
waiter_thread = threading.Thread(target=waiter, daemon=True)
waiter_thread.start()
assert waiting_started.wait(5)
time.sleep(0.2)  # let the waiter queue its ticket
assert waiter_thread.is_alive()


def child():
    assert GlobalRateLimiter._policy.max_concurrent == 1
    assert GlobalRateLimiter._adaptive is False
    with rate_limit_acquire_sync(Provider.OLLAMA):
        pass
    assert call("child") == "ok"


code = reap(fork_child(child))
release.set()
waiter_thread.join(5)
assert code == 0, code
print("done")
""",
        tmp_path,
    )


def test_stream_opened_before_the_fork_raises_in_the_child(tmp_path: Path) -> None:
    """An inherited open sync stream is the parent's: its next step raises."""
    _run(
        r"""
from llmkit import text_llm_call_stream_sync

_fake.__exit__(None, None, None)
_fake = install(lambda request: Reply(chunks=("a", "b", "c")))
_fake.__enter__()
stream = text_llm_call_stream_sync("hi", feature="fork")
first = next(stream)


def child():
    try:
        next(stream)
    except RuntimeError as exc:
        assert "opened before this process forked" in str(exc), exc
    else:
        raise AssertionError("the inherited stream advanced in the child")
    stream.close()
    assert call("child") == "abc"


assert reap(fork_child(child)) == 0
# The parent's stream is unaffected.
rest = list(stream)
assert first is not None and rest, (first, rest)
print("done")
""",
        tmp_path,
    )


def test_litellm_logging_worker_has_the_state_the_fork_reset_clears() -> None:
    """The LiteLLM attributes the child-side detach relies on still exist.

    The detach skips a name LiteLLM does not have, because the newest names are
    absent from the oldest supported LiteLLM; the ones checked here exist on
    every supported version, and a LiteLLM without them would leave the child's
    worker bound to the parent's loop.
    """
    required = {
        "_queue",
        "_worker_task",
        "_sem",
        "_bound_loop",
        "_running_tasks",
        "_timeout_summary_task",
        "_dequeued_tasks",
    }
    assert required <= set(_litellm._LOGGING_WORKER_LOOP_STATE)
    missing = sorted(name for name in required if not hasattr(GLOBAL_LOGGING_WORKER, name))
    assert not missing, missing
