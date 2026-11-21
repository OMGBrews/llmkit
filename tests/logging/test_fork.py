"""Managed log delivery in a forked child, in fresh interpreters.

A child process inherits a copy of the parent's memory but only the thread that
called ``fork``. Each scenario forks from a fresh interpreter, gives the child a
bounded time to submit, flush and report, and kills it if it does not finish,
so a child that inherited a dead worker or a held lock fails its case instead
of hanging the suite. Children never call a ``*_sync`` helper: the inherited
sync loop is a separate defect. Each child ends with an explicit flush and
``os._exit``, so no result depends on an exit drain.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="fork is POSIX-only")

_ROOT = Path(__file__).resolve().parents[2]

# Runs in the forking interpreter. ``child`` is the body run in the forked
# child; ``reap`` waits for it with a bound and reports a hang as a failure.
_COMMON = r"""
import os
import sys
import time
from pathlib import Path

from llmkit import flush_llm_logs, get_diagnostics_health, get_logging_health
from llmkit.capture import submit_record_call
from tests._support import make_record

LOG_DIR = Path(sys.argv[1])


def labels():
    return sorted(
        line.split(":", 1)[1].strip()
        for path in LOG_DIR.rglob("*.yaml")
        for line in path.read_text().splitlines()
        if line.startswith("label:")
    )


def fork_child(body):
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            body()
            code = 0
        except BaseException as exc:
            print(f"child failed: {exc!r}", flush=True)
        finally:
            os._exit(code)
    return pid


def reap(pid, bound=5.0):
    deadline = time.monotonic() + bound
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            _ = os.waitpid(pid, 0)
            return "hung"
        time.sleep(0.01)


def child_logs_and_flushes():
    health = get_logging_health()
    assert (health.queued, health.in_flight, health.written, health.dropped) == (0, 0, 0, 0), health
    submit_record_call(make_record(label="child"))
    assert flush_llm_logs(timeout=3), get_logging_health()
    health = get_logging_health()
    assert (health.written, health.queued, health.failed) == (1, 0, 0), health
    _ = get_diagnostics_health()
"""


def _run(source: str, log_dir: Path) -> None:
    """Run *source* after the common helpers; it must print ``done`` and exit 0."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLMKIT_LOG")}
    env["LLMKIT_LOG_DIR"] = str(log_dir)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", _COMMON + source, str(log_dir)],
            cwd=_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = cast("str | None", exc.output)
        pytest.fail(f"forking interpreter did not finish:\n{output}")
    assert result.returncode == 0, result.stdout
    assert "done" in result.stdout.splitlines(), result.stdout


def test_child_forked_after_worker_started_writes_its_own_records(tmp_path: Path) -> None:
    """The parent's worker is running; the child gets a fresh one and fresh counts."""
    _run(
        r"""
submit_record_call(make_record(label="parent"))
assert flush_llm_logs(timeout=3)
assert get_logging_health().written == 1
code = reap(fork_child(child_logs_and_flushes))
assert code == 0, code
# The child's write is counted in the child only; the parent keeps its own.
health = get_logging_health()
assert (health.written, health.queued) == (1, 0), health
submit_record_call(make_record(label="parent-after"))
assert flush_llm_logs(timeout=3)
assert labels() == ["child", "parent", "parent-after"], labels()
print("done")
""",
        tmp_path,
    )


def test_multiprocessing_fork_worker_writes_its_own_records(tmp_path: Path) -> None:
    """A fork-context pool started after a logged call delivers each worker's record."""
    _run(
        r"""
import multiprocessing


def work(label):
    submit_record_call(make_record(label=label))
    return flush_llm_logs(timeout=3)


submit_record_call(make_record(label="parent"))
assert flush_llm_logs(timeout=3)
with multiprocessing.get_context("fork").Pool(2) as pool:
    result = pool.map_async(work, ["a", "b", "c"])
    assert result.get(timeout=10) == [True, True, True]
assert labels() == ["a", "b", "c", "parent"], labels()
print("done")
""",
        tmp_path,
    )


# Every lock the fork hooks replace, as an expression evaluated in the parent
# before fork. The default sink is still implicit (``LLMKIT_LOG`` unread), so the
# child's first submit takes ``_switch_lock``, and the sink has not resolved its
# retention, so the child's first write takes ``_retention_lock``. Those two are
# held by the forking thread itself across the fork, which then waits for the
# holder's timed release; the others are still held by the parent thread.
_LOCKS = {
    "delivery-condition": "_delivery._delivery.condition",
    "switch-lock": "registry._switch_lock",
    "policy-lock": "_policy._lock",
    "health-lock": "registry._installed.health._lock",
    "diagnostics-lock": "_diagnostics._lock",
    "retention-lock": "registry._installed.sink._retention_lock",
}


@pytest.mark.parametrize("expression", list(_LOCKS.values()), ids=list(_LOCKS))
def test_child_forked_while_a_parent_thread_holds_a_delivery_lock_does_not_hang(
    tmp_path: Path, expression: str
) -> None:
    """A parent thread inside a critical section at fork leaves the lock held in the child."""
    _run(
        rf"""
import threading

import llmkit._diagnostics as _diagnostics
import llmkit.logging._delivery as _delivery
import llmkit.logging._policy as _policy
import llmkit.logging.registry as registry

lock = {expression}
held = threading.Event()
release = threading.Event()


def hold():
    with lock:
        held.set()
        release.wait()


threading.Thread(target=hold, daemon=True).start()
assert held.wait(3)
timer = threading.Timer(1.0, release.set)
timer.start()
pid = fork_child(child_logs_and_flushes)
release.set()
code = reap(pid)
assert code == 0, code
assert labels() == ["child"], labels()
print("done")
""",
        tmp_path,
    )


def test_child_forked_during_configure_keeps_the_explicit_sink(tmp_path: Path) -> None:
    """A fork never sees a configure half applied, so ``LLMKIT_LOG`` cannot undo it.

    The parent thread reproduces ``configure_llm_logging``'s critical section
    and pauses between its two assignments; the fork waits for it to finish.
    """
    _run(
        r"""
import os
import threading

import llmkit.logging.registry as registry
from llmkit import LocalYamlLogSink, get_log_sink

os.environ["LLMKIT_LOG"] = "off"
explicit = LocalYamlLogSink(log_dir=LOG_DIR)
inside = threading.Event()
release = threading.Event()


def configure():
    with registry._switch_lock:
        registry._installed = registry._install(explicit)
        inside.set()
        release.wait()
        registry._implicit = False


threading.Thread(target=configure, daemon=True).start()
assert inside.wait(3)
threading.Timer(1.0, release.set).start()


def child():
    assert get_log_sink() is explicit, get_log_sink()
    child_logs_and_flushes()


code = reap(fork_child(child))
assert code == 0, code
assert labels() == ["child"], labels()
print("done")
""",
        tmp_path,
    )
