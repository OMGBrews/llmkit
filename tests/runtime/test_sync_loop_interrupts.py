"""An interrupt raised on the sync loop thread reaches the caller; the loop survives.

``KeyboardInterrupt`` and ``SystemExit`` are the two exceptions asyncio lets
escape ``loop.run_forever()``: a task stores one as its outcome and then
re-raises it out of the loop. Code on the loop raises one itself when a host
callback or tool calls ``sys.exit()``, or a library re-raises an interrupt (a
real SIGINT is delivered to the main thread, never to the loop thread).

Before the loop thread contained them, such an exception ended
``llmkit-sync-loop``: the caller still received it, but the loop was closed
while ``llmkit.sync`` kept handing it out, so every later sync call in the
process raised ``RuntimeError: Event loop is closed`` or waited on a loop that
never ran again, and the exit hook printed the same error.

These tests pin, for both exception types and for both bridges (``run_sync``
and a step of an ``iterate_sync`` stream):

* the interrupt reaches the calling thread unchanged;
* the *same* loop thread object is still alive afterwards;
* a following sync call returns on it;
* process exit after the interrupt is clean;
* a restart never swallows the stop that ``shutdown`` queued.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import textwrap
from collections.abc import AsyncGenerator, Iterator

import pytest

from llmkit import sync as sync_module
from llmkit.sync import iterate_sync, run_sync

_INTERRUPTS = [KeyboardInterrupt, SystemExit]
_IDS = [exc.__name__ for exc in _INTERRUPTS]

# Bounds the follow-up call, which without containment raises at once or waits
# on a loop that never runs again.
_FOLLOW_UP_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _fresh_sync_loop() -> Iterator[None]:
    """Start each test on its own sync loop and leave a usable one behind.

    If containment regresses, the test kills the shared loop. The finalizer then
    discards the dead loop so that later tests start a new one.
    """
    sync_module.shutdown()
    yield
    loop = sync_module._loop
    if loop is not None and loop.is_closed():
        with sync_module._lock:
            sync_module._loop = None
            sync_module._thread = None
            sync_module._stop_requested = None
    else:
        with contextlib.suppress(RuntimeError):
            sync_module.shutdown()


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def _assert_loop_survived(thread: object) -> None:
    """The follow-up call returns, on the same thread, which is still alive."""
    assert run_sync(_answer(), timeout=_FOLLOW_UP_TIMEOUT) == 42
    assert sync_module._thread is thread
    assert sync_module._thread is not None
    assert sync_module._thread.is_alive()


@pytest.mark.parametrize("interrupt", _INTERRUPTS, ids=_IDS)
def test_an_interrupt_in_run_sync_reaches_the_caller_and_the_loop_survives(
    interrupt: type[BaseException],
) -> None:
    assert run_sync(_answer()) == 42  # start the loop
    thread = sync_module._thread
    assert thread is not None

    async def _raise() -> None:
        await asyncio.sleep(0)
        raise interrupt("raised on the loop thread")

    with pytest.raises(interrupt, match="raised on the loop thread"):
        run_sync(_raise(), timeout=_FOLLOW_UP_TIMEOUT)

    _assert_loop_survived(thread)


@pytest.mark.parametrize("interrupt", _INTERRUPTS, ids=_IDS)
def test_an_interrupt_in_a_stream_step_reaches_the_caller_and_the_loop_survives(
    interrupt: type[BaseException],
) -> None:
    assert run_sync(_answer()) == 42  # start the loop
    thread = sync_module._thread
    assert thread is not None

    async def _items() -> AsyncGenerator[int]:
        yield 1
        raise interrupt("raised in a stream step")

    stream = iterate_sync(_items())
    assert next(stream) == 1
    with pytest.raises(interrupt, match="raised in a stream step"):
        _ = next(stream)

    _assert_loop_survived(thread)


def test_an_interrupt_in_the_same_turn_as_shutdowns_stop_still_ends_the_loop() -> None:
    """A restart must not swallow the ``loop.stop`` that ``shutdown`` queued.

    ``loop.stop`` only sets a flag that ``run_forever`` reads after the current
    batch of callbacks. When a later callback in that batch raises an
    interrupt, ``run_forever`` ends by exception and asyncio clears the flag.
    A plain restart would then run forever and the loop would never close.
    One callback that requests the stop the way ``_shutdown_loop`` does and
    then raises reproduces that ordering deterministically.
    """
    loop = sync_module._ensure_loop()
    thread = sync_module._thread
    stop_requested = sync_module._stop_requested
    assert thread is not None
    assert stop_requested is not None
    with sync_module._lock:
        sync_module._loop = None
        sync_module._thread = None
        sync_module._stop_requested = None

    def _stop_then_interrupt() -> None:
        stop_requested.set()
        loop.stop()
        raise SystemExit("raised in the same turn as the stop")

    _ = loop.call_soon_threadsafe(_stop_then_interrupt)
    thread.join(_FOLLOW_UP_TIMEOUT)

    assert not thread.is_alive()
    assert loop.is_closed()


# Interrupt once through one bridge, then exit at once: the exit hook is the
# first thing to touch the loop afterwards, so a dead loop shows there.
_EXIT_SCRIPT = """
import sys

from llmkit.sync import iterate_sync, run_sync

interrupt = {"KeyboardInterrupt": KeyboardInterrupt, "SystemExit": SystemExit}[sys.argv[1]]


async def _raise():
    raise interrupt("call")


async def _items():
    yield 1
    raise interrupt("stream")


try:
    if sys.argv[2] == "call":
        run_sync(_raise(), timeout=5)
    else:
        stream = iterate_sync(_items())
        next(stream)
        next(stream)
except interrupt:
    print("caught")
"""


@pytest.mark.parametrize("bridge", ["call", "stream"])
@pytest.mark.parametrize("interrupt", _INTERRUPTS, ids=_IDS)
def test_process_exit_after_an_interrupt_on_the_loop_is_clean(
    interrupt: type[BaseException], bridge: str
) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EXIT_SCRIPT), interrupt.__name__, bridge],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "caught", proc.stdout + proc.stderr
    assert "Event loop is closed" not in proc.stderr, proc.stderr
    assert "Exception in thread" not in proc.stderr, proc.stderr
    assert "never awaited" not in proc.stderr, proc.stderr
