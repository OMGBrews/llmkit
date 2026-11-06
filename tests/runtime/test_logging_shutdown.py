"""Managed sink delivery remains bounded across loop shutdown and process exit.

Each case gets its own interpreter: delivery policy and the single worker are
process-wide, and a deliberately blocked daemon must not contaminate later tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.mark.parametrize("start_sync_loop", [False, True])
def test_shutdown_flushes_without_disabling_or_replacing_worker(start_sync_loop: bool) -> None:
    script = (
        "START_SYNC_LOOP = "
        + repr(start_sync_loop)
        + "\n"
        + textwrap.dedent(
            """
        import asyncio
        import threading
        import time

        from llmkit import LogDeliveryPolicy, configure_llm_logging, flush_llm_logs
        from llmkit.capture import record_call_async
        from llmkit import sync
        from tests._support import make_record

        entered = threading.Event()
        release = threading.Event()
        threads = []

        class Sink:
            def write(self, record):
                threads.append(threading.get_ident())
                entered.set()
                release.wait()

        configure_llm_logging(
            Sink(), delivery_policy=LogDeliveryPolicy(delivery_wait=0, drain_timeout=0.5)
        )
        if START_SYNC_LOOP:
            sync.run_sync(record_call_async(make_record()))
        else:
            asyncio.run(record_call_async(make_record()))
            assert sync._loop is None
        assert entered.wait(5), "sink never entered"
        start = time.monotonic()
        sync.shutdown(timeout=0.03)
        elapsed = time.monotonic() - start
        assert elapsed < 0.3, f"shutdown ignored smaller timeout: {elapsed}"
        assert not release.is_set(), "blocked-write control unexpectedly released"
        assert len(threads) == 1
        release.set()
        assert flush_llm_logs(timeout=2), "timed-out write was not retained"
        sync.run_sync(record_call_async(make_record()))
        assert flush_llm_logs(timeout=2)
        assert len(threads) == 2
        assert threads[0] == threads[1], "shutdown replaced logging worker"
        sync.shutdown()
        print("RESTARTED_SAME_WORKER")
        """
        )
    )
    assert "RESTARTED_SAME_WORKER" in _run(script)


def test_shutdown_without_sync_loop_actually_waits_for_pending_delivery() -> None:
    stdout = _run(
        """
        import asyncio
        import threading

        from llmkit import LogDeliveryPolicy, configure_llm_logging, get_logging_health
        from llmkit.capture import record_call_async
        from llmkit import sync
        from tests._support import make_record

        entered = threading.Event()
        release = threading.Event()
        shutdown_returned = threading.Event()

        class Sink:
            def write(self, record):
                entered.set()
                release.wait()

        configure_llm_logging(Sink(), delivery_policy=LogDeliveryPolicy(delivery_wait=0))
        asyncio.run(record_call_async(make_record()))
        assert entered.wait(5)
        assert sync._loop is None

        def close():
            sync.shutdown()
            shutdown_returned.set()

        closer = threading.Thread(target=close)
        closer.start()
        assert not shutdown_returned.wait(0.05), "no-loop shutdown skipped logging flush"
        release.set()
        assert shutdown_returned.wait(5), "shutdown did not observe released write"
        closer.join(5)
        assert get_logging_health().handed_off == 1
        print("ASYNC_ONLY_FLUSHED")
        """
    )
    assert "ASYNC_ONLY_FLUSHED" in stdout


def test_async_only_exit_runs_one_bounded_terminal_drain() -> None:
    stdout = _run(
        """
        import atexit
        import time

        def observe():
            elapsed = time.monotonic() - exiting_at
            assert elapsed < 0.5, f"exit spent more than one drain budget: {elapsed}"
            assert get_logging_health().dropped == 1
            print("EXIT_DRAINED", flush=True)

        # Registered first so this observes the completed llmkit exit hook.
        atexit.register(observe)
        import asyncio
        import threading
        from llmkit import LogDeliveryPolicy, configure_llm_logging, get_logging_health
        from llmkit.capture import record_call_async
        from llmkit import sync
        from tests._support import make_record

        entered = threading.Event()
        never_released = threading.Event()

        class Sink:
            def write(self, record):
                entered.set()
                never_released.wait()

        configure_llm_logging(
            Sink(), delivery_policy=LogDeliveryPolicy(delivery_wait=0, drain_timeout=0.3)
        )
        asyncio.run(record_call_async(make_record()))
        assert entered.wait(5)
        asyncio.run(record_call_async(make_record()))
        assert sync._loop is None
        exiting_at = time.monotonic()
        print("EXIT_STARTED", flush=True)
        """
    )
    assert "EXIT_STARTED" in stdout
    assert "EXIT_DRAINED" in stdout
