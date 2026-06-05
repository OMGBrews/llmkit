"""Synchronous wrapper for running async coroutines."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine


def run_sync[T](coro: Coroutine[object, object, T], *, timeout: float = 60) -> T:
    """Run an async coroutine synchronously, handling event loop detection.

    If an event loop is already running (e.g. inside an async framework),
    runs the coroutine in a new thread with its own event loop.
    Otherwise, uses ``asyncio.run()`` directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():

        def run_in_new_loop() -> T:
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result(timeout=timeout)
    else:
        return asyncio.run(coro)
