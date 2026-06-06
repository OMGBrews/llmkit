"""Synchronous wrapper for running async coroutines."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine


def run_sync[T](coro: Coroutine[object, object, T], *, timeout: float | None = 600) -> T:
    """Run an async coroutine synchronously, handling event loop detection.

    If an event loop is already running (e.g. inside an async framework),
    runs the coroutine in a new thread with its own event loop and waits up
    to ``timeout`` seconds for it. Otherwise, uses ``asyncio.run()`` directly
    (in which case ``timeout`` does not apply — the coroutine runs to
    completion in the current thread).

    ``timeout`` defaults to **600 seconds** (10 minutes), not a few seconds:
    structured LLM generations routinely run tens of seconds, and a large
    capped prompt can run minutes, so a short default would raise
    :class:`concurrent.futures.TimeoutError` mid-flight on a slow-but-valid
    call. The global rate limiter (:class:`~llmkit.rate_limiting.GlobalRateLimiter`)
    is the real concurrency backpressure; this ceiling exists only to bound a
    *hung* provider rather than to pace healthy calls. Pass ``None`` to wait
    unbounded (relying on LiteLLM's own request timeout), or a smaller value
    when the caller has a tighter latency budget.

    .. warning::
       When the worker-thread path times out, the raised
       :class:`concurrent.futures.TimeoutError` only abandons the *wait* —
       the worker thread keeps executing the coroutine (and its in-flight
       provider request) to completion, because a coroutine driven by
       ``asyncio.run`` in another thread cannot be cancelled from here. A
       timed-out call therefore leaks one thread and one provider request
       until they finish on their own. This is why the default is generous:
       timing out is a last-resort signal that something is wrong, not a
       routine control-flow path.
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
