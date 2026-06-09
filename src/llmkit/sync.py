"""Synchronous wrapper for running async coroutines."""

import asyncio
import concurrent.futures
import contextlib
import contextvars
from collections.abc import Coroutine


def _run_and_drain[T](coro: Coroutine[object, object, T], *, timeout: float | None) -> T:
    """Run ``coro`` to completion on a fresh loop, then drain stragglers.

    A plain :func:`asyncio.run` closes its loop the instant the *driving*
    coroutine returns. But LiteLLM logs asynchronously without awaiting inline:
    after a successful call it eagerly builds the
    ``Logging.async_success_handler`` coroutine and queues it on a background
    worker. That coroutine can still be pending — created but never awaited —
    when the loop closes, and Python then prints
    ``RuntimeWarning: coroutine 'Logging.async_success_handler' was never
    awaited`` to stderr: visible noise on a perfectly successful sync call.

    So before closing the loop we drain that pending logging two ways:

    * :func:`~llmkit._litellm.drain_async_logging` flushes LiteLLM's logging
      worker queue, awaiting the queued ``async_success_handler`` coroutines so
      none is destroyed unawaited. This is the real fix — the leaked coroutine
      lives in a queue, not in a task, so a generic task-drain alone can't reach
      it.
    * :func:`_drain_pending` then cancels-and-settles any remaining loop tasks
      (LiteLLM's logging-worker loop is an intentionally infinite task; the
      streaming path schedules its handler as a fire-and-forget ``create_task``
      directly). By this point every such task has already been *started*, so
      cancelling it is warning-free — only a coroutine that was never started
      triggers the warning, and the queue flush above ran those.

    Both drains are bounded by the same ``timeout`` budget as the call, so a
    hung callback can't wedge the bridge; the call's own result (or exception)
    always propagates ahead of any straggler outcome. The drains run whether
    ``coro`` succeeded or raised — an error path leaves the same pending
    logging — and never hide the call's result or error.
    """
    from llmkit import _litellm

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        # Wrap the call itself in a task so it shares the loop with whatever
        # LiteLLM schedules, and so the drain below can see it complete.
        main = loop.create_task(coro)
        try:
            return loop.run_until_complete(main)
        finally:
            # Drain pending LiteLLM logging whether the call succeeded or
            # raised — an undrained coroutine/task is otherwise destroyed
            # (and warned about) at ``loop.close()``.
            loop.run_until_complete(_litellm.drain_async_logging(timeout=timeout))
            _drain_pending(loop, timeout=timeout)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _drain_pending(loop: asyncio.AbstractEventLoop, *, timeout: float | None) -> None:
    """Cancel and settle any tasks still pending on ``loop`` before teardown.

    After :func:`~llmkit._litellm.drain_async_logging` has flushed LiteLLM's
    logging queue, the loop can still hold *background* tasks — most notably
    LiteLLM's logging-**worker loop**, an intentionally infinite
    ``while True`` that blocks on ``queue.get()`` and never completes on its
    own. Awaiting those to completion would hang the bridge, so we instead
    **cancel** every remaining task and gather the cancellations. By this point
    each task has already been entered (stepped at least once), so cancelling it
    is warning-free — the "never awaited" warning only fires for a coroutine
    that was *never started*, and the logging-queue flush above already ran the
    queued handler coroutines.

    The settle is bounded by ``timeout`` so a task that swallows cancellation
    can't wedge teardown; past the deadline we stop waiting and let
    ``loop.close()`` finish up. Task exceptions are ignored
    (``return_exceptions=True``): these are best-effort background tasks, not
    the call's result.
    """
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if not pending:
        return
    for task in pending:
        _ = task.cancel()
    drain = asyncio.gather(*pending, return_exceptions=True)
    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
        _ = loop.run_until_complete(asyncio.wait_for(drain, timeout=timeout))


def run_sync[T](coro: Coroutine[object, object, T], *, timeout: float | None = 600) -> T:
    """Run an async coroutine synchronously, handling event loop detection.

    If an event loop is already running (e.g. inside an async framework),
    runs the coroutine in a new thread with its own event loop and waits up
    to ``timeout`` seconds for it. Otherwise, drives the coroutine on a fresh
    event loop in the current thread (``timeout`` then bounds only the
    post-call straggler drain, not the call itself — the coroutine runs to
    completion).

    Either way the bridge **drains** LiteLLM's pending async logging before
    closing the loop. LiteLLM queues its success logging
    (``Logging.async_success_handler``) to run in the background and does not
    await it inline; a plain ``asyncio.run`` would close the loop with that
    coroutine still un-awaited, leaking a ``RuntimeWarning: coroutine '…' was
    never awaited`` to stderr on an otherwise clean sync call. The drain flushes
    that logging (or, past the ``timeout`` deadline, gives up) before teardown.
    See :func:`_run_and_drain`.

    ``timeout`` defaults to **600 seconds** (10 minutes), not a few seconds:
    structured LLM generations routinely run tens of seconds, and a large
    capped prompt can run minutes, so a short default would raise
    :class:`concurrent.futures.TimeoutError` mid-flight on a slow-but-valid
    call. The global rate limiter (:class:`~llmkit.rate_limiting.GlobalRateLimiter`)
    is the real concurrency backpressure; this ceiling exists only to bound a
    *hung* provider rather than to pace healthy calls. Pass ``None`` to wait
    unbounded (relying on LiteLLM's own request timeout) — which also leaves the
    post-call logging drain unbounded, so a genuinely wedged logging callback
    could block teardown — or a smaller value when the caller has a tighter
    latency budget.

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
            return _run_and_drain(coro, timeout=timeout)

        # Run the worker inside a *copy* of the caller's context so the
        # context-scoped state the call layer relies on — ``capture_llm_records``
        # / ``capture_llm_log_paths`` (both record into a list held in a
        # ``ContextVar``) and the retry ``_progress_callback`` — crosses into the
        # worker thread. A bare ``executor.submit`` does not propagate
        # ``contextvars``, so without this a ``*_sync`` call made from inside a
        # running loop would silently capture nothing (the lists live in the
        # parent context the worker never sees). The captured lists are shared by
        # reference, so the worker's appends are visible to the caller on return.
        ctx = contextvars.copy_context()
        # Deliberately *not* a ``with`` block. The context manager's ``__exit__``
        # calls ``executor.shutdown(wait=True)``, which blocks until the worker
        # thread (still driving the in-flight provider request) finishes — so on
        # the timeout path the ``TimeoutError`` could not leave this frame until
        # the very call it was meant to abandon completed, silently defeating
        # ``timeout``. Instead we shut down with ``wait=False`` so the error
        # reaches the caller at the deadline while the abandoned worker drains on
        # its own, matching the ``.. warning::`` contract in this docstring.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(ctx.run, run_in_new_loop)
        try:
            return future.result(timeout=timeout)
        finally:
            # ``wait=False``: an already-running future cannot be cancelled, so
            # never block teardown on it. On timeout this lets the ``TimeoutError``
            # propagate immediately; on success the worker is already done and
            # this is a no-op. The abandoned worker is a non-daemon thread and
            # keeps the interpreter alive at exit until the call completes.
            executor.shutdown(wait=False)
    else:
        return _run_and_drain(coro, timeout=timeout)
