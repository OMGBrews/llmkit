"""Synchronous bridge: drive async coroutines on one persistent event loop.

The sync call wrappers (``*_sync`` in :mod:`llmkit.structured_output`) need to
drive an ``async`` coroutine to completion from synchronous code. :func:`run_sync`
is that bridge.

**Why one persistent loop, not a fresh loop per call.** LiteLLM logs successful
calls asynchronously through a *process-global singleton*,
``GLOBAL_LOGGING_WORKER`` (``litellm.litellm_core_utils.logging_worker``). That
worker holds per-loop mutable state — a queue, a bound-loop reference, a
background worker task — and **rebinds it, without a lock, whenever it sees a new
event loop**. That is safe when one server loop is replaced sequentially over
time, but it is *not* safe when many short-lived loops touch the singleton at
once. A sync fan-out (N threads each making one sync call) driven by a
*fresh loop per call* is exactly that adversarial regime: the loops tear each
other's queue / worker-task apart, flooding stderr (``Task was destroyed but it
is pending``, ``<Queue …> is bound to a different event loop``, ``coroutine
'Logging.async_success_handler' was never awaited``) and — the damaging one —
occasionally wedging a teardown ``flush()`` on an orphaned queue whose
``task_done()`` never fires, which only releases when LiteLLM's ~600 s request
timeout expires: a ~600 s hang on an otherwise-successful run.

Routing **every** sync call through one long-lived loop ``L`` on one dedicated
daemon thread sidesteps the whole bug class: LiteLLM's worker binds to ``L``
once and is drained exactly once, at shutdown — the bind-once / drain-once mode
the worker is actually built for. It also removes per-call loop
creation/teardown overhead; concurrent sync calls become true cooperative async
concurrency on one loop (LiteLLM's intended mode), with the per-provider
concurrency cap enforced inside the async path by ``GlobalRateLimiter``'s
now-shared async semaphore (keyed on ``L``).

**The reentrant fallback.** :func:`run_sync` cannot block the calling thread on
``L`` when that thread *already* has a running loop — an async framework calling
a ``*_sync`` helper, or a coroutine running on ``L`` itself that calls back into
sync code. Blocking would either freeze the caller's own loop forever or, in the
``L``-on-``L`` case, deadlock. So when a loop is already running in the calling
thread, the coroutine is offloaded to a one-shot worker thread with its own
fresh loop (:func:`_run_and_drain`), which drains LiteLLM's logging before
closing that loop. This path keeps the old fresh-loop semantics — including the
documented "a timed-out worker keeps running" caveat — and is the only remaining
fresh-loop path; because it runs on a non-main worker thread it can never be
interrupted by a signal mid-call.

**Context propagation.** Submitting to ``L`` crosses a thread boundary, so the
caller's :mod:`contextvars` context is captured and the coroutine runs inside a
copy of it (``loop.create_task(coro, context=ctx)``). That carries llmkit's own
context-scoped state across the bridge — ``capture_llm_records`` /
``capture_llm_log_paths`` (ContextVar-held lists) and the retry
``_progress_callback``. A bare :func:`asyncio.run_coroutine_threadsafe` does
**not** propagate context and would silently break log/record capture.
"""

import asyncio
import atexit
import concurrent.futures
import contextlib
import contextvars
import logging
import threading
from collections.abc import Coroutine

logger = logging.getLogger(__name__)

# The single persistent loop and the daemon thread that runs it, started lazily
# on first use under ``_lock`` so concurrent first-callers can't race two loops
# into existence. ``shutdown`` clears both back to ``None``.
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _loop_main(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    """Run the persistent loop forever on its dedicated daemon thread.

    Signals *ready* once the loop is installed as this thread's loop (so the
    first :func:`asyncio.AbstractEventLoop.call_soon_threadsafe` has a loop to
    wake), then blocks in ``run_forever`` until :func:`shutdown` stops it. On
    stop it settles async generators and closes the loop — the close must happen
    on this thread, after ``run_forever`` returns, so it pairs with the
    drain-and-cancel :func:`shutdown` runs *before* stopping the loop.
    """
    asyncio.set_event_loop(loop)
    ready.set()
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # pragma: no cover - best-effort teardown
            logger.debug("shutdown_asyncgens on the persistent loop raised", exc_info=True)
        finally:
            loop.close()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the process-global persistent loop, starting it on first use.

    Thread-safe and lazy: the loop and its daemon thread are created under
    ``_lock`` so two first-callers can't construct competing loops, and an
    :mod:`atexit` hook is registered once to drain and close it on a clean exit.
    """
    global _loop, _thread
    with _lock:
        if _loop is not None:
            return _loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        thread = threading.Thread(
            target=_loop_main, args=(loop, ready), name="llmkit-sync-loop", daemon=True
        )
        thread.start()
        _loop = loop
        _thread = thread
    # Wait outside the lock: the thread sets ``ready`` once the loop is its
    # current loop, so a coroutine scheduled immediately after isn't lost.
    _ = ready.wait()
    return loop


def _submit_to_loop[T](coro: Coroutine[object, object, T], *, timeout: float | None) -> T:
    """Drive *coro* on the persistent loop, preserving the caller's context.

    Captures ``contextvars.copy_context()`` in the calling thread and creates the
    task on the loop *inside that copy*, so context-scoped state (the capture
    lists, the retry progress callback) crosses the thread boundary — a bare
    :func:`asyncio.run_coroutine_threadsafe` would not. The caller blocks on a
    :class:`concurrent.futures.Future` bridged from the task; ``timeout`` bounds
    that wait.

    On timeout (or any other exception escaping the wait, e.g. a
    :class:`KeyboardInterrupt` landing in the calling thread), the task is asked
    to **cancel** on the loop rather than leaked — cancelling the in-flight
    provider request too — and the original exception propagates. The call's own
    result or exception always propagates ahead of any straggler outcome.
    """
    loop = _ensure_loop()
    ctx = contextvars.copy_context()
    result: concurrent.futures.Future[T] = concurrent.futures.Future()
    task_box: list[asyncio.Task[T]] = []

    def _relay(fut: asyncio.Future[T]) -> None:
        # Runs on the loop thread when the task settles. If the caller already
        # abandoned the wait (timeout / interrupt), ``result`` is still pending
        # but unwatched — retrieve the task outcome so it isn't flagged
        # unretrieved, then drop it; otherwise copy the outcome across.
        if result.done():
            if not fut.cancelled():
                _ = fut.exception()
            return
        if fut.cancelled():
            result.set_exception(asyncio.CancelledError())
            return
        exc = fut.exception()
        if exc is not None:
            result.set_exception(exc)
        else:
            result.set_result(fut.result())

    def _schedule() -> None:
        # Runs on the loop thread: task creation must happen there, not from the
        # calling thread. The context copy makes the coroutine see the caller's
        # ContextVars (shared by reference, so its appends are visible on return).
        try:
            task = loop.create_task(coro, context=ctx)
        except BaseException as exc:  # pragma: no cover - create_task rarely raises
            result.set_exception(exc)
            return
        task_box.append(task)
        task.add_done_callback(_relay)

    def _cancel() -> None:
        # Runs on the loop thread: cancel the in-flight task if it exists and is
        # still pending. A no-op once the task is done, so it is safe to schedule
        # unconditionally from the exception path below.
        if task_box:
            _ = task_box[0].cancel()

    _ = loop.call_soon_threadsafe(_schedule)
    try:
        return result.result(timeout)
    except BaseException:
        _ = loop.call_soon_threadsafe(_cancel)
        raise


def _run_and_drain[T](coro: Coroutine[object, object, T], *, timeout: float | None) -> T:
    """Run *coro* on a fresh loop in *this* (worker) thread, then drain stragglers.

    The fresh-loop path retained **only** for the reentrant fallback in
    :func:`run_sync` — when the calling thread already has a running loop, so the
    coroutine cannot be driven on the persistent loop or in-thread. Runs on a
    one-shot worker thread, never the main thread, so it cannot be interrupted by
    a signal mid-call (which is why no KeyboardInterrupt settling is needed here).

    A plain :func:`asyncio.run` would close its loop the instant the driving
    coroutine returns, but LiteLLM logs without awaiting inline: it eagerly
    builds the ``Logging.async_success_handler`` coroutine and queues it on a
    background worker, which can still be pending when the loop closes — Python
    then prints ``RuntimeWarning: coroutine '…' was never awaited`` to stderr. So
    before closing the loop this drains that pending logging two ways:

    * :func:`~llmkit._litellm.drain_async_logging` flushes LiteLLM's logging
      worker queue, awaiting the queued coroutines so none is destroyed
      unawaited (the leaked coroutine lives in a queue, not a task, so a generic
      task-drain alone can't reach it);
    * :func:`_drain_pending` then cancels-and-settles any remaining loop tasks
      (LiteLLM's logging-worker loop is an intentionally infinite task). By this
      point every such task has been *started*, so cancelling it is warning-free.

    Both drains are bounded by the same ``timeout`` as the call, so a hung
    callback can't wedge the bridge; the call's own result/exception always
    propagates ahead of any straggler outcome. The drains run whether *coro*
    succeeded or raised.
    """
    from llmkit import _litellm

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Drain pending LiteLLM logging whether the call succeeded or raised —
            # an undrained coroutine/task is otherwise destroyed (and warned
            # about) at ``loop.close()``.
            loop.run_until_complete(_litellm.drain_async_logging(timeout=timeout))
            _drain_pending(loop, timeout=timeout)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _drain_pending(loop: asyncio.AbstractEventLoop, *, timeout: float | None) -> None:
    """Cancel and settle any tasks still pending on *loop* before teardown.

    After :func:`~llmkit._litellm.drain_async_logging` flushes LiteLLM's logging
    queue, the loop can still hold *background* tasks — most notably LiteLLM's
    logging-**worker loop**, an intentionally infinite ``while True`` that blocks
    on ``queue.get()`` and never completes on its own. Awaiting those to
    completion would hang the bridge, so we **cancel** every remaining task and
    gather the cancellations. By this point each task has been entered (stepped
    at least once), so cancelling it is warning-free — the "never awaited"
    warning only fires for a coroutine that was *never started*, and the
    logging-queue flush above already ran the queued handler coroutines.

    The settle is bounded by ``timeout`` so a task that swallows cancellation
    can't wedge teardown; past the deadline we stop waiting and let
    ``loop.close()`` finish up. Task exceptions are ignored
    (``return_exceptions=True``): these are best-effort background tasks, not the
    call's result.
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
    """Run an async coroutine synchronously, handling event-loop detection.

    Drives *coro* on llmkit's single **persistent** event loop (one dedicated
    daemon thread, started lazily) and blocks up to ``timeout`` seconds for the
    result. Routing every sync call through one loop is what keeps LiteLLM's
    process-global logging worker bound to a single loop — eliminating the
    cross-loop logging race a fresh-loop-per-call strategy triggers under a
    concurrent sync fan-out (flood + an occasional ~600 s hang). See the module
    docstring.

    If an event loop is **already running in the calling thread** (an async
    framework calling a ``*_sync`` helper, or a coroutine on the persistent loop
    calling back into sync code), the coroutine cannot be driven here without
    freezing that loop or deadlocking, so it is offloaded to a one-shot worker
    thread with its own fresh loop (:func:`_run_and_drain`).

    ``timeout`` bounds the **wait** and defaults to **600 seconds** (10 minutes),
    not a few seconds: structured generations routinely run tens of seconds and a
    large capped prompt can run minutes, so a short default would raise
    :class:`concurrent.futures.TimeoutError` mid-flight on a slow-but-valid call.
    The global rate limiter (:class:`~llmkit.rate_limiting.GlobalRateLimiter`) is
    the real concurrency backpressure; this ceiling exists only to bound a *hung*
    provider. Pass ``None`` to wait unbounded (relying on LiteLLM's own request
    timeout).

    On the persistent-loop path a timeout (or a :class:`KeyboardInterrupt` in the
    calling thread) **cancels** the coroutine on the loop — tearing down the
    in-flight provider request — before the exception propagates, so a timed-out
    call does not leak. On the reentrant worker-thread fallback that cancellation
    is not possible (a coroutine driven by another thread's loop can't be
    cancelled from here):

    .. warning::
       On the **worker-thread fallback**, a :class:`concurrent.futures.TimeoutError`
       only abandons the *wait* — the worker thread keeps executing the coroutine
       (and its in-flight provider request) to completion. A timed-out reentrant
       call therefore leaks one thread and one request until they finish on their
       own. This is why the default is generous: timing out is a last-resort
       signal that something is wrong, not a routine control-flow path.
    """
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread: the common path → the persistent loop.
        return _submit_to_loop(coro, timeout=timeout)

    # A loop is already running in this thread → reentrant fallback. Run the
    # worker inside a *copy* of the caller's context so the context-scoped state
    # the call layer relies on crosses into the worker thread (a bare
    # ``executor.submit`` would not propagate ``contextvars``).
    ctx = contextvars.copy_context()

    def _drive() -> T:
        return _run_and_drain(coro, timeout=timeout)

    # Deliberately *not* a ``with`` block: the context manager's ``__exit__``
    # calls ``executor.shutdown(wait=True)``, which blocks until the worker
    # finishes — so on the timeout path the ``TimeoutError`` could not leave this
    # frame until the very call it was meant to abandon completed. ``wait=False``
    # lets the error reach the caller at the deadline while the abandoned worker
    # drains on its own, matching the ``.. warning::`` contract above.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(ctx.run, _drive)
    try:
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)


async def _drain_and_close(timeout: float | None) -> None:
    """On the persistent loop: drain LiteLLM logging, then cancel background tasks.

    Run once by :func:`shutdown`. Flushes LiteLLM's queued success-logging so no
    coroutine is destroyed unawaited, then cancels and settles every remaining
    task on the loop (most importantly LiteLLM's intentionally-infinite
    logging-worker loop) so the subsequent ``loop.stop`` / ``close`` is clean.
    """
    from llmkit import _litellm

    await _litellm.drain_async_logging(timeout=timeout)
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        _ = task.cancel()
    if pending:
        _ = await asyncio.gather(*pending, return_exceptions=True)


def shutdown(*, timeout: float | None = 10.0) -> None:
    """Drain and close the persistent loop. Idempotent; registered via :mod:`atexit`.

    Drains LiteLLM's pending logging on the loop **once**, cancels its background
    worker task, runs ``shutdown_asyncgens``, closes the loop, and joins its
    thread — the bind-once / drain-once teardown that keeps a clean exit free of
    ``coroutine … was never awaited``. A no-op when the loop was never started or
    has already been shut down (the globals are cleared under ``_lock``, so a
    second call returns immediately). Exposed for hosts that want to reclaim the
    thread explicitly (e.g. before forking); most callers rely on the atexit hook.
    """
    global _loop, _thread
    with _lock:
        loop = _loop
        thread = _thread
        _loop = None
        _thread = None
    if loop is None:
        return
    try:
        cleanup = asyncio.run_coroutine_threadsafe(_drain_and_close(timeout), loop)
        _ = cleanup.result(timeout)
    except Exception:  # pragma: no cover - best-effort shutdown drain
        logger.debug("persistent-loop shutdown drain failed", exc_info=True)
    finally:
        _ = loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout if timeout is not None else 5.0)


_ = atexit.register(shutdown)
