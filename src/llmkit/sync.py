"""Synchronous bridge: drive async coroutines on one persistent event loop.

The sync call wrappers (``*_sync`` in :mod:`llmkit.calls`) need to
drive an ``async`` coroutine to completion from synchronous code. :func:`run_sync`
is that bridge. The two sync streams (``*_stream_sync``) need to drive an async
generator one item at a time; :func:`iterate_sync` is that bridge, running each
generator step as a task on the same persistent loop (see its docstring for
what a stream adds: one context for the whole stream, closing on abandonment,
and a loop-thread check on every step).

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

**Reentrant calls.** A loop already running in the calling thread — an async
framework calling a ``*_sync`` helper, or a coroutine running on ``L`` itself
that calls back into sync code — splits into two cases:

* **A foreign loop** (any loop that is not ``L``): the coroutine is submitted to
  the persistent loop like any other sync call. Blocking the calling thread is
  inherent to calling sync-from-async — every strategy blocks that thread for
  the call's duration, freezing the caller's loop meanwhile — and ``L`` runs on
  its own thread, so the wait is bounded by the call, not a deadlock. Routing
  here (rather than a fresh loop per call) is what keeps a *concurrent*
  reentrant fan-out from re-creating the multi-loop regime above, and it keeps
  the persistent-loop guarantees: cancel-on-timeout and the shared per-provider
  concurrency cap.
* **``L`` itself** (a coroutine on ``L`` calls back into sync code): blocking
  ``L``'s thread on ``L`` would deadlock, so the coroutine is offloaded to a
  one-shot worker thread with its own fresh loop (:func:`_run_and_drain`), which
  drains LiteLLM's logging before closing that loop. This is the **only**
  remaining fresh-loop path, and it cannot run concurrently with itself — the
  call blocks ``L``'s thread, so no other coroutine on ``L`` can enter it, and
  ``_fallback_lock`` pins that at-most-one-fresh-loop invariant mechanically —
  which keeps LiteLLM's worker in the *sequential* rebind regime it tolerates.
  This path keeps the old fresh-loop semantics — including the documented "a
  timed-out worker keeps running" caveat — and because it runs on a non-main
  worker thread it can never be interrupted by a signal mid-call.

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
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Generator
from types import TracebackType
from typing import cast, final, overload, override

from llmkit.options import UNSET, Unset

logger = logging.getLogger(__name__)

# The wait :func:`run_sync` applies when its caller passes no ``timeout``. Read
# at call time, not bound as the parameter default, so it is one place to
# change. The call functions' sync forms do not use it: they pass
# ``timeout=None``, because a sync call must not add a limit the async call
# lacks (the per-request ``timeout`` and the retry budget bound both forms).
_DEFAULT_TIMEOUT_SECONDS: float = 600.0

# The bound on each logging drain the loop-on-loop fallback runs after its
# call settles. Independent of the call's own wait, which may be unbounded: an
# unbounded LiteLLM ``flush()`` is the hang the module docstring describes.
# Ten seconds matches :func:`shutdown`'s default drain bound.
_DRAIN_TIMEOUT_SECONDS: float = 10.0

# The error a sync stream raises when the loop its async generator is bound to
# has been shut down (see ``_submit_to_loop``'s ``loop`` keyword).
_LOOP_GONE_MESSAGE = "llmkit's sync loop was shut down while this stream was open"

# The single persistent loop and the daemon thread that runs it, started lazily
# on first use under ``_lock`` so concurrent first-callers can't race two loops
# into existence. ``shutdown`` clears both back to ``None``.
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None

# Serializes the loop-on-loop fallback's fresh-loop lifecycle: at most one fresh
# loop may ever exist at a time, so LiteLLM's logging worker only ever rebinds
# *sequentially* (the regime it tolerates), never across concurrent loops. The
# path is naturally serial today — the fallback blocks the persistent loop's own
# thread — but that invariant lives far from this code, so it is pinned here.
_fallback_lock = threading.Lock()


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


def _submit_to_loop[T](
    coro: Coroutine[object, object, T],
    *,
    timeout: float | None,
    context: contextvars.Context | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> T:
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

    Obtaining the loop and submitting to it are made atomic against
    :func:`shutdown` via ``_lock``: a caller that races shutdown retries onto a
    freshly restarted persistent loop instead of enqueueing a callback onto a
    stopped-but-not-closed loop (which would hang for the full ``timeout`` with
    the coroutine leaked) or a closed one (which would raise ``RuntimeError``).

    The sync stream bridge (:class:`_SyncGenerator`) passes two more keywords.
    ``context`` replaces the per-submission copy with one :class:`contextvars.Context`
    the caller reuses, so successive submissions share one context. ``loop``
    pins the submission to the loop the stream's async generator is bound to:
    the pinned path never starts a loop, and when that loop has been shut down
    it raises :class:`RuntimeError` instead of retrying, because a generator
    bound to a closed loop cannot be resumed on a fresh one.
    """
    ctx = contextvars.copy_context() if context is None else context
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
            task = target.create_task(coro, context=ctx)
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

    # Obtain the loop and submit to it atomically against ``shutdown()``. The
    # identity check ``_loop is loop`` under ``_lock`` is the generation check:
    # ``shutdown`` clears ``_loop`` under ``_lock`` *before* it stops or closes
    # the loop, and only ``_ensure_loop`` sets it (always to a brand-new object,
    # so no ABA). Under the lock the identity therefore implies the loop is live
    # and our ``_schedule`` is enqueued ahead of any subsequent ``loop.stop``
    # (FIFO) — so a racing caller is either swept as an in-flight call (prompt
    # ``CancelledError``) or retried onto a fresh loop, never left to hang or
    # raise. One uncontended lock acquire per call; no retry cap is needed
    # because livelock would require an adversarial ``shutdown()`` every
    # iteration. A pinned submission makes the same check but raises instead
    # of retrying, and never calls ``_ensure_loop``.
    target: asyncio.AbstractEventLoop
    if loop is not None:
        target = loop
        with _lock:
            if _loop is not target:
                coro.close()  # never started: close it so no "never awaited" warning
                raise RuntimeError(_LOOP_GONE_MESSAGE)
            _ = target.call_soon_threadsafe(_schedule)
    else:
        while True:
            target = _ensure_loop()
            with _lock:
                if _loop is target:
                    _ = target.call_soon_threadsafe(_schedule)
                    break
            # shutdown() claimed the loop between obtaining and submitting:
            # retry — _ensure_loop lazily starts a fresh persistent loop.
    try:
        return result.result(timeout)
    except BaseException:
        # If ``shutdown()`` closed the loop between the submit and here, the task
        # was already cancelled by shutdown's sweep, so suppressing the resulting
        # ``RuntimeError("Event loop is closed")`` loses nothing and stops it from
        # masking the real exception being propagated.
        with contextlib.suppress(RuntimeError):
            _ = target.call_soon_threadsafe(_cancel)
        raise


def _run_and_drain[T](coro: Coroutine[object, object, T]) -> T:
    """Run *coro* on a fresh loop in *this* (worker) thread, then drain stragglers.

    The fresh-loop path retained **only** for the loop-on-loop fallback in
    :func:`run_sync` — when the *persistent loop itself* is running in the
    calling thread, so the coroutine can be driven neither on the persistent
    loop nor in-thread. Runs on a one-shot worker thread, never the main thread,
    so it cannot be interrupted by a signal mid-call (which is why no
    KeyboardInterrupt settling is needed here). The whole fresh-loop lifecycle
    holds ``_fallback_lock``, so two fresh loops can never coexist and race
    LiteLLM's logging worker.

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

    Each drain is bounded by ``_DRAIN_TIMEOUT_SECONDS``, independent of the
    call's own wait (which may be unbounded), so a hung callback can't wedge
    the bridge; the call's own result/exception always propagates ahead of any
    straggler outcome. The drains run whether *coro* succeeded or raised.
    """
    # Function-local + module-bound; see :mod:`llmkit._litellm`.
    import llmkit._litellm as _litellm

    with _fallback_lock:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                # Drain pending LiteLLM logging whether the call succeeded or
                # raised — an undrained coroutine/task is otherwise destroyed
                # (and warned about) at ``loop.close()``.
                loop.run_until_complete(
                    _litellm.drain_async_logging(timeout=_DRAIN_TIMEOUT_SECONDS)
                )
                _drain_pending(loop, timeout=_DRAIN_TIMEOUT_SECONDS)
        finally:
            try:
                # Asyncgen finalizers may schedule one last off-loop log write
                # (``record_call_async``), so the executor drain must come after
                # them — and before ``close``, so no llmkit log write is ever
                # abandoned in a worker thread of a closed loop.
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
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


def run_sync[T](coro: Coroutine[object, object, T], *, timeout: float | None | Unset = UNSET) -> T:
    """Run an async coroutine synchronously, handling event-loop detection.

    Drives *coro* on llmkit's single **persistent** event loop (one dedicated
    daemon thread, started lazily) and blocks up to ``timeout`` seconds for the
    result. Routing every sync call through one loop is what keeps LiteLLM's
    process-global logging worker bound to a single loop — eliminating the
    cross-loop logging race a fresh-loop-per-call strategy triggers under a
    concurrent sync fan-out (flood + an occasional ~600 s hang). See the module
    docstring. A caller that races :func:`shutdown` — obtaining the loop just as
    it is torn down — transparently lands on a lazily restarted persistent loop
    rather than hanging or raising.

    If an event loop is **already running in the calling thread**, what happens
    depends on *which* loop it is:

    * **Any loop other than the persistent one** (an async framework calling a
      ``*_sync`` helper): the call is submitted to the persistent loop exactly
      like the common path. The calling thread blocks for the call's duration —
      freezing the caller's own loop meanwhile, which is inherent to calling a
      sync helper from async code under *any* strategy — but keeps every
      persistent-loop guarantee: one loop for LiteLLM's logging worker even
      under a concurrent reentrant fan-out, cancel-on-timeout, and the shared
      per-provider concurrency cap. (From async code prefer the async call
      functions — the ones *without* the ``_sync`` suffix; this path exists so
      a sync helper reached through an async host is *safe*, not to make it a
      good idea.)
    * **The persistent loop itself** (a coroutine on it calls back into sync
      code): blocking its own thread would deadlock, so the coroutine is
      offloaded to a one-shot worker thread with its own fresh loop
      (:func:`_run_and_drain`), serialized so fresh loops never coexist.

    ``timeout`` bounds the **wait**. Unset, it is **600 seconds** (10 minutes),
    not a few seconds: structured generations routinely run tens of seconds and a
    large capped prompt can run minutes, so a short default would raise
    :class:`TimeoutError` mid-flight on a slow-but-valid call. The global rate
    limiter (:class:`~llmkit.rate_limiting.GlobalRateLimiter`) is the real
    concurrency backpressure; this ceiling exists only to bound a *hung*
    coroutine. Pass ``None`` to wait unbounded. The call functions' sync forms
    pass ``None``: like their async forms, they are bounded by the per-request
    ``timeout`` and the retry budget, not by a whole-call ceiling.

    On the persistent-loop path — including reentrant calls from a foreign
    loop — a timeout (or a :class:`KeyboardInterrupt` in the calling thread)
    **cancels** the coroutine on the loop — tearing down the in-flight provider
    request — before the exception propagates, so a timed-out call does not
    leak. Only on the loop-on-loop worker-thread fallback is that cancellation
    not possible (a coroutine driven by another thread's loop can't be cancelled
    from here):

    .. warning::
       On the **loop-on-loop worker-thread fallback** (a coroutine on the
       persistent loop calling back into sync code), a
       :class:`concurrent.futures.TimeoutError` only abandons the *wait* — the
       worker thread keeps executing the coroutine (and its in-flight provider
       request) to completion. A timed-out call on this path therefore leaks one
       thread and one request until they finish on their own — and, because
       fresh-loop workers are serialized, a *subsequent* fallback call waits for
       the abandoned worker to finish before starting. This is why the default
       is generous: timing out is a last-resort signal that something is wrong,
       not a routine control-flow path.
    """
    wait = _DEFAULT_TIMEOUT_SECONDS if isinstance(timeout, Unset) else timeout
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread: the common path → the persistent loop.
        return _submit_to_loop(coro, timeout=wait)

    # A stale read of ``_loop`` (no lock) is benign here: if the persistent loop
    # was never started or was concurrently shut down, ``running`` is a foreign
    # loop and ``_submit_to_loop`` (re)starts a persistent loop as usual. Only a
    # coroutine literally executing on the persistent loop can observe identity.
    if running is not _loop:
        # Foreign-loop reentrancy (an async host calling a ``*_sync`` helper):
        # the persistent loop is free, so submit there like any other sync call.
        # Blocking this thread is what the old fresh-loop fallback did anyway —
        # and routing here keeps LiteLLM's logging worker on one loop even when
        # many reentrant calls overlap.
        return _submit_to_loop(coro, timeout=wait)

    # The persistent loop itself is running in this thread → loop-on-loop
    # fallback. Run the worker inside a *copy* of the caller's context so the
    # context-scoped state the call layer relies on crosses into the worker
    # thread (a bare ``executor.submit`` would not propagate ``contextvars``).
    ctx = contextvars.copy_context()

    def _drive() -> T:
        return _run_and_drain(coro)

    # Deliberately *not* a ``with`` block: the context manager's ``__exit__``
    # calls ``executor.shutdown(wait=True)``, which blocks until the worker
    # finishes — so on the timeout path the ``TimeoutError`` could not leave this
    # frame until the very call it was meant to abandon completed. ``wait=False``
    # lets the error reach the caller at the deadline while the abandoned worker
    # drains on its own, matching the ``.. warning::`` contract above.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(ctx.run, _drive)
    try:
        return future.result(timeout=wait)
    finally:
        executor.shutdown(wait=False)


def _reject_loop_thread(loop: asyncio.AbstractEventLoop | None) -> None:
    """Raise when the calling thread is running *loop*, llmkit's sync loop.

    A sync stream advanced from a coroutine on the persistent loop would enqueue
    each step onto the loop it is blocking and wait forever, and the
    loop-on-loop fallback cannot help: its fresh loop is closed after one call,
    finalizing the generator with it.
    """
    if loop is None:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    if running is loop:
        raise RuntimeError(
            "a sync stream cannot be created or iterated on llmkit's own sync loop "
            + "thread, where it would deadlock; iterate the async form with "
            + "`async for` instead"
        )


@final
class _StreamState[T]:
    """What the loop-side coroutines of one :class:`_SyncGenerator` share.

    Kept apart from the iterator object so that nothing the loop holds refers
    back to the iterator: a reference cycle would delay its finalizer from the
    ``break`` that drops it to the next garbage-collector pass.
    """

    __slots__ = ("agen", "finished", "step")

    def __init__(self, agen: AsyncGenerator[T]) -> None:
        self.agen: AsyncGenerator[T] = agen
        # Written on the loop thread; read by the consumer after a step's
        # result has crossed back, which orders the two.
        self.finished: bool = False
        # The most recent step's task, so a close can wait out a step that an
        # interrupted consumer left in flight.
        self.step: asyncio.Task[object] | None = None


async def _step[T](
    state: _StreamState[T], make_step: Callable[[], Awaitable[T]]
) -> tuple[bool, T | None]:
    """On the loop: run one generator step, reporting exhaustion as a flag.

    *make_step* is called here, inside the task, so the generator method is
    first *called* on the loop thread: that is what binds the generator to the
    persistent loop's finalization hooks.
    """
    state.step = asyncio.current_task()
    try:
        return True, await make_step()
    except StopAsyncIteration:
        state.finished = True
        return False, None
    except BaseException:
        # An exception out of the generator body finishes the generator.
        state.finished = True
        raise


async def _aclose[T](state: _StreamState[T]) -> None:
    """On the loop: wait out an in-flight step, then close the generator.

    ``aclose()`` on a generator that is still running raises
    :class:`RuntimeError`, and a step can still be settling its cancellation
    when the consumer that abandoned it asks for a close.
    """
    step = state.step
    if step is not None and not step.done():
        _ = await asyncio.wait([step])
    await state.agen.aclose()
    state.finished = True


@final
class _SyncGenerator[T](Generator[T, None, None]):
    """A synchronous generator that drives an async generator on the sync loop.

    The bridge behind the ``*_stream_sync`` call functions. It mirrors the
    async generator protocol method for method: ``__next__`` and ``send`` drive
    ``asend``, ``throw`` drives ``athrow``, and ``close`` drives ``aclose``.
    Each step runs as a task on llmkit's persistent loop, created inside the
    one :class:`contextvars.Context` captured when the iterator was made, so
    the stream's retry scope, queue-wait stamp and record capture see a single
    context from the first item to the last, as they do on the async form.

    Like a generator, it is not safe to advance from two threads at once; a
    second concurrent call raises :class:`ValueError`.
    """

    def __init__(
        self,
        agen: AsyncGenerator[T],
        context: contextvars.Context,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._state: _StreamState[T] = _StreamState(agen)
        self._context: contextvars.Context = context
        self._loop: asyncio.AbstractEventLoop = loop
        self._running: threading.Lock = threading.Lock()

    def _run[R](self, coro: Coroutine[object, object, R]) -> R:
        if not self._running.acquire(blocking=False):
            coro.close()
            raise ValueError("generator already executing")
        try:
            return _submit_to_loop(coro, timeout=None, context=self._context, loop=self._loop)
        finally:
            self._running.release()

    def _advance(self, make_step: Callable[[], Awaitable[T]]) -> T:
        _reject_loop_thread(self._loop)
        has_item, item = self._run(_step(self._state, make_step))
        if not has_item:
            raise StopIteration
        return cast("T", item)

    @override
    def __next__(self) -> T:
        return self.send(None)

    @override
    def send(self, value: None, /) -> T:
        state = self._state
        if state.finished:
            raise StopIteration
        agen = state.agen
        return self._advance(lambda: agen.asend(value))

    @overload
    def throw(
        self,
        typ: type[BaseException],
        val: BaseException | object = None,
        tb: TracebackType | None = None,
        /,
    ) -> T: ...
    @overload
    def throw(
        self, typ: BaseException, val: None = None, tb: TracebackType | None = None, /
    ) -> T: ...
    @override
    def throw(
        self,
        typ: type[BaseException] | BaseException,
        val: object = None,
        tb: TracebackType | None = None,
        /,
    ) -> T:
        if isinstance(typ, BaseException):
            exc = typ
        elif isinstance(val, BaseException):
            exc = val
        else:
            exc = typ() if val is None else typ(val)
        if tb is not None:
            exc = exc.with_traceback(tb)
        state = self._state
        if state.finished:
            raise exc
        agen = state.agen
        return self._advance(lambda: agen.athrow(exc))

    @override
    def close(self) -> None:
        """Close the async generator on the loop and wait until it has closed.

        This is the deterministic abandonment path: the stream's ``finally``
        blocks run, and its abandoned-stream record is written, before
        ``close`` returns. A no-op once the stream has finished, when it was
        never started, or when the sync loop has already been shut down (its
        own shutdown finalized the generator).
        """
        state = self._state
        if state.finished:
            return
        if state.step is None:
            # Never started: the body never ran, so there is nothing to close.
            state.finished = True
            return
        _reject_loop_thread(self._loop)
        with _lock:
            loop_gone = _loop is not self._loop
        if loop_gone:
            return
        self._run(_aclose(state))

    def __del__(self) -> None:
        # A dropped iterator closes its generator best-effort. Garbage
        # collection can run this on any thread, at any allocation, so it never
        # takes ``_lock`` (the thread may already hold it), never raises, and
        # never waits unbounded. An iterator still alive at interpreter exit
        # is finalized after the atexit ``shutdown()``, whose
        # ``shutdown_asyncgens`` has already closed the generator.
        with contextlib.suppress(BaseException):
            state = self._state
            if state.finished or state.step is None:
                return
            loop = self._loop
            if _loop is not loop or loop.is_closed():
                return
            context = self._context
            done = threading.Event()

            def _settled(task: asyncio.Task[None]) -> None:
                if not task.cancelled():
                    _ = task.exception()
                done.set()

            def _schedule() -> None:
                task = loop.create_task(_aclose(state), context=context)
                task.add_done_callback(_settled)

            _ = loop.call_soon_threadsafe(_schedule)
            try:
                on_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_loop = False
            if not on_loop:
                _ = done.wait(_DRAIN_TIMEOUT_SECONDS)


def iterate_sync[T](agen: AsyncGenerator[T]) -> Generator[T]:
    """Iterate an async generator synchronously, one item per ``next()``.

    The sync stream bridge, as :func:`run_sync` is the coroutine bridge: the
    ``*_stream_sync`` call functions return ``iterate_sync(<async stream>)``.
    Each item is produced by a step on llmkit's persistent loop, so the stream
    holds no item it has not been asked for and every guarantee of the async
    stream applies unchanged. The caller's :mod:`contextvars` context is
    captured **once, here**, and every step runs inside that one copy: a
    context entered after this call (a :func:`~llmkit.capture_llm_records`
    block, a run scope) does not reach the stream.

    No step has a wait ceiling of its own; the async stream's per-request
    timeout, which llmkit enforces on each wait for the next wire frame inside
    the step, bounds each item. Closing the returned generator — explicitly, by
    ``break`` out of a ``for`` loop that holds its only reference, or by
    dropping it — closes the async generator on the loop, so the stream
    records its abandonment. :func:`shutdown` cancels a step in flight, which
    the caller sees as :class:`asyncio.CancelledError`; a stream left between
    items is closed by the shutdown, and its next ``next()`` raises
    :class:`RuntimeError`.

    Raises :class:`RuntimeError` when called, or iterated, on the persistent
    loop's own thread, where it would deadlock.
    """
    _reject_loop_thread(_loop)
    context = contextvars.copy_context()
    loop = _ensure_loop()
    return _SyncGenerator(agen, context, loop)


async def _drain_and_close(timeout: float | None) -> None:
    """On the persistent loop: drain LiteLLM logging, then cancel background tasks.

    Run once by :func:`shutdown`. Flushes LiteLLM's queued success-logging so no
    coroutine is destroyed unawaited, then cancels and settles every remaining
    task on the loop (most importantly LiteLLM's intentionally-infinite
    logging-worker loop) so the subsequent ``loop.stop`` / ``close`` is clean.
    """
    # Function-local + module-bound; see :mod:`llmkit._litellm`.
    import llmkit._litellm as _litellm

    await _litellm.drain_async_logging(timeout=timeout)
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        _ = task.cancel()
    if pending:
        _ = await asyncio.gather(*pending, return_exceptions=True)
    # Join any in-flight ``to_thread`` offloads — llmkit's own log writes run
    # there (``record_call_async``) — so a clean process exit never drops the
    # final call's record. Bounded overall by the ``shutdown()`` caller's
    # timeout on this whole coroutine.
    await asyncio.get_running_loop().shutdown_default_executor()


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
