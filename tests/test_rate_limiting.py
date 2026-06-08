"""Offline tests for the global, per-provider concurrency limiter.

These tests never touch the network and never read a provider credential.
They pin the behaviour the rate limiter promises:

* limiting is **on by default** with a per-provider cap of 8 (zero config),
* each provider gets an **independent** budget (keyed by provider name),
* reconfiguring the cap / disabling takes effect for subsequent acquires,
* a :meth:`configure` swap never strands in-flight callers on the old
  semaphore (they release back onto their snapshot), and
* the call layer accounts a slot under the *effective* provider's name —
  the same value logging records.

The limiter is a process-global; the ``reset_rate_limiter`` autouse fixture
restores a known default state around every test so nothing leaks.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncGenerator, Iterator
from unittest.mock import MagicMock, patch

import pytest

from llmkit import _litellm
from llmkit.rate_limiting import (
    GlobalRateLimiter,
    configure_rate_limit,
    get_rate_limit_config,
    rate_limit_acquire_async,
    rate_limit_acquire_sync,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    """Reset the process-global limiter to its shipped default around each test.

    The default state is on-by-default with a per-provider cap of 8.
    ``configure`` also clears the semaphore registries, so no semaphore
    constructed in one test can leak into the next.
    """
    GlobalRateLimiter.configure(max_concurrent=8, enabled=True)
    try:
        yield
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)


class _ConcurrencyProbe:
    """Instrumented coroutine that records observed peak concurrency per key."""

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def run(self, key: str, hold: asyncio.Event) -> None:
        """Acquire a slot for ``key``, record peak concurrency, hold until set."""
        async with GlobalRateLimiter.acquire_async(key):
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                _ = await hold.wait()
            finally:
                async with self._lock:
                    self.current -= 1


async def test_default_on_caps_at_eight_with_zero_config() -> None:
    """With no configuration at all, one key caps concurrency at 8."""
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(11)]

    # Let the scheduler run; exactly 8 should get in, the rest block.
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8
    assert probe.peak == 8

    hold.set()
    _ = await asyncio.gather(*tasks)
    assert probe.peak == 8


async def test_providers_have_independent_budgets() -> None:
    """Saturating one key does not block a different key; each caps at 8."""
    probe = _ConcurrencyProbe()
    openai_hold = asyncio.Event()
    ollama_hold = asyncio.Event()

    # Saturate openai with 8 holders + 1 waiter.
    openai_tasks = [asyncio.create_task(probe.run("openai", openai_hold)) for _ in range(9)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8  # openai is full

    # An ollama acquire must proceed immediately despite openai being full.
    ollama_proceeded = asyncio.Event()

    async def ollama_call() -> None:
        async with GlobalRateLimiter.acquire_async("ollama"):
            ollama_proceeded.set()
            _ = await ollama_hold.wait()

    ollama_task = asyncio.create_task(ollama_call())
    for _ in range(20):
        await asyncio.sleep(0)
    assert ollama_proceeded.is_set(), "ollama was blocked by openai's full budget"

    # Independent caps: openai still holds exactly 8 (its ninth is queued).
    assert probe.current == 8

    openai_hold.set()
    ollama_hold.set()
    _ = await asyncio.gather(*openai_tasks, ollama_task)


async def test_reconfigure_raises_cap_for_new_acquires() -> None:
    """configure(max_concurrent=5) lets 5 run concurrently on one key."""
    configure_rate_limit(max_concurrent=5)
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(5)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 5
    assert probe.peak == 5

    hold.set()
    _ = await asyncio.gather(*tasks)


async def test_disabled_is_unbounded_no_op() -> None:
    """configure(enabled=False) makes acquire a no-op: no cap at all."""
    configure_rate_limit(enabled=False)
    assert GlobalRateLimiter.is_enabled() is False
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(7)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 7  # all in flight, unbounded
    assert probe.peak == 7

    hold.set()
    _ = await asyncio.gather(*tasks)


async def test_inflight_caller_not_stranded_by_configure_swap() -> None:
    """A held slot releases cleanly even after configure() clears the registry.

    The acquirer snapshots its semaphore at acquire time, so the swap that
    configure() performs (clearing the dict) cannot strand it; the block
    exits without deadlock and a fresh acquire works afterward.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def holder() -> None:
        async with GlobalRateLimiter.acquire_async("openai"):
            entered.set()
            _ = await release.wait()
        finished.set()

    task = asyncio.create_task(holder())
    _ = await entered.wait()

    # Swap/clear the registry while the slot is held.
    configure_rate_limit(max_concurrent=3)

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert finished.is_set()

    # A fresh acquire after the swap still works (no deadlock, new cap in play).
    async with GlobalRateLimiter.acquire_async("openai"):
        pass


async def test_call_layer_accounts_under_effective_provider_name() -> None:
    """acompletion_text acquires under the running provider's ``.name``.

    Driven fully offline: the rate-limit acquire is monkeypatched to record
    the key, and ``litellm.acompletion`` is faked so no network call happens.
    A per-call ``provider=`` override must drive the recorded key.
    """
    recorded: list[str] = []

    @contextlib.asynccontextmanager
    async def _record(provider_key: str) -> AsyncGenerator[None]:
        recorded.append(provider_key)
        yield

    provider = MagicMock()
    provider.name = "ollama"
    provider.completion_kwargs.return_value = {"api_key": "k", "api_base": None}
    provider.litellm_model.return_value = "ollama/fake"
    provider.reasoning_effort = None

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi"))],
            _hidden_params={},
        )

    with (
        patch.object(GlobalRateLimiter, "acquire_async", _record),
        patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion),
    ):
        text, _cost = await _litellm.acompletion_text(
            "hi",
            temperature=0.0,
            model=None,
            provider=provider,
        )

    assert text == "hi"
    assert recorded == ["ollama"]


def test_get_rate_limit_config_reports_effective_values() -> None:
    """``get_rate_limit_config`` reflects the default and any reconfiguration,
    so a host can read its effective limits without touching ``_max_concurrent``."""
    default = get_rate_limit_config()
    assert default.enabled is True
    assert default.max_concurrent == 8

    configure_rate_limit(max_concurrent=3, enabled=False)
    updated = get_rate_limit_config()
    assert updated.enabled is False
    assert updated.max_concurrent == 3


class _AsyncFunctionProbe:
    """``_ConcurrencyProbe`` analogue that joins via ``rate_limit_acquire_async``.

    Exercises the module-level public function rather than
    ``GlobalRateLimiter.acquire_async`` directly, so the test pins the function
    path a host actually uses to join the global limit by hand.
    """

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def run(self, key: str, hold: asyncio.Event) -> None:
        """Join via the public async function, record peak, hold until set."""
        async with rate_limit_acquire_async(key):
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                _ = await hold.wait()
            finally:
                async with self._lock:
                    self.current -= 1


async def test_rate_limit_acquire_async_respects_configured_cap() -> None:
    """The public async function bounds concurrency at the configured cap."""
    probe = _AsyncFunctionProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(11)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 8  # default per-provider cap
    assert probe.peak == 8

    hold.set()
    _ = await asyncio.gather(*tasks)
    assert probe.peak == 8


async def test_rate_limit_acquire_async_acquires_and_releases() -> None:
    """A held slot releases on block exit, so the next acquire proceeds.

    Saturate the cap by hand through the public function; a further acquire
    blocks until one holder exits, then runs.
    """
    configure_rate_limit(max_concurrent=1)
    entered_second = asyncio.Event()
    release_first = asyncio.Event()

    async def first() -> None:
        async with rate_limit_acquire_async("openai"):
            _ = await release_first.wait()

    async def second() -> None:
        async with rate_limit_acquire_async("openai"):
            entered_second.set()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())

    for _ in range(20):
        await asyncio.sleep(0)
    # The single slot is held by first(); second() must be blocked.
    assert not entered_second.is_set()

    release_first.set()  # free the slot
    _ = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1.0)
    assert entered_second.is_set()


async def test_rate_limit_acquire_async_disabled_bypass() -> None:
    """With limiting disabled, the public async function is an unbounded no-op."""
    configure_rate_limit(enabled=False)
    probe = _AsyncFunctionProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(7)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 7  # all in flight, unbounded
    assert probe.peak == 7

    hold.set()
    _ = await asyncio.gather(*tasks)


def test_rate_limit_acquire_sync_respects_configured_cap() -> None:
    """The public sync function bounds concurrency at the configured cap.

    Saturate a cap of 1 from one thread, then assert a second thread cannot
    enter until the first releases — proving the sync function holds the slot.
    """
    configure_rate_limit(max_concurrent=1)
    entered_second = threading.Event()
    release_first = threading.Event()
    first_holds = threading.Event()

    def first() -> None:
        with rate_limit_acquire_sync("openai"):
            first_holds.set()
            _ = release_first.wait(timeout=1.0)

    def second() -> None:
        with rate_limit_acquire_sync("openai"):
            entered_second.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_holds.wait(timeout=1.0)

    second_thread.start()
    # The single slot is held; second() must block.
    assert not entered_second.wait(timeout=0.1)

    release_first.set()  # free the slot
    second_thread.join(timeout=1.0)
    first_thread.join(timeout=1.0)
    assert entered_second.is_set()


def test_rate_limit_acquire_sync_disabled_bypass() -> None:
    """With limiting disabled, the public sync function is an unbounded no-op.

    A cap of 1 would otherwise serialise; disabled, two threads hold the
    "slot" simultaneously.
    """
    configure_rate_limit(max_concurrent=1, enabled=False)
    # If limiting were active (cap 1) the barrier would never reach 2 and time
    # out; disabled, both threads sit inside the acquire at once and trip it.
    both_in = threading.Barrier(2, timeout=1.0)
    release = threading.Event()
    reached_barrier = threading.Event()

    def worker() -> None:
        with rate_limit_acquire_sync("openai"):
            _ = both_in.wait()  # both threads must be inside at once
            reached_barrier.set()
            _ = release.wait(timeout=1.0)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert reached_barrier.wait(timeout=1.0), "disabled sync acquire serialised callers"

    release.set()
    for thread in threads:
        thread.join(timeout=1.0)
