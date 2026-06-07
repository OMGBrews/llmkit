"""Offline tests for the global, per-provider concurrency limiter.

These tests never touch the network and never read a provider credential.
They pin the behaviour the rate limiter promises:

* limiting is **on by default** with a per-provider cap of 2 (zero config),
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
from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

import pytest

from llmkit import _litellm
from llmkit.rate_limiting import GlobalRateLimiter, configure_rate_limit


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Iterator[None]:
    """Reset the process-global limiter to its shipped default around each test.

    The default state is on-by-default with a per-provider cap of 2.
    ``configure`` also clears the semaphore registries, so no semaphore
    constructed in one test can leak into the next.
    """
    GlobalRateLimiter.configure(max_concurrent=2, enabled=True)
    try:
        yield
    finally:
        GlobalRateLimiter.configure(max_concurrent=2, enabled=True)


class _ConcurrencyProbe:
    """Instrumented coroutine that records observed peak concurrency per key."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def run(self, key: str, hold: asyncio.Event) -> None:
        """Acquire a slot for ``key``, record peak concurrency, hold until set."""
        async with GlobalRateLimiter.acquire_async(key):
            async with self._lock:
                self.current += 1
                self.peak = max(self.peak, self.current)
            try:
                await hold.wait()
            finally:
                async with self._lock:
                    self.current -= 1


async def test_default_on_caps_at_two_with_zero_config() -> None:
    """With no configuration at all, one key caps concurrency at 2."""
    probe = _ConcurrencyProbe()
    hold = asyncio.Event()

    tasks = [asyncio.create_task(probe.run("openai", hold)) for _ in range(5)]

    # Let the scheduler run; exactly 2 should get in, the rest block.
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 2
    assert probe.peak == 2

    hold.set()
    await asyncio.gather(*tasks)
    assert probe.peak == 2


async def test_providers_have_independent_budgets() -> None:
    """Saturating one key does not block a different key; each caps at 2."""
    probe = _ConcurrencyProbe()
    openai_hold = asyncio.Event()
    ollama_hold = asyncio.Event()

    # Saturate openai with 2 holders + 1 waiter.
    openai_tasks = [asyncio.create_task(probe.run("openai", openai_hold)) for _ in range(3)]
    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.current == 2  # openai is full

    # An ollama acquire must proceed immediately despite openai being full.
    ollama_proceeded = asyncio.Event()

    async def ollama_call() -> None:
        async with GlobalRateLimiter.acquire_async("ollama"):
            ollama_proceeded.set()
            await ollama_hold.wait()

    ollama_task = asyncio.create_task(ollama_call())
    for _ in range(20):
        await asyncio.sleep(0)
    assert ollama_proceeded.is_set(), "ollama was blocked by openai's full budget"

    # Independent caps: openai still holds exactly 2 (its third is queued).
    assert probe.current == 2

    openai_hold.set()
    ollama_hold.set()
    await asyncio.gather(*openai_tasks, ollama_task)


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
    await asyncio.gather(*tasks)


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
    await asyncio.gather(*tasks)


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
            await release.wait()
        finished.set()

    task = asyncio.create_task(holder())
    await entered.wait()

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
    async def _record(provider_key: str) -> AsyncIterator[None]:
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
