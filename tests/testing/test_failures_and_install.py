"""Failure helpers, delays, install scoping and the cache, through the public API.

The failure helpers must be treated exactly like a real provider's errors:
the retry layer retries them (honoring ``Retry-After``), adaptive concurrency
and the circuit breaker classify the 429 and 503 as throttles, and a host's
``except LLM_RECOVERABLE_ERRORS`` catches them.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import openai
import pytest

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    RetryPolicy,
    configure_llm_cache,
    configure_rate_limit,
    text_llm_call,
)
from llmkit.exceptions import (
    LLM_RECOVERABLE_ERRORS,
    LLM_TRANSPORT_ERRORS,
    CircuitOpenError,
    ServiceUnavailableError,
)
from llmkit.rate_limiting._outcomes import is_throttle_signal
from llmkit.rate_limiting._tuning import BREAKER_MIN_SAMPLES
from llmkit.testing import (
    Reply,
    TransportRequest,
    install,
    rate_limit_error,
    server_error,
    service_unavailable_error,
    timeout_error,
)


def test_helpers_are_the_transport_errors_the_library_classifies() -> None:
    throttles = [rate_limit_error(), rate_limit_error(2), service_unavailable_error()]
    neutral = [server_error(), timeout_error()]
    for error in [*throttles, *neutral]:
        assert isinstance(error, LLM_TRANSPORT_ERRORS)
        assert isinstance(error, LLM_RECOVERABLE_ERRORS)
    assert all(is_throttle_signal(error) for error in throttles)
    assert not any(is_throttle_signal(error) for error in neutral)
    assert isinstance(rate_limit_error(), openai.RateLimitError)
    assert isinstance(service_unavailable_error(), ServiceUnavailableError)


async def test_rate_limit_is_retried_after_the_requested_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def fake(request: TransportRequest) -> Reply:
        if request.attempt == 1:
            raise rate_limit_error(retry_after=7)
        return Reply.text("recovered")

    with install(fake) as installed:
        text = await text_llm_call(
            "x", feature="f", retry=RetryPolicy(max_attempts=3, backoff_base_seconds=1.0)
        )
    assert text == "recovered"
    assert [request.attempt for request in installed.requests] == [1, 2]
    # The server's wait plus at most one backoff base of jitter.
    assert len(slept) == 1
    assert 7.0 <= slept[0] <= 8.0


async def test_service_unavailable_reaches_the_host_after_the_budget() -> None:
    with (
        install(lambda _request: (_ for _ in ()).throw(service_unavailable_error())) as installed,
        pytest.raises(ServiceUnavailableError),
    ):
        _ = await text_llm_call("x", feature="f", retry=RetryPolicy(max_attempts=2))
    assert len(installed.requests) == 2


async def test_throttles_from_the_fake_trip_the_circuit_breaker() -> None:
    configure_rate_limit(breaker=True)

    def fake(_request: TransportRequest) -> Reply:
        raise rate_limit_error()

    with install(fake) as installed:
        for _ in range(BREAKER_MIN_SAMPLES):
            with pytest.raises(openai.RateLimitError):
                _ = await text_llm_call("x", feature="f", retry=NO_RETRY)
        with pytest.raises(CircuitOpenError):
            _ = await text_llm_call("x", feature="f", retry=NO_RETRY)
    # The open breaker fast-failed the last call before it reached the fake.
    assert len(installed.requests) == BREAKER_MIN_SAMPLES


async def test_delay_is_spent_inside_the_rate_limiter_slot() -> None:
    configure_rate_limit(max_concurrent=1, adaptive=False)
    with install(lambda _request: Reply.text("slow", delay=0.1)):
        started = time.monotonic()
        results = await asyncio.gather(
            text_llm_call("a", feature="f"), text_llm_call("b", feature="f")
        )
        elapsed = time.monotonic() - started
    assert results == ["slow", "slow"]
    # Serialized by the one slot: two delays, not one.
    assert elapsed >= 0.2


async def test_nested_installs_stack_and_restore() -> None:
    with install(lambda _request: Reply.text("outer")):
        assert await text_llm_call("x", feature="f") == "outer"
        with install(lambda _request: Reply.text("inner")):
            assert await text_llm_call("x", feature="f") == "inner"
        assert await text_llm_call("x", feature="f") == "outer"


async def test_without_a_fake_the_real_transport_runs() -> None:
    with install(lambda _request: Reply.text("faked")):
        pass

    async def provider_call(**_kwargs: object) -> object:
        raise ConnectionRefusedError("the real transport ran")

    # Internal seam, used here only to observe that the provider call happens.
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=provider_call),
        pytest.raises(ConnectionRefusedError, match="real transport"),
    ):
        _ = await text_llm_call("x", feature="f", retry=NO_RETRY)


async def test_a_cache_hit_never_reaches_the_fake() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.text("answer")) as installed:
        first = await text_llm_call("same prompt", feature="f")
        second = await text_llm_call("same prompt", feature="other-feature", label="l")
    assert first == second == "answer"
    assert len(installed.requests) == 1
