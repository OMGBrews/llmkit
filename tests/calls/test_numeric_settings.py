"""Numeric boundary regressions: reject before work and preserve usable loops."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import cast

import httpx
import pytest
from instructor.core.exceptions import InstructorRetryException

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCallOptions,
    LocalYamlLogSink,
    calls,
    capture_llm_records,
    configure_llm_cache,
)
from llmkit._litellm import _deadline
from llmkit.providers import OpenAIProvider
from llmkit.sync import run_sync, shutdown
from llmkit.testing import Reply, install
from tests._support import OkSchema, ollama_call_defaults
from tests.calls.test_deadline import _run

_BAD_NUMBERS: list[object] = [True, False, float("nan"), float("inf"), -float("inf"), "1", 10**1000]
_BAD_DURATIONS = [*_BAD_NUMBERS, -1]
_BAD_COUNTS: list[object] = [
    True,
    False,
    float("nan"),
    float("inf"),
    -float("inf"),
    "1",
    0,
    -1,
    1.0,
]


@pytest.mark.parametrize("lane", ["text", "structured", "tool", "text_stream", "tool_stream"])
@pytest.mark.parametrize("field", ["timeout", "temperature", "max_tokens"])
@pytest.mark.parametrize("bundled", [False, True])
async def test_invalid_call_number_precedes_requests_and_records(
    lane: str, field: str, bundled: bool
) -> None:
    with (
        ollama_call_defaults(),
        install(lambda _: Reply.text("unexpected")) as fake,
        capture_llm_records() as records,
    ):
        kwargs = {field: float("nan")}
        options = cast("Callable[..., LLMCallOptions]", LLMCallOptions)(**kwargs)
        # Options are intentionally validated when used, not when constructed.
        with pytest.raises(ValueError, match=field):
            if bundled:
                _ = await _run(lane, options=options)
            elif field == "timeout":
                _ = await _run(lane, timeout=float("nan"))
            else:
                call = cast("Callable[..., Coroutine[object, object, str]]", calls.text_llm_call)
                _ = await call("hi", feature="numeric", **kwargs)
        assert fake.requests == []
        assert list(records) == []


@pytest.mark.parametrize(
    "field,values",
    [("timeout", _BAD_DURATIONS), ("temperature", _BAD_NUMBERS), ("max_tokens", _BAD_COUNTS)],
)
async def test_call_numeric_domains(field: str, values: list[object]) -> None:
    call = cast("Callable[..., object]", LLMCallOptions)
    with ollama_call_defaults(), install(lambda _: Reply.text("ok")) as fake:
        for value in values:
            options = cast("LLMCallOptions", call(**{field: value}))
            with pytest.raises(ValueError, match=field):
                _ = await _run("text", options=options)
        assert fake.requests == []


async def test_invalid_timeout_cannot_be_hidden_by_cache_hit() -> None:
    with ollama_call_defaults(), install(lambda _: Reply.text("ok")) as fake:
        configure_llm_cache(InMemoryLLMCache())
        assert await _run("text", timeout=None) == "ok"
        with pytest.raises(ValueError, match="timeout"):
            _ = await _run("text", timeout=float("nan"))
        assert len(fake.requests) == 1


@pytest.mark.parametrize("value", _BAD_DURATIONS)
def test_provider_deadline_rejects_unsafe_numbers(value: object) -> None:
    if isinstance(value, str):
        assert _deadline({"timeout": value}) is None
    else:
        with pytest.raises(ValueError, match="timeout"):
            _ = _deadline({"timeout": value})


def test_provider_non_number_timeout_remains_transport_owned() -> None:
    assert _deadline({"timeout": httpx.Timeout(2)}) is None
    assert _deadline({"timeout": 0}) == 0


@pytest.mark.parametrize("structured", [False, True])
async def test_provider_invalid_timeout_error_shape(
    structured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OpenAIProvider.completion_kwargs

    def kwargs(self: OpenAIProvider) -> dict[str, object]:
        return {**original(self), "timeout": float("nan")}

    monkeypatch.setattr(OpenAIProvider, "completion_kwargs", kwargs)
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    with ollama_call_defaults(), install(lambda _: Reply.text("unexpected")) as fake:
        with pytest.raises(InstructorRetryException if structured else ValueError):
            if structured:
                _ = await calls.structured_llm_call(
                    "hi", OkSchema, feature="numeric", provider=provider, retry=NO_RETRY
                )
            else:
                _ = await calls.text_llm_call(
                    "hi", feature="numeric", provider=provider, retry=NO_RETRY
                )
        assert fake.requests == []


@pytest.mark.parametrize("value", [*_BAD_DURATIONS, threading.TIMEOUT_MAX * 2])
def test_sync_rejection_closes_coroutine_and_preserves_loop(value: object) -> None:
    async def answer() -> int:
        return 42

    coro = answer()
    with pytest.raises(ValueError, match="timeout"):
        _ = run_sync(coro, timeout=cast("float", value))
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
    assert run_sync(answer(), timeout=None) == 42
    with pytest.raises(ValueError, match="timeout"):
        shutdown(timeout=cast("float", value))
    assert run_sync(answer(), timeout=None) == 42


@pytest.mark.parametrize("value", _BAD_COUNTS)
def test_cache_bound_rejected_before_state(value: object) -> None:
    cache = InMemoryLLMCache.__new__(InMemoryLLMCache)
    with pytest.raises(ValueError, match="max_entries"):
        cache.__init__(cast("int", value))
    assert vars(cache) == {}


@pytest.mark.parametrize("field", ["retention_days", "max_index_bytes"])
@pytest.mark.parametrize("value", _BAD_COUNTS)
def test_sink_bound_rejected_before_state(tmp_path: Path, field: str, value: object) -> None:
    sink = LocalYamlLogSink.__new__(LocalYamlLogSink)
    directory = tmp_path / "absent"
    initialize = cast("Callable[..., None]", sink.__init__)
    with pytest.raises(ValueError, match=field):
        initialize(directory, **{field: value})
    assert vars(sink) == {}
    assert not directory.exists()


@pytest.mark.parametrize("field", ["delay", "cost"])
@pytest.mark.parametrize("value", _BAD_DURATIONS)
def test_reply_numeric_domain(field: str, value: object) -> None:
    make = cast("Callable[..., Reply]", Reply.text)
    with pytest.raises(ValueError, match=field):
        _ = make("ok", **{field: value})


async def test_valid_numeric_boundaries() -> None:
    with ollama_call_defaults(), install(lambda _: Reply.text("ok", delay=0, cost=0)):
        for temperature in [-100.0, 0, 100.0, None]:
            assert (
                await calls.text_llm_call(
                    "hi", feature="numeric", temperature=temperature, max_tokens=1, timeout=None
                )
                == "ok"
            )
    assert Reply.text("ok", cost=None).cost is None
    assert InMemoryLLMCache(max_entries=1).max_entries == 1
    await asyncio.sleep(0)


@pytest.mark.parametrize("operation", ["run", "shutdown"])
def test_boolean_sync_wait_is_rejected(operation: str) -> None:
    """Safe negative control: old boolean waits finish in at most one second."""
    if operation == "shutdown":
        with pytest.raises(ValueError, match="timeout"):
            shutdown(timeout=True)
    else:

        async def answer() -> int:
            return 42

        coro = answer()
        try:
            with pytest.raises(ValueError, match="timeout"):
                _ = run_sync(coro, timeout=True)
        finally:
            coro.close()
