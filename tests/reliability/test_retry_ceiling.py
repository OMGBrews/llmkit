"""The same outer ceiling applies across error classes, helpers and call forms."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import openai
import pytest

from llmkit import (
    DEFAULT_RETRY_POLICY,
    NO_RETRY,
    ResultValidationError,
    RetryPolicy,
    capture_llm_records,
    text_llm_call,
    text_llm_call_sync,
)
from llmkit.retry import retry_progress_callback, with_retries, with_retries_stream
from llmkit.testing import Reply, TransportRequest, install, timeout_error


def _max_jitter(_low: float, high: float) -> float:
    return high


class _TransportGlitch(Exception):
    pass


class _InvalidReply(Exception):
    pass


_CASES = [
    pytest.param("TTT", 3, 2, 3, False, id="all-transport"),
    pytest.param("VV", 3, 2, 2, False, id="all-validation"),
    pytest.param("TTV", 3, 2, 3, False, id="mixed-ceiling"),
    pytest.param("TV", 3, 2, 3, True, id="mixed-success"),
    pytest.param("VTV", 5, 2, 3, False, id="validation-does-not-reset"),
    pytest.param("V", 1, 2, 1, False, id="one-means-one"),
    pytest.param("VV", 2, 10, 2, False, id="sublimit-cannot-raise-ceiling"),
]


@pytest.mark.parametrize("failures,total,validation,expected,success", _CASES)
@pytest.mark.parametrize("stream", [False, True], ids=["awaitable", "pre-yield-stream"])
async def test_mixed_failures_obey_one_ceiling(
    failures: str,
    total: int,
    validation: int,
    expected: int,
    success: bool,
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []
    progress: list[tuple[int, int]] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    def report(*, label: str, attempt: int, max_attempts: int, error: BaseException) -> None:
        _ = label, error
        progress.append((attempt, max_attempts))

    monkeypatch.setattr("llmkit.retry.asyncio.sleep", sleep)
    monkeypatch.setattr("llmkit.retry.random.uniform", _max_jitter)
    policy = RetryPolicy(
        max_attempts=total,
        validation_max_attempts=validation,
        retry_on=(_TransportGlitch,),
        validation_retry_on=(_InvalidReply,),
        backoff_base_seconds=0.5,
    )

    async def attempt() -> str:
        nonlocal calls
        calls += 1
        if calls <= len(failures):
            error = _TransportGlitch if failures[calls - 1] == "T" else _InvalidReply
            raise error(f"pass {calls}")
        return "ok"

    async def factory(number: int) -> AsyncGenerator[str]:
        assert number == calls + 1
        yield await attempt()

    async def invoke() -> str:
        if stream:
            return "".join(
                [
                    item
                    async for item in with_retries_stream(
                        factory, policy=policy, label="ceiling", surface="test"
                    )
                ]
            )
        return await with_retries(attempt, policy=policy)

    with retry_progress_callback(report):
        if success:
            assert await invoke() == "ok"
        else:
            error = _TransportGlitch if failures[expected - 1] == "T" else _InvalidReply
            with pytest.raises(error, match=f"pass {expected}"):
                _ = await invoke()
    assert calls == expected
    assert progress == [(number, total) for number in range(1, expected)]
    assert sleeps == [0.5 * 2 ** (number - 1) for number in range(1, expected)]


@pytest.mark.parametrize("failures,total,validation,expected,success", _CASES)
@pytest.mark.parametrize("sync", [False, True], ids=["async-call", "sync-call"])
def test_call_ceiling_preserves_hook_allowance_and_attempt_identity(
    failures: str, total: int, validation: int, expected: int, success: bool, sync: bool
) -> None:
    calls = 0
    policy = RetryPolicy(
        max_attempts=total, validation_max_attempts=validation, backoff_base_seconds=0
    )

    def fake(_request: TransportRequest) -> Reply:
        nonlocal calls
        calls += 1
        failure = failures[calls - 1] if calls <= len(failures) else ""
        if failure == "T":
            raise timeout_error()
        return Reply.text("reject" if failure == "V" else "ok")

    def accept(text: str) -> None:
        if text == "reject":
            raise ResultValidationError("rejected")

    def invoke() -> str:
        if sync:
            return text_llm_call_sync("q", feature="ceiling", retry=policy, on_result=accept)
        return asyncio.run(text_llm_call("q", feature="ceiling", retry=policy, on_result=accept))

    with install(fake), capture_llm_records() as records:
        if success:
            assert invoke() == "ok"
        else:
            error = (
                openai.APITimeoutError if failures[expected - 1] == "T" else ResultValidationError
            )
            with pytest.raises(error):
                _ = invoke()
    assert calls == expected
    assert [record.attempt for record in records] == list(range(1, expected + 1))
    assert len({record.call_id for record in records}) == 1


async def test_helper_defaults_are_the_call_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async def fail() -> str:
        nonlocal attempts
        attempts += 1
        raise timeout_error()

    monkeypatch.setattr("llmkit.retry.asyncio.sleep", sleep)
    monkeypatch.setattr("llmkit.retry.random.uniform", _max_jitter)
    with pytest.raises(openai.APITimeoutError):
        _ = await with_retries(fail)
    assert attempts == DEFAULT_RETRY_POLICY.max_attempts == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("policy", [DEFAULT_RETRY_POLICY, NO_RETRY])
async def test_helper_permanent_errors_fail_immediately(policy: RetryPolicy) -> None:
    attempts = 0

    async def fail() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("programming error")

    with pytest.raises(ValueError, match="programming error"):
        _ = await with_retries(fail, policy=policy)
    assert attempts == 1


async def test_helper_no_retry_disables_transient_retries() -> None:
    attempts = 0

    async def fail() -> str:
        nonlocal attempts
        attempts += 1
        raise timeout_error()

    with pytest.raises(openai.APITimeoutError):
        _ = await with_retries(fail, policy=NO_RETRY)
    assert attempts == 1
