"""Tests for default-on transient-error retries in the call functions.

The call functions (:func:`structured_llm_call`, :func:`text_llm_call`,
:func:`text_llm_call_stream`) now retry *transient* provider errors on
their own, without the caller wrapping every call. These tests pin that
contract end-to-end through an :mod:`llmkit.testing` fake, so the real rate
limiter and parsing run around every attempt:

* a transient error is retried then succeeds, with no caller wrapping;
* a non-recoverable (programming) error is never retried;
* ``retry=NO_RETRY`` opts a call out (a single attempt);
* a custom :class:`~llmkit.RetryPolicy` budget is honoured;
* each attempt is its own logged call (one log path per attempt);
* streaming retries only a failure *before* the first chunk is yielded.

A canonical transient error is :func:`~llmkit.testing.timeout_error` (in
``LLM_RECOVERABLE_ERRORS``); a canonical non-recoverable one is
:class:`TypeError` (outside the set). A structured call's failure escapes as
instructor's ``InstructorRetryException`` with the original on ``__cause__``
(see ``llmkit._litellm._schema_repair_retrying``). The
:class:`~llmkit.RetryPolicy` value object itself, the recoverable-error set
membership, and the validation-budget split are pinned in
``test_retry_policy.py``.

The mid-stream failure test stays on the private ``astream_text`` seam: the
public fake cannot fail a stream after its first chunk.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import openai
import pytest
from instructor.core import InstructorRetryException

from llmkit import (
    NO_RETRY,
    LocalYamlLogSink,
    RetryPolicy,
    configure_llm_logging,
)
from llmkit import (
    calls as llm_calls,
)
from llmkit.capture import capture_llm_log_paths
from llmkit.testing import Fake, Reply, TransportRequest, install, timeout_error
from tests._support import OkSchema

# Backoff set to 0.0 so these tests retry without real sleeps (the default
# backoff itself is pinned in test_retry_policy.py).
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)

_OK = Reply.json(OkSchema(ok=True))


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    """Consume an async text stream into a list of chunks."""
    return [chunk async for chunk in stream]


def _times_out_then(reply: Reply, *, failures: int = 1) -> Fake:
    """A fake raising ``timeout_error()`` on the first *failures* attempts."""

    def fake(request: TransportRequest) -> Reply:
        if request.attempt is None or request.attempt <= failures:
            raise timeout_error()
        return reply

    return fake


def _always_raises(error: Exception) -> Fake:
    """A fake raising *error* on every round trip."""

    def fake(_request: TransportRequest) -> Reply:
        raise error

    return fake


def _attempts(requests: list[TransportRequest]) -> list[int | None]:
    return [request.attempt for request in requests]


# --- structured_llm_call -------------------------------------------------


async def test_structured_transient_error_is_retried_then_succeeds() -> None:
    """A transient timeout raised once then success is retried and succeeds
    with no caller wrapping; the provider is asked twice."""
    with install(_times_out_then(_OK)) as installed:
        result = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert _attempts(installed.requests) == [1, 2]


async def test_structured_non_recoverable_error_is_not_retried() -> None:
    """A non-recoverable ``TypeError`` propagates after a single attempt."""
    with (
        install(_always_raises(TypeError("programming error"))) as installed,
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=_NO_BACKOFF)

    assert isinstance(excinfo.value.__cause__, TypeError)
    assert str(excinfo.value.__cause__) == "programming error"
    assert _attempts(installed.requests) == [1]


async def test_structured_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` makes a transient error propagate after one attempt."""
    with (
        install(_always_raises(timeout_error())) as installed,
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=NO_RETRY)

    assert isinstance(excinfo.value.__cause__, openai.APITimeoutError)
    assert _attempts(installed.requests) == [1]


async def test_structured_custom_policy_succeeds_on_last_attempt() -> None:
    """A custom ``RetryPolicy(max_attempts=N)`` retries N-1 transient
    failures before a success on the final attempt."""
    policy = RetryPolicy(max_attempts=4, backoff_base_seconds=0.0)
    with install(_times_out_then(_OK, failures=3)) as installed:
        result = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=policy)

    assert result.ok is True
    assert _attempts(installed.requests) == [1, 2, 3, 4]


async def test_structured_custom_policy_exhaustion_re_raises() -> None:
    """A custom budget that never succeeds exhausts and re-raises the last
    transient error, having attempted exactly ``max_attempts`` times."""
    policy = RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)
    with (
        install(_always_raises(timeout_error())) as installed,
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=policy)

    assert isinstance(excinfo.value.__cause__, openai.APITimeoutError)
    assert _attempts(installed.requests) == [1, 2, 3]


async def test_each_attempt_is_its_own_logged_call(tmp_path: Path) -> None:
    """A transient-then-success run writes one log per attempt: under
    ``capture_llm_log_paths`` the two attempts yield two captured paths."""
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with install(_times_out_then(_OK)), capture_llm_log_paths() as paths:
            result = await llm_calls.structured_llm_call(
                "hi", OkSchema, feature="test", retry=_NO_BACKOFF
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert result.ok is True
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


# --- text_llm_call -------------------------------------------------------


async def test_text_transient_error_is_retried_then_succeeds() -> None:
    """``text_llm_call`` retries a transient error then returns the text."""
    with install(_times_out_then(Reply.text("hello"))) as installed:
        result = await llm_calls.text_llm_call("hi", feature="test", retry=_NO_BACKOFF)

    assert result == "hello"
    assert _attempts(installed.requests) == [1, 2]


async def test_text_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` on ``text_llm_call`` runs a single attempt."""
    with (
        install(_always_raises(timeout_error())) as installed,
        pytest.raises(openai.APITimeoutError),
    ):
        _ = await llm_calls.text_llm_call("hi", feature="test", retry=NO_RETRY)

    assert _attempts(installed.requests) == [1]


# --- text_llm_call_stream ------------------------------------------------


async def test_stream_failure_before_first_chunk_is_retried() -> None:
    """A transient failure *before* the first chunk is retried; the second
    attempt yields the full output and the provider is asked twice."""
    reply = Reply(chunks=["he", "llo"])
    with install(_times_out_then(reply)) as installed:
        chunks = await _drain(
            llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF)
        )

    assert chunks == ["he", "llo"]
    assert _attempts(installed.requests) == [1, 2]


async def test_stream_each_attempt_is_its_own_logged_call(tmp_path: Path) -> None:
    """A retried stream writes one log per attempt: the pre-first-chunk
    failure and the successful retry each produce a captured path, so
    ``capture_llm_log_paths`` sees one record per attempt (parity with the
    structured path)."""
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            install(_times_out_then(Reply(chunks=["he", "llo"]))),
            capture_llm_log_paths() as paths,
        ):
            chunks = await _drain(
                llm_calls.text_llm_call_stream("hi", feature="test", retry=_NO_BACKOFF)
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert chunks == ["he", "llo"]
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


@pytest.mark.asyncio
async def test_stream_failure_after_first_chunk_is_not_retried() -> None:
    """A mid-stream failure (after a chunk reached the caller) is NOT
    retried: the already-yielded chunk is delivered, then the error
    propagates, and the transport is invoked exactly once."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        yield "he"
        raise TimeoutError("mid-stream")

    delivered: list[str] = []
    configure_llm_logging(None)
    try:
        with (
            patch("llmkit._litellm.astream_text", _transport),
            pytest.raises(TimeoutError, match="mid-stream"),
        ):
            async for chunk in llm_calls.text_llm_call_stream(
                "hi", feature="test", retry=_NO_BACKOFF
            ):
                delivered.append(chunk)
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert delivered == ["he"]
    assert calls[0] == 1


async def test_stream_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` for streaming runs a single attempt even when the
    failure occurs before the first chunk."""
    with (
        install(_always_raises(timeout_error())) as installed,
        pytest.raises(openai.APITimeoutError),
    ):
        _ = await _drain(llm_calls.text_llm_call_stream("hi", feature="test", retry=NO_RETRY))

    assert _attempts(installed.requests) == [1]


async def test_custom_max_backoff_propagates_through_call_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom ``RetryPolicy.max_backoff_seconds`` reaches the backoff sleeps
    of a call function, not just direct ``with_retries`` users. Jitter is
    pinned to its max so the asserted values are the exact ceilings: base 2.0
    would permit 2.0 then 4.0; the 1.5s cap bounds both."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    def _max_jitter(_lo: float, hi: float) -> float:
        return hi

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", _max_jitter)

    policy = RetryPolicy(max_attempts=3, backoff_base_seconds=2.0, max_backoff_seconds=1.5)
    with (
        install(_always_raises(timeout_error())) as installed,
        pytest.raises(InstructorRetryException) as excinfo,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=policy)

    assert isinstance(excinfo.value.__cause__, openai.APITimeoutError)
    assert _attempts(installed.requests) == [1, 2, 3]
    assert slept == [1.5, 1.5]
