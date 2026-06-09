"""Tests for default-on transient-error retries in the call functions.

The call functions (:func:`structured_llm_call`, :func:`text_llm_call`,
:func:`stream_text_with_log`) now retry *transient* provider errors on
their own, without the caller wrapping every call. These tests pin that
contract end-to-end over the patched transport seam:

* a transient error is retried then succeeds, with no caller wrapping;
* a non-recoverable (programming) error is never retried;
* ``retry=NO_RETRY`` opts a call out (a single attempt);
* a custom :class:`~llmkit.RetryPolicy` budget is honoured;
* each attempt is its own logged call (one log path per attempt);
* streaming retries only a failure *before* the first chunk is yielded.

A canonical transient error is :class:`TimeoutError` (in
``LLM_RECOVERABLE_ERRORS``); a canonical non-recoverable one is
:class:`TypeError` (outside the set). The :class:`~llmkit.RetryPolicy` value
object itself, the recoverable-error set membership, and the validation-budget
split are pinned in ``test_retry_policy.py``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import pytest

from llmkit import (
    NO_RETRY,
    LocalYamlLogSink,
    RetryPolicy,
    configure_llm_logging,
    structured_output,
)
from llmkit.capture import capture_llm_log_paths
from tests._support import OkSchema

# Backoff set to 0.0 so these tests retry without real sleeps (the default
# backoff itself is pinned in test_retry_policy.py).
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    """Consume an async text stream into a list of chunks."""
    return [chunk async for chunk in stream]


# --- structured_llm_call -------------------------------------------------


@pytest.mark.asyncio
async def test_structured_transient_error_is_retried_then_succeeds() -> None:
    """A transient ``TimeoutError`` raised once then success is retried and
    succeeds with no caller wrapping; the transport is invoked twice."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_structured_non_recoverable_error_is_not_retried() -> None:
    """A non-recoverable ``TypeError`` propagates after a single attempt."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TypeError("programming error")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TypeError, match="programming error"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_structured_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` makes a transient error propagate after one attempt."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("transient")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TimeoutError, match="transient"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=NO_RETRY
        )

    assert calls[0] == 1


@pytest.mark.asyncio
async def test_structured_custom_policy_succeeds_on_last_attempt() -> None:
    """A custom ``RetryPolicy(max_attempts=N)`` retries N-1 transient
    failures before a success on the final attempt."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] < 4:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    policy = RetryPolicy(max_attempts=4, backoff_base_seconds=0.0)
    with patch("llmkit._litellm.acompletion_structured", side_effect=_transport):
        result = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=policy
        )

    assert result.ok is True
    assert calls[0] == 4


@pytest.mark.asyncio
async def test_structured_custom_policy_exhaustion_re_raises() -> None:
    """A custom budget that never succeeds exhausts and re-raises the last
    transient error, having attempted exactly ``max_attempts`` times."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("always")

    policy = RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=policy
        )

    assert calls[0] == 3


@pytest.mark.asyncio
async def test_each_attempt_is_its_own_logged_call(tmp_path: Path) -> None:
    """A transient-then-success run writes one log per attempt: under
    ``capture_llm_log_paths`` the two attempts yield two captured paths."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
            capture_llm_log_paths() as paths,
        ):
            result = await structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", retry=_NO_BACKOFF
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert result.ok is True
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


# --- text_llm_call -------------------------------------------------------


@pytest.mark.asyncio
async def test_text_transient_error_is_retried_then_succeeds() -> None:
    """``text_llm_call`` retries a transient error then returns the text."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
        return "hello", None

    configure_llm_logging(None)
    try:
        with patch("llmkit._litellm.acompletion_text", side_effect=_transport):
            result = await structured_output.text_llm_call("hi", feature="test", retry=_NO_BACKOFF)
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert result == "hello"
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_text_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` on ``text_llm_call`` runs a single attempt."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        calls[0] += 1
        raise TimeoutError("transient")

    configure_llm_logging(None)
    try:
        with (
            patch("llmkit._litellm.acompletion_text", side_effect=_transport),
            pytest.raises(TimeoutError, match="transient"),
        ):
            _ = await structured_output.text_llm_call("hi", feature="test", retry=NO_RETRY)
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert calls[0] == 1


# --- stream_text_with_log ------------------------------------------------


@pytest.mark.asyncio
async def test_stream_failure_before_first_chunk_is_retried() -> None:
    """A transient failure *before* the first chunk is retried; the second
    attempt yields the full output and the transport runs twice."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
            # Unreachable by design: the bare ``yield`` makes this an async
            # generator; the ``raise`` above precedes it on the first attempt.
            yield  # pyright: ignore[reportUnreachable]  # pragma: no cover
        for delta in ("he", "llo"):
            yield delta

    configure_llm_logging(None)
    try:
        with patch("llmkit._litellm.astream_text", _transport):
            chunks = await _drain(
                structured_output.stream_text_with_log("hi", feature="test", retry=_NO_BACKOFF)
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert chunks == ["he", "llo"]
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_stream_each_attempt_is_its_own_logged_call(tmp_path: Path) -> None:
    """A retried stream writes one log per attempt: the pre-first-chunk
    failure and the successful retry each produce a captured path, so
    ``capture_llm_log_paths`` sees one record per attempt (parity with the
    structured path)."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        if calls[0] == 1:
            raise TimeoutError("transient")
            # Unreachable by design: the bare ``yield`` makes this an async
            # generator; the ``raise`` above precedes it on the first attempt.
            yield  # pyright: ignore[reportUnreachable]  # pragma: no cover
        for delta in ("he", "llo"):
            yield delta

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        with (
            patch("llmkit._litellm.astream_text", _transport),
            capture_llm_log_paths() as paths,
        ):
            chunks = await _drain(
                structured_output.stream_text_with_log("hi", feature="test", retry=_NO_BACKOFF)
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
            async for chunk in structured_output.stream_text_with_log(
                "hi", feature="test", retry=_NO_BACKOFF
            ):
                delivered.append(chunk)
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert delivered == ["he"]
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_stream_no_retry_opt_out_runs_single_attempt() -> None:
    """``retry=NO_RETRY`` for streaming runs a single attempt even when the
    failure occurs before the first chunk."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        raise TimeoutError("transient")
        # Unreachable by design: the bare ``yield`` makes this an async
        # generator; the ``raise`` above precedes it on every attempt.
        yield  # pyright: ignore[reportUnreachable]  # pragma: no cover

    configure_llm_logging(None)
    try:
        with (
            patch("llmkit._litellm.astream_text", _transport),
            pytest.raises(TimeoutError, match="transient"),
        ):
            _ = await _drain(
                structured_output.stream_text_with_log("hi", feature="test", retry=NO_RETRY)
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert calls[0] == 1


@pytest.mark.asyncio
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

    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        raise TimeoutError("always")

    policy = RetryPolicy(max_attempts=3, backoff_base_seconds=2.0, max_backoff_seconds=1.5)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        pytest.raises(TimeoutError, match="always"),
    ):
        _ = await structured_output.structured_llm_call(
            "hi", OkSchema, feature="test", retry=policy
        )

    assert calls[0] == 3
    assert slept == [1.5, 1.5]
