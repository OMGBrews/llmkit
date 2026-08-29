"""Tests for the correlation + timing fields on :class:`LLMCallRecord`.

Pins the July-2026 audit fixes for log-record fidelity:

* **Correlation** — every logical call mints one ``call_id`` (uuid4 hex) and
  numbers its attempts 1-based, so the N records a retried call produces
  join on ``call_id`` instead of feature + timestamp proximity (which breaks
  under concurrent same-feature fan-out). All three call surfaces
  (structured, text, stream) and the sync bridge carry them.
* **Queue wait** — ``queue_wait_ms`` records the time an attempt spent
  queued behind llmkit's own rate limiter, reset per attempt (never a stale
  carry-over), ``0.0`` when the limiter is disabled, and ``None`` when the
  attempt failed before completing an acquire. ``duration_ms`` keeps its
  meaning; provider latency ≈ ``duration_ms - queue_wait_ms``.
* **Rendering** — the three fields land in the per-call YAML and
  ``index.jsonl``, and header line 2 gains a ``call=<id[:8]> attempt=<n>``
  suffix while line 1 (the pinned ``head -1`` triage shape) is untouched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import yaml

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    RetryPolicy,
    capture_llm_records,
)
from llmkit import (
    calls as llm_calls,
)
from llmkit.rate_limiting import GlobalRateLimiter
from llmkit.rate_limiting._observability import _queue_wait_ms
from tests._support import OkSchema, provider_mock
from tests._support import make_record as _record

_NO_BACKOFF = RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)


# 1. Correlation across retries.


@pytest.mark.asyncio
async def test_retried_structured_call_shares_call_id_and_numbers_attempts() -> None:
    """Three attempts of one logical call → three records joined by one
    ``call_id``, with attempts numbered 1..3."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] < 3:
            raise TimeoutError("transient")
        return OkSchema(ok=True), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        result = await llm_calls.structured_llm_call(
            "hi", OkSchema, feature="test", retry=_NO_BACKOFF
        )

    assert result.ok is True
    assert [r.attempt for r in records] == [1, 2, 3]
    call_ids = {r.call_id for r in records}
    assert len(call_ids) == 1
    (call_id,) = call_ids
    assert call_id is not None and len(call_id) == 32


@pytest.mark.asyncio
async def test_separate_logical_calls_get_distinct_call_ids() -> None:
    """Two calls — even identical ones — never share a ``call_id``."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")

    assert len(records) == 2
    assert records[0].call_id != records[1].call_id
    assert [r.attempt for r in records] == [1, 1]


@pytest.mark.asyncio
async def test_text_call_records_correlation() -> None:
    """The buffered text surface carries the same correlation fields."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "hello", None

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await llm_calls.text_llm_call("hi", feature="test")

    assert len(records) == 1
    assert records[0].call_id is not None
    assert records[0].attempt == 1


@pytest.mark.asyncio
async def test_stream_retry_shares_call_id_across_attempts() -> None:
    """A pre-first-chunk transient failure retries under the same ``call_id``;
    each attempt is its own record (one-attempt-one-log), numbered in order."""
    from collections.abc import AsyncIterator

    calls = [0]

    def _fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        calls[0] += 1
        first = calls[0] == 1

        async def _gen() -> AsyncIterator[str]:
            if first:
                raise TimeoutError("pre-first-chunk blip")
            yield "ok"

        return _gen()

    with (
        patch("llmkit._litellm.astream_text", side_effect=_fake_stream),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        chunks = [
            chunk
            async for chunk in llm_calls.text_llm_call_stream(
                "hi", feature="test", retry=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0)
            )
        ]

    assert chunks == ["ok"]
    assert [r.attempt for r in records] == [1, 2]
    assert records[0].call_id == records[1].call_id is not None
    assert records[0].error is not None and "pre-first-chunk" in records[0].error
    assert records[1].error is None


def test_sync_bridge_carries_correlation() -> None:
    """``structured_llm_call_sync`` records ``call_id``/``attempt`` exactly
    like the async path — the fields cross the ``run_sync`` bridge."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        result = llm_calls.structured_llm_call_sync("hi", OkSchema, feature="test")

    assert result.ok is True
    assert len(records) == 1
    assert records[0].call_id is not None
    assert records[0].attempt == 1


# 2. Queue wait.


@pytest.mark.asyncio
async def test_queue_wait_recorded_when_transport_acquires() -> None:
    """An attempt whose transport passed through the limiter records a
    non-``None`` ``queue_wait_ms`` (≈0 when uncontended)."""

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        async with GlobalRateLimiter.acquire_async("Google AI Studio"):
            return OkSchema(ok=True), None

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test")

    assert len(records) == 1
    assert records[0].queue_wait_ms is not None
    assert records[0].queue_wait_ms >= 0.0


@pytest.mark.asyncio
async def test_queue_wait_is_reset_between_attempts_never_stale() -> None:
    """An attempt that fails *before* completing an acquire records ``None`` —
    never the previous attempt's stale stamp."""
    calls = [0]

    async def _transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        calls[0] += 1
        if calls[0] == 1:
            async with GlobalRateLimiter.acquire_async("Google AI Studio"):
                raise TimeoutError("transient, after acquiring")
        raise RuntimeError("failed before any acquire")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        capture_llm_records() as records,
        pytest.raises(RuntimeError, match="before any acquire"),
    ):
        _ = await llm_calls.structured_llm_call("hi", OkSchema, feature="test", retry=_NO_BACKOFF)

    assert len(records) == 2
    assert records[0].queue_wait_ms is not None
    assert records[1].queue_wait_ms is None


@pytest.mark.asyncio
async def test_queue_wait_measures_contended_wait() -> None:
    """With the concurrency gate saturated, the queued caller's stamp reflects
    the real wait behind the limiter."""
    GlobalRateLimiter.configure(max_concurrent=1, enabled=True)
    try:
        release = asyncio.Event()
        holding = asyncio.Event()

        async def _holder() -> None:
            async with GlobalRateLimiter.acquire_async("qw-test"):
                holding.set()
                _ = await release.wait()

        async def _waiter() -> float | None:
            async with GlobalRateLimiter.acquire_async("qw-test"):
                pass
            return _queue_wait_ms.get()

        holder = asyncio.create_task(_holder())
        _ = await holding.wait()
        waiter = asyncio.create_task(_waiter())
        await asyncio.sleep(0.05)
        release.set()
        wait_ms = await waiter
        await holder
        assert wait_ms is not None
        assert wait_ms >= 25.0
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)


@pytest.mark.asyncio
async def test_queue_wait_is_zero_when_limiter_disabled() -> None:
    """A disabled limiter stamps ``0.0`` — no queue exists, which is different
    from "never acquired" (``None``)."""
    GlobalRateLimiter.configure(max_concurrent=8, enabled=False)
    try:
        async with GlobalRateLimiter.acquire_async("qw-test"):
            pass
        assert _queue_wait_ms.get() == 0.0
    finally:
        GlobalRateLimiter.configure(max_concurrent=8, enabled=True)


# 3. Rendering in the sink.


def test_yaml_and_header_carry_correlation_fields(tmp_path: Path) -> None:
    """The three fields land in the YAML body, and header line 2 gains the
    ``call=<id[:8]> attempt=<n>`` suffix — line 1's pinned shape unchanged."""
    record = _record(call_id="a1b2c3d4" + "0" * 24, attempt=2, queue_wait_ms=12.34)
    path = LocalYamlLogSink(tmp_path).write_returning_path(record)
    assert path is not None
    text = path.read_text()
    comment_lines = [ln for ln in text.splitlines() if ln.startswith("# ")]
    assert len(comment_lines) == 2
    assert comment_lines[0].startswith("# ok | extraction/summary |")
    assert "call=" not in comment_lines[0]
    assert "call=a1b2c3d4 attempt=2" in comment_lines[1]
    doc = cast("dict[str, object]", yaml.safe_load(text))
    assert doc["call_id"] == "a1b2c3d4" + "0" * 24
    assert doc["attempt"] == 2
    assert doc["queue_wait_ms"] == 12.3


def test_default_record_renders_without_correlation_suffix(tmp_path: Path) -> None:
    """A directly-constructed record (all three fields ``None``) keeps the old
    two-line header verbatim and serializes explicit nulls in the body."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    text = path.read_text()
    comment_lines = [ln for ln in text.splitlines() if ln.startswith("# ")]
    assert len(comment_lines) == 2
    assert "call=" not in text.splitlines()[1]
    doc = cast("dict[str, object]", yaml.safe_load(text))
    assert "call_id" in doc and doc["call_id"] is None
    assert "attempt" in doc and doc["attempt"] is None
    assert "queue_wait_ms" in doc and doc["queue_wait_ms"] is None


def test_index_line_carries_correlation_fields(tmp_path: Path) -> None:
    """``index.jsonl`` gains the three keys, so cross-call triage can join
    retries without opening the per-call YAMLs."""
    record = _record(call_id="feedface" + "0" * 24, attempt=3, queue_wait_ms=7.77)
    assert LocalYamlLogSink(tmp_path).write_returning_path(record) is not None
    line = cast(
        "dict[str, object]",
        json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0]),
    )
    assert line["call_id"] == "feedface" + "0" * 24
    assert line["attempt"] == 3
    assert line["queue_wait_ms"] == 7.8


def test_record_stays_constructible_without_new_fields() -> None:
    """The three fields are defaulted, so every existing direct constructor
    (custom sinks, test helpers) keeps working unchanged."""
    record = LLMCallRecord(
        started_at=_record().started_at,
        feature="f",
        label=None,
        model=None,
        provider=None,
        temperature=0.0,
        duration_ms=1.0,
        schema="text",
        prompt="hi",
        response=None,
        error=None,
    )
    assert record.call_id is None
    assert record.attempt is None
    assert record.queue_wait_ms is None
