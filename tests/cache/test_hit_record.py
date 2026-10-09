"""What a hit leaves behind: its record, its log file, and nothing else.

A hit still writes one :class:`~llmkit.LLMCallRecord` — the log is how an
operator sees that a call happened — but it takes no rate-limiter slot, runs no
retry loop, and pays nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel

from llmkit import (
    InMemoryLLMCache,
    LocalYamlLogSink,
    capture_llm_records,
    configure_llm_cache,
    configure_llm_logging,
    structured_llm_call,
    text_llm_call,
)
from llmkit.rate_limiting import GlobalRateLimiter
from llmkit.rate_limiting._observability import stamp_queue_wait
from llmkit.retry import with_retries
from tests._support import UsageCounts
from tests.cache.conftest import cache_provider


class _Answer(BaseModel):
    verdict: str


async def _structured(*_a: object, **_k: object) -> tuple[_Answer, float | None, UsageCounts]:
    return _Answer(verdict="yes"), 0.004, (100, 20, 120)


@pytest.mark.asyncio
async def test_hit_record_fields() -> None:
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_structured),
        capture_llm_records() as records,
    ):
        _ = await structured_llm_call("q", _Answer, feature="coach", label="a", provider=provider)
        _ = await structured_llm_call("q", _Answer, feature="coach", label="b", provider=provider)

    paid, hit = records
    assert paid.cache_hit is False
    assert paid.source_call_id is None
    assert paid.approximate_cost == 0.004

    assert hit.cache_hit is True
    assert hit.source_call_id == paid.call_id
    assert hit.call_id is not None and hit.call_id != paid.call_id
    assert hit.approximate_cost == 0.0
    assert hit.usage is None
    assert hit.queue_wait_ms is None
    assert hit.attempt == 1
    assert hit.error is None
    assert cast("object", hit.response) == {"verdict": "yes"}
    assert hit.schema == "_Answer"
    assert (hit.feature, hit.label) == ("coach", "b")
    assert (hit.model, hit.provider) == ("gemini-3.1-flash-lite", "Google AI Studio")
    assert hit.duration_ms >= 0


@pytest.mark.asyncio
async def test_text_hit_record_never_reads_the_previous_queue_wait() -> None:
    """The queue-wait stamp still holds the paid attempt's value when the hit
    is recorded; the hit must not inherit it."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()

    async def _text(*_a: object, **_k: object) -> tuple[str, float | None, UsageCounts]:
        stamp_queue_wait(42.0)  # what the limiter does once a slot is held
        return "yes", 0.001, (1, 1, 2)

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_text),
        capture_llm_records() as records,
    ):
        _ = await text_llm_call("q", feature="f", provider=provider)
        _ = await text_llm_call("q", feature="f", provider=provider)

    paid, hit = records
    assert paid.queue_wait_ms == 42.0
    assert (hit.cache_hit, hit.queue_wait_ms, hit.schema) == (True, None, "text")
    assert cast("object", hit.response) == "yes"
    assert (hit.approximate_cost, hit.usage, hit.source_call_id) == (0.0, None, paid.call_id)


@pytest.mark.asyncio
async def test_records_without_a_cache_say_so() -> None:
    provider = cache_provider()
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_structured),
        capture_llm_records() as records,
    ):
        _ = await structured_llm_call("q", _Answer, feature="f", provider=provider)

    (record,) = records
    assert (record.cache_hit, record.source_call_id) == (False, None)


@pytest.mark.asyncio
async def test_header_line_two_and_index_carry_the_flag(tmp_path: Path) -> None:
    configure_llm_cache(InMemoryLLMCache())
    configure_llm_logging(LocalYamlLogSink(tmp_path))
    provider = cache_provider()
    with patch("llmkit._litellm.acompletion_structured", side_effect=_structured):
        _ = await structured_llm_call("q", _Answer, feature="f", label="paid", provider=provider)
        _ = await structured_llm_call("q", _Answer, feature="f", label="hit", provider=provider)

    paid_file = next(tmp_path.glob("*_paid_*.yaml"))
    hit_file = next(tmp_path.glob("*_hit_*.yaml"))
    paid_head = paid_file.read_text().splitlines()[:2]
    hit_head = hit_file.read_text().splitlines()[:2]

    # Line 1 keeps its pinned verdict shape; the hit shows its zero cost.
    assert hit_head[0].startswith("# ok | f/hit | gemini-3.1-flash-lite | _Answer | ")
    assert hit_head[0].endswith(" | $0")
    assert "cache" not in hit_head[0]
    # Line 2 gains the suffix after the correlation fields, on hits only.
    assert hit_head[1].endswith(" attempt=1 cache=hit")
    assert "cache=" not in paid_head[1]

    body = cast("dict[str, object]", yaml.safe_load(hit_file.read_text()))
    assert body["cache_hit"] is True
    paid_body = cast("dict[str, object]", yaml.safe_load(paid_file.read_text()))
    assert body["source_call_id"] == paid_body["call_id"]
    assert paid_body["cache_hit"] is False and paid_body["source_call_id"] is None

    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (tmp_path / "index.jsonl").read_text().splitlines()
    ]
    assert [(line["label"], line["cache_hit"]) for line in lines] == [
        ("paid", False),
        ("hit", True),
    ]
    assert all("source_call_id" not in line for line in lines)


@pytest.mark.asyncio
async def test_a_hit_takes_no_rate_limiter_slot_and_enters_no_retry_loop() -> None:
    """Driven through the real transport over a faked ``litellm.acompletion``,
    so the limiter is genuinely acquired on the miss and observably skipped on
    the hit."""
    configure_llm_cache(InMemoryLLMCache())
    provider = cache_provider()
    response = MagicMock(_hidden_params={})
    response.choices = [MagicMock(message=MagicMock(content="answer"))]

    async def _respond(**_kwargs: object) -> MagicMock:
        return response

    acompletion = MagicMock(side_effect=_respond)
    with (
        patch("llmkit._litellm.litellm.acompletion", acompletion),
        patch.object(
            GlobalRateLimiter, "acquire_async", side_effect=GlobalRateLimiter.acquire_async
        ) as acquire,
        patch("llmkit.calls._shared.with_retries", side_effect=with_retries) as retries,
    ):
        first = await text_llm_call("q", feature="f", provider=provider)
        assert (acquire.call_count, retries.call_count, acompletion.call_count) == (1, 1, 1)
        second = await text_llm_call("q", feature="f", provider=provider)

    assert first == second == "answer"
    assert (acquire.call_count, retries.call_count, acompletion.call_count) == (1, 1, 1)
