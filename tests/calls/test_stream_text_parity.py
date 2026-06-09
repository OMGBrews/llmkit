"""Tests for ``max_tokens`` / ``reasoning_effort`` parity on the stream path.

``stream_text_with_log`` previously omitted ``max_tokens`` and
``reasoning_effort`` that every other public call function accepts. These
tests pin the now-symmetric seams:

* the public ``stream_text_with_log`` carries both parameters and threads
  them to the transport (``astream_text``),
* the transport only puts each on the provider request when set
  (``reasoning_effort`` resolved against the provider's value) — so the
  default stream request is byte-identical to before, and
* the ``LLMCallRecord`` written for a stream records both settings.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from unittest.mock import patch

from llmkit import structured_output
from tests._support import capture_stream_provider_kwargs, capturing_sink


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    """Consume an async text stream into a list of chunks."""
    return [chunk async for chunk in stream]


def test_signature_exposes_max_tokens_and_reasoning_effort() -> None:
    """Acceptance: the stream call function carries both parameters."""
    params = inspect.signature(structured_output.stream_text_with_log).parameters
    assert "max_tokens" in params
    assert "reasoning_effort" in params


def test_stream_threads_kwargs_to_transport() -> None:
    """``stream_text_with_log(..., max_tokens=N, reasoning_effort=...)`` forwards
    both down to ``astream_text``."""
    seen: dict[str, object] = {}

    async def _fake_astream(*_args: object, **kwargs: object) -> AsyncIterator[str]:
        seen.update(kwargs)
        for delta in ("he", "llo"):
            yield delta

    with patch("llmkit._litellm.astream_text", _fake_astream):
        chunks = asyncio.run(
            _drain(
                structured_output.stream_text_with_log(
                    "hi", feature="test", max_tokens=5, reasoning_effort="low"
                )
            )
        )

    assert chunks == ["he", "llo"]
    assert seen["max_tokens"] == 5
    assert seen["reasoning_effort"] == "low"


def test_transport_omits_kwargs_when_unset() -> None:
    """No cap and a provider configured with ``None`` effort → neither kwarg
    reaches the provider request, byte-identical to the pre-parity stream."""
    seen = capture_stream_provider_kwargs(
        max_tokens=None, reasoning_effort=None, provider_effort=None
    )
    assert "max_tokens" not in seen
    assert "reasoning_effort" not in seen


def test_transport_includes_kwargs_when_set() -> None:
    """A concrete cap and a per-call effort both reach the provider request."""
    seen = capture_stream_provider_kwargs(
        max_tokens=8, reasoning_effort="high", provider_effort=None
    )
    assert seen["max_tokens"] == 8
    assert seen["reasoning_effort"] == "high"


def test_transport_falls_back_to_provider_reasoning_effort() -> None:
    """With no per-call effort, the provider's configured value is forwarded."""
    seen = capture_stream_provider_kwargs(
        max_tokens=None, reasoning_effort=None, provider_effort="disable"
    )
    assert seen["reasoning_effort"] == "disable"


def test_log_record_carries_kwargs() -> None:
    """The ``LLMCallRecord`` written for a stream records both settings."""

    async def _fake_astream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "hello"

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.astream_text", _fake_astream),
    ):
        _ = asyncio.run(
            _drain(
                structured_output.stream_text_with_log(
                    "hi", feature="test", max_tokens=128, reasoning_effort="disable"
                )
            )
        )

    assert len(captured) == 1
    assert captured[0].max_tokens == 128
    assert captured[0].reasoning_effort == "disable"
