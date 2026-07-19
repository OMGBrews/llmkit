"""Tests for ``max_tokens`` / ``reasoning_effort`` parity on the stream path.

``text_llm_call_stream`` previously omitted ``max_tokens`` and
``reasoning_effort`` that every other public call function accepts. These
tests pin the now-symmetric seams:

* the public ``text_llm_call_stream`` carries both parameters and threads
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

import pytest

from llmkit import structured_output
from tests._support import capture_stream_provider_kwargs, capturing_sink


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    """Consume an async text stream into a list of chunks."""
    return [chunk async for chunk in stream]


def test_signature_exposes_max_tokens_and_reasoning_effort() -> None:
    """Acceptance: the stream call function carries both parameters."""
    params = inspect.signature(structured_output.text_llm_call_stream).parameters
    assert "max_tokens" in params
    assert "reasoning_effort" in params


def test_stream_threads_kwargs_to_transport() -> None:
    """``text_llm_call_stream(..., max_tokens=N, reasoning_effort=...)`` forwards
    both down to ``astream_text``."""
    seen: dict[str, object] = {}

    async def _fake_astream(*_args: object, **kwargs: object) -> AsyncIterator[str]:
        seen.update(kwargs)
        for delta in ("he", "llo"):
            yield delta

    with patch("llmkit._litellm.astream_text", _fake_astream):
        chunks = asyncio.run(
            _drain(
                structured_output.text_llm_call_stream(
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
                structured_output.text_llm_call_stream(
                    "hi", feature="test", max_tokens=128, reasoning_effort="disable"
                )
            )
        )

    assert len(captured) == 1
    assert captured[0].max_tokens == 128
    assert captured[0].reasoning_effort == "disable"


def test_deprecated_alias_warns_at_call_time_and_matches_new_name() -> None:
    """``stream_text_with_log`` is a deprecated alias for ``text_llm_call_stream``.

    It warns ``DeprecationWarning`` **eagerly at call time** (a plain ``def``, so
    the warning fires before the first ``__anext__``, not on first iteration) and
    yields chunks identical to the canonical name over the same faked transport.
    """

    async def _fake_astream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for delta in ("he", "llo"):
            yield delta

    with (
        capturing_sink(),
        patch("llmkit._litellm.astream_text", _fake_astream),
    ):
        canonical = asyncio.run(
            _drain(structured_output.text_llm_call_stream("hi", feature="test"))
        )
        # The warning fires when the alias is *called*, before any chunk is
        # pulled — so wrapping only the call (not the drain) captures it, which
        # is exactly the eager-at-call-time contract.
        with pytest.warns(DeprecationWarning, match="text_llm_call_stream"):
            stream = structured_output.stream_text_with_log("hi", feature="test")
        aliased = asyncio.run(_drain(stream))

    assert aliased == canonical == ["he", "llo"]
