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
from unittest.mock import MagicMock, patch

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
    structured_output,
)
from llmkit.logging import LogSink


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

    configure_llm_logging(None)  # logging is exercised elsewhere; keep this test pure
    try:
        with patch("llmkit._litellm.astream_text", _fake_astream):
            chunks = asyncio.run(
                _drain(
                    structured_output.stream_text_with_log(
                        "hi", feature="test", max_tokens=5, reasoning_effort="low"
                    )
                )
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert chunks == ["he", "llo"]
    assert seen["max_tokens"] == 5
    assert seen["reasoning_effort"] == "low"


def test_transport_omits_kwargs_when_unset() -> None:
    """No cap and a provider configured with ``None`` effort → neither kwarg
    reaches the provider request, byte-identical to the pre-parity stream."""
    seen = _capture_provider_kwargs(max_tokens=None, reasoning_effort=None, provider_effort=None)
    assert "max_tokens" not in seen
    assert "reasoning_effort" not in seen


def test_transport_includes_kwargs_when_set() -> None:
    """A concrete cap and a per-call effort both reach the provider request."""
    seen = _capture_provider_kwargs(max_tokens=8, reasoning_effort="high", provider_effort=None)
    assert seen["max_tokens"] == 8
    assert seen["reasoning_effort"] == "high"


def test_transport_falls_back_to_provider_reasoning_effort() -> None:
    """With no per-call effort, the provider's configured value is forwarded."""
    seen = _capture_provider_kwargs(
        max_tokens=None, reasoning_effort=None, provider_effort="disable"
    )
    assert seen["reasoning_effort"] == "disable"


def test_log_record_carries_kwargs() -> None:
    """The ``LLMCallRecord`` written for a stream records both settings."""
    captured: list[LLMCallRecord] = []

    class _CapturingSink:
        def write(self, record: LLMCallRecord) -> None:
            captured.append(record)

    async def _fake_astream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "hello"

    sink: LogSink = _CapturingSink()
    configure_llm_logging(sink)
    try:
        with patch("llmkit._litellm.astream_text", _fake_astream):
            _ = asyncio.run(
                _drain(
                    structured_output.stream_text_with_log(
                        "hi", feature="test", max_tokens=128, reasoning_effort="disable"
                    )
                )
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(captured) == 1
    assert captured[0].max_tokens == 128
    assert captured[0].reasoning_effort == "disable"


def _capture_provider_kwargs(
    *, max_tokens: int | None, reasoning_effort: str | None, provider_effort: str | None
) -> dict[str, object]:
    """Drive ``astream_text`` over a faked LiteLLM stream and return the kwargs
    the provider call (``litellm.acompletion``) saw.

    ``provider_effort`` is the provider's configured value; ``reasoning_effort``
    is the per-call override.
    """
    from llmkit import _litellm

    seen: dict[str, object] = {}

    class _FakeStream:
        def __aiter__(self) -> AsyncIterator[MagicMock]:
            async def _gen() -> AsyncIterator[MagicMock]:
                for delta in ("he", "llo"):
                    yield MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])

            return _gen()

    async def _fake_acompletion(**kwargs: object) -> _FakeStream:
        seen.update(kwargs)
        return _FakeStream()

    provider = MagicMock()
    provider.completion_kwargs.return_value = {"api_key": "k", "api_base": None}
    provider.litellm_model.return_value = "fake/model"
    provider.reasoning_effort = provider_effort

    async def _drive() -> list[str]:
        return [
            chunk
            async for chunk in _litellm.astream_text(
                "hi",
                temperature=0.0,
                model=None,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        ]

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        chunks = asyncio.run(_drive())

    assert chunks == ["he", "llo"]
    return seen
