"""Tests for the optional ``max_tokens`` cap on the structured-output path.

``structured_llm_call`` / ``structured_llm_call_sync`` gained a keyword-only
``max_tokens`` (parity with ``text_llm_call``). These tests pin two seams:

* the public call functions thread ``max_tokens`` down to the transport, and
* the transport (``acompletion_structured``) only puts ``max_tokens`` on the
  provider request when it is not ``None`` — so the default produces a
  request byte-identical to the prior behaviour (no ``max_tokens`` key).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
    structured_output,
)
from llmkit.logging import LogSink


class _Schema(BaseModel):
    ok: bool


def test_signatures_expose_max_tokens() -> None:
    """Acceptance: both public functions carry a ``max_tokens`` parameter."""
    assert "max_tokens" in inspect.signature(structured_output.structured_llm_call).parameters
    assert "max_tokens" in inspect.signature(structured_output.structured_llm_call_sync).parameters


def test_sync_call_threads_max_tokens_to_transport() -> None:
    """``structured_llm_call_sync(..., max_tokens=256)`` forwards the cap."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        seen.update(kwargs)
        return _Schema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = structured_output.structured_llm_call_sync(
            "hi", _Schema, feature="test", max_tokens=256
        )

    assert seen["max_tokens"] == 256
    # max_tokens does not disturb parsing — a validated instance comes back.
    assert isinstance(result, _Schema)
    assert result.ok is True


def test_async_call_threads_max_tokens_to_transport() -> None:
    """The async ``structured_llm_call(..., max_tokens=N)`` does the same."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        seen.update(kwargs)
        return _Schema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = asyncio.run(
            structured_output.structured_llm_call("hi", _Schema, feature="test", max_tokens=42)
        )

    assert seen["max_tokens"] == 42
    assert result.ok is True


def test_transport_omits_max_tokens_when_none() -> None:
    """At the provider seam, a ``None`` cap sends **no** ``max_tokens`` kwarg
    (absent, not ``None``) — byte-identical to the pre-feature request."""
    seen = _capture_provider_kwargs(max_tokens=None)
    assert "max_tokens" not in seen


def test_transport_includes_max_tokens_when_set() -> None:
    """A concrete cap reaches the provider call kwargs unchanged."""
    seen = _capture_provider_kwargs(max_tokens=8)
    assert seen["max_tokens"] == 8


def test_log_record_carries_max_tokens(tmp_path: Path) -> None:
    """The ``LLMCallRecord`` built for a structured call records the cap."""
    captured: list[LLMCallRecord] = []

    class _CapturingSink:
        def write(self, record: LLMCallRecord) -> Path | None:
            captured.append(record)
            return None

    async def _fake_transport(*_args: object, **_kwargs: object) -> tuple[_Schema, float | None]:
        return _Schema(ok=True), None

    sink: LogSink = _CapturingSink()
    configure_llm_logging(sink)
    try:
        with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
            asyncio.run(
                structured_output.structured_llm_call("hi", _Schema, feature="test", max_tokens=256)
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(captured) == 1
    assert captured[0].max_tokens == 256


def _capture_provider_kwargs(*, max_tokens: int | None) -> dict[str, object]:
    """Drive ``acompletion_structured`` over a faked instructor client and
    return the kwargs the provider call (``create_with_completion``) saw."""
    from llmkit import _litellm

    seen: dict[str, object] = {}

    async def _fake_create_with_completion(
        **kwargs: object,
    ) -> tuple[_Schema, MagicMock]:
        seen.update(kwargs)
        return _Schema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat.completions.create_with_completion = _fake_create_with_completion

    provider = MagicMock()
    provider.completion_kwargs.return_value = {"api_key": "k", "api_base": None}
    provider.instructor_mode = "json"
    provider.litellm_model.return_value = "fake/model"
    provider.reasoning_effort = None

    with patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client):
        parsed, _cost = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                _Schema,
                temperature=0.0,
                model=None,
                max_tokens=max_tokens,
                provider=provider,
            )
        )
    assert parsed.ok is True
    return seen
