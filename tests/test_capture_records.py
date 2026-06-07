"""Tests for the records-oriented capture primitive.

``capture_llm_records()`` lets a host receive the per-call
:class:`~llmkit.logging.LLMCallRecord` objects — ``approximate_cost``,
resolved model/provider, duration — without authoring a custom
:class:`~llmkit.logging.LogSink`. These tests pin that it captures across
both the async call path and the ``run_sync`` sync bridge, that it carries
the cost a host needs, and that it works even with logging disabled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
    structured_output,
)

# Imported from the submodule rather than the package root: the integration
# phase owns adding ``capture_llm_records`` to ``llmkit.__all__``.
from llmkit.structured_output import capture_llm_records


class _Schema(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    """Point logging at a no-op sink so these tests don't touch the disk.

    Capture is sink-independent, so a ``None`` sink still exercises the
    records path fully while keeping the suite from writing YAML files.
    """
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.model = "gemini-2.5-flash-lite"
    provider.name = "Google AI Studio"
    return provider


async def _fake_structured(*_args: object, **_kwargs: object) -> tuple[_Schema, float | None]:
    return _Schema(ok=True), 0.0042


def test_capture_records_yields_record_with_cost_async() -> None:
    """A host gets the per-call record — including ``approximate_cost`` —
    from an async call with no custom sink."""
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_structured),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                _ = await structured_output.structured_llm_call("hi", _Schema, feature="extraction")
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    record = captured[0]
    assert isinstance(record, LLMCallRecord)
    assert record.approximate_cost == 0.0042
    assert record.feature == "extraction"
    assert record.model == "gemini-2.5-flash-lite"
    assert record.provider == "Google AI Studio"
    assert record.error is None


def test_capture_records_crosses_the_sync_bridge() -> None:
    """Capture survives the ``run_sync`` sync-bridge boundary: a record is
    captured for a ``structured_llm_call_sync`` driven from the sync side."""
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_structured),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):
        with capture_llm_records() as records:
            result = structured_output.structured_llm_call_sync(
                "hi", _Schema, feature="classification"
            )

    assert result.ok is True
    assert len(records) == 1
    assert records[0].approximate_cost == 0.0042
    assert records[0].feature == "classification"


def test_capture_records_captures_text_calls() -> None:
    """``text_llm_call`` records flow into the capture buffer too."""

    async def _fake_text(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "hello", 0.001

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_fake_text),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                _ = await structured_output.text_llm_call("hi", feature="summary")
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    assert captured[0].approximate_cost == 0.001
    assert captured[0].schema == "stream"


def test_capture_records_records_error_on_failure() -> None:
    """A failing call still produces a captured record with the error set."""

    async def _boom(*_args: object, **_kwargs: object) -> tuple[_Schema, float | None]:
        raise RuntimeError("provider exploded")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_boom),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                with pytest.raises(RuntimeError, match="provider exploded"):
                    _ = await structured_output.structured_llm_call(
                        "hi", _Schema, feature="extraction"
                    )
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    assert captured[0].error is not None
    assert "provider exploded" in captured[0].error


def test_capture_records_independent_of_paths_capture() -> None:
    """Records-capture works with logging disabled — proving it does not
    depend on a file sink the way path-capture does."""
    configure_llm_logging(None)
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_structured),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> tuple[list[LLMCallRecord], list[object]]:
            with capture_llm_records() as records:
                with structured_output.capture_llm_log_paths() as paths:
                    _ = await structured_output.structured_llm_call(
                        "hi", _Schema, feature="extraction"
                    )
                    return records, list(paths)

        captured_records, captured_paths = asyncio.run(_run())

    # The record is captured regardless of the sink; no file path is, since
    # logging is disabled.
    assert len(captured_records) == 1
    assert captured_records[0].approximate_cost == 0.0042
    assert captured_paths == []


def test_capture_record_carries_cost_metadata_without_a_sink() -> None:
    """Pin the contract FiW relied on: the record exposes both
    ``approximate_cost`` and the per-call metadata a host needs for cost
    attribution, with no sink authored."""
    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_structured),
        patch("llmkit.providers.build_provider", return_value=_provider()),
    ):

        async def _run() -> LLMCallRecord:
            with capture_llm_records() as records:
                _ = await structured_output.structured_llm_call(
                    "hi", _Schema, feature="extraction", label="risk_register", model="custom-model"
                )
            return records[0]

        record = asyncio.run(_run())

    assert record.approximate_cost == 0.0042
    assert record.model == "custom-model"
    assert record.provider == "Google AI Studio"
    assert record.feature == "extraction"
    assert record.label == "risk_register"
    assert record.duration_ms >= 0.0
