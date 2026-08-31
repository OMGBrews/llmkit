"""Tests for the resolved-model/provider fields in the LLM invocation log.

The log used to record ``model: null`` whenever a caller passed
``model=None`` (provider default), so cost attribution had to be
reverse-engineered from code. These tests pin the fix: the log records
the *resolved* effective model + the provider name, so a sweep's tier is
a ``grep`` away.

Also pins two response-fidelity contracts: a failed buffered text attempt
logs ``response: None`` (not ``""``, which would be indistinguishable from
a successful empty completion), and a schema whose custom serializer raises
degrades the *logged* response to ``None`` without breaking the call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, model_serializer

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    capture_llm_records,
    configure_llm_logging,
)
from llmkit import (
    calls as llm_calls,
)
from llmkit.calls._shared import resolve_model_and_provider
from tests._support import make_record as _record
from tests._support import provider_mock


def test_resolve_substitutes_provider_default_when_model_none() -> None:
    fake_provider = MagicMock()
    fake_provider.model = "gemini-3.1-flash-lite"
    fake_provider.name = "Google AI Studio"
    with patch("llmkit.providers.build_provider", return_value=fake_provider):
        resolved, provider = resolve_model_and_provider(None)
    assert resolved == "gemini-3.1-flash-lite"
    assert provider == "Google AI Studio"


def test_resolve_keeps_explicit_model_and_records_provider() -> None:
    fake_provider = MagicMock()
    fake_provider.model = "gemini-3.1-flash-lite"
    fake_provider.name = "Google AI Studio"
    with patch("llmkit.providers.build_provider", return_value=fake_provider):
        resolved, provider = resolve_model_and_provider("gemini-2.5-flash")
    assert resolved == "gemini-2.5-flash"
    assert provider == "Google AI Studio"


def test_resolve_uses_explicit_provider_without_global_lookup() -> None:
    """An explicit per-call provider is recorded as-is, and the global
    ``build_provider`` source is never consulted."""
    override = MagicMock()
    override.model = "anthropic/claude-sonnet-4-6"
    override.name = "OpenRouter"
    with patch(
        "llmkit.providers.build_provider",
        side_effect=AssertionError("global build_provider must not be called"),
    ):
        resolved, provider = resolve_model_and_provider(None, override)
    assert resolved == "anthropic/claude-sonnet-4-6"
    assert provider == "OpenRouter"


def test_structured_call_forwards_provider_override_to_transport() -> None:
    """``structured_llm_call`` threads a per-call provider down to the
    transport so the call routes through the override, not the global."""
    import asyncio

    from pydantic import BaseModel

    class _Schema(BaseModel):
        ok: bool

    override = MagicMock()
    override.model = "some-model"
    override.name = "OpenRouter"

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        assert kwargs["provider"] is override
        return _Schema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = asyncio.run(
            llm_calls.structured_llm_call("hi", _Schema, feature="test", provider=override)
        )
    assert result.ok is True


def test_resolve_degrades_gracefully_when_provider_unavailable() -> None:
    """A provider-resolution failure must not break the log write — it
    degrades to (model, None) rather than raising into the call's finally
    block."""
    with patch(
        "llmkit.providers.build_provider",
        side_effect=RuntimeError("no provider configured"),
    ):
        resolved, provider = resolve_model_and_provider("explicit-model")
    assert resolved == "explicit-model"
    assert provider is None


def test_local_yaml_sink_records_model_and_provider(tmp_path: Path) -> None:
    """The written YAML carries both the resolved model and provider."""
    import yaml

    path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert path is not None
    doc = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    assert doc["model"] == "gemini-3.1-flash-lite"
    assert doc["provider"] == "Google AI Studio"


def test_sink_write_returns_none_per_contract(tmp_path: Path) -> None:
    """The ``LogSink`` contract is ``write(record) -> None``: the file
    sink's ``write`` writes the YAML but returns nothing; the path is
    exposed via the file-specific ``write_returning_path`` instead."""
    sink = LocalYamlLogSink(tmp_path)
    assert sink.write(_record()) is None
    path = sink.write_returning_path(_record())
    assert path is not None and path.exists()


def test_write_llm_log_returns_path_for_file_sink(tmp_path: Path) -> None:
    """``write_llm_log`` returns the file path for the configured file
    sink (so path-capture keeps working) even though the shared contract
    return is ``None``."""
    from llmkit.logging import write_llm_log

    configure_llm_logging(LocalYamlLogSink(tmp_path))
    try:
        path = write_llm_log(_record())
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert path is not None and path.exists()


def test_write_llm_log_returns_none_for_third_party_sink() -> None:
    """A third-party sink implements only ``write(record) -> None``;
    ``write_llm_log`` honors that and returns ``None`` (no leaked path),
    while the sink still receives the record."""
    from llmkit.logging import write_llm_log

    received: list[LLMCallRecord] = []

    class _MemorySink:
        def write(self, record: LLMCallRecord) -> None:
            received.append(record)

    configure_llm_logging(_MemorySink())
    try:
        result = write_llm_log(_record())
    finally:
        configure_llm_logging(LocalYamlLogSink())
    assert result is None
    assert len(received) == 1


def test_local_yaml_sink_includes_approximate_cost_field(tmp_path: Path) -> None:
    """The record gains an ``approximate_cost`` field — ``None`` until Step 2c."""
    import yaml

    default_path = LocalYamlLogSink(tmp_path).write_returning_path(_record())
    assert default_path is not None
    doc = cast("dict[str, object]", yaml.safe_load(default_path.read_text()))
    assert "approximate_cost" in doc
    assert doc["approximate_cost"] is None

    priced_path = LocalYamlLogSink(tmp_path).write_returning_path(_record(approximate_cost=0.0123))
    assert priced_path is not None
    priced = cast("dict[str, object]", yaml.safe_load(priced_path.read_text()))
    assert priced["approximate_cost"] == 0.0123


def test_configure_llm_logging_none_disables_writes() -> None:
    """A ``None`` sink makes ``write_llm_log`` a no-op (returns None)."""
    from llmkit.logging import write_llm_log

    configure_llm_logging(None)
    try:
        assert write_llm_log(_record()) is None
    finally:
        configure_llm_logging(LocalYamlLogSink())


def test_summary_header_is_verdict_first(tmp_path: Path) -> None:
    """The first comment line is a single-glance ``ok`` verdict with the
    key metadata; an errored call leads with ``ERROR``."""
    ok_path = LocalYamlLogSink(tmp_path).write_returning_path(_record(approximate_cost=5.9e-06))
    assert ok_path is not None
    first_line = ok_path.read_text().splitlines()[0]
    assert first_line.startswith("# ok | extraction/summary | gemini-3.1-flash-lite | Schema |")
    assert "$5.9e-06" in first_line

    err_path = LocalYamlLogSink(tmp_path).write_returning_path(_record(error="APIError: boom"))
    assert err_path is not None
    assert err_path.read_text().splitlines()[0].startswith("# ERROR |")


def test_yaml_body_puts_blobs_last(tmp_path: Path) -> None:
    """High-signal metadata (incl. error/cost) precedes the big
    response/prompt blobs, and response precedes prompt."""
    path = LocalYamlLogSink(tmp_path).write_returning_path(
        _record(prompt="PROMPT_TEXT", response="RESP_TEXT")
    )
    assert path is not None
    body = path.read_text()
    assert body.index("error:") < body.index("response:") < body.index("prompt:")
    assert body.index("approximate_cost:") < body.index("response:")


def test_index_jsonl_appends_one_line_per_call(tmp_path: Path) -> None:
    """Every call appends a compact summary line to ``index.jsonl``."""
    import json

    sink = LocalYamlLogSink(tmp_path)
    p1 = sink.write_returning_path(_record(label="first", approximate_cost=1e-06))
    p2 = sink.write_returning_path(_record(label="second", error="Timeout: slow"))
    assert p1 is not None and p2 is not None

    index = tmp_path / "index.jsonl"
    lines = index.read_text().strip().splitlines()
    assert len(lines) == 2

    first = cast("dict[str, object]", json.loads(lines[0]))
    assert first["file"] == p1.name
    assert first["feature"] == "extraction"
    assert first["label"] == "first"
    assert first["model"] == "gemini-3.1-flash-lite"
    assert first["approximate_cost"] == 1e-06
    assert first["error"] is None
    # The big prompt/response blobs are deliberately NOT in the index.
    assert "prompt" not in first and "response" not in first

    second = cast("dict[str, object]", json.loads(lines[1]))
    assert second["label"] == "second"
    assert second["error"] == "Timeout: slow"


def test_failed_text_call_logs_response_none() -> None:
    """A text attempt that raises before any content logs ``response: None``
    (per the ``LLMCallRecord`` contract), not ``""`` — an empty string would
    be indistinguishable from a successful empty completion."""

    async def _boom(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        raise RuntimeError("transport exploded")

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_boom),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                with pytest.raises(RuntimeError, match="transport exploded"):
                    _ = await llm_calls.text_llm_call("hi", feature="summary")
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    assert cast("object", captured[0].response) is None
    assert captured[0].error is not None
    assert "transport exploded" in captured[0].error


def test_successful_text_call_still_logs_the_text() -> None:
    """Regression for the ``None``-sentinel fix: a successful text call
    records the actual completion text, exactly as before."""

    async def _fake_text(*_args: object, **_kwargs: object) -> tuple[str, float | None]:
        return "hello world", 0.001

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_fake_text),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                result = await llm_calls.text_llm_call("hi", feature="summary")
                assert result == "hello world"
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    assert cast("object", captured[0].response) == "hello world"
    assert captured[0].error is None


def test_mid_stream_failure_still_logs_partial_transcript() -> None:
    """The streaming surface is deliberately different from the buffered
    one: a mid-stream error logs the partial transcript accumulated so far
    (alongside the error), not ``None``."""
    from collections.abc import AsyncIterator

    def _fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield "par"
            yield "tial"
            raise RuntimeError("stream died")

        return _gen()

    with (
        patch("llmkit._litellm.astream_text", side_effect=_fake_stream),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> tuple[list[str], list[LLMCallRecord]]:
            chunks: list[str] = []
            with capture_llm_records() as records:
                with pytest.raises(RuntimeError, match="stream died"):
                    async for chunk in llm_calls.text_llm_call_stream("hi", feature="summary"):
                        chunks.append(chunk)
            return chunks, records

        chunks, captured = asyncio.run(_run())

    assert chunks == ["par", "tial"]
    assert len(captured) == 1
    assert cast("object", captured[0].response) == "partial"
    assert captured[0].error is not None
    assert "stream died" in captured[0].error


def test_abandoned_stream_logs_abandonment_marker_not_ok(tmp_path: Path) -> None:
    """A consumer that breaks out of a stream mid-flight never logs as ``ok``.

    Regression: ``GeneratorExit`` is a ``BaseException``, so it bypassed
    ``_stream_once``'s ``except Exception`` and the ``finally`` logged the
    partial transcript with ``error=None`` — a ``# ok`` verdict over a
    truncated response, indistinguishable from a stream the model finished.
    An abandoned stream now records ``STREAM_ABANDONED_ERROR`` (alongside the
    partial transcript), so the written YAML's ``head -1`` verdict reads
    ``ERROR``, never ``ok``.
    """
    from collections.abc import AsyncIterator

    def _fake_stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for delta in ("a", "b", "c", "d"):
                yield delta

        return _gen()

    with (
        patch("llmkit._litellm.astream_text", side_effect=_fake_stream),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
    ):

        async def _run() -> list[LLMCallRecord]:
            with capture_llm_records() as records:
                stream = llm_calls.text_llm_call_stream("hi", feature="summary")
                async for _chunk in stream:
                    break  # abandon after the first chunk
                await stream.aclose()
            return records

        captured = asyncio.run(_run())

    assert len(captured) == 1
    assert cast("object", captured[0].response) == "a"  # the partial transcript is kept
    assert captured[0].error == llm_calls.STREAM_ABANDONED_ERROR

    # End to end through the file sink: the head -1 triage line must carry the
    # ERROR verdict, not read like a completed call.
    path = LocalYamlLogSink(tmp_path).write_returning_path(captured[0])
    assert path is not None
    verdict_line = path.read_text().splitlines()[0]
    assert verdict_line.startswith("# ERROR")
    assert "Abandoned" in path.read_text()


class _ExplodingDumpSchema(BaseModel):
    """A schema whose custom serializer raises — only ``model_dump`` breaks;
    construction and field access stay fine."""

    ok: bool

    @model_serializer
    def _boom(self) -> dict[str, object]:
        raise ValueError("serializer exploded")


def test_raising_model_serializer_degrades_logged_response_not_the_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A custom ``@model_serializer`` that raises must not replace a
    successful return value (or mask a real provider error) — the dump
    exists only for the log, so the logged response degrades to ``None``
    with a warning, and the call still returns the parsed model."""

    async def _fake_structured(
        *_args: object, **_kwargs: object
    ) -> tuple[_ExplodingDumpSchema, float | None]:
        return _ExplodingDumpSchema(ok=True), 0.001

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_structured),
        patch("llmkit.providers.build_provider", return_value=provider_mock()),
        caplog.at_level("WARNING", logger="llmkit.calls"),
    ):

        async def _run() -> tuple[_ExplodingDumpSchema, list[LLMCallRecord]]:
            with capture_llm_records() as records:
                result = await llm_calls.structured_llm_call(
                    "hi", _ExplodingDumpSchema, feature="extraction"
                )
            return result, records

        result, captured = asyncio.run(_run())

    # The call itself succeeds — the serializer failure stays in the log path.
    assert result.ok is True
    assert len(captured) == 1
    assert cast("object", captured[0].response) is None
    assert captured[0].error is None
    assert any("Failed to serialize LLM response" in m for m in caplog.messages)
