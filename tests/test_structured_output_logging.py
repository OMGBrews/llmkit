"""Tests for the resolved-model/provider fields in the LLM invocation log.

The log used to record ``model: null`` whenever a caller passed
``model=None`` (provider default), so cost attribution had to be
reverse-engineered from code. These tests pin the fix: the log records
the *resolved* effective model + the provider name, so a sweep's tier is
a ``grep`` away.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmkit import (
    LLMCallRecord,
    LocalYamlLogSink,
    configure_llm_logging,
    structured_output,
)


def test_resolve_substitutes_provider_default_when_model_none() -> None:
    fake_provider = MagicMock()
    fake_provider.model = "gemini-2.5-flash-lite"
    fake_provider.name = "Google AI Studio"
    with patch("llmkit.providers.get_provider", return_value=fake_provider):
        resolved, provider = structured_output._resolve_model_and_provider(None)
    assert resolved == "gemini-2.5-flash-lite"
    assert provider == "Google AI Studio"


def test_resolve_keeps_explicit_model_and_records_provider() -> None:
    fake_provider = MagicMock()
    fake_provider.model = "gemini-2.5-flash-lite"
    fake_provider.name = "Google AI Studio"
    with patch("llmkit.providers.get_provider", return_value=fake_provider):
        resolved, provider = structured_output._resolve_model_and_provider("gemini-2.5-flash")
    assert resolved == "gemini-2.5-flash"
    assert provider == "Google AI Studio"


def test_resolve_uses_explicit_provider_without_global_lookup() -> None:
    """An explicit per-call provider is recorded as-is, and the global
    ``get_provider`` source is never consulted."""
    override = MagicMock()
    override.model = "anthropic/claude-sonnet-4-6"
    override.name = "OpenRouter"
    with patch(
        "llmkit.providers.get_provider",
        side_effect=AssertionError("global get_provider must not be called"),
    ):
        resolved, provider = structured_output._resolve_model_and_provider(None, override)
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
            structured_output.structured_llm_call("hi", _Schema, feature="test", provider=override)
        )
    assert result.ok is True


def test_resolve_degrades_gracefully_when_provider_unavailable() -> None:
    """A provider-resolution failure must not break the log write — it
    degrades to (model, None) rather than raising into the call's finally
    block."""
    with patch(
        "llmkit.providers.get_provider",
        side_effect=RuntimeError("no provider configured"),
    ):
        resolved, provider = structured_output._resolve_model_and_provider("explicit-model")
    assert resolved == "explicit-model"
    assert provider is None


def _record(**overrides: object) -> LLMCallRecord:
    base: dict[str, object] = {
        "started_at": datetime(2026, 5, 31, tzinfo=UTC),
        "feature": "extraction",
        "label": "summary",
        "model": "gemini-2.5-flash-lite",
        "provider": "Google AI Studio",
        "temperature": 0.0,
        "duration_ms": 12.3,
        "schema": "Schema",
        "prompt": "hi",
        "response": None,
        "error": None,
    }
    base.update(overrides)
    return LLMCallRecord(**base)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat


def test_local_yaml_sink_records_model_and_provider(tmp_path: Path) -> None:
    """The written YAML carries both the resolved model and provider."""
    import yaml

    path = LocalYamlLogSink(tmp_path).write(_record())
    assert path is not None
    doc = yaml.safe_load(path.read_text())
    assert doc["model"] == "gemini-2.5-flash-lite"
    assert doc["provider"] == "Google AI Studio"


def test_local_yaml_sink_includes_approximate_cost_field(tmp_path: Path) -> None:
    """The record gains an ``approximate_cost`` field — ``None`` until Step 2c."""
    import yaml

    default_path = LocalYamlLogSink(tmp_path).write(_record())
    assert default_path is not None
    doc = yaml.safe_load(default_path.read_text())
    assert "approximate_cost" in doc
    assert doc["approximate_cost"] is None

    priced_path = LocalYamlLogSink(tmp_path).write(_record(approximate_cost=0.0123))
    assert priced_path is not None
    priced = yaml.safe_load(priced_path.read_text())
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
    ok_path = LocalYamlLogSink(tmp_path).write(_record(approximate_cost=5.9e-06))
    assert ok_path is not None
    first_line = ok_path.read_text().splitlines()[0]
    assert first_line.startswith("# ok | extraction/summary | gemini-2.5-flash-lite | Schema |")
    assert "$5.9e-06" in first_line

    err_path = LocalYamlLogSink(tmp_path).write(_record(error="APIError: boom"))
    assert err_path is not None
    assert err_path.read_text().splitlines()[0].startswith("# ERROR |")


def test_yaml_body_puts_blobs_last(tmp_path: Path) -> None:
    """High-signal metadata (incl. error/cost) precedes the big
    response/prompt blobs, and response precedes prompt."""
    path = LocalYamlLogSink(tmp_path).write(_record(prompt="PROMPT_TEXT", response="RESP_TEXT"))
    assert path is not None
    body = path.read_text()
    assert body.index("error:") < body.index("response:") < body.index("prompt:")
    assert body.index("approximate_cost:") < body.index("response:")


def test_index_jsonl_appends_one_line_per_call(tmp_path: Path) -> None:
    """Every call appends a compact summary line to ``index.jsonl``."""
    import json

    sink = LocalYamlLogSink(tmp_path)
    p1 = sink.write(_record(label="first", approximate_cost=1e-06))
    p2 = sink.write(_record(label="second", error="Timeout: slow"))
    assert p1 is not None and p2 is not None

    index = tmp_path / "index.jsonl"
    lines = index.read_text().strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["file"] == p1.name
    assert first["feature"] == "extraction"
    assert first["label"] == "first"
    assert first["model"] == "gemini-2.5-flash-lite"
    assert first["approximate_cost"] == 1e-06
    assert first["error"] is None
    # The big prompt/response blobs are deliberately NOT in the index.
    assert "prompt" not in first and "response" not in first

    second = json.loads(lines[1])
    assert second["label"] == "second"
    assert second["error"] == "Timeout: slow"
