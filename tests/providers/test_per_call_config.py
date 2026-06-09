"""Tests for fully per-call hosts that never register a global config source.

The per-call ``provider=`` override is a standout feature for multi-tenant
hosts that inject a different credential per request. A host that passes
``provider=`` on *every* call should not have to register a global
:func:`~llmkit.configure_llm_client` source it never uses — the per-call
pattern must stand on its own: the call runs, and logging resolves the
*effective* provider from the per-call provider rather than from the global
config.

These tests pin that with the global config source explicitly unset (so a
spurious read would raise), while confirming the global-config path is
unchanged — a call with neither a per-call provider nor a registered source
still raises the clear "not configured" ``RuntimeError``.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import llmkit.providers.base as base_mod
from llmkit import (
    LocalYamlLogSink,
    Provider,
    capture_llm_records,
    configure_llm_logging,
    make_provider,
)
from llmkit.structured_output import structured_llm_call, text_llm_call


class _Schema(BaseModel):
    ok: bool


@pytest.fixture
def _no_global_config(  # pyright: ignore[reportUnusedFunction]  # pytest fixture, injected by name
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Unset the module-level config source for the duration of the test.

    There is no public *unregister*, and other tests may have registered a
    source in this process; resetting the private global guarantees the
    per-call path is exercised with genuinely no global config available, so a
    spurious fallback read would surface as the "not configured" ``RuntimeError``.
    """
    monkeypatch.setattr(base_mod, "_config_source", None)
    configure_llm_logging(None)
    try:
        yield
    finally:
        configure_llm_logging(LocalYamlLogSink())


@pytest.mark.asyncio
async def test_structured_per_call_only_runs_and_logs_effective_provider(
    _no_global_config: None,
) -> None:
    """A structured call with only a per-call ``provider=`` (no global source)
    runs, and the log records the effective per-call provider + model."""

    async def _transport(*_a: object, **_k: object) -> tuple[_Schema, float | None]:
        return _Schema(ok=True), 0.0

    provider = make_provider(Provider.OPENAI, api_key="sk-test", model="gpt-4.1")

    with (
        patch("llmkit._litellm.acompletion_structured", side_effect=_transport),
        capture_llm_records() as records,
    ):
        result = await structured_llm_call("hi", _Schema, feature="t", provider=provider)

    assert result.ok is True
    assert len(records) == 1
    # Logging resolved the effective provider purely from the per-call provider —
    # no global source was consulted (there was none).
    assert records[0].provider == "OpenAI"
    assert records[0].model == "gpt-4.1"


@pytest.mark.asyncio
async def test_text_per_call_only_runs_and_logs_effective_provider(
    _no_global_config: None,
) -> None:
    """The text path is per-call-only too: it runs and logs the per-call provider."""

    async def _transport(*_a: object, **_k: object) -> tuple[str, float | None]:
        return "hello", 0.0

    provider = make_provider(Provider.OPENROUTER, api_key="sk-test", model="x/y")

    with (
        patch("llmkit._litellm.acompletion_text", side_effect=_transport),
        capture_llm_records() as records,
    ):
        result = await text_llm_call("hi", feature="t", provider=provider)

    assert result == "hello"
    assert records[0].provider == "OpenRouter"
    assert records[0].model == "x/y"


@pytest.mark.asyncio
async def test_no_provider_and_no_global_config_still_raises_clearly(
    _no_global_config: None,
) -> None:
    """The global-config path is unchanged: a call with neither a per-call
    provider nor a registered source raises the clear "not configured" error
    (the transport reaches ``build_provider`` -> ``active_config``). The error
    is non-transient, so it propagates immediately rather than being retried."""
    with pytest.raises(RuntimeError, match="not configured"):
        _ = await structured_llm_call("hi", _Schema, feature="t")
