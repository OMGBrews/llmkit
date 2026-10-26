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

The provider round trip is faked through :mod:`llmkit.testing`; the per-call
provider objects are real, so they run the real request path up to the fake.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

import llmkit.providers.base as base_mod
from llmkit import Provider, capture_llm_records, make_provider
from llmkit.calls import structured_llm_call, text_llm_call
from llmkit.testing import Reply, install


class _Schema(BaseModel):
    ok: bool


@pytest.fixture
def _no_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Unset the module-level config source for the duration of the test.

    There is no public *unregister*, and other tests may have registered a
    source in this process; resetting the private global guarantees the
    per-call path is exercised with genuinely no global config available, so a
    spurious fallback read would surface as the "not configured" ``RuntimeError``.
    Applied on top of the directory's autouse Ollama setup, which mutes logging
    and restores the previous config source afterwards.
    """
    monkeypatch.setattr(base_mod, "_config_source", None)
    yield


@pytest.mark.asyncio
async def test_structured_per_call_only_runs_and_logs_effective_provider(
    _no_global_config: None,
) -> None:
    """A structured call with only a per-call ``provider=`` (no global source)
    runs, and the log records the effective per-call provider + model."""
    provider = make_provider(Provider.OPENAI, api_key="sk-test", model="gpt-4.1")

    with (
        install(lambda _request: Reply.json(_Schema(ok=True), cost=0.0)) as installed,
        capture_llm_records() as records,
    ):
        result = await structured_llm_call("hi", _Schema, feature="t", provider=provider)

    assert result.ok is True
    # The round trip itself went out through the per-call provider.
    assert [r.provider for r in installed.requests] == ["OpenAI"]
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
    provider = make_provider(Provider.OPENROUTER, api_key="sk-test", model="x/y")

    with (
        install(lambda _request: Reply.text("hello", cost=0.0)) as installed,
        capture_llm_records() as records,
    ):
        result = await text_llm_call("hi", feature="t", provider=provider)

    assert result == "hello"
    assert [r.provider for r in installed.requests] == ["OpenRouter"]
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
