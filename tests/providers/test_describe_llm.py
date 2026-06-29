"""Unit tests for :func:`describe_llm` and its :class:`LLMInfo` snapshot.

``describe_llm`` is the read accessor (renamed from ``get_llm_config`` in
0.2.0) that builds a provider only to read back the resolved provider + model
for display/telemetry. These tests pin that it resolves the provider's own
default model when the config leaves ``model`` unset, names the effective
provider, and reports ``is_local`` as a provider trait (true only for the
local Ollama provider).
"""

from __future__ import annotations

from llmkit.providers import (
    LLMClientConfig,
    LLMInfo,
    Provider,
    describe_llm,
)


def test_describe_llm_resolves_provider_default_model() -> None:
    """An unset ``model`` resolves to the provider's own default in the snapshot."""
    info = describe_llm(LLMClientConfig(provider=Provider.GOOGLE, api_key="k"))
    assert isinstance(info, LLMInfo)
    assert info.provider == Provider.GOOGLE
    assert info.provider_name == "Google AI Studio"
    assert info.model == "gemini-2.5-flash-lite"
    assert info.is_local is False


def test_describe_llm_honors_explicit_model() -> None:
    """An explicit ``model`` is carried through to the snapshot verbatim."""
    info = describe_llm(
        LLMClientConfig(provider=Provider.GOOGLE, api_key="k", model="gemini-2.5-pro")
    )
    assert info.model == "gemini-2.5-pro"


def test_describe_llm_marks_only_ollama_local() -> None:
    """``is_local`` is a provider trait: true for Ollama, false for cloud providers."""
    ollama = describe_llm(LLMClientConfig(provider=Provider.OLLAMA))
    assert ollama.is_local is True
    assert ollama.provider_name == "Ollama"

    google = describe_llm(LLMClientConfig(provider=Provider.GOOGLE, api_key="k"))
    assert google.is_local is False


def test_describe_llm_snapshots_vertex() -> None:
    """Vertex resolves its name + default model in the snapshot; residency
    fields don't affect the read accessor (they're routing, not display)."""
    info = describe_llm(LLMClientConfig(provider=Provider.VERTEX, vertex_location="europe-west4"))
    assert isinstance(info, LLMInfo)
    assert info.provider == Provider.VERTEX
    assert info.provider_name == "Google Vertex AI"
    assert info.model == "gemini-2.5-flash-lite"
    assert info.is_local is False
