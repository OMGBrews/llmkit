"""Live, end-to-end smoke tests against every supported provider.

Unlike the mocked unit tests, these make a *real* structured call through
each provider family and assert the library gets a validated Pydantic
instance back — the one thing mocks can never prove: that our model
strings, ``instructor.Mode`` pinning, and credential kwargs are actually
accepted by the provider's API.

These cost money and require network + credentials, so they are **opt-in**:
each test reads its provider's key from an environment variable and *skips*
(never fails) when that key is absent. With no keys set, the whole module
skips and ``pytest`` stays green — safe to ship in the public repo.

Run just these (after exporting keys — see ``docs/operations`` in the
maintainer workspace, or the table below):

    OPENROUTER_API_KEY=sk-or-...  \
    GEMINI_API_KEY=...            \
    ANTHROPIC_API_KEY=...         \
    uv run pytest tests/integration -v

Provider -> credential it reads:

    OpenRouter  OPENROUTER_API_KEY          https://openrouter.ai/keys
    Google      GEMINI_API_KEY              https://aistudio.google.com/apikey
    Anthropic   ANTHROPIC_API_KEY           https://console.anthropic.com/settings/keys
    Ollama      (local server on :11434)    https://ollama.com  (no key)
"""

from __future__ import annotations

import os
import socket

import pytest
from pydantic import BaseModel

from llmkit import (
    AnthropicProvider,
    GoogleProvider,
    LLMProviderInterface,
    OllamaProvider,
    OpenRouterProvider,
    structured_llm_call,
)


class Capital(BaseModel):
    """Tiny schema: cheap to generate, unambiguous to verify."""

    country: str
    capital: str


# Cheap, broadly-available default models per provider. Override via the
# matching ``*_SMOKE_MODEL`` env var if a key lacks access to one of these.
_OPENROUTER_MODEL = os.getenv("OPENROUTER_SMOKE_MODEL", "google/gemini-2.0-flash-001")
_GOOGLE_MODEL = os.getenv("GOOGLE_SMOKE_MODEL", "gemini-2.5-flash-lite")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_SMOKE_MODEL", "claude-haiku-4-5-20251001")
_OLLAMA_MODEL = os.getenv("OLLAMA_SMOKE_MODEL", "llama3.2")


async def _assert_structured_roundtrip(provider: LLMProviderInterface) -> None:
    """Drive one real structured call and check the parsed result."""
    result = await structured_llm_call(
        "What is the capital of France? Answer with the country and its capital.",
        Capital,
        feature="integration-smoke",
        label=provider.name,
        provider=provider,
        # Keep Gemini's default-on thinking from eating a tight token budget.
        reasoning_effort="disable",
    )
    assert isinstance(result, Capital)
    assert result.capital.strip().lower() == "paris"


def _ollama_up() -> bool:
    """True if an Ollama server answers on localhost:11434."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    netloc = host.split("://", 1)[-1]
    name, _, port = netloc.partition(":")
    try:
        with socket.create_connection((name or "localhost", int(port or 11434)), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.asyncio
async def test_openrouter_live() -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    await _assert_structured_roundtrip(OpenRouterProvider(api_key=key, model=_OPENROUTER_MODEL))


@pytest.mark.asyncio
async def test_google_live() -> None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    await _assert_structured_roundtrip(GoogleProvider(api_key=key, model=_GOOGLE_MODEL))


@pytest.mark.asyncio
async def test_anthropic_live() -> None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    await _assert_structured_roundtrip(AnthropicProvider(api_key=key, model=_ANTHROPIC_MODEL))


@pytest.mark.asyncio
async def test_ollama_live() -> None:
    if not _ollama_up():
        pytest.skip("no Ollama server on localhost:11434")
    await _assert_structured_roundtrip(OllamaProvider(model=_OLLAMA_MODEL))
