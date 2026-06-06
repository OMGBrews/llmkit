"""Live, end-to-end smoke tests against every supported provider.

Unlike the mocked unit tests, these make a *real* structured call through
each provider family and assert the library gets a validated Pydantic
instance back — the one thing mocks can never prove: that our model
strings, ``instructor.Mode`` pinning, and credential kwargs are actually
accepted by the provider's API.

**These are the live half of an explicit split.** Every test in this module is
marked ``@pytest.mark.live`` (via ``pytestmark`` below) and runs *only* when
``--run-live`` is passed. The ``conftest.py`` switch — never the presence of a
key — decides whether they run, so the same command behaves identically on every
machine. Plain ``pytest`` (CI, contributors, no keys) never collects a live
call; offline tests, conversely, never touch the network or read a credential.
There is no in-between and no test that quietly changes mode with the
environment.

Under ``--run-live`` every test here **must pass**: a missing key or an
unreachable Ollama server is a hard **failure**, not a skip — you asked for live
coverage, so a provider can't be silently dropped. Export every credential and
start a local Ollama server first (this is the release gate; see
``docs/operations`` in the maintainer workspace):

    OPENROUTER_API_KEY=sk-or-...  \
    GEMINI_API_KEY=...            \
    ANTHROPIC_API_KEY=...         \
    OPENAI_API_KEY=sk-...         \
    DEEPSEEK_API_KEY=sk-...       \
    uv run pytest tests/integration --run-live -v

To exercise a single provider you have a key for, select it explicitly rather
than relying on which keys happen to be set, e.g.
``uv run pytest tests/integration --run-live -k openai``.

Provider -> credential it reads:

    OpenRouter  OPENROUTER_API_KEY          https://openrouter.ai/keys
    Google      GEMINI_API_KEY              https://aistudio.google.com/apikey
    Anthropic   ANTHROPIC_API_KEY           https://console.anthropic.com/settings/keys
    OpenAI      OPENAI_API_KEY              https://platform.openai.com/api-keys
    DeepSeek    DEEPSEEK_API_KEY            https://platform.deepseek.com/api_keys
    Ollama      (local server on :11434)    https://ollama.com  (no key)

Ollama reads no key — it needs a reachable server instead. Point ``OLLAMA_HOST``
at it (default ``http://localhost:11434``); in the maintainer devcontainer
Ollama runs on the host, reached at ``http://host.docker.internal:11434``.
"""

from __future__ import annotations

import os
import socket
from typing import NoReturn

import pytest
from pydantic import BaseModel

from llmkit import (
    AnthropicProvider,
    DeepSeekProvider,
    GoogleProvider,
    LLMProviderInterface,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    structured_llm_call,
)

# Every test in this module makes a real API call. The marker is what the
# ``--run-live`` switch in conftest.py keys off of: without the flag these are
# skipped at collection (deterministically, regardless of any credentials in the
# environment); with it, they all run and must pass.
pytestmark = pytest.mark.live


class CountryProfile(BaseModel):
    """Multi-field, multi-type smoke schema: cheap to generate, strict to verify.

    Deliberately exercises a string, an integer, a boolean, and a list so a
    structured-output mode that silently drops or empties fields (the
    ``Mode.TOOLS_STRICT`` regression on ``gpt-4.1-mini`` that returned empty
    fields yet flaked green) cannot satisfy the assertion by luck — every
    field is checked below. The values are *dictated by the prompt* (an
    extraction, not a knowledge recall), so the correct answer is fully
    deterministic and within reach of every supported model regardless of how
    much geography it knows.
    """

    country: str
    capital: str
    continent: str
    eu_member: bool
    land_borders: int
    largest_cities: list[str]


# Cheap, broadly-available default models per provider. Override via the
# matching ``*_SMOKE_MODEL`` env var if a key lacks access to one of these.
# An open model hosted by many OpenRouter providers (so it won't 404 when a
# single vendor slug is retired) that supports OpenRouter structured outputs.
_OPENROUTER_MODEL = os.getenv("OPENROUTER_SMOKE_MODEL", "mistralai/mistral-nemo")
_GOOGLE_MODEL = os.getenv("GOOGLE_SMOKE_MODEL", "gemini-2.5-flash-lite")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_SMOKE_MODEL", "claude-haiku-4-5-20251001")
_OPENAI_MODEL = os.getenv("OPENAI_SMOKE_MODEL", "gpt-4.1-mini")
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_SMOKE_MODEL", "deepseek-chat")
_OLLAMA_MODEL = os.getenv("OLLAMA_SMOKE_MODEL", "llama3.2")
# Where the Ollama server lives. Default is an in-process localhost server; in
# the maintainer devcontainer Ollama runs on the *host*, so the container sets
# OLLAMA_HOST=http://host.docker.internal:11434. Both the up-check and the
# provider read this same value, so the guard and the real call can't disagree.
# This configures *where* the one server is — it never changes whether the test
# runs (that's --run-live's job).
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _missing(dep: str) -> NoReturn:
    """Fail a live test whose credential or server is unavailable.

    Only reachable under ``--run-live`` (otherwise the whole module is skipped at
    collection), so a missing dependency is an *error*, not a skip: opting into
    live coverage and then silently dropping a provider is exactly the
    looks-green-but-untested trap this split exists to prevent. Returns
    ``NoReturn`` so callers get type-narrowing on a just-checked key.
    """
    pytest.fail(f"{dep} — required when --run-live is set")


async def _assert_structured_roundtrip(
    provider: LLMProviderInterface, reasoning_effort: str | None = None
) -> None:
    """Drive one real structured call and check the parsed result.

    ``reasoning_effort`` is opt-in per provider: only some provider/model
    pairs accept it (Gemini's 2.5 thinking models do; e.g. OpenRouter's
    ``gemini-2.0-flash-001`` does not, and litellm raises rather than drop
    it). Default ``None`` sends nothing, so the call works everywhere.

    Every field of :class:`CountryProfile` is asserted: a half-working mode
    that empties or drops any field fails *deterministically* here rather
    than flaking green on the one easy field.
    """
    result = await structured_llm_call(
        # An extraction with the answer fully specified, so the result is
        # deterministic and a dropped/empty field is unambiguously wrong.
        "Extract the following facts about France into the schema. "
        "Country: France. Capital: Paris. Continent: Europe. "
        "Member of the European Union: yes. Number of bordering countries: 8. "
        "Three largest cities by population: Paris, Marseille, Lyon.",
        CountryProfile,
        feature="integration-smoke",
        label=provider.name,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )
    assert isinstance(result, CountryProfile)
    assert result.country.strip().lower() == "france"
    assert result.capital.strip().lower() == "paris"
    assert result.continent.strip().lower() == "europe"
    assert result.eu_member is True
    assert result.land_borders == 8
    assert {city.strip().lower() for city in result.largest_cities} == {
        "paris",
        "marseille",
        "lyon",
    }


def _ollama_up() -> bool:
    """True if an Ollama server answers at ``OLLAMA_HOST`` (default localhost)."""
    netloc = _OLLAMA_HOST.split("://", 1)[-1]
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
        _missing("OPENROUTER_API_KEY not set")
    await _assert_structured_roundtrip(OpenRouterProvider(api_key=key, model=_OPENROUTER_MODEL))


@pytest.mark.asyncio
async def test_google_live() -> None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        _missing("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    # Gemini 2.5 thinks by default; disable it so the smoke call doesn't
    # burn reasoning tokens on a trivial prompt.
    await _assert_structured_roundtrip(
        GoogleProvider(api_key=key, model=_GOOGLE_MODEL), reasoning_effort="disable"
    )


@pytest.mark.asyncio
async def test_anthropic_live() -> None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        _missing("ANTHROPIC_API_KEY not set")
    await _assert_structured_roundtrip(AnthropicProvider(api_key=key, model=_ANTHROPIC_MODEL))


@pytest.mark.asyncio
async def test_openai_live() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        _missing("OPENAI_API_KEY not set")
    await _assert_structured_roundtrip(OpenAIProvider(api_key=key, model=_OPENAI_MODEL))


@pytest.mark.asyncio
async def test_deepseek_live() -> None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        _missing("DEEPSEEK_API_KEY not set")
    # Mode.JSON (not JSON_SCHEMA, which DeepSeek's API rejects) validates on
    # both deepseek-chat and deepseek-reasoner; the default smoke model is
    # deepseek-chat (V3). reasoning_effort is omitted: it's only meaningful for
    # deepseek-reasoner, and the provider forwards it when configured.
    await _assert_structured_roundtrip(DeepSeekProvider(api_key=key, model=_DEEPSEEK_MODEL))


@pytest.mark.asyncio
async def test_ollama_live() -> None:
    if not _ollama_up():
        _missing(f"no Ollama server at {_OLLAMA_HOST} (run `ollama serve`)")
    await _assert_structured_roundtrip(OllamaProvider(base_url=_OLLAMA_HOST, model=_OLLAMA_MODEL))
