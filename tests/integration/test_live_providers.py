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

Set ``LLMKIT_REQUIRE_ALL_PROVIDERS=1`` to flip every skip into a hard
**failure**. A silent skip is indistinguishable from a pass at the summary
line, so a typo'd key or a forgotten provider would otherwise leave a provider
untested while the run still looks green. Maintainers export every credential
(and start a local Ollama server) and set this flag as a release gate, so
"tested against all supported providers" is enforced rather than hoped for.
Default-off keeps the contributor/CI experience unchanged.

Run just these (after exporting keys — see ``docs/operations`` in the
maintainer workspace, or the table below):

    OPENROUTER_API_KEY=sk-or-...  \
    GEMINI_API_KEY=...            \
    ANTHROPIC_API_KEY=...         \
    OPENAI_API_KEY=sk-...         \
    DEEPSEEK_API_KEY=sk-...       \
    uv run pytest tests/integration -v

Provider -> credential it reads:

    OpenRouter  OPENROUTER_API_KEY          https://openrouter.ai/keys
    Google      GEMINI_API_KEY              https://aistudio.google.com/apikey
    Anthropic   ANTHROPIC_API_KEY           https://console.anthropic.com/settings/keys
    OpenAI      OPENAI_API_KEY              https://platform.openai.com/api-keys
    DeepSeek    DEEPSEEK_API_KEY            https://platform.deepseek.com/api_keys
    Ollama      (local server on :11434)    https://ollama.com  (no key)
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

# Maintainer release gate: when set, a missing key/server is a hard failure
# instead of a silently-green skip, so the release can't ship with a provider
# left untested. Off by default — contributors and CI without keys stay green.
_REQUIRE_ALL = os.getenv("LLMKIT_REQUIRE_ALL_PROVIDERS", "") not in ("", "0", "false", "False")


def _unavailable(reason: str) -> NoReturn:
    """Skip this provider, or fail it under ``LLMKIT_REQUIRE_ALL_PROVIDERS``.

    Always raises (both branches call out of ``pytest``), so callers can treat
    it like ``pytest.skip`` for type-narrowing a just-checked key to non-None.
    """
    if _REQUIRE_ALL:
        pytest.fail(f"{reason} — required by LLMKIT_REQUIRE_ALL_PROVIDERS")
    pytest.skip(reason)


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
        _unavailable("OPENROUTER_API_KEY not set")
    await _assert_structured_roundtrip(OpenRouterProvider(api_key=key, model=_OPENROUTER_MODEL))


@pytest.mark.asyncio
async def test_google_live() -> None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        _unavailable("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
    # Gemini 2.5 thinks by default; disable it so the smoke call doesn't
    # burn reasoning tokens on a trivial prompt.
    await _assert_structured_roundtrip(
        GoogleProvider(api_key=key, model=_GOOGLE_MODEL), reasoning_effort="disable"
    )


@pytest.mark.asyncio
async def test_anthropic_live() -> None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        _unavailable("ANTHROPIC_API_KEY not set")
    await _assert_structured_roundtrip(AnthropicProvider(api_key=key, model=_ANTHROPIC_MODEL))


@pytest.mark.asyncio
async def test_openai_live() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        _unavailable("OPENAI_API_KEY not set")
    await _assert_structured_roundtrip(OpenAIProvider(api_key=key, model=_OPENAI_MODEL))


@pytest.mark.asyncio
async def test_deepseek_live() -> None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        _unavailable("DEEPSEEK_API_KEY not set")
    # Mode.JSON (not JSON_SCHEMA, which DeepSeek's API rejects) validates on
    # both deepseek-chat and deepseek-reasoner; the default smoke model is
    # deepseek-chat (V3). reasoning_effort is omitted: it's only meaningful for
    # deepseek-reasoner, and the provider forwards it when configured.
    await _assert_structured_roundtrip(DeepSeekProvider(api_key=key, model=_DEEPSEEK_MODEL))


@pytest.mark.asyncio
async def test_ollama_live() -> None:
    if not _ollama_up():
        _unavailable("no Ollama server on localhost:11434 (run `ollama serve`)")
    await _assert_structured_roundtrip(OllamaProvider(model=_OLLAMA_MODEL))
