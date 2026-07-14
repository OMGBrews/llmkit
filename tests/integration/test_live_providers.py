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
    Bedrock     AWS chain + AWS_REGION_NAME https://console.aws.amazon.com  (see below)
    Vertex      Google ADC + VERTEXAI_*     https://console.cloud.google.com  (see below)

Ollama reads no key — it needs a reachable server instead. Point ``OLLAMA_HOST``
at it (default ``http://localhost:11434``); in the maintainer devcontainer
Ollama runs on the host, reached at ``http://host.docker.internal:11434``.

Bedrock is the one deliberate exception to "every live test must pass under
``--run-live``". It needs both an AWS account *and* the optional ``boto3``
dependency, which ships only with the ``omg-llmkit[bedrock]`` extra. Its test
is therefore gated with ``pytest.importorskip("boto3")`` — a *structural*
skip (the dependency isn't installed), not the env-driven mode switch this
module otherwise forbids. Install the extra (``uv sync --extra dev --extra
bedrock``) and export ``AWS_REGION_NAME`` plus ambient AWS credentials to run
it; once boto3 is present, a missing region/credential is a hard failure like
every other provider, not a skip. Secrets come from the AWS chain — they are
never passed through ``LLMClientConfig``.

Vertex AI is the same kind of structural exception. It needs both a GCP project
*and* the optional ``google-auth`` dependency (``omg-llmkit[vertex]``), so its
test is gated with ``pytest.importorskip("google.auth")``. Install the extra
(``uv sync --extra dev`` already pulls it) and export ``VERTEXAI_PROJECT`` /
``VERTEXAI_LOCATION`` plus resolvable Application Default Credentials
(``gcloud auth application-default login``) to run it; once google-auth is
present, a missing project/location/credential is a hard failure like every
other provider, not a skip. Credentials come from the ADC chain — they are
never passed through ``LLMClientConfig``; ``vertex_location`` selects the
data-residency region. This is also where the ``Mode.JSON_SCHEMA`` pin for
Vertex Gemini is *measured*, not trusted.
"""

from __future__ import annotations

import os
import socket
from typing import NoReturn, Protocol, cast

import pytest
from pydantic import BaseModel

from llmkit import (
    LLMProviderInterface,
    structured_llm_call,
)
from llmkit.providers import (
    AnthropicProvider,
    BedrockProvider,
    DeepSeekProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VertexProvider,
)

# Every test in this module makes a real API call. The marker is what the
# ``--run-live`` switch in conftest.py keys off of: without the flag these are
# skipped at collection (deterministically, regardless of any credentials in the
# environment); with it, they all run and must pass.
pytestmark = pytest.mark.live


class _Boto3Session(Protocol):
    """The slice of ``boto3.Session`` the Bedrock credential probe touches."""

    def get_credentials(self) -> object | None: ...


class _Boto3Module(Protocol):
    """Typed view of the dynamically-imported ``boto3`` module.

    ``pytest.importorskip`` returns ``Any``; narrowing it to this Protocol keeps
    the credential check below precisely typed instead of leaking ``reportAny``.
    """

    def Session(self) -> _Boto3Session: ...


class _GoogleAuthModule(Protocol):
    """Typed view of the dynamically-imported ``google.auth`` module.

    ``google.auth.default()`` resolves the ambient Application Default
    Credentials, returning ``(credentials, project_id)`` or raising when none
    resolve. Narrowing the ``importorskip`` result to this Protocol keeps the
    credential probe precisely typed instead of leaking ``reportAny``.
    """

    def default(self) -> tuple[object, object]: ...


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
# OpenRouter is deliberately covered TWICE — see the two tests below. This
# constant is the *override* half: an open model hosted by many OpenRouter
# providers (so it won't 404 when a single vendor slug is retired) and the one
# measured to stochastically echo the schema under a non-strict json_schema, so
# it doubles as the canary for the provider's strict-enforcement upgrade. The
# provider's own default model is covered separately, and on purpose is NOT
# overridable from the environment.
_OPENROUTER_MODEL = os.getenv("OPENROUTER_SMOKE_MODEL", "mistralai/mistral-nemo")
_GOOGLE_MODEL = os.getenv("GOOGLE_SMOKE_MODEL", "gemini-2.5-flash-lite")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_SMOKE_MODEL", "claude-haiku-4-5-20251001")
_OPENAI_MODEL = os.getenv("OPENAI_SMOKE_MODEL", "gpt-4.1-mini")
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_SMOKE_MODEL", "deepseek-chat")
# The provider's default: Claude Haiku 4.5 via its cross-region inference
# profile. Override with another profile- or partition-prefixed id (e.g.
# ``eu.anthropic.claude-...``) when the account routes elsewhere.
_BEDROCK_MODEL = os.getenv("BEDROCK_SMOKE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
# Gemini on Vertex: same model family as the AI Studio smoke model. Override
# with another Vertex-available Gemini id if the project lacks access.
_VERTEX_MODEL = os.getenv("VERTEX_SMOKE_MODEL", "gemini-2.5-flash-lite")
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
    pairs accept it (Gemini's 2.5 thinking models do; e.g. Mistral's models
    do not, and litellm raises rather than drop it). Default ``None`` sends
    nothing, so the call works everywhere.

    ``max_tokens`` is capped: OpenRouter checks *affordability* against the
    requested budget, and with no cap that budget defaults to the model's
    full context window — so a low account balance rejects the call with a
    402 that names credits, not the real coupling (observed 2026-07-14:
    "You requested up to 65536 tokens, but can only afford 12176"). The cap
    makes the gate's verdict depend on the pins, not the balance. 1024 is
    ~15x what the smoke schema needs, so it can't cause the truncation
    fail-fast (``OutputLimitError``) on a healthy provider.

    Every field of :class:`CountryProfile` is asserted: a half-working mode
    that empties or drops any field fails *deterministically* here rather
    than flaking green on the one easy field.
    """
    result = await structured_llm_call(
        # An extraction with the answer fully specified, so the result is
        # deterministic and a dropped/empty field is unambiguously wrong.
        "Extract the following facts about France into the schema. "
        + "Country: France. Capital: Paris. Continent: Europe. "
        + "Member of the European Union: yes. Number of bordering countries: 8. "
        + "Three largest cities by population: Paris, Marseille, Lyon.",
        CountryProfile,
        feature="integration-smoke",
        label=provider.name,
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_tokens=1024,
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
    """OpenRouter on the *override* model — the strict-enforcement canary.

    ``mistralai/mistral-nemo`` is the model measured to stochastically echo the
    schema back (values nested under ``properties``) when the ``json_schema``
    ``response_format`` is sent non-strict, which is what instructor's core
    ``JSON_SCHEMA`` handler does. So this is the test that fails if the
    provider's ``strict_json_schema`` upgrade ever regresses. It is *not*
    redundant with the default-model test below: that one pins the model id,
    this one pins the wire shape.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        _missing("OPENROUTER_API_KEY not set")
    await _assert_structured_roundtrip(OpenRouterProvider(api_key=key, model=_OPENROUTER_MODEL))


@pytest.mark.asyncio
async def test_openrouter_live_default_model() -> None:
    """OpenRouter on its **own default** model — no ``model=``, no env override.

    This test exists because ``OpenRouterProvider._default_model`` was
    ``google/gemini-2.0-flash-001`` for several releases *after* OpenRouter
    retired it: the slug left the catalog, so every call that took the default
    died with ``NotFoundError: No endpoints found``, and no consumer could
    construct the provider without naming a model. The live suite never caught
    it because every OpenRouter test overrode the model.

    Passing no ``model`` is therefore the entire point — the provider must
    resolve its own default, so a retired default fails *this* release gate
    instead of shipping. For the same reason there is deliberately no
    ``*_SMOKE_MODEL`` env override here: one that could repoint this call at a
    model the default isn't would restore the exact blind spot it closes.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        _missing("OPENROUTER_API_KEY not set")
    await _assert_structured_roundtrip(OpenRouterProvider(api_key=key))


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


@pytest.mark.asyncio
async def test_bedrock_live() -> None:
    # boto3 ships only with the optional `omg-llmkit[bedrock]` extra. Skipping
    # when it's absent is a *structural* gate (the dependency isn't installed),
    # not the env-driven mode switch this module forbids — and Bedrock uniquely
    # needs both that extra and an AWS account, which plain `--run-live` cannot
    # assume. Once the extra IS installed, the usual hard-fail rule applies:
    # a missing region/credential fails the test rather than skipping it.
    boto3 = cast(
        "_Boto3Module",
        pytest.importorskip("boto3", reason="install omg-llmkit[bedrock] for live Bedrock"),
    )
    region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_REGION")
    if not region:
        _missing("AWS_REGION_NAME (or AWS_REGION) not set")
    if boto3.Session().get_credentials() is None:
        _missing("no resolvable AWS credentials in the chain (env / profile / role)")
    # Secrets resolve from the ambient AWS chain; only the region is explicit.
    await _assert_structured_roundtrip(
        BedrockProvider(model=_BEDROCK_MODEL, aws_region_name=region)
    )


@pytest.mark.asyncio
async def test_vertex_live() -> None:
    # google-auth ships only with the optional `omg-llmkit[vertex]` extra.
    # Skipping when it's absent is a *structural* gate (the dependency isn't
    # installed), not the env-driven mode switch this module forbids — and Vertex
    # uniquely needs both that extra and a GCP project, which plain `--run-live`
    # cannot assume. Once the extra IS installed, the usual hard-fail rule
    # applies: a missing project/location/credential fails the test, not skips it.
    google_auth = cast(
        "_GoogleAuthModule",
        pytest.importorskip("google.auth", reason="install omg-llmkit[vertex] for live Vertex"),
    )
    project = os.getenv("VERTEXAI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEXAI_LOCATION")
    if not project:
        _missing("VERTEXAI_PROJECT (or GOOGLE_CLOUD_PROJECT) not set")
    if not location:
        _missing("VERTEXAI_LOCATION not set (it selects the data-residency region)")
    try:
        # Only the no-raise matters: this resolves the ambient ADC or errors.
        _ = google_auth.default()
    except Exception:  # google.auth.exceptions.DefaultCredentialsError and friends
        _missing("no resolvable Google ADC (run `gcloud auth application-default login`)")
    # Credentials resolve from the ambient ADC chain; only project + the
    # residency location are explicit. Gemini 2.5 thinks by default — disable it
    # so the smoke call doesn't burn reasoning tokens on a trivial prompt. This
    # call is what *measures* the Mode.JSON_SCHEMA pin against live Vertex.
    await _assert_structured_roundtrip(
        VertexProvider(model=_VERTEX_MODEL, vertex_project=project, vertex_location=location),
        reasoning_effort="disable",
    )
