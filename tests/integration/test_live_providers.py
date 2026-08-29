"""Live, end-to-end smoke tests against every supported provider.

Unlike the mocked unit tests, these make a *real* structured call through
each provider family and assert the library gets a validated Pydantic
instance back — the one thing mocks can never prove: that our model
strings, ``instructor.Mode`` pinning, and credential kwargs are actually
accepted by the provider's API.

Two kinds of test live here, and the difference is the model id:

* ``test_live_default_model`` is parametrized over the whole :class:`Provider`
  enum and passes **no** ``model=``, so each provider resolves its own
  ``_default_model``. This is the gate that a default the vendor has retired
  cannot ship (it is the one OpenRouter's dead ``google/gemini-2.0-flash-001``
  slipped through for several releases). Its model ids are deliberately *not*
  env-overridable.
* The per-provider ``test_<name>_live`` tests drive an explicit, env-overridable
  ``*_SMOKE_MODEL``. They pin wire-level behaviour (OpenRouter's strict
  ``json_schema`` enforcement, Vertex's ``Mode.JSON_SCHEMA``) and give a
  maintainer whose key lacks access to a default a way to still exercise the
  provider.

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
module otherwise forbids. ``uv sync`` installs boto3 (the ``dev`` dependency
group pulls the ``[bedrock]`` extra); export ``AWS_REGION_NAME`` plus ambient
AWS credentials to run
it; once boto3 is present, a missing region/credential is a hard failure like
every other provider, not a skip. Secrets come from the AWS chain — they are
never passed through ``LLMClientConfig``.

Vertex AI is the same kind of structural exception. It needs both a GCP project
*and* the optional ``google-auth`` dependency (``omg-llmkit[vertex]``), so its
test is gated with ``pytest.importorskip("google.auth")``. Install the extra
(``uv sync`` already pulls it) and export ``VERTEXAI_PROJECT`` /
``VERTEXAI_LOCATION`` plus resolvable Application Default Credentials
(``gcloud auth application-default login``) to run it; once google-auth is
present, a missing project/location/credential is a hard failure like every
other provider, not a skip. Credentials come from the ADC chain — they are
never passed through ``LLMClientConfig``; ``vertex_location`` selects the
data-residency region. This is also where the ``Mode.JSON_SCHEMA`` pin for
Vertex Gemini is *measured*, not trusted.
"""

from __future__ import annotations

import enum
import os
import socket
from typing import NoReturn, Protocol, TypedDict, assert_never, cast

import pytest
from pydantic import BaseModel

from llmkit import (
    ChatMessage,
    LLMProviderInterface,
    Provider,
    ToolDefinition,
    ToolName,
    model_from_json_schema,
    structured_llm_call,
    tool_llm_call,
    tool_result_message,
)
from llmkit.providers import make_provider

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


class _AdditionArgs(BaseModel):
    """Arguments for the deliberately tiny, deterministic live tool."""

    left: int
    right: int


class _ToolAnswer(BaseModel):
    total: int


class _TemperatureUnit(enum.StrEnum):
    """A string enum: the one enum shape Gemini's ``Schema`` subset accepts."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class _ForecastLocation(BaseModel):
    city: str
    country: str


class _ForecastArgs(BaseModel):
    """A nested model plus a string enum — the shape that carries ``$defs``.

    ``model_json_schema()`` renders ``where`` and ``unit`` as ``$ref``s into a
    ``$defs`` block, and llmkit forwards ``parameters`` verbatim, so this is the
    schema whose survival to the wire the Gemini guarantee is about.
    """

    where: _ForecastLocation
    unit: _TemperatureUnit
    days: int


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


class _LiveCredentials(TypedDict, total=False):
    """The credential kwargs a live provider needs, as accepted by ``make_provider``.

    Deliberately excludes ``model``: these are the *credentials only*, so the
    same resolved value can build either a default-model provider or an
    override-model one without a caller being able to smuggle a model in.
    """

    api_key: str
    base_url: str
    aws_region_name: str
    vertex_project: str
    vertex_location: str


def _require_env(*names: str) -> str:
    """First non-empty value among ``names``; fail the live test if none is set."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    _missing(f"{' / '.join(names)} not set")


def _live_credentials(provider: Provider) -> _LiveCredentials:
    """Resolve one provider's real credentials, failing loudly when any is absent.

    The **single** place the live suite probes for a credential — both the
    default-model gate and the override-model tests below build from this, so
    the two can never disagree about what a provider needs.

    The ``match`` has no catch-all and ends in :func:`typing.assert_never`, which
    is what makes the default-model coverage *structural* rather than a list
    someone must remember to extend: a newly-added :class:`Provider` member with
    no ``case`` here is a basedpyright error, and an unwired one raises rather
    than silently resolving to nothing.
    """
    match provider:
        case Provider.OPENROUTER:
            return {"api_key": _require_env("OPENROUTER_API_KEY")}
        case Provider.OLLAMA:
            if not _ollama_up():
                _missing(f"no Ollama server at {_OLLAMA_HOST} (run `ollama serve`)")
            return {"base_url": _OLLAMA_HOST}
        case Provider.GOOGLE:
            return {"api_key": _require_env("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        case Provider.ANTHROPIC:
            return {"api_key": _require_env("ANTHROPIC_API_KEY")}
        case Provider.OPENAI:
            return {"api_key": _require_env("OPENAI_API_KEY")}
        case Provider.DEEPSEEK:
            return {"api_key": _require_env("DEEPSEEK_API_KEY")}
        case Provider.BEDROCK:
            # boto3 ships only with the optional `omg-llmkit[bedrock]` extra.
            # Skipping when it's absent is a *structural* gate (the dependency
            # isn't installed), not the env-driven mode switch this module
            # forbids — Bedrock uniquely needs both that extra and an AWS
            # account, which plain `--run-live` cannot assume. Once the extra IS
            # installed the usual hard-fail rule applies: a missing region or
            # credential fails the test rather than skipping it.
            boto3 = cast(
                "_Boto3Module",
                pytest.importorskip("boto3", reason="install omg-llmkit[bedrock] for live Bedrock"),
            )
            region = _require_env("AWS_REGION_NAME", "AWS_REGION")
            if boto3.Session().get_credentials() is None:
                _missing("no resolvable AWS credentials in the chain (env / profile / role)")
            # Secrets resolve from the ambient AWS chain; only the region is explicit.
            return {"aws_region_name": region}
        case Provider.VERTEX:
            # Same structural gate as Bedrock: google-auth ships only with the
            # optional `omg-llmkit[vertex]` extra, and Vertex also needs a GCP
            # project that `--run-live` cannot assume.
            google_auth = cast(
                "_GoogleAuthModule",
                pytest.importorskip(
                    "google.auth", reason="install omg-llmkit[vertex] for live Vertex"
                ),
            )
            project = _require_env("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT")
            location = os.getenv("VERTEXAI_LOCATION")
            if not location:
                _missing("VERTEXAI_LOCATION not set (it selects the data-residency region)")
            try:
                # Only the no-raise matters: this resolves the ambient ADC or errors.
                _ = google_auth.default()
            except Exception:  # google.auth.exceptions.DefaultCredentialsError and friends
                _missing("no resolvable Google ADC (run `gcloud auth application-default login`)")
            # Credentials resolve from the ambient ADC chain; only project + the
            # residency location are explicit.
            return {"vertex_project": project, "vertex_location": location}
    assert_never(provider)


# Gemini 2.5 (AI Studio *and* Vertex) thinks by default; disable it so the smoke
# call doesn't burn reasoning tokens on a trivial extraction. Every other
# provider sends nothing — some models reject the knob outright.
_REASONING_EFFORT: dict[Provider, str] = {
    Provider.GOOGLE: "disable",
    Provider.VERTEX: "disable",
}


async def _override_roundtrip(provider: Provider, model: str) -> None:
    """Drive one real structured call on an explicitly-named (``*_SMOKE_MODEL``) model.

    The override half of the split: this is what a maintainer whose key lacks
    access to a provider's default can repoint, and what pins the *wire shape*
    (see ``test_openrouter_live``). The model id it drives is deliberately
    env-overridable — which is exactly why it can never stand in for
    :func:`test_live_default_model`, whose model id must not be.
    """
    built = make_provider(provider, model=model, **_live_credentials(provider))
    await _assert_structured_roundtrip(built, reasoning_effort=_REASONING_EFFORT.get(provider))


async def _assert_tool_roundtrip(provider: LLMProviderInterface) -> None:
    """Offer a tool, return its result, then require a natural-language finish."""
    add = ToolDefinition.from_model("add", _AdditionArgs, "Add two integer values.")
    history: list[ChatMessage] = [
        {"role": "user", "content": "Use the add tool to calculate 2 plus 3."}
    ]
    requested = await tool_llm_call(
        history,
        [add],
        feature="integration-tool-smoke",
        label=provider.name,
        provider=provider,
        tool_choice=ToolName("add"),
        max_tokens=256,
    )
    assert len(requested.tool_calls) == 1
    call = requested.tool_calls[0]
    assert call.name == "add"
    assert call.validated == _AdditionArgs(left=2, right=3)
    history.extend([requested.to_message(), tool_result_message(call.id, "5")])
    finished = await tool_llm_call(
        history,
        [add],
        feature="integration-tool-smoke",
        label=provider.name,
        provider=provider,
        tool_choice="none",
        max_tokens=256,
    )
    assert finished.tool_calls == []
    assert finished.text is not None
    assert "5" in finished.text


async def _assert_unnormalised_schema_roundtrip(provider: LLMProviderInterface) -> None:
    """Two schemas llmkit does *not* pre-process still reach the provider intact.

    The tool lane forwards ``ToolDefinition.parameters`` verbatim, so a
    ``from_model()`` schema arrives at the transport carrying ``$defs``,
    ``$ref`` and ``additionalProperties`` — none of which Gemini's ``Schema``
    subset accepts. ``tests/providers/test_gemini_tool_schema_transport.py``
    pins that LiteLLM normalises them; this asserts the *provider* accepts the
    result, which is the half no offline test can prove. Both tools are offered
    in one request, so the no-argument declaration and the nested-model one are
    measured on the same wire payload.

    A regression here is a 400 from the provider, not a wrong answer — the
    assertions are deliberately about the call being made, not about forecasting.
    """
    forecast = ToolDefinition.from_model(
        "forecast", _ForecastArgs, "Look up a weather forecast for a city."
    )
    # The portable no-argument schema, documented in the README: an empty
    # ``properties`` map, which the transport reduces to a bare OBJECT.
    clock = ToolDefinition(
        "current_time", "Get the current time.", {"type": "object", "properties": {}}
    )

    requested = await tool_llm_call(
        [
            {
                "role": "user",
                "content": ("Use the forecast tool for Ottawa, Canada, in celsius, for 3 days."),
            }
        ],
        [forecast, clock],
        feature="integration-tool-schema-smoke",
        label=provider.name,
        provider=provider,
        tool_choice=ToolName("forecast"),
        max_tokens=256,
    )

    assert len(requested.tool_calls) == 1
    call = requested.tool_calls[0]
    assert call.name == "forecast"
    # The nested model and the enum survived the transform semantically: the
    # provider filled both, and llmkit validated the result against the model.
    assert isinstance(call.validated, _ForecastArgs)
    assert call.validated.where.city.lower() == "ottawa"
    assert call.validated.unit is _TemperatureUnit.CELSIUS

    # The no-arg tool travelled in the same payload; ask for it explicitly to
    # prove its bare OBJECT declaration is callable, not merely accepted.
    history: list[ChatMessage] = [{"role": "user", "content": "What time is it right now?"}]
    timed = await tool_llm_call(
        history,
        [forecast, clock],
        feature="integration-tool-schema-smoke",
        label=provider.name,
        provider=provider,
        tool_choice=ToolName("current_time"),
        max_tokens=256,
    )
    assert [requested_call.name for requested_call in timed.tool_calls] == ["current_time"]
    assert timed.tool_calls[0].arguments == {}


async def _assert_compose_roundtrip(provider: LLMProviderInterface) -> None:
    """One request can ask for a tool or return a validated final answer."""
    add = ToolDefinition.from_model("add", _AdditionArgs, "Add two integer values.")
    result = await tool_llm_call(
        "Return the final answer 5 as JSON. Do not call a tool.",
        [add],
        feature="integration-compose-smoke",
        label=provider.name,
        provider=provider,
        tool_choice="none",
        output_schema=_ToolAnswer,
        max_tokens=128,
    )
    assert result.tool_calls == []
    assert result.parsed == _ToolAnswer(total=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [Provider.ANTHROPIC, Provider.OPENAI])
async def test_live_compose_tools_with_structured_final(provider: Provider) -> None:
    """The two measured compose routes produce a validated final answer."""
    built = make_provider(provider, **_live_credentials(provider))
    await _assert_compose_roundtrip(built)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(Provider))
async def test_live_default_model(provider: Provider) -> None:
    """Every provider's own ``_default_model`` survives a real structured call.

    This test exists because ``OpenRouterProvider._default_model`` was
    ``google/gemini-2.0-flash-001`` for several releases *after* OpenRouter
    retired it: the slug left the catalog, so every call that took the default
    died with ``NotFoundError: No endpoints found``, and no consumer could
    construct the provider without naming a model. The live suite never caught
    it because every OpenRouter test passed an explicit ``model=``.

    Passing **no model** is therefore the entire point — each provider must
    resolve its own default, so a default the vendor has retired fails *this*
    release gate instead of shipping. For the same reason there is deliberately
    no ``*_SMOKE_MODEL`` override honoured here: one that could repoint this call
    at a model the default isn't would restore the exact blind spot it closes.
    The override models live in the per-provider tests below.

    Parametrizing over :class:`Provider` (rather than hand-writing one test per
    provider) is what keeps the coverage structural: a provider added later is
    picked up here the moment its enum member exists, and :func:`_live_credentials`
    will not type-check until someone says how to authenticate it.
    """
    built = make_provider(provider, model=None, **_live_credentials(provider))
    try:
        await _assert_structured_roundtrip(built, reasoning_effort=_REASONING_EFFORT.get(provider))
    except Exception as exc:
        # Vendors name a dead model inconsistently (OpenRouter's was a bare
        # "404 No endpoints found"). Naming it ourselves means a red gate always
        # says *which* id to go replace, whatever the provider chose to say.
        exc.add_note(
            f"This call used {provider.value}'s own _default_model, which resolved "
            + f"to {built.litellm_model()!r}. If the vendor has retired that id, "
            + f"replace {type(built).__name__}._default_model."
        )
        raise


@pytest.mark.asyncio
async def test_openrouter_strict_schema_ref_hygiene_live() -> None:
    """A ``model_from_json_schema`` model with described enum + object fields
    survives OpenAI's strict ``response_format`` validator.

    Regression for the 0.7.0 emission: the enum factored into ``$defs`` with
    the ``description`` beside the ``$ref``, and every OpenAI-served model
    rejected the call with a 400 (``$ref cannot have keywords
    {'description'}``) — reported by found-in-words, whose judge rubrics all
    use exactly this shape via ``openai/gpt-4o-mini`` on OpenRouter. The
    model id is deliberately NOT env-overridable: it must stay an
    OpenAI-served slug, because OpenAI's strict validator (reached through
    OpenRouter's ``strict_json_schema`` upgrade) is the thing under test.
    Non-OpenAI models accept the broken shape, which is what masked the bug.
    """
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recognizability_score": {
                "type": "integer",
                "enum": [-1, 1, 2, 3, 4, 5],
                "description": "How recognizable the phrase is; -1 means not a real phrase.",
            },
            "assessment": {
                "type": "object",
                "description": "Structured detail behind the score.",
                "additionalProperties": False,
                "properties": {
                    "confident": {
                        "type": "boolean",
                        "description": "Whether the judgement is confident.",
                    }
                },
                "required": ["confident"],
            },
        },
        "required": ["recognizability_score", "assessment"],
    }
    judgment = model_from_json_schema(schema, name="phrase_judgment")
    provider = make_provider(
        Provider.OPENROUTER,
        model="openai/gpt-4o-mini",
        **_live_credentials(Provider.OPENROUTER),
    )
    result = await structured_llm_call(
        # Dictate the answer so the assertion is deterministic: a 400 from the
        # schema validator is the failure this test exists to catch, and a
        # correctly-parsed dictated value proves the round trip.
        "Judge the phrase 'the quick brown fox'. It is extremely recognizable: "
        + "score it exactly 5, and mark the assessment as confident.",
        judgment,
        feature="integration-smoke",
        label="openrouter-strict-ref-hygiene",
        provider=provider,
        max_tokens=1024,
    )
    assert result.model_dump() == {
        "recognizability_score": 5,
        "assessment": {"confident": True},
    }


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
    await _override_roundtrip(Provider.OPENROUTER, _OPENROUTER_MODEL)


@pytest.mark.asyncio
async def test_openrouter_reasoning_effort_live() -> None:
    """OpenRouter accepts native translated disable controls on Gemini.

    The ids are deliberately hard-coded: Gemini 3.5 Flash is the
    mandatory-thinking incident model, while Gemini 2.5 Flash-Lite exercises
    OpenRouter's true ``none`` branch. The offline routing tests capture the
    corresponding outgoing native bodies; this smoke proves each reaches the
    gateway and returns visible structured output under the small shared cap.
    """
    for model in ("google/gemini-3.5-flash", "google/gemini-2.5-flash-lite"):
        provider = make_provider(
            Provider.OPENROUTER, model=model, **_live_credentials(Provider.OPENROUTER)
        )
        await _assert_structured_roundtrip(provider, reasoning_effort="disable")


@pytest.mark.asyncio
async def test_google_live() -> None:
    await _override_roundtrip(Provider.GOOGLE, _GOOGLE_MODEL)


@pytest.mark.asyncio
async def test_anthropic_live() -> None:
    await _override_roundtrip(Provider.ANTHROPIC, _ANTHROPIC_MODEL)


@pytest.mark.asyncio
async def test_openai_live() -> None:
    await _override_roundtrip(Provider.OPENAI, _OPENAI_MODEL)


@pytest.mark.asyncio
async def test_deepseek_live() -> None:
    # Mode.JSON (not JSON_SCHEMA, which DeepSeek's API rejects) validates on
    # both deepseek-chat and deepseek-reasoner; the default smoke model is
    # deepseek-chat (V3). reasoning_effort is omitted: it's only meaningful for
    # deepseek-reasoner, and the provider forwards it when configured.
    await _override_roundtrip(Provider.DEEPSEEK, _DEEPSEEK_MODEL)


@pytest.mark.asyncio
async def test_ollama_live() -> None:
    await _override_roundtrip(Provider.OLLAMA, _OLLAMA_MODEL)


@pytest.mark.asyncio
async def test_bedrock_live() -> None:
    await _override_roundtrip(Provider.BEDROCK, _BEDROCK_MODEL)


@pytest.mark.asyncio
async def test_vertex_live() -> None:
    # This call is what *measures* the Mode.JSON_SCHEMA pin against live Vertex.
    await _override_roundtrip(Provider.VERTEX, _VERTEX_MODEL)


@pytest.mark.asyncio
async def test_anthropic_tool_roundtrip_live() -> None:
    provider = make_provider(
        Provider.ANTHROPIC, model=_ANTHROPIC_MODEL, **_live_credentials(Provider.ANTHROPIC)
    )
    await _assert_tool_roundtrip(provider)


@pytest.mark.asyncio
async def test_vertex_tool_roundtrip_live() -> None:
    provider = make_provider(
        Provider.VERTEX, model=_VERTEX_MODEL, **_live_credentials(Provider.VERTEX)
    )
    await _assert_tool_roundtrip(provider)


@pytest.mark.asyncio
async def test_vertex_tool_schema_roundtrip_live() -> None:
    """Gemini accepts a ``from_model`` schema and a no-arg tool, unprocessed.

    The Vertex leg of the guarantee README states: a consumer hands llmkit a
    Pydantic model with a nested model and an enum, or a no-argument tool, and
    needs no schema pre-processing of their own. Measured on the provider family
    that would reject the raw schema, rather than inferred from Anthropic/OpenAI,
    which accept ``$ref``/``$defs`` natively and so prove nothing here.
    """
    provider = make_provider(
        Provider.VERTEX, model=_VERTEX_MODEL, **_live_credentials(Provider.VERTEX)
    )
    await _assert_unnormalised_schema_roundtrip(provider)
