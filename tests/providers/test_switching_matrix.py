"""One unchanged structured call site, switched across every provider by config alone.

"Switching the provider for a whole program is a config edit" is pinned here
offline: the same call site, ``structured_llm_call("q", _Answer, feature="f")``,
runs once per :class:`~llmkit.Provider` member through the
:mod:`llmkit.testing` fake, and the only thing that changes between rows is
the ``provider`` field of the registered :class:`~llmkit.LLMClientConfig`.
The environment is the same for every row (every bearer key set), the
model is left unset so each route resolves its own default, and
``reasoning_effort`` is left unset because ``"disable"`` raises on four
routes' default models; per-route reasoning translation is pinned in
``test_portable_reasoning_every_route.py``.

Each row then asserts that the switch carried the route's own decisions with
it, against a hard-coded expected table rather than values read back from the
provider classes, so a changed pin fails here:

- the routed model id the fake received (``<prefix><default model>``);
- the default model itself, on the built instance and on the log record,
  which carries the unprefixed id;
- the provider display name on the request, the charged rate-limit key and
  the log record (``"AWS Bedrock"`` and ``"Google Agent Platform"``, not the
  enum values);
- the structured-output mode and the strict-schema flag of the instance the
  call actually ran on, captured by wrapping ``build_provider``. The fake's
  :class:`~llmkit.testing.TransportRequest` carries neither trait.

The expected modes and strict flags come from the per-route table in the
structured-output contract audit record (wrapper
``docs/planning/structured-output-contract-audit-2026-11-02.md``), which
agrees with the code at the time of writing: ``Mode.JSON`` on DeepSeek,
``Mode.JSON_SCHEMA`` everywhere else (Gemini routes on their default
``"schema"`` strategy), and the strict upgrade on OpenRouter only.

Bedrock and Vertex need no route field to construct: ``aws_region_name``,
``vertex_project`` and ``vertex_location`` are optional. Unset, Bedrock's
region comes from ``AWS_REGION_NAME`` or ``AWS_REGION`` through the AWS
credential chain; Vertex's project comes from ``VERTEXAI_PROJECT``, else the
Application Default Credentials project, and its location from
``VERTEXAI_LOCATION``, else LiteLLM's ``us-central1``. So the one
provider-only row per member is the literal "only ``provider`` changed" switch, and two
extra rows set those fields as a deployed host usually would, to show they
change nothing the switch is meant to carry. A guard on the rows,
``test_the_matrix_covers_every_provider``, holds both halves of that: every
member has an expected route, and every member's own row is exactly
``LLMClientConfig(provider=member)``.

Mutation check: with ``DeepSeekProvider._mode`` set to ``Mode.JSON_SCHEMA``,
the DeepSeek row fails on the mode assertion; with
``AnthropicProvider._default_model`` set to ``"claude-haiku-4-5"``, the
Anthropic row fails on the routed model id; with
``OpenRouterProvider._strict_json_schema`` set to ``False``, the OpenRouter
row fails on the strict flag; with the DeepSeek row given an ``api_key``,
which changes nothing the row asserts, only the guard on the rows fails.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import instructor
import pytest
from pydantic import BaseModel

import llmkit.providers
from llmkit import (
    LLMClientConfig,
    Provider,
    capture_llm_records,
    configure_llm_client,
    structured_llm_call,
)
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import GlobalRateLimiter, RateLimitSlot
from llmkit.testing import Reply, install


class _Answer(BaseModel):
    ok: bool


async def _the_call_site() -> _Answer:
    """The one call site every row runs, unchanged. It never names a vendor."""
    return await structured_llm_call("q", _Answer, feature="f")


@dataclass(frozen=True)
class _Route:
    """What a switch to one provider must carry with it."""

    display_name: str
    prefix: str
    default_model: str
    mode: instructor.Mode
    strict_json_schema: bool


_EXPECTED: dict[Provider, _Route] = {
    Provider.ANTHROPIC: _Route(
        "Anthropic", "anthropic/", "claude-sonnet-4-6", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.BEDROCK: _Route(
        "AWS Bedrock",
        "bedrock/",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        instructor.Mode.JSON_SCHEMA,
        False,
    ),
    Provider.CEREBRAS: _Route(
        "Cerebras", "cerebras/", "gpt-oss-120b", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.DEEPSEEK: _Route(
        "DeepSeek", "deepseek/", "deepseek-chat", instructor.Mode.JSON, False
    ),
    Provider.MOONSHOT: _Route(
        "Moonshot", "moonshot/", "kimi-k3.1-pro", instructor.Mode.JSON, False
    ),
    Provider.ALIBABA: _Route(
        "Alibaba", "qwencloud/", "qwen3.7-plus", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.GOOGLE: _Route(
        "Google AI Studio", "gemini/", "gemini-3.1-flash-lite", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.OLLAMA: _Route(
        "Ollama", "ollama_chat/", "llama3.2", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.OPENAI: _Route(
        "OpenAI", "openai/", "gpt-4.1-mini", instructor.Mode.JSON_SCHEMA, False
    ),
    Provider.OPENROUTER: _Route(
        "OpenRouter",
        "openrouter/",
        "google/gemini-3.1-flash-lite",
        instructor.Mode.JSON_SCHEMA,
        True,
    ),
    Provider.VERTEX: _Route(
        "Google Agent Platform",
        "vertex_ai/",
        "gemini-3.1-flash-lite",
        instructor.Mode.JSON_SCHEMA,
        False,
    ),
}

#: The program-wide switch, one row per registered config. One row per
#: ``Provider`` member differs only in ``provider``; the last two add the route
#: fields a deployed Bedrock or Vertex host usually sets.
_SWITCHES: dict[str, LLMClientConfig] = {
    Provider.ANTHROPIC.value: LLMClientConfig(provider=Provider.ANTHROPIC),
    Provider.BEDROCK.value: LLMClientConfig(provider=Provider.BEDROCK),
    Provider.CEREBRAS.value: LLMClientConfig(provider=Provider.CEREBRAS),
    Provider.DEEPSEEK.value: LLMClientConfig(provider=Provider.DEEPSEEK),
    Provider.GOOGLE.value: LLMClientConfig(provider=Provider.GOOGLE),
    Provider.MOONSHOT.value: LLMClientConfig(provider=Provider.MOONSHOT),
    Provider.ALIBABA.value: LLMClientConfig(provider=Provider.ALIBABA),
    Provider.OLLAMA.value: LLMClientConfig(provider=Provider.OLLAMA),
    Provider.OPENAI.value: LLMClientConfig(provider=Provider.OPENAI),
    Provider.OPENROUTER.value: LLMClientConfig(provider=Provider.OPENROUTER),
    Provider.VERTEX.value: LLMClientConfig(provider=Provider.VERTEX),
    "bedrock-with-region": LLMClientConfig(provider=Provider.BEDROCK, aws_region_name="us-east-1"),
    "vertex-with-project-and-location": LLMClientConfig(
        provider=Provider.VERTEX, vertex_project="p", vertex_location="us-central1"
    ),
}

#: The bearer-key variables llmkit reads, one per keyed route. Set for
#: every row, so the environment never varies with the switch.
_BEARER_KEY_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "CEREBRAS_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "QWENCLOUD_API_KEY",
)


def test_the_matrix_covers_every_provider() -> None:
    """Guard on the rows: every member is covered, and its own row changes only ``provider``.

    A new ``Provider`` member must join both the expected table and the switch
    rows, and a member's own row that gained any other field would no longer be
    the "only ``provider`` changed" switch the matrix exists to run.
    """
    assert set(_EXPECTED) == set(Provider)
    assert {p: _SWITCHES.get(p.value) for p in Provider} == {
        p: LLMClientConfig(provider=p) for p in Provider
    }


@pytest.mark.parametrize("row", sorted(_SWITCHES))
async def test_one_call_site_switches_provider_by_config_alone(
    row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _SWITCHES[row]
    expected = _EXPECTED[config.provider]
    for name in _BEARER_KEY_ENV_VARS:
        monkeypatch.setenv(name, "test-key")

    built: list[LLMProviderInterface] = []
    real_build = llmkit.providers.build_provider

    def recording_build(config: LLMClientConfig | None = None) -> llmkit.providers.BaseProvider:
        instance = real_build(config)
        built.append(instance)
        return instance

    charged: list[str] = []
    real_acquire: Callable[[str], AbstractAsyncContextManager[RateLimitSlot]] = (
        GlobalRateLimiter.acquire_async
    )

    def recording_acquire(provider_key: str) -> AbstractAsyncContextManager[RateLimitSlot]:
        charged.append(provider_key)
        return real_acquire(provider_key)

    monkeypatch.setattr(llmkit.providers, "build_provider", recording_build)
    monkeypatch.setattr(GlobalRateLimiter, "acquire_async", recording_acquire)

    previous = configure_llm_client(config)
    try:
        with (
            install(lambda _request: Reply.json(_Answer(ok=True), cost=0.0)) as fake,
            capture_llm_records() as records,
        ):
            result = await _the_call_site()
    finally:
        _ = configure_llm_client(previous)

    assert result == _Answer(ok=True)

    # The round trip went out on the route's own spelling of its default model.
    assert [request.model for request in fake.requests] == [
        f"{expected.prefix}{expected.default_model}"
    ]
    assert [request.provider for request in fake.requests] == [expected.display_name]

    # The instance the call ran on carries the route's structured-output pins.
    assert len(built) == 1
    instance = built[0]
    assert instance.name == expected.display_name
    assert instance.model == expected.default_model
    assert instance.instructor_mode is expected.mode
    assert instance.strict_json_schema is expected.strict_json_schema

    # The budget charged is the route's own, keyed by its display name.
    assert charged == [expected.display_name]

    # The log record names the route and the unprefixed model.
    assert [(record.provider, record.model) for record in records] == [
        (expected.display_name, expected.default_model)
    ]
