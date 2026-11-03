"""A cross-provider comparison in one program keeps each call's decisions with its provider.

The switching principle names comparing vendors inside one program as a normal
use: no global source is registered, and every call passes its own provider
instance as ``provider=``. That is only safe if each call's correctness
decisions travel with the instance it was given and never leak to a sibling.
This module runs one such program and pins, per call:

* the structured-output **mode pin** and the **strict** wire upgrade, against a
  hard-coded table (not read back from the instance), and on the wire;
* the **reasoning translation** of the portable effort, through the public
  per-call path with only ``litellm.acompletion`` stubbed, because the
  :mod:`llmkit.testing` fake never sees the translated kwargs;
* the **rate-limit bucket** each call is charged to;
* the **circuit breaker**: one per provider, and tripping one leaves the
  others closed, including the router that serves the same model;
* the **log identity**: each record names its own provider and model.

The routes are chosen to separate on purpose. Google AI Studio and Vertex
serve the same model under the same spelling, and llmkit and LiteLLM's shared
Gemini transform both translate their reasoning effort the same way, so the pair
separates by name, routing, bucket and breaker rather than by translation.
Anthropic and OpenRouter serve the same Claude model directly and through the
router, and they translate ``"disable"`` differently. DeepSeek adds a route with a
different mode pin, so the mode assertion is not the same value five times.

The Gemini pair is driven with ``"low"`` rather than ``"disable"``: ``"disable"``
raises before transport on Gemini 3, so it cannot drive a comparison there.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import instructor
import litellm
import openai
import pytest
from instructor.core import InstructorRetryException
from pydantic import BaseModel

from llmkit import (
    CircuitOpenError,
    LLMClientNotConfiguredError,
    Provider,
    capture_llm_records,
    configure_llm_client,
    configure_rate_limit,
    make_provider,
)
from llmkit._types import ReasoningEffort
from llmkit.calls import structured_llm_call
from llmkit.exceptions import underlying_provider_error
from llmkit.providers import LLMProviderInterface
from llmkit.providers.base import client_configured
from llmkit.rate_limiting import GlobalRateLimiter, RateLimitSlot, _tuning
from llmkit.rate_limiting._breaker import CircuitState
from llmkit.rate_limiting._tuning import BREAKER_MIN_SAMPLES
from llmkit.retry import RetryPolicy
from llmkit.testing import Reply, TransportRequest, install, rate_limit_error


class _Answer(BaseModel):
    """Which provider the fake saw the call arrive on."""

    answered_by: str


@dataclass(frozen=True)
class _Route:
    """One comparison participant and what each of its decisions must be."""

    build: Callable[[], LLMProviderInterface]
    name: str
    #: The model id the caller and the log record see.
    model: str
    #: The provider-routed id the wire and the fake see.
    routed: str
    mode: instructor.Mode
    strict: bool
    one_tool_call: bool
    effort: ReasoningEffort | None
    #: The effort's translation on the wire, as :func:`_reasoning_part` reads it.
    reasoning: dict[str, object]


_GEMINI = "gemini-3.1-flash-lite"
_CLAUDE = "claude-sonnet-4-6"
#: The same model as Anthropic's ``claude-sonnet-4-6``, under OpenRouter's
#: catalogue spelling (dots, not hyphens).
_OPENROUTER_CLAUDE = "anthropic/claude-sonnet-4.6"

#: The expected decisions, written out rather than read from the instances, so
#: a provider whose trait drifts fails here instead of agreeing with itself.
_ROUTES: dict[str, _Route] = {
    "google": _Route(
        build=lambda: make_provider(Provider.GOOGLE, api_key="k", model=_GEMINI),
        name="Google AI Studio",
        model=_GEMINI,
        routed=f"gemini/{_GEMINI}",
        mode=instructor.Mode.JSON_SCHEMA,
        strict=False,
        one_tool_call=False,
        effort="low",
        reasoning={"reasoning_effort": "low"},
    ),
    "vertex": _Route(
        build=lambda: make_provider(
            Provider.VERTEX, vertex_project="p", vertex_location="us-central1", model=_GEMINI
        ),
        name="Google Agent Platform",
        model=_GEMINI,
        routed=f"vertex_ai/{_GEMINI}",
        mode=instructor.Mode.JSON_SCHEMA,
        strict=False,
        one_tool_call=False,
        effort="low",
        reasoning={"reasoning_effort": "low"},
    ),
    "anthropic": _Route(
        build=lambda: make_provider(Provider.ANTHROPIC, api_key="k", model=_CLAUDE),
        name="Anthropic",
        model=_CLAUDE,
        routed=f"anthropic/{_CLAUDE}",
        mode=instructor.Mode.JSON_SCHEMA,
        strict=False,
        one_tool_call=True,
        effort="disable",
        reasoning={"reasoning_effort": "none"},
    ),
    "openrouter": _Route(
        build=lambda: make_provider(Provider.OPENROUTER, api_key="k", model=_OPENROUTER_CLAUDE),
        name="OpenRouter",
        model=_OPENROUTER_CLAUDE,
        routed=f"openrouter/{_OPENROUTER_CLAUDE}",
        mode=instructor.Mode.JSON_SCHEMA,
        strict=True,
        one_tool_call=False,
        effort="disable",
        reasoning={"extra_body.reasoning": {"effort": "none"}},
    ),
    "deepseek": _Route(
        build=lambda: make_provider(Provider.DEEPSEEK, api_key="k"),
        name="DeepSeek",
        model="deepseek-chat",
        routed="deepseek/deepseek-chat",
        mode=instructor.Mode.JSON,
        strict=False,
        one_tool_call=False,
        effort=None,
        reasoning={},
    ),
}

#: The response-format type each mode pin puts on the wire.
_WIRE_FORMAT = {instructor.Mode.JSON_SCHEMA: "json_schema", instructor.Mode.JSON: "json_object"}

#: The limiter key the patched ``acquire_async`` is holding for the current task.
_CHARGED: ContextVar[str | None] = ContextVar("_CHARGED", default=None)

_ONE_ATTEMPT = RetryPolicy(max_attempts=1)


def test_the_routes_separate_where_the_comparison_claims_they_do() -> None:
    """Fixture guard: the table has the shape the module docstring describes.

    These assert nothing about llmkit. They fail if an edit to the table makes
    the comparison weaker than claimed: the Gemini pair stops sharing a model
    or a translation, the Claude pair stops translating differently, the mode
    pin becomes one value, or two routes share a display name.
    """
    google, vertex = _ROUTES["google"], _ROUTES["vertex"]
    assert google.model == vertex.model
    assert google.routed != vertex.routed
    assert (google.effort, google.reasoning) == (vertex.effort, vertex.reasoning)
    anthropic, openrouter = _ROUTES["anthropic"], _ROUTES["openrouter"]
    assert anthropic.effort == openrouter.effort
    assert anthropic.reasoning != openrouter.reasoning
    assert len({route.mode for route in _ROUTES.values()}) > 1
    assert len({route.name.casefold() for route in _ROUTES.values()}) == len(_ROUTES)


async def _drive_structured_call(
    provider: LLMProviderInterface, effort: ReasoningEffort | None
) -> dict[str, object]:
    """Make one public per-call ``structured_llm_call`` and return the kwargs the
    stubbed ``litellm.acompletion`` received.

    Only the LiteLLM call is stubbed, so the provider's real mode handler, strict
    upgrade and reasoning translation build the request.
    """
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"answered_by": "stub"}'},
                }
            ]
        )

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        answer = await structured_llm_call(
            "q", _Answer, feature="f", provider=provider, reasoning_effort=effort
        )
    # Harness guard: the stub's reply came back, so the request was really sent.
    assert answer.answered_by == "stub"
    return captured


def _extra_body(kwargs: dict[str, object]) -> dict[str, object]:
    """The request's ``extra_body``, or an empty dict when none was sent."""
    return cast("dict[str, object]", kwargs.get("extra_body", {}))


def _reasoning_part(kwargs: dict[str, object]) -> dict[str, object]:
    """The effort's translation on the wire: a flat ``reasoning_effort`` kwarg, or
    the native ``reasoning`` object inside ``extra_body``."""
    part: dict[str, object] = {}
    if "reasoning_effort" in kwargs:
        part["reasoning_effort"] = kwargs["reasoning_effort"]
    body = _extra_body(kwargs)
    if "reasoning" in body:
        part["extra_body.reasoning"] = body["reasoning"]
    return part


async def test_a_comparison_keeps_every_decision_with_its_own_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five providers in one program with no global source, each passed per call."""
    monkeypatch.setattr(_tuning, "now", lambda: 1_000.0)  # the breaker cooldown never elapses
    previous_source = configure_llm_client(None)
    previous_breaker = GlobalRateLimiter.breaker()
    try:
        await _compare(monkeypatch)
    finally:
        configure_rate_limit(breaker=previous_breaker)
        _ = configure_llm_client(previous_source)


async def _compare(monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparison program: the stubbed-LiteLLM half first, then the fake-transport half."""
    providers = {key: route.build() for key, route in _ROUTES.items()}

    # Mode pin, strict upgrade and one-tool-call rule: the table, not the instance.
    for key, route in _ROUTES.items():
        provider = providers[key]
        assert provider.name == route.name, key
        assert provider.instructor_mode == route.mode, key
        assert provider.strict_json_schema is route.strict, key
        assert provider.single_tool_call_per_schema is route.one_tool_call, key

    # --- Half 1: the translated request, with only LiteLLM stubbed ------------
    # The fake transport discards the translated kwargs, so this half makes the
    # same public per-call calls with the same instances, before any fake is
    # installed, and reads the request LiteLLM would have sent.
    assert not client_configured()
    wire = {
        key: await _drive_structured_call(providers[key], _ROUTES[key].effort) for key in _ROUTES
    }
    assert not client_configured()

    for key, route in _ROUTES.items():
        kwargs = wire[key]
        assert kwargs["model"] == route.routed, key
        response_format = cast("dict[str, object]", kwargs["response_format"])
        assert response_format["type"] == _WIRE_FORMAT[route.mode], key
        if route.mode is instructor.Mode.JSON_SCHEMA:
            json_schema = cast("dict[str, object]", response_format["json_schema"])
            assert json_schema.get("strict") is (True if route.strict else None), key
        if route.one_tool_call:
            assert kwargs["parallel_tool_calls"] is False, key
        else:
            assert "parallel_tool_calls" not in kwargs, key
        assert _reasoning_part(kwargs) == route.reasoning, key
        # OpenRouter's routing preference shares extra_body with its reasoning
        # object but comes from the route, not the effort, so it is pinned apart.
        expected_preference = {"require_parameters": True} if key == "openrouter" else None
        assert _extra_body(kwargs).get("provider") == expected_preference, key

    # The Gemini pair shares a model and a translation. On the wire it differs by
    # routed id (checked above), credentials, and Vertex's project and location.
    # Its buckets and breakers are checked in the second half.
    assert wire["google"]["api_key"] == "k"
    assert "api_key" not in wire["vertex"]
    assert wire["vertex"]["vertex_project"] == "p"
    assert wire["vertex"]["vertex_location"] == "us-central1"
    assert "vertex_project" not in wire["google"]
    assert "vertex_location" not in wire["google"]

    # --- Half 2: the same per-call path, through the fake transport ----------
    # Arming the breaker clears the limiter registries, so every breaker below
    # was created by this half's calls.
    configure_rate_limit(breaker=True)
    original_acquire = GlobalRateLimiter.acquire_async

    @contextlib.asynccontextmanager
    async def _charging(provider_key: str) -> AsyncGenerator[RateLimitSlot]:
        token = _CHARGED.set(provider_key)
        try:
            async with original_acquire(provider_key) as slot:
                yield slot
        finally:
            _CHARGED.reset(token)

    monkeypatch.setattr(GlobalRateLimiter, "acquire_async", _charging)

    charged: dict[str | None, list[str | None]] = {}

    def _fake(request: TransportRequest) -> Reply:
        charged.setdefault(request.label, []).append(_CHARGED.get())
        if request.label == "trip":
            raise rate_limit_error()
        return Reply.json(_Answer(answered_by=request.provider), cost=0.0)

    with install(_fake) as fake:
        # No global source: a call that names no provider is refused before any attempt.
        assert not client_configured()
        with pytest.raises(LLMClientNotConfiguredError):
            _ = await structured_llm_call("q", _Answer, feature="compare")
        assert fake.requests == []

        # Every provider answers its own call, concurrently, and is logged as itself.
        with capture_llm_records() as records:
            answers = await asyncio.gather(
                *(
                    structured_llm_call(
                        "q", _Answer, feature="compare", label=key, provider=providers[key]
                    )
                    for key in _ROUTES
                )
            )

        for key, answer in zip(_ROUTES, answers, strict=True):
            assert answer.answered_by == _ROUTES[key].name, key

        by_label = {request.label: request for request in fake.requests}
        assert len(by_label) == len(fake.requests) == len(_ROUTES)
        for key, route in _ROUTES.items():
            request = by_label[key]
            assert request.provider == route.name, key
            assert request.model == route.routed, key
            # Rate-limit bucket: the call was held under its own provider's key.
            assert charged[key] == [route.name], key

        logged = {record.label: record for record in records}
        assert set(logged) == set(_ROUTES)
        for key, route in _ROUTES.items():
            assert logged[key].provider == route.name, key
            assert logged[key].model == route.model, key

        # Breaker: one per provider, distinct objects, and nothing else was created.
        keys = {route.name.casefold() for route in _ROUTES.values()}
        assert set(GlobalRateLimiter._breaker_states) == keys
        assert len({id(GlobalRateLimiter._get_breaker(k)) for k in keys}) == len(keys)

        # Trip Anthropic's breaker with throttles on Anthropic calls only.
        anthropic_key = _ROUTES["anthropic"].name.casefold()
        anthropic_breaker = GlobalRateLimiter._get_breaker(anthropic_key)
        for _ in range(BREAKER_MIN_SAMPLES):
            if anthropic_breaker._state is CircuitState.OPEN:
                break
            with pytest.raises(InstructorRetryException) as throttled:
                _ = await structured_llm_call(
                    "q",
                    _Answer,
                    feature="compare",
                    label="trip",
                    provider=providers["anthropic"],
                    retry=_ONE_ATTEMPT,
                )
            assert isinstance(underlying_provider_error(throttled.value), openai.RateLimitError)
        assert anthropic_breaker._state is CircuitState.OPEN
        assert set(charged["trip"]) == {_ROUTES["anthropic"].name}

        # Anthropic now fails fast, before the fake...
        sent = len(fake.requests)
        with pytest.raises(CircuitOpenError) as opened:
            _ = await structured_llm_call(
                "q", _Answer, feature="compare", label="open", provider=providers["anthropic"]
            )
        assert opened.value.provider == anthropic_key
        assert len(fake.requests) == sent

        # ...while every other provider, the same Claude model on OpenRouter
        # included, still answers and keeps its breaker closed.
        for key, route in _ROUTES.items():
            if key == "anthropic":
                continue
            answer = await structured_llm_call(
                "q", _Answer, feature="compare", label=f"after-{key}", provider=providers[key]
            )
            assert answer.answered_by == route.name, key
            assert GlobalRateLimiter._get_breaker(route.name.casefold())._state is (
                CircuitState.CLOSED
            ), key

        assert not client_configured()
