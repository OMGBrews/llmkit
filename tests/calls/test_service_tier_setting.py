"""The ``service_tier`` setting: where it is read, how it resolves, and when it refuses.

A call asks for a provider service tier through one setting that lives in three
places: the per-call ``service_tier`` keyword, :class:`~llmkit.LLMCallOptions`,
and the provider's configured default (``LLMClientConfig.service_tier`` or
``make_provider(..., service_tier=...)``). These tests pin:

* precedence, keyword over options over the provider's default, on all five
  lanes, observed through the :mod:`llmkit.testing` fake's
  ``TransportRequest.service_tier`` and the call record;
* that an explicit ``None`` on the keyword or the options cancels a configured
  default, and that nothing set anywhere asks for no tier;
* that an invalid value raises ``ValueError`` before any request;
* that a tier the route cannot honour raises
  :class:`~llmkit.ServiceTierUnsupportedError` before any request, record,
  cache read or retry-loop claim, even when the response cache already holds
  the answer;
* that the tier is kept out of the response-cache key, so a cached answer
  serves a call that differs only in tier;
* the request arguments: a host-written provider forwards the flat
  ``service_tier`` key, an explicit ``None`` sends no key, the temperature
  re-issue keeps the tier, and a translation's ``extra_body`` merges with
  OpenRouter's own.

The wire-level tests patch ``llmkit._litellm.litellm.acompletion``, as
``test_temperature_refusal.py`` does, so the real transport builds the request.
No test reaches the network.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterator
from typing import ClassVar, cast, final, override
from unittest.mock import patch

import instructor
import openai
import pytest
from litellm import exceptions as litellm_errors
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

import llmkit.providers.anthropic as anthropic_module
from llmkit import (
    NO_RETRY,
    UNSET,
    InMemoryLLMCache,
    LLMCallOptions,
    LLMCallRecord,
    LLMClientConfig,
    LLMProviderInterface,
    Provider,
    ServiceTier,
    ServiceTierUnsupportedError,
    ToolDefinition,
    Unset,
    capture_llm_records,
    configure_llm_cache,
    configure_llm_client,
    make_provider,
)
from llmkit import calls as llm_calls
from llmkit._litellm import _completion_request_kwargs
from llmkit._types import ServiceTierRequest
from llmkit.calls._shared import resolve_tier_request
from llmkit.options import resolve_call_args
from llmkit.providers import (
    AnthropicProvider,
    BaseProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from llmkit.retry import RetryPolicy, with_retries
from llmkit.testing import FakeTransport, Reply, TransportRequest, install, timeout_error

type Kind = str
KINDS: tuple[Kind, ...] = ("text", "structured", "tool", "text_stream", "tool_stream")

#: What a call function's ``service_tier`` keyword accepts, as these tests pass it.
type TierArg = ServiceTier | None | Unset
type ProviderArg = LLMProviderInterface | Provider | None | Unset


class Answer(BaseModel):
    value: int


TOOL = ToolDefinition.from_model("answer", Answer)
ANSWER = '{"value": 1}'


def _answer(_request: TransportRequest) -> Reply:
    return Reply.text(ANSWER)


def _openai(service_tier: ServiceTier | None = None) -> OpenAIProvider:
    """A provider that forwards every tier as the flat ``service_tier`` key."""
    return OpenAIProvider(api_key="k", model="gpt-4.1-mini", service_tier=service_tier)


def _ollama() -> OllamaProvider:
    """A provider that refuses every tier."""
    return OllamaProvider(model="fake-model")


@pytest.fixture
def configured_flex() -> Iterator[None]:
    """A registered client config whose provider asks for ``"flex"`` by default.

    The autouse fixture restores the keyless Ollama config afterwards.
    """
    _ = configure_llm_client(
        LLMClientConfig(
            provider=Provider.OPENAI, api_key="k", model="gpt-4.1-mini", service_tier="flex"
        )
    )
    yield


async def _call(
    kind: Kind,
    *,
    service_tier: TierArg = UNSET,
    options: LLMCallOptions | None = None,
    provider: ProviderArg = UNSET,
    retry: RetryPolicy = NO_RETRY,
    feature: str = "tier",
) -> None:
    """Run one call of *kind* to completion with the given tier inputs."""
    if kind == "text":
        _ = await llm_calls.text_llm_call(
            "q",
            feature=feature,
            service_tier=service_tier,
            options=options,
            provider=provider,
            retry=retry,
        )
    elif kind == "structured":
        _ = await llm_calls.structured_llm_call(
            "q",
            Answer,
            feature=feature,
            service_tier=service_tier,
            options=options,
            provider=provider,
            retry=retry,
        )
    elif kind == "tool":
        _ = await llm_calls.tool_llm_call(
            "q",
            [TOOL],
            feature=feature,
            service_tier=service_tier,
            options=options,
            provider=provider,
            retry=retry,
        )
    elif kind == "text_stream":
        async for _delta in llm_calls.text_llm_call_stream(
            "q",
            feature=feature,
            service_tier=service_tier,
            options=options,
            provider=provider,
            retry=retry,
        ):
            pass
    else:
        assert kind == "tool_stream"
        async for _event in llm_calls.tool_llm_call_stream(
            "q",
            [TOOL],
            feature=feature,
            service_tier=service_tier,
            options=options,
            provider=provider,
            retry=retry,
        ):
            pass


def _only(fake: FakeTransport, records: list[LLMCallRecord]) -> tuple[str | None, str | None]:
    """The single request's tier and the single record's requested tier."""
    [request] = fake.requests
    [record] = records
    return request.service_tier, record.requested_service_tier


# --- the setting's surface ----------------------------------------------------


def _call_functions() -> dict[str, object]:
    return {
        name: cast("object", getattr(llm_calls, name))
        for name in llm_calls.__all__
        if name.endswith("_llm_call") or "_llm_call_" in name
    }


def test_every_call_function_takes_a_keyword_only_unset_service_tier() -> None:
    functions = _call_functions()
    assert len(functions) == 10, "the scan found every call function and sync twin"
    for name, fn in functions.items():
        parameter = inspect.signature(cast("type", fn)).parameters["service_tier"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert cast("object", parameter.default) is UNSET, name
        assert str(cast("object", parameter.annotation)) == "ServiceTier | None | Unset", name


def test_options_config_and_make_provider_carry_the_setting() -> None:
    assert LLMCallOptions().service_tier is UNSET
    assert LLMClientConfig(provider=Provider.OLLAMA).service_tier is None
    assert dataclasses.fields(LLMClientConfig)[-1].name == "service_tier"
    default = cast("object", inspect.signature(make_provider).parameters["service_tier"].default)
    assert default is None
    built = make_provider(Provider.OPENAI, api_key="k", service_tier="priority")
    assert isinstance(built, BaseProvider)
    assert built.service_tier == "priority"
    assert make_provider(Provider.OPENAI, api_key="k").service_tier is None


# --- precedence: keyword > options > provider default, on every lane ---------


@pytest.mark.parametrize("kind", KINDS)
async def test_the_configured_default_applies_when_nothing_else_is_set(
    kind: Kind, configured_flex: None
) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind)
    assert _only(fake, records) == ("flex", "flex")


@pytest.mark.parametrize("kind", KINDS)
async def test_options_override_the_configured_default(kind: Kind, configured_flex: None) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, options=LLMCallOptions(service_tier="priority"))
    assert _only(fake, records) == ("priority", "priority")


@pytest.mark.parametrize("kind", KINDS)
async def test_the_keyword_overrides_options_and_the_configured_default(
    kind: Kind, configured_flex: None
) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, service_tier="default", options=LLMCallOptions(service_tier="priority"))
    assert _only(fake, records) == ("default", "default")


@pytest.mark.parametrize("kind", KINDS)
async def test_a_per_call_provider_brings_its_own_default(
    kind: Kind, configured_flex: None
) -> None:
    """The default belongs to the provider the call runs on, not to the registered config."""
    del configured_flex
    provider = make_provider(Provider.OPENAI, api_key="k", service_tier="priority")
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, provider=provider)
        await _call(kind, provider=_openai())
        await _call(kind, provider=provider, service_tier="default")
    assert [r.service_tier for r in fake.requests] == ["priority", None, "default"]
    assert [r.requested_service_tier for r in records] == ["priority", None, "default"]


@pytest.mark.parametrize("kind", KINDS)
async def test_an_explicit_none_keyword_cancels_the_configured_default(
    kind: Kind, configured_flex: None
) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, service_tier=None, options=LLMCallOptions(service_tier="priority"))
    assert _only(fake, records) == (None, None)


@pytest.mark.parametrize("kind", KINDS)
async def test_an_explicit_none_in_options_cancels_the_configured_default(
    kind: Kind, configured_flex: None
) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, options=LLMCallOptions(service_tier=None))
    assert _only(fake, records) == (None, None)


@pytest.mark.parametrize("kind", KINDS)
async def test_a_keyword_tier_overrides_a_none_in_options(kind: Kind) -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(
            kind, provider=_openai(), service_tier="flex", options=LLMCallOptions(service_tier=None)
        )
    assert _only(fake, records) == ("flex", "flex")


@pytest.mark.parametrize("kind", KINDS)
async def test_unset_everywhere_asks_for_no_tier(kind: Kind) -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, provider=_openai())
    assert _only(fake, records) == (None, None)


def test_the_sync_twins_resolve_the_same_way(configured_flex: None) -> None:
    del configured_flex
    with install(_answer) as fake, capture_llm_records() as records:
        _ = llm_calls.text_llm_call_sync("q", feature="tier", retry=NO_RETRY)
        _ = llm_calls.structured_llm_call_sync(
            "q", Answer, feature="tier", retry=NO_RETRY, service_tier="priority"
        )
        _ = llm_calls.tool_llm_call_sync(
            "q", [TOOL], feature="tier", retry=NO_RETRY, service_tier=None
        )
        for _delta in llm_calls.text_llm_call_stream_sync(
            "q", feature="tier", retry=NO_RETRY, options=LLMCallOptions(service_tier="default")
        ):
            pass
        for _event in llm_calls.tool_llm_call_stream_sync(
            "q", [TOOL], feature="tier", retry=NO_RETRY
        ):
            pass
    assert [r.service_tier for r in fake.requests] == ["flex", "priority", None, "default", "flex"]
    assert [r.requested_service_tier for r in records] == [
        "flex",
        "priority",
        None,
        "default",
        "flex",
    ]


def test_resolution_keeps_unset_distinct_from_none() -> None:
    """The merge keeps an unset tier unset, so the provider's default can still apply."""

    def resolved(keyword: TierArg, options: TierArg) -> TierArg:
        return resolve_call_args(
            LLMCallOptions(service_tier=options),
            temperature=UNSET,
            model=UNSET,
            max_tokens=UNSET,
            reasoning_effort=UNSET,
            retry=UNSET,
            provider=UNSET,
            cache=UNSET,
            timeout=UNSET,
            service_tier=keyword,
        ).service_tier

    assert resolved(UNSET, UNSET) is UNSET
    assert resolved(UNSET, None) is None
    assert resolved(None, "flex") is None
    assert resolved(UNSET, "flex") == "flex"
    assert resolved("priority", "flex") == "priority"
    assert resolved("vendor-tier", UNSET) == "vendor-tier"


# --- invalid values -----------------------------------------------------------

_INVALID: list[object] = ["", 3, True, b"x"]
_INVALID_IDS = ["empty", "int", "bool", "bytes"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", _INVALID, ids=_INVALID_IDS)
@pytest.mark.parametrize("where", ["keyword", "options"])
async def test_an_invalid_tier_raises_value_error_before_any_request(
    kind: Kind, value: object, where: str
) -> None:
    tier = cast("str", value)
    keyword: TierArg = tier if where == "keyword" else UNSET
    options = LLMCallOptions(service_tier=tier) if where == "options" else None
    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ValueError, match="service_tier must be a non-empty string or None"),
    ):
        await _call(kind, provider=_openai(), service_tier=keyword, options=options)
    assert fake.requests == []
    assert records == []


@pytest.mark.parametrize("value", _INVALID, ids=_INVALID_IDS)
def test_an_invalid_configured_tier_raises_value_error_at_construction(value: object) -> None:
    tier = cast("str", value)
    with pytest.raises(ValueError, match=r"LLMClientConfig\.service_tier must be"):
        _ = LLMClientConfig(provider=Provider.OPENAI, api_key="k", service_tier=tier)
    with pytest.raises(ValueError, match=r"LLMClientConfig\.service_tier must be"):
        _ = make_provider(Provider.OPENAI, api_key="k", service_tier=tier)


# --- a tier the route cannot honour -------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tier", ["default", "flex", "priority", "auto"])
async def test_a_refused_tier_raises_before_any_request_or_record(kind: Kind, tier: str) -> None:
    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError, match=rf"service_tier='{tier}' on 'fake-model'"),
    ):
        await _call(kind, provider=_ollama(), service_tier=tier)
    assert fake.requests == []
    assert records == []


@pytest.mark.parametrize("kind", KINDS)
async def test_a_refused_configured_default_raises_too(kind: Kind) -> None:
    _ = configure_llm_client(
        LLMClientConfig(provider=Provider.OLLAMA, model="fake-model", service_tier="flex")
    )
    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError, match="Ollama"),
    ):
        await _call(kind)
    assert fake.requests == []
    assert records == []


@pytest.mark.parametrize("kind", KINDS)
async def test_an_explicit_none_cancels_a_default_the_route_would_refuse(kind: Kind) -> None:
    _ = configure_llm_client(
        LLMClientConfig(provider=Provider.OLLAMA, model="fake-model", service_tier="flex")
    )
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, service_tier=None)
    assert _only(fake, records) == (None, None)


@pytest.mark.parametrize("kind", KINDS)
async def test_anthropic_flex_is_refused_before_any_request(kind: Kind) -> None:
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-6")
    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError, match="no flex tier"),
    ):
        await _call(kind, provider=provider, service_tier="flex")
    assert fake.requests == []
    assert records == []


class _OwnTierOpenAI(OpenAIProvider):
    """A provider whose own request arguments carry a tier, which llmkit refuses."""

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {**super().completion_kwargs(), "service_tier": "priority"}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("setting", [UNSET, None, "flex"], ids=["unset", "none", "flex"])
async def test_a_tier_in_the_providers_own_request_arguments_is_refused(
    kind: Kind, setting: TierArg
) -> None:
    provider = _OwnTierOpenAI(api_key="k", model="gpt-4.1-mini")
    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError, match="completion_kwargs"),
    ):
        await _call(kind, provider=provider, service_tier=setting)
    assert fake.requests == []
    assert records == []


async def test_a_refusal_is_not_retried_by_an_outer_loop() -> None:
    passes = [0]

    async def _refused() -> None:
        passes[0] += 1
        await _call("text", provider=_ollama(), service_tier="priority")

    with (
        install(_answer) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError),
    ):
        await with_retries(_refused, policy=RetryPolicy(backoff_base_seconds=0.0))
    assert passes == [1]
    assert fake.requests == []
    assert records == []


async def test_a_refusal_counts_no_attempt_in_an_outer_loop() -> None:
    """A refused call on pass 1 claims nothing, so the next call with the same
    ``(feature, label)`` stays the first occurrence on both passes.

    Had the refusal claimed, pass 1's real call would be the second occurrence
    and pass 2's the first, so the two records would carry different call ids.
    """
    passes = [0]
    sent = [0]

    def _fake(_request: TransportRequest) -> Reply:
        sent[0] += 1
        if sent[0] == 1:
            raise timeout_error()
        return Reply.text("hello")

    async def _pass() -> None:
        passes[0] += 1
        if passes[0] == 1:
            with pytest.raises(ServiceTierUnsupportedError):
                await _call("text", provider=_ollama(), service_tier="flex", feature="same")
        await _call("text", provider=_ollama(), feature="same")

    with install(_fake), capture_llm_records() as records:
        await with_retries(
            _pass,
            policy=RetryPolicy(
                max_attempts=3, retry_on=(openai.APITimeoutError,), backoff_base_seconds=0.0
            ),
        )
    assert passes == [2]
    assert [r.attempt for r in records] == [1, 2]
    assert len({r.call_id for r in records}) == 1


# --- the response cache -------------------------------------------------------

_CACHED_KINDS: tuple[Kind, ...] = ("text", "structured", "text_stream")


@pytest.mark.parametrize("kind", _CACHED_KINDS)
async def test_a_refused_tier_raises_even_when_the_cache_holds_the_answer(kind: Kind) -> None:
    """The refusal comes before the cache read: a cached answer does not mask it.

    The third call, without a tier, is answered from the cache, which shows the
    refused call's request had the same key as the paid one.
    """
    configure_llm_cache(InMemoryLLMCache())
    provider = _ollama()
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, provider=provider)
        with pytest.raises(ServiceTierUnsupportedError):
            await _call(kind, provider=provider, service_tier="priority")
        await _call(kind, provider=provider)
    assert len(fake.requests) == 1
    assert [r.cache_hit for r in records] == [False, True]


@pytest.mark.parametrize("kind", _CACHED_KINDS)
async def test_a_cached_answer_serves_a_call_that_differs_only_in_tier(kind: Kind) -> None:
    """The tier is not in the key; the hit's record reports the tier its call asked for."""
    configure_llm_cache(InMemoryLLMCache())
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, provider=_openai())
        await _call(kind, provider=_openai(), service_tier="priority")
        await _call(kind, provider=_openai(service_tier="flex"))
    assert len(fake.requests) == 1
    assert [(r.cache_hit, r.requested_service_tier) for r in records] == [
        (False, None),
        (True, "priority"),
        (True, "flex"),
    ]


# --- the request arguments ----------------------------------------------------


def _response(content: str = "ok") -> ModelResponse:
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )
    resp._hidden_params = {}  # mirror litellm's cost-lookup attribute
    return resp


@final
class _ScriptedLiteLLM:
    """Raise the scripted errors in order, then answer; record every request."""

    def __init__(self, *errors: Exception) -> None:
        self.errors: list[Exception] = list(errors)
        self.requests: list[dict[str, object]] = []

    async def __call__(self, **request: object) -> ModelResponse:
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        return _response()


async def _wire_text(
    provider: LLMProviderInterface,
    *,
    service_tier: TierArg = UNSET,
    temperature: float | None | Unset = UNSET,
    errors: tuple[Exception, ...] = (),
) -> tuple[list[dict[str, object]], LLMCallRecord]:
    """Send one text call through the real transport; return what LiteLLM was asked."""
    scripted = _ScriptedLiteLLM(*errors)
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=scripted.__call__),
        capture_llm_records() as records,
    ):
        _ = await llm_calls.text_llm_call(
            "q",
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            temperature=temperature,
            retry=NO_RETRY,
        )
    [record] = records
    return scripted.requests, record


async def test_a_forwarded_tier_is_the_flat_key_on_the_wire() -> None:
    [request], record = await _wire_text(_openai(), service_tier="priority")
    assert request["service_tier"] == "priority"
    assert record.requested_service_tier == "priority"


async def test_an_explicit_none_sends_no_tier_argument() -> None:
    [request], record = await _wire_text(_openai(service_tier="flex"), service_tier=None)
    assert "service_tier" not in request
    assert record.requested_service_tier is None


async def test_the_record_reports_the_tier_as_written_not_its_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The declared LiteLLM floor sends no tier on Anthropic, so the route's
    # capability check is pinned here: this test is about the record.
    def _sends(_model: str) -> bool:
        return True

    monkeypatch.setattr(anthropic_module, "_litellm_sends_service_tier", _sends)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-6")
    [request], record = await _wire_text(provider, service_tier="default")
    assert request["service_tier"] == "standard_only"
    assert record.requested_service_tier == "default"


class _HostProvider(BaseProvider):
    """A host-written provider that overrides nothing tier-related."""

    _provider_name: ClassVar[str] = "Host"
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON
    _default_model: ClassVar[str] = "host-model"
    _model_prefix: ClassVar[str] = "openai/"
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset()

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_key": "k", "api_base": "https://host.example/v1"}

    @classmethod
    @override
    def build(cls, config: LLMClientConfig) -> _HostProvider:
        return cls(config.model)


@final
class _PlainProvider:
    """A host-written provider satisfying ``LLMProviderInterface`` and nothing more.

    It has neither ``service_tier`` nor ``service_tier_kwargs``.
    """

    @property
    def name(self) -> str:
        return "Plain"

    @property
    def model(self) -> str:
        return "plain-model"

    @property
    def instructor_mode(self) -> instructor.Mode:
        return instructor.Mode.JSON

    @property
    def strict_json_schema(self) -> bool:
        return False

    @property
    def single_tool_call_per_schema(self) -> bool:
        return False

    @property
    def compose_tools_schema(self) -> bool:
        return False

    def guard_compose_tools_schema(self, model: str | None) -> None:
        del model

    @property
    def supports_tool_choice(self) -> bool:
        return True

    @property
    def reasoning_effort(self) -> str | None:
        return None

    def reasoning_kwargs(self, effort: str, model: str) -> dict[str, object]:
        del model
        return {"reasoning_effort": effort}

    def litellm_model(self, model: str | None = None) -> str:
        return f"openai/{model or self.model}"

    def completion_kwargs(self) -> dict[str, object]:
        return {"api_key": "k", "api_base": "https://plain.example/v1"}


def test_the_plain_provider_satisfies_the_protocol_without_the_tier_members() -> None:
    plain = _PlainProvider()
    assert isinstance(plain, LLMProviderInterface)
    assert not hasattr(plain, "service_tier")
    assert not hasattr(plain, "service_tier_kwargs")


@pytest.mark.parametrize(
    "provider", [_HostProvider(), _PlainProvider()], ids=["base-subclass", "plain"]
)
async def test_a_host_written_provider_forwards_the_flat_key(
    provider: LLMProviderInterface,
) -> None:
    [request], record = await _wire_text(provider, service_tier="vendor-tier")
    assert request["service_tier"] == "vendor-tier"
    assert record.requested_service_tier == "vendor-tier"
    [untiered], record = await _wire_text(provider)
    assert "service_tier" not in untiered
    assert record.requested_service_tier is None


async def test_a_host_subclass_default_applies() -> None:
    [request], record = await _wire_text(_HostProvider(service_tier="flex"))
    assert request["service_tier"] == "flex"
    assert record.requested_service_tier == "flex"


@pytest.mark.parametrize("kind", KINDS)
async def test_a_plain_provider_reaches_the_fake_with_the_tier(kind: Kind) -> None:
    with install(_answer) as fake, capture_llm_records() as records:
        await _call(kind, provider=_PlainProvider(), service_tier="flex")
        await _call(kind, provider=_PlainProvider())
    assert [r.service_tier for r in fake.requests] == ["flex", None]
    assert [r.requested_service_tier for r in records] == ["flex", None]


def test_resolution_falls_back_to_the_flat_key_without_the_hook() -> None:
    args = resolve_call_args(
        None,
        temperature=UNSET,
        model=UNSET,
        max_tokens=UNSET,
        reasoning_effort=UNSET,
        retry=UNSET,
        provider=UNSET,
        cache=UNSET,
        timeout=UNSET,
        service_tier="flex",
    )
    assert resolve_tier_request(args, _PlainProvider()) == ServiceTierRequest(
        "flex", {"service_tier": "flex"}
    )
    unset = dataclasses.replace(args, service_tier=UNSET)
    assert resolve_tier_request(unset, _PlainProvider()) is None


def _client_refusal() -> Exception:
    return litellm_errors.UnsupportedParamsError(
        message="gpt-5-mini doesn't support temperature=0.0 while reasoning is active.",
        llm_provider="openai",
        model="gpt-5-mini",
    )


async def test_the_temperature_reissue_keeps_the_tier() -> None:
    provider = OpenAIProvider(api_key="k", model="gpt-5-mini")
    (refused, reissued), record = await _wire_text(
        provider, service_tier="priority", temperature=0.0, errors=(_client_refusal(),)
    )
    assert reissued["additional_drop_params"] == ["temperature"]
    assert refused["service_tier"] == reissued["service_tier"] == "priority"
    assert record.temperature is None
    assert record.requested_service_tier == "priority"


class _BodyTierOpenRouter(OpenRouterProvider):
    """An OpenRouter route whose tier translation travels in ``extra_body``."""

    @override
    def service_tier_kwargs(self, tier: ServiceTier, model: str) -> dict[str, object]:
        del model
        return {"extra_body": {"service_tier": tier}}


def test_a_tier_translations_extra_body_merges_with_openrouters_own() -> None:
    provider = _BodyTierOpenRouter(api_key="k", model="openai/gpt-4.1-mini")
    tier = ServiceTierRequest("flex", provider.service_tier_kwargs("flex", provider.model))
    merged = _completion_request_kwargs(provider, "high", None, tier)
    assert merged["extra_body"] == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "high"},
        "service_tier": "flex",
    }
    assert "service_tier" not in merged
    # The provider's own arguments are not mutated by the merge.
    assert provider.completion_kwargs()["extra_body"] == {"provider": {"require_parameters": True}}


def test_openrouters_flat_tier_leaves_its_extra_body_alone() -> None:
    provider = OpenRouterProvider(api_key="k", model="openai/gpt-4.1-mini")
    tier = ServiceTierRequest("flex", provider.service_tier_kwargs("flex", provider.model))
    merged = _completion_request_kwargs(provider, None, None, tier)
    assert merged["service_tier"] == "flex"
    assert merged["extra_body"] == {"provider": {"require_parameters": True}}
    assert "service_tier" not in _completion_request_kwargs(provider, None, None, None)


async def test_openrouter_sends_both_end_to_end() -> None:
    provider = _BodyTierOpenRouter(api_key="k", model="openai/gpt-4.1-mini")
    [request], record = await _wire_text(provider, service_tier="priority")
    assert request["extra_body"] == {
        "provider": {"require_parameters": True},
        "service_tier": "priority",
    }
    assert record.requested_service_tier == "priority"
