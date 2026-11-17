"""The call record carries the requested and the served service tier.

The requested tier is the ``service_tier`` setting as the host wrote it (here
the per-call keyword, on an OpenAI provider, which forwards any tier); the
served tier is what the response, or a stream frame, reported. A top-level
``service_tier`` in a provider's own ``completion_kwargs()`` is refused with
:class:`~llmkit.ServiceTierUnsupportedError` before any request, and Ollama
refuses every tier. Most tests drive every public lane through the
:mod:`llmkit.testing` fake; the ``real LiteLLM`` tests send through LiteLLM's
own OpenAI handler and stream processing to an in-process HTTP transport, so
the conversion from LiteLLM's objects is measured at whichever LiteLLM is
installed (the declared floor under the lowest-direct run). No test reaches the
network.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, cast, override

import httpx
import pytest
import yaml
from instructor.core import InstructorRetryException
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage
from openai import AsyncOpenAI
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    LLMCallRecord,
    LocalYamlLogSink,
    OutputLimitError,
    ProviderFailureError,
    RetryPolicy,
    ServiceTierUnsupportedError,
    ToolDefinition,
    capture_llm_records,
    configure_llm_cache,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_stream,
    text_llm_call_stream_sync,
    text_llm_call_sync,
    tool_llm_call,
    tool_llm_call_stream,
    tool_llm_call_stream_sync,
)
from llmkit._litellm import (
    _served_service_tier,
)
from llmkit.options import UNSET, Unset
from llmkit.providers import OllamaProvider, OpenAIProvider
from llmkit.testing import Reply, TransportRequest, install

Kind = Literal["text", "structured", "tool", "text_stream", "tool_stream"]
KINDS: tuple[Kind, ...] = ("text", "structured", "tool", "text_stream", "tool_stream")


class Answer(BaseModel):
    value: int


TOOL = ToolDefinition.from_model("answer", Answer)
ANSWER = '{"value": 1}'


def _openai() -> OpenAIProvider:
    """A provider that forwards any tier; the fake answers it without network."""
    return OpenAIProvider(api_key="sk-test", model="fake-model")


OPENAI = _openai()


class _OwnTierOpenAI(OpenAIProvider):
    """An OpenAI provider whose own request arguments carry *extra*."""

    def __init__(self, extra: dict[str, object]) -> None:
        super().__init__(api_key="sk-test", model="fake-model")
        self._extra: dict[str, object] = extra

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {**super().completion_kwargs(), **self._extra}


async def _call(
    kind: Kind,
    *,
    provider: OllamaProvider | OpenAIProvider | None = OPENAI,
    service_tier: str | None | Unset = "priority",
    timeout: float | None = None,
    retry: RetryPolicy = NO_RETRY,
    between_chunks: Callable[[], Awaitable[object]] | None = None,
) -> None:
    if kind == "text":
        _ = await text_llm_call(
            "q",
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            timeout=timeout,
            retry=retry,
        )
    elif kind == "structured":
        _ = await structured_llm_call(
            "q",
            Answer,
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            timeout=timeout,
            retry=retry,
        )
    elif kind == "tool":
        _ = await tool_llm_call(
            "q",
            [TOOL],
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            timeout=timeout,
            retry=retry,
        )
    elif kind == "text_stream":
        async for _delta in text_llm_call_stream(
            "q",
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            timeout=timeout,
            retry=retry,
        ):
            if between_chunks is not None:
                _ = await between_chunks()
    else:
        async for _event in tool_llm_call_stream(
            "q",
            [TOOL],
            feature="tier",
            provider=provider,
            service_tier=service_tier,
            timeout=timeout,
            retry=retry,
        ):
            pass


# --- every lane, through the public fake ------------------------------------


@pytest.mark.parametrize("kind", KINDS)
async def test_every_lane_records_the_requested_and_the_served_tier(kind: Kind) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")),
        capture_llm_records() as records,
    ):
        await _call(kind)
    [record] = records
    assert record.requested_service_tier == "priority"
    assert record.served_service_tier == "default"


@pytest.mark.parametrize("kind", KINDS)
async def test_a_response_reporting_no_tier_records_a_null_served_tier(kind: Kind) -> None:
    """Null, not the requested tier: silence is distinguishable from a match."""
    with install(lambda _request: Reply.text(ANSWER)), capture_llm_records() as records:
        await _call(kind)
    [record] = records
    assert record.requested_service_tier == "priority"
    assert record.served_service_tier is None


@pytest.mark.parametrize("kind", KINDS)
async def test_a_response_reporting_the_requested_tier_records_it(kind: Kind) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="priority")),
        capture_llm_records() as records,
    ):
        await _call(kind)
    [record] = records
    assert record.requested_service_tier == record.served_service_tier == "priority"


@pytest.mark.parametrize("kind", KINDS)
async def test_an_explicit_none_requests_no_tier(kind: Kind) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")) as fake,
        capture_llm_records() as records,
    ):
        await _call(kind, service_tier=None)
    assert [request.service_tier for request in fake.requests] == [None]
    [record] = records
    assert record.requested_service_tier is None
    assert record.served_service_tier == "default"


@pytest.mark.parametrize("kind", KINDS)
async def test_the_fake_sees_the_tier_the_host_wrote(kind: Kind) -> None:
    with install(lambda _request: Reply.text(ANSWER)) as fake:
        await _call(kind)
    assert [request.service_tier for request in fake.requests] == ["priority"]


@pytest.mark.parametrize("kind", KINDS)
async def test_the_registered_provider_requests_no_tier(kind: Kind) -> None:
    with install(lambda _request: Reply.text(ANSWER)), capture_llm_records() as records:
        await _call(kind, provider=None, service_tier=UNSET)
    [record] = records
    assert record.requested_service_tier is None
    assert record.served_service_tier is None


@pytest.mark.parametrize("kind", KINDS)
async def test_an_attempt_that_failed_before_any_response_records_the_requested_tier(
    kind: Kind,
) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default", delay=5.0)),
        capture_llm_records() as records,
        # instructor wraps the structured lane's transport error.
        pytest.raises((TimeoutError, InstructorRetryException)),
    ):
        await _call(kind, timeout=0.05)
    [record] = records
    assert record.error is not None
    assert record.requested_service_tier == "priority"
    assert record.served_service_tier is None


@pytest.mark.parametrize("kind", KINDS)
async def test_a_failed_attempt_that_received_a_response_records_the_served_tier(
    kind: Kind,
) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, finish_reason="length", service_tier="flex")),
        capture_llm_records() as records,
        pytest.raises(OutputLimitError),
    ):
        await _call(kind)
    [record] = records
    assert record.requested_service_tier == "priority"
    assert record.served_service_tier == "flex"


@pytest.mark.parametrize("kind", ["text", "structured", "tool"])
async def test_each_retry_attempt_records_its_own_served_tier(kind: Kind) -> None:
    """A provider failure is retried; the retry's silence is not the failure's tier."""
    answers = iter(
        [
            Reply.text(
                "failed",
                finish_reason="stop",
                native_finish_reason="MALFORMED_FUNCTION_CALL",
                service_tier="flex",
            ),
            Reply.text(ANSWER),
        ]
    )
    with install(lambda _request: next(answers)), capture_llm_records() as records:
        await _call(kind, retry=RetryPolicy(backoff_base_seconds=0.0))
    assert [(r.attempt, r.served_service_tier) for r in records] == [(1, "flex"), (2, None)]
    assert records[0].error is not None
    assert records[0].error.startswith(ProviderFailureError.__name__)


async def test_a_structured_repair_records_the_last_completions_tier() -> None:
    answers = iter(
        [Reply.text("not json", service_tier="flex"), Reply.text(ANSWER, service_tier="default")]
    )
    with install(lambda _request: next(answers)) as fake, capture_llm_records() as records:
        await _call("structured")
    assert [request.repair for request in fake.requests] == [False, True]
    [record] = records
    assert record.served_service_tier == "default"


async def test_a_streams_record_is_not_written_by_the_consumers_own_calls() -> None:
    """A consumer's llmkit call between chunks keeps its own tiers, and the stream its."""

    def _answer(request: TransportRequest) -> Reply:
        if request.kind == "text_stream":
            return Reply(content="a b c", service_tier="flex")
        return Reply.text("inner", service_tier="default")

    async def _inner() -> object:
        return await text_llm_call("inner", feature="inner", provider=None, retry=NO_RETRY)

    with install(_answer), capture_llm_records() as records:
        await _call("text_stream", between_chunks=_inner)
    by_feature = {record.feature: record for record in records}
    inner = [record for record in records if record.feature == "inner"]
    assert len(inner) == 3  # one per delta of "a b c"
    assert all(r.requested_service_tier is None for r in inner)
    assert all(r.served_service_tier == "default" for r in inner)
    assert by_feature["tier"].requested_service_tier == "priority"
    assert by_feature["tier"].served_service_tier == "flex"


@pytest.mark.parametrize("kind", ["text", "structured", "text_stream"])
async def test_a_cache_hit_records_the_requested_tier_and_no_served_tier(kind: Kind) -> None:
    configure_llm_cache(InMemoryLLMCache())
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")) as fake,
        capture_llm_records() as records,
    ):
        await _call(kind)
        await _call(kind)
    assert len(fake.requests) == 1
    paid, hit = records
    assert not paid.cache_hit and hit.cache_hit
    assert (paid.requested_service_tier, paid.served_service_tier) == ("priority", "default")
    assert (hit.requested_service_tier, hit.served_service_tier) == ("priority", None)


def test_the_sync_forms_record_both_tiers() -> None:
    """The sync bridge runs the call on llmkit's own loop in a captured context."""
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")),
        capture_llm_records() as records,
    ):
        _ = text_llm_call_sync(
            "q", feature="tier", provider=OPENAI, service_tier="priority", retry=NO_RETRY
        )
        _ = structured_llm_call_sync(
            "q", Answer, feature="tier", provider=OPENAI, service_tier="priority", retry=NO_RETRY
        )
        _ = list(
            text_llm_call_stream_sync(
                "q", feature="tier", provider=OPENAI, service_tier="priority", retry=NO_RETRY
            )
        )
        _ = list(
            tool_llm_call_stream_sync(
                "q",
                [TOOL],
                feature="tier",
                provider=OPENAI,
                service_tier="priority",
                retry=NO_RETRY,
            )
        )
    assert [(r.requested_service_tier, r.served_service_tier) for r in records] == [
        ("priority", "default")
    ] * 4


# --- refusals before any request --------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("setting", [UNSET, None, "priority"])
@pytest.mark.parametrize("own_tier", ["priority", "", 3, {"nested": "priority"}])
async def test_a_tier_in_the_providers_own_request_arguments_is_refused(
    kind: Kind, setting: str | None | Unset, own_tier: object
) -> None:
    """With or without the setting: the setting is the only way to ask for a tier."""
    provider = _OwnTierOpenAI({"service_tier": own_tier})
    with (
        install(lambda _request: Reply.text(ANSWER)) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError, match="completion_kwargs"),
    ):
        await _call(kind, provider=provider, service_tier=setting)
    assert fake.requests == []
    assert records == []


@pytest.mark.parametrize("kind", KINDS)
async def test_a_none_tier_in_the_providers_own_request_arguments_is_not_refused(
    kind: Kind,
) -> None:
    provider = _OwnTierOpenAI({"service_tier": None})
    with install(lambda _request: Reply.text(ANSWER)), capture_llm_records() as records:
        await _call(kind, provider=provider)
    [record] = records
    assert record.requested_service_tier == "priority"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("setting", [UNSET, "priority"])
async def test_a_tier_nested_in_extra_body_is_not_refused(kind: Kind, setting: str | Unset) -> None:
    provider = _OwnTierOpenAI({"extra_body": {"service_tier": "flex"}})
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")) as fake,
        capture_llm_records() as records,
    ):
        await _call(kind, provider=provider, service_tier=setting)
    assert len(fake.requests) == 1
    [record] = records
    assert record.error is None
    expected = None if isinstance(setting, Unset) else setting
    assert record.requested_service_tier == expected
    assert record.served_service_tier == "default"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tier", ["default", "flex", "priority", "auto"])
async def test_ollama_refuses_every_tier_before_any_request(kind: Kind, tier: str) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER)) as fake,
        capture_llm_records() as records,
        pytest.raises(ServiceTierUnsupportedError),
    ):
        await _call(kind, provider=OllamaProvider(model="fake-model"), service_tier=tier)
    assert fake.requests == []
    assert records == []


@pytest.mark.parametrize("kind", ["text_stream", "tool_stream"])
async def test_a_stream_raises_the_refusal_at_its_first_iteration(kind: Kind) -> None:
    provider = _OwnTierOpenAI({"service_tier": "priority"})
    with install(lambda _request: Reply.text(ANSWER)) as fake:
        stream = (
            text_llm_call_stream("q", feature="tier", provider=provider, retry=NO_RETRY)
            if kind == "text_stream"
            else tool_llm_call_stream(
                "q", [TOOL], feature="tier", provider=provider, retry=NO_RETRY
            )
        )
        # Opening the stream sends nothing and raises nothing.
        with pytest.raises(ServiceTierUnsupportedError):
            _ = await anext(aiter(stream))
    assert fake.requests == []


def test_the_sync_forms_refuse_before_any_request() -> None:
    provider = _OwnTierOpenAI({"service_tier": "priority"})
    with install(lambda _request: Reply.text(ANSWER)) as fake, capture_llm_records() as records:
        with pytest.raises(ServiceTierUnsupportedError):
            _ = text_llm_call_sync("q", feature="tier", provider=provider, retry=NO_RETRY)
        with pytest.raises(ServiceTierUnsupportedError):
            _ = list(text_llm_call_stream_sync("q", feature="tier", provider=provider))
    assert fake.requests == []
    assert records == []


# --- the YAML body and the index line ---------------------------------------


def _written(record: LLMCallRecord, log_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    path = LocalYamlLogSink(log_dir).write_returning_path(record)
    assert path is not None
    body = cast("dict[str, object]", yaml.safe_load(path.read_text()))
    [line] = (log_dir / "index.jsonl").read_text().splitlines()
    return body, cast("dict[str, object]", json.loads(line))


@pytest.mark.parametrize("kind", KINDS)
async def test_the_yaml_body_and_the_index_line_carry_both_tiers(
    kind: Kind, tmp_path: Path
) -> None:
    with (
        install(lambda _request: Reply.text(ANSWER, service_tier="default")),
        capture_llm_records() as records,
    ):
        await _call(kind)
    body, line = _written(records[0], tmp_path)
    for written in (body, line):
        assert written["requested_service_tier"] == "priority"
        assert written["served_service_tier"] == "default"


async def test_an_unreported_tier_is_an_explicit_null_in_both_outputs(tmp_path: Path) -> None:
    with install(lambda _request: Reply.text(ANSWER)), capture_llm_records() as records:
        await _call("text_stream")
    body, line = _written(records[0], tmp_path)
    for written in (body, line):
        assert written["requested_service_tier"] == "priority"
        assert "served_service_tier" in written and written["served_service_tier"] is None


def test_a_record_built_without_tiers_defaults_both_to_none() -> None:
    from tests._support import make_record

    record = make_record()
    assert record.requested_service_tier is None
    assert record.served_service_tier is None


# --- the adapter's readers ---------------------------------------------------


def test_the_served_tier_is_read_from_the_response_then_usage_then_traffic_type() -> None:
    response = ModelResponse(usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    assert _served_service_tier(response) is None
    response._hidden_params = {"provider_specific_fields": {"traffic_type": "ON_DEMAND"}}
    assert _served_service_tier(response) == "ON_DEMAND"
    usage = cast("Usage", response.usage)  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — set above
    usage.service_tier = "flex"  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    assert _served_service_tier(response) == "flex"
    response.service_tier = "priority"  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    assert _served_service_tier(response) == "priority"


def test_an_empty_served_tier_falls_through_and_nothing_is_mapped() -> None:
    chunk = ModelResponseStream()
    chunk.service_tier = ""  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    chunk._hidden_params = {"provider_specific_fields": {"traffic_type": "PROVISIONED_THROUGHPUT"}}
    assert _served_service_tier(chunk) == "PROVISIONED_THROUGHPUT"
    assert _served_service_tier(object()) is None


def _fast_mode_response() -> ModelResponse:
    """Anthropic's answer to fast mode: the speed, its capacity tier, a top-level tier."""
    response = ModelResponse(usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
    usage = cast("Usage", response.usage)  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — set above
    usage.speed = "fast"  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    usage.service_tier = "standard"  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    response.service_tier = "priority"  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    return response


def test_when_speed_was_sent_the_usage_speed_comes_first() -> None:
    assert _served_service_tier(_fast_mode_response(), speed_sent=True) == "fast"


def test_when_speed_was_not_sent_the_usage_speed_is_ignored() -> None:
    response = _fast_mode_response()
    assert _served_service_tier(response) == "priority"
    assert _served_service_tier(response, speed_sent=False) == "priority"
    response.service_tier = None  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    assert _served_service_tier(response) == "standard"


def test_when_speed_was_sent_but_not_reported_the_usual_order_applies() -> None:
    response = _fast_mode_response()
    usage = cast("Usage", response.usage)  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — set above
    usage.speed = ""  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    assert _served_service_tier(response, speed_sent=True) == "priority"
    response.service_tier = None  # pyright: ignore[reportAttributeAccessIssue]  # test-helper — a pydantic extra
    assert _served_service_tier(response, speed_sent=True) == "standard"
    assert _served_service_tier(object(), speed_sent=True) is None


# --- real LiteLLM, no fake ---------------------------------------------------

_CHUNK_BASE = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}


def _sse(frames: list[dict[str, object]]) -> bytes:
    return ("".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n").encode()


def _stream_body(tier_on: Literal["every_frame", "usage_frame_only"]) -> bytes:
    frames: list[dict[str, object]] = [
        {
            **_CHUNK_BASE,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
        },
        {**_CHUNK_BASE, "choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {**_CHUNK_BASE, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **_CHUNK_BASE,
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]
    for frame in frames if tier_on == "every_frame" else frames[-1:]:
        frame["service_tier"] = "default"
    return _sse(frames)


_BUFFERED_BODY = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "service_tier": "default",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


class _InProcessOpenAI(OpenAIProvider):
    """OpenAI through LiteLLM's real handler, answered by an in-process transport."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        super().__init__(
            api_key="sk-test", model="gpt-4o-mini", base_url="https://api.openai.com/v1"
        )
        self.sent: list[dict[str, object]] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            self.sent.append(cast("dict[str, object]", json.loads(request.content)))
            return respond(request)

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {**super().completion_kwargs(), "client": self._client}


async def test_real_litellm_buffered_openai_response_records_its_tier() -> None:
    provider = _InProcessOpenAI(lambda _request: httpx.Response(200, json=_BUFFERED_BODY))
    with capture_llm_records() as records:
        assert (
            await text_llm_call(
                "q", feature="tier", provider=provider, service_tier="priority", retry=NO_RETRY
            )
            == "Hello"
        )
    assert provider.sent[0]["service_tier"] == "priority"
    [record] = records
    assert (record.requested_service_tier, record.served_service_tier) == ("priority", "default")


@pytest.mark.parametrize(
    ("tier_on", "served"),
    # Measured at LiteLLM 1.109.1 and at the 1.104.0 floor: both copy a content
    # frame's ``service_tier`` onto the chunk, and neither delivers one that
    # only the final usage frame reports.
    [("every_frame", "default"), ("usage_frame_only", None)],
)
async def test_real_litellm_openai_stream_records_its_tier(
    tier_on: Literal["every_frame", "usage_frame_only"], served: str | None
) -> None:
    provider = _InProcessOpenAI(
        lambda _request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_stream_body(tier_on)
        )
    )
    with capture_llm_records() as records:
        deltas = [
            delta
            async for delta in text_llm_call_stream(
                "q", feature="tier", provider=provider, service_tier="priority", retry=NO_RETRY
            )
        ]
    assert "".join(deltas) == "Hello"
    assert provider.sent[0]["service_tier"] == "priority"
    [record] = records
    assert record.error is None
    assert (record.requested_service_tier, record.served_service_tier) == ("priority", served)
