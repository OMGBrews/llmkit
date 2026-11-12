"""Finish classification survives every public lane without repair or cache leaks.

Synthetic post-LiteLLM replies deliberately vary mapped and native reasons.
The mapper guard catches dependency changes that invalidate the failure list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.types.llms.openai import OpenAIChatCompletionChoices
from litellm.types.llms.vertex_ai import Candidates
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from llmkit import (
    NO_RETRY,
    ContentFilterError,
    FinishReasonError,
    InMemoryLLMCache,
    LocalYamlLogSink,
    OutputLimitError,
    ProviderFailureError,
    RetryPolicy,
    StructuredOutputError,
    TextDeltaEvent,
    TokenUsage,
    ToolCallResult,
    ToolDefinition,
    capture_llm_records,
    configure_llm_cache,
    structured_llm_call,
    text_llm_call,
    text_llm_call_stream,
    tool_llm_call,
    tool_llm_call_stream,
)
from llmkit._litellm import FAILURE_NATIVES, classify_finish
from llmkit.testing import FakeToolCall, Reply, install

Kind = Literal["text", "structured", "tool", "text_stream", "tool_stream"]
KINDS: tuple[Kind, ...] = ("text", "structured", "tool", "text_stream", "tool_stream")
BUFFERED: tuple[Kind, ...] = ("text", "structured", "tool")


class Answer(BaseModel):
    value: int


TOOL = ToolDefinition.from_model("answer", Answer)
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


async def _call(
    kind: Kind, *, retry: RetryPolicy = NO_RETRY, seen: list[object] | None = None
) -> object:
    if kind == "text":
        return await text_llm_call("q", feature="finish", retry=retry)
    if kind == "structured":
        return await structured_llm_call("q", Answer, feature="finish", retry=retry)
    if kind == "tool":
        return await tool_llm_call("q", [TOOL], feature="finish", retry=retry)
    collected = seen if seen is not None else []
    if kind == "text_stream":
        async for delta in text_llm_call_stream("q", feature="finish", retry=retry):
            collected.append(delta)
    else:
        async for event in tool_llm_call_stream("q", [TOOL], feature="finish", retry=retry):
            collected.append(event)
    return collected


@pytest.mark.parametrize("native", sorted(FAILURE_NATIVES))
def test_failure_native_still_maps_to_stop(native: str) -> None:
    assert map_finish_reason(native) == "stop"
    assert classify_finish("stop", native) == "failed"


@pytest.mark.parametrize("mapped", ["tool_calls", "function_call", None, "new_reason"])
@pytest.mark.parametrize("native", ["MALFORMED_FUNCTION_CALL", "unrecognized", None])
def test_only_mapped_stop_consults_failure_natives(mapped: str | None, native: str | None) -> None:
    assert classify_finish(mapped, native) == "complete"
    assert classify_finish("length", native) == "truncated"
    assert classify_finish("content_filter", native) == "filtered"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("mapped", "native", "error_type"),
    [
        ("stop", "MALFORMED_FUNCTION_CALL", ProviderFailureError),
        ("content_filter", "SAFETY", ContentFilterError),
    ],
)
async def test_every_lane_raises_owned_finish_error_and_logs_evidence(
    kind: Kind,
    mapped: str,
    native: str,
    error_type: type[ProviderFailureError | ContentFilterError],
) -> None:
    reply = Reply.text(
        "partial", finish_reason=mapped, native_finish_reason=native, usage=TokenUsage(2, 3, 5)
    )
    seen: list[object] = []
    with install(lambda _request: reply) as fake, capture_llm_records() as records:
        with pytest.raises(error_type) as caught:
            _ = await _call(kind, seen=seen)
    assert len(fake.requests) == 1  # structured must not spend an in-call parse repair
    assert caught.value.finish_reason == mapped
    assert caught.value.native_finish_reason == native
    assert caught.value.partial_text == "partial"
    assert not hasattr(caught.value, "status_code")
    [record] = records
    assert record.error is not None and record.error.startswith(error_type.__name__ + ":")
    assert record.partial_text == "partial"
    assert record.native_finish_reason == native
    assert record.output_limit is None
    assert record.usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    if kind == "text_stream":
        assert seen == ["partial"]
    elif kind == "tool_stream":
        assert seen == [TextDeltaEvent("partial")]
        assert not any(isinstance(item, ToolCallResult) for item in seen)


@pytest.mark.parametrize("kind", BUFFERED)
async def test_provider_failure_retries_under_transient_budget(kind: Kind) -> None:
    answers = iter(
        [
            Reply.text(
                "failed", finish_reason="stop", native_finish_reason="MALFORMED_FUNCTION_CALL"
            ),
            Reply.text('{"value": 1}'),
        ]
    )
    with install(lambda _request: next(answers)) as fake:
        _ = await _call(kind, retry=_NO_BACKOFF)
    assert [(request.attempt, request.repair) for request in fake.requests] == [
        (1, False),
        (2, False),
    ]


@pytest.mark.parametrize("kind", BUFFERED)
async def test_persistent_provider_failure_spends_three_outer_attempts(kind: Kind) -> None:
    reply = Reply.text(
        "failed", finish_reason="stop", native_finish_reason="MALFORMED_FUNCTION_CALL"
    )
    with install(lambda _request: reply) as fake:
        with pytest.raises(ProviderFailureError):
            _ = await _call(kind, retry=_NO_BACKOFF)
    assert [(request.attempt, request.repair) for request in fake.requests] == [
        (1, False),
        (2, False),
        (3, False),
    ]


@pytest.mark.parametrize("kind", KINDS)
async def test_filter_is_zero_budget_by_default(kind: Kind) -> None:
    with install(lambda _request: Reply.text("filtered", finish_reason="content_filter")) as fake:
        with pytest.raises(ContentFilterError):
            _ = await _call(kind, retry=_NO_BACKOFF)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("kind", BUFFERED)
async def test_filter_retry_requires_explicit_opt_in(kind: Kind) -> None:
    answers = iter(
        [Reply.text("filtered", finish_reason="content_filter"), Reply.text('{"value": 1}')]
    )
    policy = RetryPolicy(retry_on=(ContentFilterError,), backoff_base_seconds=0.0)
    with install(lambda _request: next(answers)) as fake:
        _ = await _call(kind, retry=policy)
    assert len(fake.requests) == 2


@pytest.mark.parametrize("kind", ["text_stream", "tool_stream"])
@pytest.mark.parametrize("reason", ["stop", "content_filter"])
async def test_stream_finish_error_is_never_replayed_after_prose(kind: Kind, reason: str) -> None:
    policy = RetryPolicy(
        retry_on=(ProviderFailureError, ContentFilterError), backoff_base_seconds=0.0
    )
    reply = Reply(
        chunks=["partial"], finish_reason=reason, native_finish_reason="MALFORMED_FUNCTION_CALL"
    )
    with install(lambda _request: reply) as fake:
        with pytest.raises((ProviderFailureError, ContentFilterError)):
            _ = await _call(kind, retry=policy)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("kind", ["text", "structured", "text_stream"])
@pytest.mark.parametrize("mapped", ["stop", "content_filter"])
async def test_finish_failures_never_enter_cache(kind: Kind, mapped: str) -> None:
    configure_llm_cache(InMemoryLLMCache())
    answers = iter(
        [
            Reply.text(
                "partial", finish_reason=mapped, native_finish_reason="MALFORMED_FUNCTION_CALL"
            ),
            Reply.text('{"value": 1}'),
        ]
    )
    with install(lambda _request: next(answers)) as fake:
        with pytest.raises((ProviderFailureError, ContentFilterError)):
            _ = await _call(kind)
        _ = await _call(kind)
        _ = await _call(kind)
    assert len(fake.requests) == 2


@pytest.mark.parametrize("kind", KINDS)
async def test_unknown_native_is_complete_and_logged(kind: Kind) -> None:
    with (
        install(lambda _request: Reply.text('{"value": 1}', native_finish_reason="FUTURE_REASON")),
        capture_llm_records() as records,
    ):
        result = await _call(kind)
    assert records[0].error is None
    assert records[0].native_finish_reason == "FUTURE_REASON"
    if isinstance(result, ToolCallResult):
        assert result.native_stop_reason == "FUTURE_REASON"


@pytest.mark.parametrize("kind", ["tool", "tool_stream"])
async def test_truncated_tool_arguments_raise_before_argument_validation(kind: Kind) -> None:
    reply = Reply(
        content="partial", tool_calls=[FakeToolCall("answer", '{"value":')], finish_reason="length"
    )
    with install(lambda _request: reply) as fake:
        with pytest.raises(OutputLimitError):
            _ = await _call(kind)
    assert len(fake.requests) == 1


async def test_litellm_choice_native_reaches_classifier_and_record() -> None:
    # Real Choices construction performs LiteLLM's normalization: an unknown
    # failure native maps to stop while remaining in provider_specific_fields.
    response = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "MALFORMED_FUNCTION_CALL",
                "message": {"role": "assistant", "content": "partial"},
            }
        ]
    )
    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=response)),
        capture_llm_records() as records,
    ):
        with pytest.raises(ProviderFailureError):
            _ = await text_llm_call("q", feature="finish", retry=NO_RETRY)
    assert records[0].native_finish_reason == "MALFORMED_FUNCTION_CALL"


def test_no_choice_cannot_claim_a_native_reason() -> None:
    with pytest.raises(ValueError):
        _ = Reply(no_choice=True, native_finish_reason="ERROR")


@pytest.mark.parametrize("route", ["openai", "gemini"])
async def test_provider_transform_native_survives_to_public_error(route: str) -> None:
    if route == "openai":
        choices = OpenAIGPTConfig()._transform_choices(  # pyright: ignore[reportUnknownMemberType]  # raw-llm — untyped provider transform under test
            [
                cast(
                    "OpenAIChatCompletionChoices",
                    cast(
                        "object",
                        {
                            "index": 0,
                            "finish_reason": "MALFORMED_FUNCTION_CALL",
                            "message": {"role": "assistant", "content": "partial"},
                        },
                    ),
                )
            ]
        )
        response = ModelResponse(choices=choices)
    else:
        response = ModelResponse()
        _ = VertexGeminiConfig._process_candidates(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # raw-llm — untyped provider transform under test
            [
                cast(
                    "Candidates",
                    cast(
                        "object",
                        {
                            "finishReason": "MALFORMED_FUNCTION_CALL",
                            "content": {"role": "model", "parts": [{"text": "partial"}]},
                        },
                    ),
                )
            ],
            response,
            {},
        )
        # Gemini appends; the default first choice is not the candidate.
        response.choices = response.choices[-1:]
    with (
        patch("llmkit._litellm.litellm.acompletion", AsyncMock(return_value=response)),
        capture_llm_records() as records,
    ):
        with pytest.raises(ProviderFailureError) as caught:
            _ = await text_llm_call("q", feature="finish", retry=NO_RETRY)
    assert caught.value.partial_text == "partial"
    assert caught.value.native_finish_reason == "MALFORMED_FUNCTION_CALL"
    assert records[0].native_finish_reason == "MALFORMED_FUNCTION_CALL"


@pytest.mark.parametrize("kind", BUFFERED)
async def test_filter_stays_zero_budget_under_broad_validation_policy(kind: Kind) -> None:
    policy = RetryPolicy(validation_retry_on=(Exception,), backoff_base_seconds=0.0)
    with install(lambda _request: Reply.text("filtered", finish_reason="content_filter")) as fake:
        with pytest.raises(ContentFilterError):
            _ = await _call(kind, retry=policy)
    assert len(fake.requests) == 1


async def test_native_reason_written_to_yaml_body_only(tmp_path: Path) -> None:
    with (
        install(lambda _request: Reply.text("ok", native_finish_reason="FUTURE_REASON")),
        capture_llm_records() as records,
    ):
        _ = await _call("text")
    path = LocalYamlLogSink(tmp_path).write_returning_path(records[0])
    assert path is not None
    assert "native_finish_reason: FUTURE_REASON" in path.read_text()
    assert "native_finish_reason" not in (tmp_path / "index.jsonl").read_text()


@pytest.mark.parametrize("kind", KINDS)
async def test_finish_failure_is_neutral_inside_rate_limit_slot(kind: Kind) -> None:
    from llmkit.rate_limiting import configure_rate_limit
    from llmkit.rate_limiting._outcomes import is_outage_signal, is_throttle_signal

    configure_rate_limit(enabled=True, adaptive=True, breaker=True)
    reply = Reply.text("partial", native_finish_reason="MALFORMED_FUNCTION_CALL")
    with (
        install(lambda _request: reply),
        patch("llmkit.rate_limiting._limiter.record_gate_outcome") as record,
    ):
        with pytest.raises(ProviderFailureError):
            _ = await _call(kind)
    assert record.call_count == 1
    outcome = cast("object", record.call_args.kwargs["outcome"])
    assert isinstance(outcome, FinishReasonError)
    assert not is_outage_signal(outcome)
    assert not is_throttle_signal(outcome)


async def test_structured_parse_failure_logs_last_repair_native() -> None:
    replies = iter(
        [
            Reply.text("invalid", native_finish_reason="FIRST_REASON"),
            Reply.text("still invalid", native_finish_reason="LAST_REASON"),
        ]
    )
    with install(lambda _request: next(replies)) as fake, capture_llm_records() as records:
        with pytest.raises(StructuredOutputError) as caught:
            _ = await _call("structured")
    assert len(fake.requests) == 2
    assert caught.value.native_finish_reason == "LAST_REASON"
    assert records[0].native_finish_reason == "LAST_REASON"
