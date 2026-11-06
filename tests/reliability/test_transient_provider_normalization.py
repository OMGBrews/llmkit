"""Real provider boundaries normalize finite transient gaps without changing retry policy.

Patch LiteLLM itself: testing.install bypasses both normalization boundaries.
The native mapping inventory lives in test_transient_provider_mapping.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import httpx
import instructor
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException
from litellm.exceptions import APIError, BadGatewayError
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from llmkit import (
    LLM_RECOVERABLE_ERRORS,
    LLM_TRANSPORT_ERRORS,
    RetryPolicy,
    TextDeltaEvent,
    ToolDefinition,
    TransientProviderError,
    structured_llm_call,
    text_llm_call,
    text_llm_call_stream,
    tool_llm_call,
    tool_llm_call_stream,
)
from llmkit.exceptions import normalize_transient_provider_error, underlying_provider_error
from llmkit.rate_limiting._outcomes import is_outage_signal, is_throttle_signal
from llmkit.retry import _retry_after_seconds, with_retries
from tests._support import OkSchema

_TOOLS = [ToolDefinition.from_model("check", OkSchema, "Check a boolean.")]
_LANES = ("text", "structured", "tool", "text_stream", "tool_stream")


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.name = "test-provider"
    provider.model = "test-model"
    provider.completion_kwargs = MagicMock(return_value={"api_key": "offline-placeholder"})
    provider.litellm_model = MagicMock(return_value="openai/test-model")
    provider.instructor_mode = instructor.Mode.JSON_SCHEMA
    provider.reasoning_effort = None
    provider.strict_json_schema = False
    provider.single_tool_call_per_schema = False
    provider.compose_tools_schema = False
    provider.supports_tool_choice = True
    return provider


def _error(status: int, *, direct: float | None = 2.0) -> Exception:
    response = httpx.Response(
        status,
        headers={"retry-after": "7", "retry-after-ms": "3000"},
        request=httpx.Request("POST", "https://offline.invalid/completions"),
    )
    if status == 502:
        error = BadGatewayError(
            "temporary", llm_provider="test-provider", model="test-model", response=response
        )
    else:
        error = APIError(status, "temporary", llm_provider="test-provider", model="test-model")
        setattr(error, "response", response)
    if direct is not None:
        setattr(error, "retry_after", direct)
    return error


async def _invoke(lane: str, policy: RetryPolicy) -> list[object]:
    provider = _provider()
    if lane == "text":
        return [await text_llm_call("hi", feature="normalization", provider=provider, retry=policy)]
    if lane == "structured":
        return [
            await structured_llm_call(
                "hi", OkSchema, feature="normalization", provider=provider, retry=policy
            )
        ]
    if lane == "tool":
        return [
            await tool_llm_call(
                "hi", _TOOLS, feature="normalization", provider=provider, retry=policy
            )
        ]
    if lane == "text_stream":
        return [
            item
            async for item in text_llm_call_stream(
                "hi", feature="normalization", provider=provider, retry=policy
            )
        ]
    assert lane == "tool_stream"
    return [
        item
        async for item in tool_llm_call_stream(
            "hi", _TOOLS, feature="normalization", provider=provider, retry=policy
        )
    ]


@pytest.mark.parametrize("status", [500, 502, 529])
@pytest.mark.parametrize("lane", _LANES)
async def test_exhaustion_preserves_budget_catchability_and_metadata(
    status: int, lane: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _error(status)
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    def min_jitter(low: float, _high: float) -> float:
        return low

    monkeypatch.setattr("llmkit.retry.asyncio.sleep", sleep)
    monkeypatch.setattr("llmkit.retry.random.uniform", min_jitter)
    caught: Exception | None = None
    with patch("llmkit._litellm.litellm.acompletion", side_effect=original) as send:
        try:
            _ = await _invoke(lane, RetryPolicy(max_attempts=3))
        except LLM_RECOVERABLE_ERRORS as error:
            caught = error
    assert send.call_count == 3
    assert sleeps == [2.0, 2.0]
    assert caught is not None
    if lane == "structured":
        assert isinstance(caught, InstructorRetryException)
    root = underlying_provider_error(caught)
    assert isinstance(root, TransientProviderError)
    assert root.__cause__ is original
    assert root.status_code == status
    assert root.provider == "test-provider"
    assert root.model == "test-model"
    assert root.response is getattr(original, "response")
    assert str(root) == str(original)
    assert _retry_after_seconds(caught) == 2.0


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize(
    "retry_on,attempts",
    [
        ((TransientProviderError,), 2),
        ((), 1),
        ((TimeoutError,), 1),
        ((BadGatewayError,), 1),
        ((openai.APIError,), 1),
    ],
)
async def test_custom_policy_matches_only_normalized_type(
    lane: str, retry_on: tuple[type[BaseException], ...], attempts: int
) -> None:
    original = _error(502, direct=0)
    policy = RetryPolicy(max_attempts=2, retry_on=retry_on, backoff_base_seconds=0)
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=original) as send,
        pytest.raises(LLM_RECOVERABLE_ERRORS) as caught,
    ):
        _ = await _invoke(lane, policy)
    assert send.call_count == attempts
    assert isinstance(underlying_provider_error(caught.value), TransientProviderError)


@pytest.mark.parametrize("status", [500, 502, 529])
@pytest.mark.parametrize("direct,expected", [(2.0, 2.0), (None, 3.0), (-1.0, 0.0)])
def test_normalization_preserves_timing_status_and_identity(
    status: int, direct: float | None, expected: float
) -> None:
    original = _error(status, direct=direct)
    normalized = normalize_transient_provider_error(original)
    assert isinstance(normalized, TransientProviderError)
    assert _retry_after_seconds(normalized) == _retry_after_seconds(original) == expected
    assert normalized.response is getattr(original, "response")
    assert is_throttle_signal(normalized) == is_throttle_signal(original) == (status == 529)
    assert is_outage_signal(normalized) == is_outage_signal(original) is True
    assert normalize_transient_provider_error(normalized) is normalized


def test_missing_metadata_and_attribute_only_wait() -> None:
    original = openai.APIError(
        "temporary", request=httpx.Request("POST", "https://offline.invalid"), body=None
    )
    setattr(original, "status_code", 529)
    setattr(original, "retry_after", 4)
    normalized = normalize_transient_provider_error(original)
    assert isinstance(normalized, TransientProviderError)
    assert normalized.status_code == 529
    assert normalized.response is None
    assert normalized.provider is None
    assert normalized.model is None
    assert _retry_after_seconds(normalized) == 4


@pytest.mark.parametrize("status", [400, 401, 403, 501, 504, 520, None, "502", 502.0])
def test_uncovered_nonselected_or_malformed_api_status_passes_through(status: object) -> None:
    original = openai.APIError(
        "permanent or malformed",
        request=httpx.Request("POST", "https://offline.invalid"),
        body=None,
    )
    if status is not None:
        setattr(original, "status_code", status)
    assert normalize_transient_provider_error(original) is original
    assert not isinstance(original, LLM_TRANSPORT_ERRORS)


@pytest.mark.parametrize("status", [500, 502, 529])
def test_programming_error_cannot_opt_in_by_status_attribute(status: int) -> None:
    original = TypeError("programming failure")
    setattr(original, "status_code", status)
    assert normalize_transient_provider_error(original) is original
    assert not isinstance(original, LLM_RECOVERABLE_ERRORS)


@pytest.mark.parametrize("status", [500, 502, 529])
def test_recognized_transport_error_retains_identity(status: int) -> None:
    original = openai.InternalServerError(
        "already retryable",
        response=httpx.Response(status, request=httpx.Request("POST", "https://offline.invalid")),
        body=None,
    )
    assert normalize_transient_provider_error(original) is original


@pytest.mark.parametrize("lane", ["text_stream", "tool_stream"])
@pytest.mark.parametrize("delivered", [False, True], ids=["keepalive", "public-output"])
async def test_stream_iteration_retries_only_before_public_output(
    lane: str, delivered: bool
) -> None:
    original = _error(529, direct=0)
    received: list[object] = []

    async def frames() -> AsyncIterator[ModelResponseStream]:
        if delivered:
            yield ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content="hello"))]
            )
        else:
            yield ModelResponseStream(choices=[])
        raise original

    async def send(**_kwargs: object) -> AsyncIterator[ModelResponseStream]:
        return frames()

    caught: Exception | None = None
    policy = RetryPolicy(max_attempts=3, backoff_base_seconds=0)
    provider = _provider()
    with patch("llmkit._litellm.litellm.acompletion", side_effect=send) as mock:
        try:
            if lane == "text_stream":
                async for item in text_llm_call_stream(
                    "hi", feature="normalization", provider=provider, retry=policy
                ):
                    received.append(item)
            else:
                async for event in tool_llm_call_stream(
                    "hi", _TOOLS, feature="normalization", provider=provider, retry=policy
                ):
                    received.append(event)
        except LLM_RECOVERABLE_ERRORS as error:
            caught = error
    assert isinstance(caught, TransientProviderError)
    assert caught.__cause__ is original
    assert mock.call_count == (1 if delivered else 3)
    assert received == (
        ["hello" if lane == "text_stream" else TextDeltaEvent("hello")] if delivered else []
    )


async def test_text_recovers_after_transient_failure() -> None:
    response = ModelResponse(
        choices=[
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ]
    )
    with patch(
        "llmkit._litellm.litellm.acompletion", side_effect=[_error(500, direct=0), response]
    ) as send:
        assert await _invoke("text", RetryPolicy(backoff_base_seconds=0)) == ["ok"]
    assert send.call_count == 2


async def test_raw_call_inside_with_retries_has_no_normalization() -> None:
    original = _error(502)
    attempts = 0

    async def raw() -> None:
        nonlocal attempts
        attempts += 1
        raise original

    with pytest.raises(BadGatewayError) as caught:
        await with_retries(raw, policy=RetryPolicy(backoff_base_seconds=0))
    assert caught.value is original
    assert attempts == 1


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize("status", [400, 401, 403, 520, 502])
async def test_permanent_and_programming_failures_do_not_retry(lane: str, status: int) -> None:
    if status == 502:
        original: Exception = TypeError("programming failure with misleading status")
        setattr(original, "status_code", status)
    else:
        original = _error(status, direct=0)
    with (
        patch("llmkit._litellm.litellm.acompletion", side_effect=original) as send,
        pytest.raises(Exception) as caught,
    ):
        _ = await _invoke(lane, RetryPolicy(backoff_base_seconds=0))
    assert send.call_count == 1
    assert underlying_provider_error(caught.value) is original
