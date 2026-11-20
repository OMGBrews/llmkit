"""A refused tool call claims nothing from an enclosing retry loop.

Inside :func:`~llmkit.retry.with_retries`, a call claims its ``(feature,
label)`` occurrence and so its ``call_id`` and attempt count (see
:func:`~llmkit.retry.claim_logical_call`). A guard that refuses the call's
arguments must raise before that claim. Had it claimed, a valid call that
follows it with the same ``feature`` would be the second occurrence on the
first pass and the first on the second pass, so its two attempts would carry
different call ids and the second id would start at attempt 2.

Each test refuses one call on pass 1 only, then makes a valid call that times
out on pass 1 and succeeds on pass 2. The two records must be attempts 1 and 2
of one call id. The service-tier refusal, which shares this property, is pinned
in ``test_service_tier_setting.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

import openai
import pytest
from pydantic import BaseModel

from llmkit import (
    ComposeUnsupportedError,
    LLMClientConfig,
    Provider,
    ToolChoiceUnsupportedError,
    ToolDefinition,
    UnsupportedSchemaError,
    capture_llm_records,
    configure_llm_client,
)
from llmkit import calls as llm_calls
from llmkit.providers import AnthropicProvider, OllamaProvider
from llmkit.retry import RetryPolicy, with_retries
from llmkit.testing import Reply, TransportRequest, install, timeout_error


class Answer(BaseModel):
    value: int


TOOL = ToolDefinition.from_model("answer", Answer)
FEATURE = "same"


def _ollama() -> OllamaProvider:
    """A provider that refuses ``tool_choice`` other than ``"auto"`` and cannot compose."""
    return OllamaProvider(model="fake-model")


async def _valid_tool_call() -> None:
    _ = await llm_calls.tool_llm_call("q", [TOOL], feature=FEATURE, provider=_ollama())


async def _valid_tool_stream() -> None:
    async for _event in llm_calls.tool_llm_call_stream(
        "q", [TOOL], feature=FEATURE, provider=_ollama()
    ):
        pass


async def _refused_tool_choice() -> None:
    _ = await llm_calls.tool_llm_call(
        "q", [TOOL], feature=FEATURE, provider=_ollama(), tool_choice="required"
    )


async def _refused_tool_choice_stream() -> None:
    async for _event in llm_calls.tool_llm_call_stream(
        "q", [TOOL], feature=FEATURE, provider=_ollama(), tool_choice="required"
    ):
        pass


async def _refused_output_schema() -> None:
    _ = await llm_calls.tool_llm_call(
        "q",
        [TOOL],
        feature=FEATURE,
        provider=_ollama(),
        output_schema=cast("type[BaseModel]", dict),
    )


async def _refused_compose_provider() -> None:
    _ = await llm_calls.tool_llm_call(
        "q", [TOOL], feature=FEATURE, provider=_ollama(), output_schema=Answer
    )


async def _refused_compose_model() -> None:
    # Anthropic composes, but not on a model without native structured output.
    provider = AnthropicProvider(api_key="k", model="claude-legacy")
    _ = await llm_calls.tool_llm_call(
        "q", [TOOL], feature=FEATURE, provider=provider, output_schema=Answer
    )


async def _refused_unbuildable_config() -> None:
    # No provider for the call: the registered config is built again for the
    # compose check, and its own build error is the refusal.
    _ = await llm_calls.tool_llm_call("q", [TOOL], feature=FEATURE, output_schema=Answer)


type Call = Callable[[], Awaitable[None]]

CASES: list[tuple[Call, type[BaseException], str, Call]] = [
    (
        _refused_tool_choice,
        ToolChoiceUnsupportedError,
        "does not support tool_choice",
        _valid_tool_call,
    ),
    (
        _refused_tool_choice_stream,
        ToolChoiceUnsupportedError,
        "does not support tool_choice",
        _valid_tool_stream,
    ),
    (
        _refused_output_schema,
        UnsupportedSchemaError,
        "must be a pydantic BaseModel",
        _valid_tool_call,
    ),
    (
        _refused_compose_provider,
        ComposeUnsupportedError,
        "does not support combining",
        _valid_tool_call,
    ),
    (_refused_compose_model, ComposeUnsupportedError, "cannot safely combine", _valid_tool_call),
    (_refused_unbuildable_config, ValueError, "requires an API key", _valid_tool_call),
]
IDS = [
    "tool_choice-buffered",
    "tool_choice-stream",
    "unsupported-output-schema",
    "compose-unsupported-provider",
    "compose-unsupported-model",
    "compose-unbuildable-config",
]


@pytest.mark.parametrize(("refused", "error", "match", "valid"), CASES, ids=IDS)
async def test_a_refused_tool_call_counts_no_attempt_in_an_outer_loop(
    monkeypatch: pytest.MonkeyPatch,
    refused: Call,
    error: type[BaseException],
    match: str,
    valid: Call,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Only the unbuildable-config case reads the registered config; every
    # other call names its own provider.
    _ = configure_llm_client(LLMClientConfig(provider=Provider.ANTHROPIC, model="m"))
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
            with pytest.raises(error, match=match):
                await refused()
        await valid()

    with install(_fake), capture_llm_records() as records:
        await with_retries(
            _pass,
            policy=RetryPolicy(
                max_attempts=3, retry_on=(openai.APITimeoutError,), backoff_base_seconds=0.0
            ),
        )
    assert passes == [2]
    assert sent == [2]
    assert [r.attempt for r in records] == [1, 2]
    assert len({r.call_id for r in records}) == 1
