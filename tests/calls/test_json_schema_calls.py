"""Schema authority survives real parsing, repair, caching, and tool lanes."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from llmkit import (
    NO_RETRY,
    InMemoryLLMCache,
    Provider,
    ResultValidationError,
    ToolDefinition,
    ToolName,
    configure_llm_cache,
    make_provider,
    model_from_json_schema,
    structured_llm_call,
    structured_llm_call_sync,
    tool_llm_call,
)
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install
from tests._support import capturing_sink

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"code": {"type": "string", "pattern": "^[0-9]+$"}},
    "required": ["code"],
}


async def test_invalid_answer_repaired_by_real_instructor() -> None:
    model = model_from_json_schema(_SCHEMA)
    answers = iter(['{"code":"bad"}', '{"code":"123"}'])
    with install(lambda _request: Reply.text(next(answers))) as installed:
        result = await structured_llm_call("q", model, feature="f", retry=NO_RETRY)
    assert result.root == {"code": "123"}
    assert len(installed.requests) == 2
    assert installed.requests[1].repair is True


async def test_invalid_answer_exhausts_bounded_repair() -> None:
    from instructor.core import InstructorRetryException

    model = model_from_json_schema(_SCHEMA)
    with install(lambda _request: Reply.text('{"code":"bad"}')) as installed:
        with pytest.raises(InstructorRetryException):
            _ = await structured_llm_call("q", model, feature="f", retry=NO_RETRY)
    assert 1 < len(installed.requests) <= 3


async def test_nonfinite_answer_is_repaired_before_normalization_fallback() -> None:
    model = model_from_json_schema({"type": "object", "properties": {"x": {"type": "number"}}})
    answers = iter(['{"x":NaN}', '{"x":1}'])
    with install(lambda _request: Reply.text(next(answers))) as installed:
        result = await structured_llm_call("q", model, feature="f", retry=NO_RETRY)
    assert result.root == {"x": 1}
    assert len(installed.requests) == 2


async def test_lossless_paid_cached_and_shared_results_and_logs() -> None:
    configure_llm_cache(InMemoryLLMCache())
    model = model_from_json_schema({"type": "object", "minProperties": 1})
    value = {"null": None, "nested": {"large": 9007199254740993, "values": [None, True]}}
    expected = model.model_validate(value)
    with (
        install(lambda _request: Reply.json(expected, delay=0.02)) as installed,
        capturing_sink() as records,
    ):
        paid, shared = await asyncio.gather(
            structured_llm_call("same", model, feature="f"),
            structured_llm_call("same", model, feature="f"),
        )
        hit = await structured_llm_call("same", model, feature="f")
    assert len(installed.requests) == 1
    assert all(
        type(result) is model and result.model_dump() == value for result in (paid, shared, hit)
    )
    assert all(cast("object", record.response) == value for record in records)


def test_sync_schema_backed_result() -> None:
    model = model_from_json_schema({"type": "object"})
    expected = model.model_validate({"null": None, "n": 9007199254740993})
    with install(lambda _request: Reply.json(expected)):
        result = structured_llm_call_sync("q", model, feature="f")
    assert type(result) is model
    assert result.root == expected.root


async def test_tool_compose_uses_original_authority() -> None:
    model = model_from_json_schema(_SCHEMA)
    provider = make_provider(Provider.OPENAI, api_key="fake")
    with install(lambda _request: Reply.text('{"code":"bad"}')):
        with pytest.raises(ResultValidationError):
            _ = await tool_llm_call(
                "q", [], output_schema=model, feature="f", provider=provider, retry=NO_RETRY
            )
    with install(lambda _request: Reply.text('{"code":"123"}')):
        result = await tool_llm_call("q", [], output_schema=model, feature="f", provider=provider)
    assert result.parsed is not None
    assert result.parsed.root == {"code": "123"}
    assert result.to_log_dict()["parsed"] == {"code": "123"}


async def test_schema_backed_tool_arguments() -> None:
    model = model_from_json_schema(_SCHEMA)
    tool = ToolDefinition.from_model("lookup", model, "Look up a code")
    provider = make_provider(Provider.OPENAI, api_key="fake")

    def answer(_request: TransportRequest) -> Reply:
        return Reply(tool_calls=[FakeToolCall("lookup", '{"code":"123"}')])

    with install(answer):
        result = await tool_llm_call(
            "q", [tool], feature="f", provider=provider, tool_choice=ToolName("lookup")
        )
    assert result.tool_calls[0].validated is not None
    assert result.tool_calls[0].validated.model_dump() == {"code": "123"}


async def test_engine_fault_is_not_reasked_as_a_bad_answer() -> None:
    from instructor.core import InstructorRetryException

    from llmkit import SchemaEngineError
    from llmkit.exceptions import underlying_provider_error

    # jsonschema's floating multipleOf arithmetic overflows when converting this
    # JSON integer to float. This is an engine limitation, not invalid model data.
    model = model_from_json_schema({"type": "object", "properties": {"x": {"multipleOf": 0.1}}})
    raw = '{"x":' + str(10**1000) + "}"
    with install(lambda _request: Reply.text(raw)) as installed:
        with pytest.raises(InstructorRetryException) as caught:
            _ = await structured_llm_call("q", model, feature="f")
    assert isinstance(underlying_provider_error(caught.value), SchemaEngineError)
    assert len(installed.requests) == 1


@pytest.mark.parametrize("tool_arguments", [False, True])
async def test_tool_lanes_do_not_retry_engine_faults(tool_arguments: bool) -> None:
    from llmkit import SchemaEngineError

    model = model_from_json_schema({"type": "object", "properties": {"x": {"multipleOf": 0.1}}})
    raw = '{"x":' + str(10**1000) + "}"
    provider = make_provider(Provider.OPENAI, api_key="fake")
    tool = ToolDefinition.from_model("lookup", model, "Look up a value")
    reply = Reply(tool_calls=[FakeToolCall("lookup", raw)]) if tool_arguments else Reply.text(raw)
    with install(lambda _request: reply) as installed:
        with pytest.raises(SchemaEngineError):
            if tool_arguments:
                _ = await tool_llm_call("q", [tool], feature="f", provider=provider)
            else:
                _ = await tool_llm_call(
                    "q", [], output_schema=model, feature="f", provider=provider
                )
    assert len(installed.requests) == 1
