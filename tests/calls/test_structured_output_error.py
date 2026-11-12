"""A failed structured answer raises :class:`~llmkit.StructuredOutputError`.

Both lanes that return a validated instance raise the one llmkit-owned type
when the answer does not validate: the structured lane re-owns instructor's
exhausted parse-cause ``InstructorRetryException``, and the compose lane
(``tool_llm_call`` with ``output_schema=``) raises it for a final answer that
fails validation or a turn with neither tool calls nor text. A compose final
answer cut off by its output limit raises :class:`~llmkit.OutputLimitError`
instead and is not re-asked. Every test answers through the public
:mod:`llmkit.testing` fake.
"""

from __future__ import annotations

import pickle
from typing import cast

import pytest
from instructor.core import InstructorRetryException
from pydantic import BaseModel, field_validator

from llmkit import (
    LLM_RECOVERABLE_ERRORS,
    LLM_SCHEMA_ERRORS,
    NO_RETRY,
    LlmkitError,
    OutputLimitError,
    Provider,
    StructuredOutputError,
    TokenUsage,
    ToolDefinition,
    capture_llm_records,
    make_provider,
    structured_llm_call,
    structured_llm_call_sync,
    tool_llm_call,
)
from llmkit.testing import FakeToolCall, Reply, install


class _Positive(BaseModel):
    n: int

    @field_validator("n")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class _Args(BaseModel):
    x: int


_PROVIDER = make_provider(Provider.OPENAI, api_key="fake")


def test_the_type_is_owned_and_in_the_schema_set() -> None:
    assert issubclass(StructuredOutputError, LlmkitError)
    assert not issubclass(StructuredOutputError, InstructorRetryException)
    assert StructuredOutputError in LLM_SCHEMA_ERRORS
    assert StructuredOutputError in LLM_RECOVERABLE_ERRORS


async def test_structured_failure_names_the_field_and_keeps_the_answer() -> None:
    with (
        install(lambda _request: Reply.text('{"n": -1}')) as fake,
        capture_llm_records() as records,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call(
            "q", _Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    exc = caught.value
    assert exc.schema_name == "_Positive"
    assert exc.raw_text == '{"n": -1}'
    assert exc.errors is not None
    assert [(error["type"], error["loc"]) for error in exc.errors] == [("value_error", ("n",))]
    assert exc.attempt == 1
    assert exc.provider == _PROVIDER.name
    assert exc.model is not None
    assert isinstance(exc.__cause__, InstructorRetryException)
    # instructor's one in-call repair re-ask ran before the failure surfaced.
    assert len(fake.requests) == 2
    [record] = records
    assert record.error is not None and record.error.startswith("StructuredOutputError: ")
    # The raw answer is on the exception, never in its message.
    assert '{"n": -1}' not in str(exc)


async def test_malformed_json_reports_one_json_invalid_error() -> None:
    with (
        install(lambda _request: Reply.text("not json")),
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call(
            "q", _Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    assert caught.value.raw_text == "not json"
    assert caught.value.errors is not None
    assert [error["type"] for error in caught.value.errors] == ["json_invalid"]


def test_the_sync_twin_raises_the_same_type() -> None:
    with (
        install(lambda _request: Reply.text("{}")),
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = structured_llm_call_sync(
            "q", _Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    assert caught.value.errors is not None
    assert [(error["type"], error["loc"]) for error in caught.value.errors] == [("missing", ("n",))]


async def test_compose_failure_carries_the_answer_and_the_attempt() -> None:
    with (
        install(lambda _request: Reply.text('{"n": 0}')) as fake,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await tool_llm_call("q", [], output_schema=_Positive, feature="f", provider=_PROVIDER)
    exc = caught.value
    assert exc.schema_name == "_Positive"
    assert exc.raw_text == '{"n": 0}'
    assert exc.errors is not None
    assert [error["loc"] for error in exc.errors] == [("n",)]
    # The default validation budget re-asked once; no in-call repair here.
    assert len(fake.requests) == 2
    assert exc.attempt == 2
    assert exc.provider == _PROVIDER.name


async def test_compose_turn_with_neither_calls_nor_text_raises_the_same_type() -> None:
    with (
        install(lambda _request: Reply()),
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await tool_llm_call(
            "q", [], output_schema=_Positive, feature="f", provider=_PROVIDER, retry=NO_RETRY
        )
    assert caught.value.raw_text is None
    assert caught.value.errors is None


async def test_truncated_compose_final_answer_fails_fast_as_an_output_limit() -> None:
    reply = Reply.text('{"n": 1', finish_reason="length", usage=TokenUsage(3, 5, 8))
    with (
        install(lambda _request: reply) as fake,
        capture_llm_records() as records,
        pytest.raises(OutputLimitError) as caught,
    ):
        _ = await tool_llm_call(
            "q", [], output_schema=_Positive, feature="f", provider=_PROVIDER, max_tokens=5
        )
    exc = caught.value
    assert exc.max_tokens == 5
    assert exc.completion_tokens == 5
    assert exc.diagnostics is not None
    assert exc.diagnostics.partial_text == '{"n": 1'
    assert exc.diagnostics.finish_reason == "length"
    # Never re-asked under the default policy: one request, one record.
    assert len(fake.requests) == 1
    [record] = records
    assert cast("object", record.response) is None
    assert record.error is not None and record.error.startswith("OutputLimitError: ")
    assert record.partial_text == '{"n": 1'
    assert record.output_limit is not None
    assert record.output_limit["finish_reason"] == "length"
    assert record.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


async def test_a_truncated_turn_that_requested_tools_raises_before_parsing() -> None:
    tool = ToolDefinition.from_model("add", _Args, "Add")
    reply = Reply(tool_calls=[FakeToolCall("add", '{"x": 1}', id="c1")], finish_reason="length")
    with install(lambda _request: reply) as fake, pytest.raises(OutputLimitError) as caught:
        _ = await tool_llm_call(
            "q", [tool], output_schema=_Positive, feature="f", provider=_PROVIDER
        )
    assert caught.value.diagnostics is not None
    assert caught.value.diagnostics.finish_reason == "length"
    assert caught.value.diagnostics.partial_text is None
    assert len(fake.requests) == 1


class _Secretive(BaseModel):
    n: int
    note: str


@pytest.mark.parametrize(
    "answer",
    ['{"n": 1, "note": "hunter2', '{"n": "hunter2", "note": "x"}'],
    ids=["malformed", "wrong-type"],
)
async def test_the_message_and_the_record_never_quote_the_answer(answer: str) -> None:
    """pydantic's own message quotes each failing input; the error's does not."""
    for call in ("structured", "compose"):
        with (
            install(lambda _request: Reply.text(answer)),
            capture_llm_records() as records,
            pytest.raises(StructuredOutputError) as caught,
        ):
            if call == "structured":
                _ = await structured_llm_call(
                    "q", _Secretive, feature="f", retry=NO_RETRY, provider=_PROVIDER
                )
            else:
                _ = await tool_llm_call(
                    "q",
                    [],
                    output_schema=_Secretive,
                    feature="f",
                    retry=NO_RETRY,
                    provider=_PROVIDER,
                )
        assert caught.value.raw_text == answer
        assert caught.value.errors
        assert "hunter2" not in str(caught.value), call
        assert all("hunter2" not in (record.error or "") for record in records), call


async def test_both_lanes_name_the_same_routed_model() -> None:
    errors: list[StructuredOutputError] = []
    with install(lambda _request: Reply.text("{}")):
        for run in (
            lambda: structured_llm_call(
                "q", _Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
            ),
            lambda: tool_llm_call(
                "q", [], output_schema=_Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
            ),
        ):
            with pytest.raises(StructuredOutputError) as caught:
                _ = await run()
            errors.append(caught.value)

    assert errors[0].model == errors[1].model == _PROVIDER.litellm_model(None)
    assert errors[0].provider == errors[1].provider == _PROVIDER.name


async def test_the_error_survives_a_pickle_round_trip() -> None:
    with (
        install(lambda _request: Reply.text('{"n": -1}')),
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call(
            "q", _Positive, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    restored = cast("object", pickle.loads(pickle.dumps(caught.value)))
    assert isinstance(restored, StructuredOutputError)

    assert str(restored) == str(caught.value)
    for name in ("schema_name", "raw_text", "attempt", "model", "provider"):
        assert getattr(restored, name) == getattr(caught.value, name), name
    # ``ctx`` holds the validator's exception object, equal only to itself.
    assert restored.errors is not None and caught.value.errors is not None
    assert [(e["type"], e["loc"], e["msg"], e["input"]) for e in restored.errors] == [
        (e["type"], e["loc"], e["msg"], e["input"]) for e in caught.value.errors
    ]
