"""Offline negative space of the structured-output contract, varied by provider.

"Structured output always validates" promises a validated instance of the
caller's schema or an exception, never a value that failed validation. This
module drives the failing answers a provider can send through the public
call functions over the :mod:`llmkit.testing` fake, on every shipped
provider plus both Gemini ``"json"`` strategies, and pins for each one that
the call raises :class:`~llmkit.StructuredOutputError`, what the exception
reports, and exactly which paid requests it took. A valid control row per
provider pins that a good answer comes back as exactly the schema's class,
and a validate-once row that a validator which transforms its input is
applied exactly once.

The fake answers each request differently, keyed on the request's attempt
and on whether it is the in-call repair re-ask, so the exception's
``raw_text`` and ``errors`` are pinned to the *last* answer the call
received: the final attempt's repair re-ask on the structured lane, the
final attempt on the compose lane. Only the empty-content and no-choices
answers have one spelling and so read the same on every request.

What the provider axis buys, honestly: the fake answers below LiteLLM, so
LiteLLM's per-provider response handling never runs here, and for these
replies every provider behaves identically today. The axis is a tripwire. It
crosses both instructor parse paths llmkit ships (``Mode.JSON_SCHEMA`` on
most providers, ``Mode.JSON`` on DeepSeek and Moonshot and on Gemini's
``"json"`` strategy), so a provider re-pinned to a mode that parses differently (a
tool-call mode, say) fails the control row; and it runs each provider's
request-side traits on the way to the fake (its mode's request shaping,
OpenRouter's strict ``response_format`` upgrade, the Claude routes'
``parallel_tool_calls=False``), so a trait that raises on that path fails
here. A trait that sends a wrong but well-formed request does not, because
the fake answers whatever it is sent. Whether a real provider's response
survives LiteLLM is a live-suite question.

The budgets pinned: with the default policy a structured call makes
``validation_max_attempts`` outer attempts, each with one in-call repair
re-ask, so ``validation_max_attempts * 2`` requests; an answer with no
choices is never re-asked in-call. The compose lane
(``tool_llm_call(..., output_schema=...)``) has no in-call repair, so it
spends ``validation_max_attempts`` requests.

Mutation check: with ``_as_output_schema`` returning instructor's instance
unchecked and the structured lane answering an exhausted parse failure with
an unvalidated ``model_construct()``, every structured failing,
control and validate-once row fails. With ``_as_output_schema`` validating
the answer again (``output_schema.model_validate_json(parsed.model_dump_json())``),
the structured validate-once row fails on every provider; with the compose
lane validating its answer twice, the compose validate-once row fails. With
the structured lane's ``raw_text`` taken from instructor's first failed
in-call attempt instead of its last, every answer-carrying row except
empty-content fails on every provider. With DeepSeek pinned to
``instructor.Mode.TOOLS``, its six answer-carrying rows, its control row and
its validate-once row fail. With
``acompletion_structured`` passing ``strict=False``, the lax-only row fails
on every provider. That the ``strict=True`` argument itself is passed is
pinned by
``tests/calls/test_single_tool_call_per_schema.py::test_structured_request_asks_instructor_for_strict_validation_explicitly``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import pytest
from pydantic import BaseModel, field_validator, model_validator

from llmkit import (
    Provider,
    RetryPolicy,
    StructuredOutputError,
    make_provider,
    structured_llm_call,
    tool_llm_call,
)
from llmkit.providers import LLMProviderInterface
from llmkit.testing import Reply, TransportRequest, install


class _Count(BaseModel):
    """One required strict ``int``, a field validator and a model validator."""

    n: int

    @field_validator("n")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n must be positive")
        return value

    @model_validator(mode="after")
    def _not_ending_in_thirteen(self) -> Self:
        if self.n % 100 == 13:
            raise ValueError("n must not end in 13")
        return self


class _Scaled(BaseModel):
    """A field validator that transforms its input: run twice, it scales twice."""

    n: int

    @field_validator("n")
    @classmethod
    def _hundredfold(cls, value: int) -> int:
        return value * 100


#: The default budgets, with no backoff sleep so the matrix stays fast.
_POLICY = RetryPolicy(backoff_base_seconds=0)

_ATTEMPTS = range(1, _POLICY.validation_max_attempts + 1)

#: ``(attempt, repair)`` of every request, in order: each outer attempt sends
#: the answer and its one in-call repair re-ask.
_STRUCTURED_REQUESTS = [(attempt, repair) for attempt in _ATTEMPTS for repair in (False, True)]
#: An answer with no choices is never re-asked in-call, and the compose lane
#: has no in-call repair: one request per outer attempt.
_ONE_REQUEST_PER_ATTEMPT = [(attempt, False) for attempt in _ATTEMPTS]

_ErrorRow = list[tuple[str, tuple[int | str, ...], object]]


def _tag(attempt: int, repair: bool) -> int:
    """A number unique to one request: 11, 12, 21, 22, ... (attempt, then repair)."""
    return 10 * attempt + (2 if repair else 1)


@dataclass(frozen=True)
class _Row:
    """One kind of failing answer, spelled per request, and the errors each spelling draws."""

    #: The fake's answer to the request ``(attempt, repair)``.
    answer: Callable[[int, bool], Reply]
    #: ``(type, loc, input)`` of each pydantic error that answer draws, or
    #: ``None`` when there are none.
    errors: Callable[[int, bool], _ErrorRow | None]


def _malformed(attempt: int, repair: bool) -> str:
    return f"not json {_tag(attempt, repair)}"


def _model_rejected(attempt: int, repair: bool) -> int:
    return 100 * _tag(attempt, repair) + 13


def _empty_object(attempt: int, repair: bool) -> str:
    return "{" + " " * _tag(attempt, repair) + "}"


_FAILING_ROWS: dict[str, _Row] = {
    "malformed-json": _Row(
        lambda a, r: Reply.text(_malformed(a, r)),
        lambda a, r: [("json_invalid", (), _malformed(a, r))],
    ),
    "field-validator": _Row(
        lambda a, r: Reply.text(f'{{"n": -{_tag(a, r)}}}'),
        lambda a, r: [("value_error", ("n",), -_tag(a, r))],
    ),
    "model-validator": _Row(
        lambda a, r: Reply.text(f'{{"n": {_model_rejected(a, r)}}}'),
        lambda a, r: [("value_error", (), {"n": _model_rejected(a, r)})],
    ),
    "empty-object": _Row(
        lambda a, r: Reply.text(_empty_object(a, r)),
        lambda _a, _r: [("missing", ("n",), {})],
    ),
    "empty-content": _Row(
        lambda _a, _r: Reply.text(""),
        lambda _a, _r: [("json_invalid", (), "")],
    ),
    # Lax validation would coerce "11" to 11; strict JSON rejects it.
    "lax-only": _Row(
        lambda a, r: Reply.text(f'{{"n": "{_tag(a, r)}"}}'),
        lambda a, r: [("int_type", ("n",), str(_tag(a, r)))],
    ),
    "no-choices": _Row(
        lambda _a, _r: Reply(no_choice=True),
        lambda _a, _r: None,
    ),
}

_VALID = Reply.text('{"n": 1}')

#: Validated once, ``_Scaled`` reads this as ``n == 200``; twice, as 20000.
_SCALED = Reply.text('{"n": 2}')


_PROVIDER_BUILDERS: dict[str, Callable[[], LLMProviderInterface]] = {
    Provider.ANTHROPIC.value: lambda: make_provider(Provider.ANTHROPIC, api_key="k"),
    Provider.BEDROCK.value: lambda: make_provider(Provider.BEDROCK, aws_region_name="us-east-1"),
    Provider.CEREBRAS.value: lambda: make_provider(Provider.CEREBRAS, api_key="k"),
    Provider.DEEPSEEK.value: lambda: make_provider(Provider.DEEPSEEK, api_key="k"),
    Provider.MOONSHOT.value: lambda: make_provider(Provider.MOONSHOT, api_key="k"),
    Provider.ALIBABA.value: lambda: make_provider(Provider.ALIBABA, api_key="k"),
    Provider.GOOGLE.value: lambda: make_provider(Provider.GOOGLE, api_key="k"),
    Provider.OLLAMA.value: lambda: make_provider(Provider.OLLAMA),
    Provider.OPENAI.value: lambda: make_provider(Provider.OPENAI, api_key="k"),
    Provider.OPENROUTER.value: lambda: make_provider(Provider.OPENROUTER, api_key="k"),
    Provider.VERTEX.value: lambda: make_provider(
        Provider.VERTEX, vertex_project="p", vertex_location="us-central1"
    ),
    "google-json": lambda: make_provider(
        Provider.GOOGLE, api_key="k", gemini_structured_output="json"
    ),
    "vertex-json": lambda: make_provider(
        Provider.VERTEX,
        vertex_project="p",
        vertex_location="us-central1",
        gemini_structured_output="json",
    ),
}

_COMPOSE_PROVIDERS = (Provider.OPENAI.value, Provider.ANTHROPIC.value)


def test_the_matrix_covers_every_provider_and_both_gemini_json_strategies() -> None:
    """A new ``Provider`` member must join the matrix."""
    assert set(_PROVIDER_BUILDERS) == {p.value for p in Provider} | {"google-json", "vertex-json"}


def test_the_compose_block_covers_every_compose_capable_provider() -> None:
    """A provider that gains the tools-plus-schema capability must join the compose block."""
    capable = {p.value for p in Provider if _PROVIDER_BUILDERS[p.value]().compose_tools_schema}
    assert capable == set(_COMPOSE_PROVIDERS)


def test_every_varied_row_answers_each_request_differently() -> None:
    """The last-answer pins below can only fail if earlier answers differ from the last."""
    for row_id in (
        "malformed-json",
        "field-validator",
        "model-validator",
        "empty-object",
        "lax-only",
    ):
        row = _FAILING_ROWS[row_id]
        texts = [row.answer(attempt, repair).content for attempt, repair in _STRUCTURED_REQUESTS]
        assert len(set(texts)) == len(texts), row_id


def _answering(row: _Row) -> Callable[[TransportRequest], Reply]:
    def answer(request: TransportRequest) -> Reply:
        assert request.attempt is not None
        return row.answer(request.attempt, request.repair)

    return answer


def _assert_reports_last_answer(
    exc: StructuredOutputError,
    row: _Row,
    provider: LLMProviderInterface,
    last: tuple[int, bool],
) -> None:
    """*exc* names the schema and provider, and describes the *last* answer sent."""
    assert exc.schema_name == "_Count"
    assert exc.provider == provider.name
    assert exc.raw_text == row.answer(*last).content
    expected = row.errors(*last)
    if expected is None:
        assert exc.errors is None
    else:
        assert exc.errors is not None
        assert [(error["type"], error["loc"], error["input"]) for error in exc.errors] == expected
    # The default validation budget was spent, not cut short or exceeded.
    assert exc.attempt == last[0] == _POLICY.validation_max_attempts


@pytest.mark.parametrize("row_id", list(_FAILING_ROWS))
@pytest.mark.parametrize("provider_id", list(_PROVIDER_BUILDERS))
async def test_a_failing_structured_answer_raises_and_reports_the_last_answer(
    provider_id: str, row_id: str
) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    row = _FAILING_ROWS[row_id]
    with (
        install(_answering(row)) as fake,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call("q", _Count, feature="f", provider=provider, retry=_POLICY)

    requests = fake.requests
    assert all(request.kind == "structured" for request in requests)
    assert all(request.provider == provider.name for request in requests)
    expected = _ONE_REQUEST_PER_ATTEMPT if row.answer(1, False).no_choice else _STRUCTURED_REQUESTS
    assert [(request.attempt, request.repair) for request in requests] == expected
    _assert_reports_last_answer(caught.value, row, provider, expected[-1])


@pytest.mark.parametrize("provider_id", list(_PROVIDER_BUILDERS))
async def test_a_valid_structured_answer_is_exactly_the_schema(provider_id: str) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    with install(lambda _request: _VALID) as fake:
        result = await structured_llm_call(
            "q", _Count, feature="f", provider=provider, retry=_POLICY
        )

    assert type(result) is _Count
    assert result == _Count(n=1)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("provider_id", list(_PROVIDER_BUILDERS))
async def test_a_structured_answer_is_validated_exactly_once(provider_id: str) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    with install(lambda _request: _SCALED) as fake:
        result = await structured_llm_call(
            "q", _Scaled, feature="f", provider=provider, retry=_POLICY
        )

    assert type(result) is _Scaled
    assert result.n == 200
    assert len(fake.requests) == 1


@pytest.mark.parametrize("row_id", list(_FAILING_ROWS))
@pytest.mark.parametrize("provider_id", _COMPOSE_PROVIDERS)
async def test_a_failing_compose_answer_raises_and_reports_the_last_answer(
    provider_id: str, row_id: str
) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    row = _FAILING_ROWS[row_id]
    with (
        install(_answering(row)) as fake,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await tool_llm_call(
            "q", [], output_schema=_Count, feature="f", provider=provider, retry=_POLICY
        )

    requests = fake.requests
    assert all(request.kind == "tool" for request in requests)
    assert all(request.provider == provider.name for request in requests)
    # No in-call repair on this lane: one request per validation attempt.
    assert [(request.attempt, request.repair) for request in requests] == _ONE_REQUEST_PER_ATTEMPT
    _assert_reports_last_answer(caught.value, row, provider, _ONE_REQUEST_PER_ATTEMPT[-1])


@pytest.mark.parametrize("provider_id", _COMPOSE_PROVIDERS)
async def test_a_valid_compose_answer_is_exactly_the_schema(provider_id: str) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    with install(lambda _request: _VALID) as fake:
        result = await tool_llm_call(
            "q", [], output_schema=_Count, feature="f", provider=provider, retry=_POLICY
        )

    assert type(result.parsed) is _Count
    assert result.parsed == _Count(n=1)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("provider_id", _COMPOSE_PROVIDERS)
async def test_a_compose_answer_is_validated_exactly_once(provider_id: str) -> None:
    provider = _PROVIDER_BUILDERS[provider_id]()
    with install(lambda _request: _SCALED) as fake:
        result = await tool_llm_call(
            "q", [], output_schema=_Scaled, feature="f", provider=provider, retry=_POLICY
        )

    assert type(result.parsed) is _Scaled
    assert result.parsed.n == 200
    assert len(fake.requests) == 1
