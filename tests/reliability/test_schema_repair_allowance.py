"""The caller-set in-call schema-repair allowance, ``schema_repairs``.

instructor's in-call repair re-ask used to be fixed at one. ``schema_repairs``
lets a caller choose it, per call or through :class:`~llmkit.LLMCallOptions`,
while the parse-only eligibility stays fixed. Each test drives the real
instructor loop through the :mod:`llmkit.testing` fake, so the sends counted
are the provider round trips a caller would pay for, and ``repair`` marks
which of them were instructor's re-asks.

Each test names the broken implementation it exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel

from llmkit import (
    DEFAULT_SCHEMA_REPAIRS,
    NO_RETRY,
    UNSET,
    LLMCallOptions,
    OutputLimitError,
    Provider,
    RetryPolicy,
    SchemaEngineError,
    StructuredOutputError,
    make_provider,
    model_from_json_schema,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    tool_llm_call,
)
from llmkit.exceptions import underlying_provider_error
from llmkit.options import resolve_call_args
from llmkit.testing import Reply, TransportRequest, install, rate_limit_error

_PROVIDER = make_provider(Provider.OPENAI, api_key="fake")

# The default budgets (three outer attempts, two of them validation) without
# the backoff sleeps.
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


class _N(BaseModel):
    n: int


def _malformed(_request: TransportRequest) -> Reply:
    return Reply.text('{"n": "not a number"}')


def _repairs(requests: list[TransportRequest]) -> list[bool]:
    return [request.repair for request in requests]


def test_the_default_is_one_repair() -> None:
    assert DEFAULT_SCHEMA_REPAIRS == 1


@pytest.mark.parametrize("allowance", [0, 1, 2, 3])
async def test_a_persistent_parse_failure_sends_allowance_plus_one(allowance: int) -> None:
    """Broken implementation caught: the builder ignores its argument, or
    stops at ``repairs`` instead of ``repairs + 1``."""
    with install(_malformed) as fake, pytest.raises(StructuredOutputError):
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=allowance
        )
    assert _repairs(fake.requests) == [False] + [True] * allowance


async def test_zero_repairs_sends_once_per_outer_attempt_on_the_validation_budget() -> None:
    """Zero repairs still validates: the failure is a ``StructuredOutputError``
    re-asked on the outer validation allowance, one send per attempt.
    Broken implementation caught: an allowance of zero that skips validation
    or routes the failure to the transport budget (three sends)."""
    with install(_malformed) as fake, pytest.raises(StructuredOutputError) as caught:
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, schema_repairs=0
        )
    assert [(r.attempt, r.repair) for r in fake.requests] == [(1, False), (2, False)]
    assert "with no in-call repair" in str(caught.value)
    assert caught.value.errors is not None


async def test_zero_repairs_under_no_retry_sends_once() -> None:
    with install(_malformed) as fake, pytest.raises(StructuredOutputError):
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=0
        )
    assert len(fake.requests) == 1


async def test_a_repair_within_the_allowance_succeeds() -> None:
    answers = iter(['{"n": "x"}', '{"n": "y"}', '{"n": 3}'])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        result = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=2
        )
    assert result.n == 3
    assert _repairs(fake.requests) == [False, True, True]


@pytest.mark.parametrize(
    ("allowance", "wording"),
    [
        (0, "with no in-call repair"),
        (1, "after 1 in-call repair re-ask ("),
        (2, "after 2 in-call repair re-asks ("),
    ],
)
async def test_the_error_names_the_repairs_that_ran(allowance: int, wording: str) -> None:
    """Broken implementation caught: a fixed "after in-call repair" message,
    which is false when no repair ran."""
    with install(_malformed), pytest.raises(StructuredOutputError) as caught:
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=allowance
        )
    assert wording in str(caught.value)


def _auth_error() -> openai.AuthenticationError:
    request = httpx.Request("POST", "https://fake/v1/chat/completions")
    return openai.AuthenticationError(
        "denied", response=httpx.Response(401, request=request), body=None
    )


def _raise(make: Callable[[], Exception]) -> Callable[[TransportRequest], Reply]:
    def fake(_request: TransportRequest) -> Reply:
        raise make()

    return fake


@pytest.mark.parametrize(
    ("fake", "cause"),
    [
        (_raise(rate_limit_error), openai.RateLimitError),
        (_raise(_auth_error), openai.AuthenticationError),
    ],
    ids=["429", "401"],
)
async def test_provider_errors_are_never_repaired(
    fake: Callable[[TransportRequest], Reply], cause: type[Exception]
) -> None:
    """Broken implementation caught: a predicate that re-asks every exception
    once the allowance is raised."""
    with install(fake) as installed, pytest.raises(InstructorRetryException) as caught:
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=3
        )
    assert isinstance(underlying_provider_error(caught.value), cause)
    assert len(installed.requests) == 1


async def test_truncation_is_never_repaired() -> None:
    with (
        install(lambda _request: Reply.text('{"n": ', finish_reason="length")) as fake,
        pytest.raises(OutputLimitError),
    ):
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=3
        )
    assert len(fake.requests) == 1


async def test_an_engine_failure_is_never_repaired() -> None:
    """``SchemaEngineError``: the JSON Schema validator itself failed while
    checking a parsed answer, so another answer cannot help.
    Broken implementation caught: a predicate that treats any ``LlmkitError``
    raised during validation as repairable."""
    # jsonschema's floating multipleOf arithmetic overflows on this integer.
    model = model_from_json_schema({"type": "object", "properties": {"x": {"multipleOf": 0.1}}})
    raw = '{"x":' + str(10**1000) + "}"
    with (
        install(lambda _request: Reply.text(raw)) as fake,
        pytest.raises(InstructorRetryException) as caught,
    ):
        _ = await structured_llm_call(
            "q", model, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=3
        )
    assert isinstance(underlying_provider_error(caught.value), SchemaEngineError)
    assert len(fake.requests) == 1


# --- the merge -------------------------------------------------------------


def _resolve(options: LLMCallOptions | None, keyword: object = UNSET) -> int:
    return resolve_call_args(
        options,
        temperature=UNSET,
        model=UNSET,
        max_tokens=UNSET,
        reasoning_effort=UNSET,
        retry=UNSET,
        provider=UNSET,
        cache=UNSET,
        timeout=UNSET,
        schema_repairs=keyword,  # pyright: ignore[reportArgumentType]  # test-helper — exercises invalid values
    ).schema_repairs


def test_keyword_beats_options_beats_default() -> None:
    """Broken implementation caught: ``_pick`` arguments in the wrong order."""
    assert _resolve(None) == DEFAULT_SCHEMA_REPAIRS
    assert _resolve(LLMCallOptions()) == DEFAULT_SCHEMA_REPAIRS
    assert _resolve(LLMCallOptions(temperature=0.5)) == DEFAULT_SCHEMA_REPAIRS
    assert _resolve(LLMCallOptions(schema_repairs=0)) == 0
    assert _resolve(LLMCallOptions(schema_repairs=0), 2) == 2
    assert _resolve(None, 3) == 3


@pytest.mark.parametrize(
    ("options", "keyword", "sends"),
    [
        (None, UNSET, 2),
        (LLMCallOptions(), UNSET, 2),
        (LLMCallOptions(schema_repairs=0), UNSET, 1),
        (LLMCallOptions(schema_repairs=0), 2, 3),
    ],
    ids=["default", "unset-option", "option", "keyword-wins"],
)
async def test_the_merged_allowance_reaches_the_transport(
    options: LLMCallOptions | None, keyword: int, sends: int
) -> None:
    with install(_malformed) as fake, pytest.raises(StructuredOutputError):
        _ = await structured_llm_call(
            "q",
            _N,
            feature="f",
            retry=NO_RETRY,
            provider=_PROVIDER,
            options=options,
            schema_repairs=keyword,
        )
    assert len(fake.requests) == sends


def test_the_sync_form_forwards_the_allowance() -> None:
    with install(_malformed) as fake, pytest.raises(StructuredOutputError):
        _ = structured_llm_call_sync(
            "q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER, schema_repairs=0
        )
    assert len(fake.requests) == 1


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, ValueError),
        (False, ValueError),
        (1.5, ValueError),
        (1.0, ValueError),
        ("1", ValueError),
        (-1, ValueError),
    ],
)
async def test_an_invalid_allowance_is_refused_before_any_request(
    value: object, error: type[Exception]
) -> None:
    """Broken implementation caught: validation removed (``True`` would read
    as one repair, ``-1`` would stop instructor before its first send)."""
    with install(_malformed) as fake:
        with pytest.raises(error, match="schema_repairs"):
            _ = await structured_llm_call(
                "q",
                _N,
                feature="f",
                provider=_PROVIDER,
                schema_repairs=value,  # pyright: ignore[reportArgumentType]  # test-helper — invalid on purpose
            )
        with pytest.raises(error, match="schema_repairs"):
            _ = await structured_llm_call(
                "q",
                _N,
                feature="f",
                provider=_PROVIDER,
                options=LLMCallOptions(schema_repairs=value),  # pyright: ignore[reportArgumentType]  # test-helper — invalid on purpose
            )
    assert fake.requests == []


async def test_an_invalid_options_allowance_is_refused_on_every_family() -> None:
    """The other families ignore a valid value but share the merge, so a
    shared options object carrying a bad value fails wherever it is used."""
    bad = LLMCallOptions(schema_repairs=-1)
    with install(lambda _request: Reply.text("ok")) as fake:
        with pytest.raises(ValueError, match="schema_repairs"):
            _ = await text_llm_call("q", feature="f", provider=_PROVIDER, options=bad)
        with pytest.raises(ValueError, match="schema_repairs"):
            _ = await tool_llm_call("q", [], feature="f", provider=_PROVIDER, options=bad)
    assert fake.requests == []


async def test_other_families_ignore_a_valid_allowance() -> None:
    with install(lambda _request: Reply.text("ok")) as fake:
        text = await text_llm_call(
            "q", feature="f", provider=_PROVIDER, options=LLMCallOptions(schema_repairs=5)
        )
    assert text == "ok"
    assert len(fake.requests) == 1
