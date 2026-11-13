"""The structured-output contract's validation rules, pinned lane by lane.

"Structured output always validates" means one rule wherever an answer becomes
a caller's model: the structured lane, the compose lane
(``tool_llm_call(output_schema=...)``), a response-cache hit and a tool's
arguments all validate the answer's JSON text in pydantic's strict JSON mode,
once. Around that rule these tests pin:

* strictness itself: ``{"n": "1"}`` fails everywhere, while ISO datetimes and
  enum strings (valid JSON spellings) pass everywhere, for a
  ``ConfigDict(strict=True)`` model and a plain one;
* ``on_result``: an ``async def`` hook is awaited, and a hook that returns a
  value (a predicate) raises :class:`TypeError` instead of being ignored;
* a cache hit: only an undecodable entry is a (warned) miss, an awaited hook's
  rejection of a hit pays for a re-roll, a hook's own error propagates without
  a paid request, and a hook's mutation is never stored;
* the stored entry: the provider's exact answer text, and nothing at all for a
  host provider pinned to ``Mode.TOOLS``, whose answer text is empty;
* the schema guard: a schema no lane can hold to the contract raises
  :class:`~llmkit.exceptions.UnsupportedSchemaError` before any request and
  before the cache is read;
* the compose lane's log dump is best-effort;
* a reply with no choices at all, per lane.

Every test answers through the public :mod:`llmkit.testing` fake and counts the
requests it received, so "no request was made" is an observation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar, TypedDict, cast, override

import instructor
import pytest
from instructor.dsl import IterableModel, Partial
from instructor.validation import async_field_validator, async_model_validator
from pydantic import BaseModel, ConfigDict, model_serializer

from llmkit import (
    LLM_RECOVERABLE_ERRORS,
    NO_RETRY,
    InMemoryLLMCache,
    LLMCacheEntry,
    LlmkitError,
    Provider,
    ResultValidationError,
    RetryPolicy,
    StructuredOutputError,
    ToolArgumentError,
    ToolDefinition,
    UnsupportedSchemaError,
    capture_llm_records,
    configure_llm_cache,
    make_provider,
    structured_llm_call,
    structured_llm_call_sync,
    text_llm_call,
    text_llm_call_sync,
    tool_llm_call,
    tool_llm_call_sync,
)
from llmkit._types import AssistantToolMessage, ChatMessage, ToolResultMessage
from llmkit.providers import OllamaProvider
from llmkit.testing import FakeToolCall, Reply, TransportRequest, install

# A provider that composes tools with an output schema; the fake answers its
# round trips, so the key is never used.
_PROVIDER = make_provider(Provider.OPENAI, api_key="fake")

# The default budgets (transport 3, validation 2) without the backoff sleeps.
_NO_BACKOFF = RetryPolicy(backoff_base_seconds=0.0)


class _N(BaseModel):
    n: int


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class _StrictEvent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

    when: datetime
    color: _Color


class _LaxEvent(BaseModel):
    when: datetime
    color: _Color


_EVENT_JSON = '{"when": "2026-11-02T18:22:04Z", "color": "red"}'
_EVENT_WHEN = datetime(2026, 11, 2, 18, 22, 4, tzinfo=UTC)


class _SpyCache(InMemoryLLMCache):
    """The reference store, counting reads and remembering the keys written."""

    def __init__(self) -> None:
        super().__init__()
        self.gets: int = 0
        self.keys: list[str] = []

    @override
    async def get(self, key: str) -> LLMCacheEntry | None:
        self.gets += 1
        return await super().get(key)

    @override
    async def set(self, key: str, entry: LLMCacheEntry) -> None:
        self.keys.append(key)
        await super().set(key, entry)


async def _replace_stored(cache: _SpyCache, response: str) -> None:
    """Rewrite the one stored entry's answer text, as a host's store might hold it."""
    [key] = set(cache.keys)
    entry = await InMemoryLLMCache.get(cache, key)
    assert entry is not None
    await InMemoryLLMCache.set(cache, key, dataclasses.replace(entry, response=response))


def _cache_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "llmkit.cache" and record.levelno >= logging.WARNING
    ]


def _answer(text: str) -> Callable[[TransportRequest], Reply]:
    return lambda _request: Reply.text(text)


def _as_hook[T](hook: Callable[[T], object]) -> Callable[[T], None]:
    """Launder a predicate-shaped hook past the ``None``-returning annotation."""
    return cast("Callable[[T], None]", hook)


# --- (a) strict JSON on every lane ------------------------------------------


async def test_structured_lane_rejects_a_string_for_an_int() -> None:
    with (
        install(_answer('{"n": "1"}')) as fake,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call("q", _N, feature="f", retry=NO_RETRY, provider=_PROVIDER)
    assert caught.value.raw_text == '{"n": "1"}'
    assert caught.value.errors is not None
    assert [(error["type"], error["loc"]) for error in caught.value.errors] == [
        ("int_type", ("n",))
    ]
    # One attempt: the answer and instructor's one in-call repair re-ask.
    assert len(fake.requests) == 2


async def test_compose_lane_rejects_a_string_for_an_int() -> None:
    with (
        install(_answer('{"n": "1"}')) as fake,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await tool_llm_call(
            "q", [], output_schema=_N, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    assert caught.value.raw_text == '{"n": "1"}'
    assert caught.value.errors is not None
    assert [(error["type"], error["loc"]) for error in caught.value.errors] == [
        ("int_type", ("n",))
    ]
    assert len(fake.requests) == 1


async def test_tool_arguments_reject_a_string_for_an_int_beside_a_good_call() -> None:
    tool = ToolDefinition.from_model("count", _N, "Count")
    reply = Reply(
        tool_calls=[
            FakeToolCall("count", '{"n": "1"}', id="bad"),
            FakeToolCall("count", '{"n": 2}', id="good"),
        ]
    )
    with install(lambda _request: reply):
        result = await tool_llm_call("q", [tool], feature="f", provider=_PROVIDER)
    assert [call.id for call in result.tool_calls] == ["good"]
    assert result.tool_calls[0].validated == _N(n=2)
    [invalid] = result.invalid_calls
    assert invalid.call_id == "bad"
    assert invalid.arguments_raw == '{"n": "1"}'
    assert "failed validation" in str(invalid)


async def test_tool_arguments_raise_when_every_call_is_lax_only() -> None:
    tool = ToolDefinition.from_model("count", _N, "Count")
    reply = Reply(tool_calls=[FakeToolCall("count", '{"n": "1"}', id="bad")])
    with install(lambda _request: reply) as fake, pytest.raises(ToolArgumentError) as caught:
        _ = await tool_llm_call("q", [tool], feature="f", provider=_PROVIDER, retry=_NO_BACKOFF)
    assert caught.value.call_id == "bad"
    # Re-asked on the validation budget, like a schema failure.
    assert len(fake.requests) == 2


@pytest.mark.parametrize("as_messages", [False, True])
async def test_tool_reask_repairs_strict_arguments(as_messages: bool) -> None:
    tool = ToolDefinition.from_model("count", _N, "Count")
    answers = iter(
        [
            Reply(tool_calls=[FakeToolCall("count", '{"n": "1"}', id="bad")]),
            Reply(tool_calls=[FakeToolCall("count", '{"n": 1}', id="good")]),
        ]
    )
    prompt: str | list[ChatMessage] = [{"role": "user", "content": "q"}] if as_messages else "q"
    with install(lambda _request: next(answers)) as fake, capture_llm_records() as records:
        result = await tool_llm_call(
            prompt, [tool], feature="f", provider=_PROVIDER, retry=_NO_BACKOFF
        )
    assert result.tool_calls[0].validated == _N(n=1)
    first, second = fake.requests
    assert second.prompt == first.prompt
    assert second.repair is True
    user, assistant, error, reminder = second.messages
    assert user == {"role": "user", "content": "q"}
    assert assistant == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "bad",
                "type": "function",
                "function": {"name": "count", "arguments": '{"n": "1"}'},
            }
        ],
    }
    assert error["role"] == "tool"
    assert error["tool_call_id"] == "bad"
    assert error["content"].startswith("ERROR: ")
    assert all(part in error["content"] for part in ("count", "bad", "int_type"))
    assert records[0].prompt == prompt
    assert reminder["role"] == "user"
    assert isinstance(reminder["content"], str)
    assert "Retry the rejected tool request" in reminder["content"]
    assert records[1].prompt == [
        user,
        assistant,
        {
            "role": "tool",
            "tool_call_id": "bad",
            "content": error["content"].removeprefix("ERROR: "),
            "is_error": True,
        },
        reminder,
    ]
    assert prompt == ([{"role": "user", "content": "q"}] if as_messages else "q")


@pytest.mark.parametrize("arguments", ["{not json", "[]", "null"])
async def test_tool_reask_normalizes_unreplayable_arguments(arguments: str) -> None:
    answers = iter(
        [
            Reply(tool_calls=[FakeToolCall("count", arguments, id="bad")]),
            Reply(tool_calls=[FakeToolCall("count", '{"n": 1}', id="good")]),
        ]
    )
    with install(lambda _request: next(answers)) as fake:
        _ = await tool_llm_call(
            "q",
            [ToolDefinition.from_model("count", _N, "Count")],
            feature="f",
            provider=_PROVIDER,
            retry=_NO_BACKOFF,
        )
    _, assistant, error, _ = fake.requests[1].messages
    assistant = cast("AssistantToolMessage", assistant)
    assert cast("dict[str, object]", assistant["tool_calls"][0]["function"])["arguments"] == "{}"
    assert arguments in cast("ToolResultMessage", error)["content"]


async def test_tool_reask_carries_every_rejected_call() -> None:
    replies = iter(
        [
            Reply(
                tool_calls=[
                    FakeToolCall("count", '{"n": "1"}', id="one"),
                    FakeToolCall("unknown", "{}", id="two"),
                ]
            ),
            Reply(tool_calls=[FakeToolCall("count", '{"n": 1}', id="good")]),
        ]
    )
    with install(lambda _request: next(replies)) as fake:
        _ = await tool_llm_call(
            "q",
            [ToolDefinition.from_model("count", _N, "Count")],
            feature="f",
            provider=_PROVIDER,
            retry=_NO_BACKOFF,
        )
    _, assistant, one, two, _ = fake.requests[1].messages
    assistant = cast("AssistantToolMessage", assistant)
    assert [call["id"] for call in assistant["tool_calls"]] == ["one", "two"]
    one = cast("ToolResultMessage", one)
    two = cast("ToolResultMessage", two)
    assert one["tool_call_id"] == "one"
    assert two["tool_call_id"] == "two"
    assert "int_type" in one["content"]
    assert all(part in two["content"] for part in ("unknown", "not offered", "count"))


async def test_a_cache_hit_whose_stored_text_is_lax_only_is_a_warned_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = _SpyCache()
    configure_llm_cache(cache)
    answers = iter(['{"n": 1}', '{"n": 2}'])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        first = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        assert _cache_warnings(caplog) == []
        # The same field a lax decode would coerce to 1.
        await _replace_stored(cache, '{"n": "1"}')
        second = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
    assert first == _N(n=1)
    assert second == _N(n=2)
    assert len(fake.requests) == 2
    [warning] = _cache_warnings(caplog)
    assert "decode" in warning.getMessage()


@pytest.mark.parametrize("schema", [_StrictEvent, _LaxEvent])
async def test_iso_datetimes_and_enum_strings_validate_on_the_structured_lane_and_its_hit(
    schema: type[_StrictEvent] | type[_LaxEvent], caplog: pytest.LogCaptureFixture
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(_answer(_EVENT_JSON)) as fake:
        paid = await structured_llm_call("q", schema, feature="f", provider=_PROVIDER)
        hit = await structured_llm_call("q", schema, feature="f", provider=_PROVIDER)
    for result in (paid, hit):
        assert type(result) is schema
        assert result.when == _EVENT_WHEN
        assert result.color is _Color.RED
    assert len(fake.requests) == 1
    assert _cache_warnings(caplog) == []


@pytest.mark.parametrize("schema", [_StrictEvent, _LaxEvent])
async def test_iso_datetimes_and_enum_strings_validate_on_the_compose_lane(
    schema: type[_StrictEvent] | type[_LaxEvent],
) -> None:
    with install(_answer(_EVENT_JSON)):
        result = await tool_llm_call(
            "q", [], output_schema=schema, feature="f", retry=NO_RETRY, provider=_PROVIDER
        )
    assert result.parsed is not None
    assert type(result.parsed) is schema
    assert result.parsed.when == _EVENT_WHEN
    assert result.parsed.color is _Color.RED


@pytest.mark.parametrize("schema", [_StrictEvent, _LaxEvent])
async def test_iso_datetimes_and_enum_strings_validate_as_tool_arguments(
    schema: type[_StrictEvent] | type[_LaxEvent],
) -> None:
    tool = ToolDefinition.from_model("schedule", schema, "Schedule")
    reply = Reply(tool_calls=[FakeToolCall("schedule", _EVENT_JSON, id="c1")])
    with install(lambda _request: reply):
        result = await tool_llm_call("q", [tool], feature="f", retry=NO_RETRY, provider=_PROVIDER)
    assert result.invalid_calls == []
    [call] = result.tool_calls
    assert isinstance(call.validated, schema)
    assert call.validated.when == _EVENT_WHEN
    assert call.validated.color is _Color.RED


# --- (b) on_result: awaited, and never a predicate ---------------------------


async def test_an_async_hook_that_rejects_rerolls_then_propagates_on_the_structured_lane() -> None:
    seen: list[int] = []

    async def _reject(result: _N) -> None:
        await asyncio.sleep(0)
        seen.append(result.n)
        raise ResultValidationError("never good enough")

    answers = iter(['{"n": 1}', '{"n": 2}'])
    with (
        install(lambda _request: Reply.text(next(answers))) as fake,
        pytest.raises(ResultValidationError),
    ):
        _ = await structured_llm_call(
            "q", _N, feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject
        )
    assert seen == [1, 2]
    assert len(fake.requests) == 2


async def test_an_async_hook_that_rejects_rerolls_then_propagates_on_the_text_lane() -> None:
    seen: list[str] = []

    async def _reject(text: str) -> None:
        await asyncio.sleep(0)
        seen.append(text)
        raise ResultValidationError("never good enough")

    answers = iter(["one", "two"])
    with (
        install(lambda _request: Reply.text(next(answers))) as fake,
        pytest.raises(ResultValidationError),
    ):
        _ = await text_llm_call(
            "q", feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject
        )
    assert seen == ["one", "two"]
    assert len(fake.requests) == 2


async def test_an_async_hook_that_rejects_once_returns_the_reroll() -> None:
    async def _reject_one(result: _N) -> None:
        if result.n == 1:
            raise ResultValidationError("too low")

    answers = iter(['{"n": 1}', '{"n": 2}'])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        result = await structured_llm_call(
            "q", _N, feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject_one
        )
    assert result == _N(n=2)
    assert len(fake.requests) == 2


async def test_an_async_hook_returning_none_accepts_on_both_lanes() -> None:
    awaited: list[object] = []

    async def _accept(result: object) -> None:
        await asyncio.sleep(0)
        awaited.append(result)

    with install(_answer('{"n": 1}')) as fake:
        structured = await structured_llm_call(
            "q", _N, feature="f", provider=_PROVIDER, on_result=_accept
        )
        text = await text_llm_call("q", feature="f", provider=_PROVIDER, on_result=_accept)
    assert structured == _N(n=1)
    assert text == '{"n": 1}'
    assert awaited == [_N(n=1), '{"n": 1}']
    assert len(fake.requests) == 2


@pytest.mark.parametrize("verdict", [True, False])
async def test_a_predicate_hook_raises_type_error_on_the_structured_lane(verdict: bool) -> None:
    with install(_answer('{"n": 1}')) as fake, pytest.raises(TypeError, match="bool"):
        _ = await structured_llm_call(
            "q",
            _N,
            feature="f",
            provider=_PROVIDER,
            on_result=_as_hook(lambda _result: verdict),
        )
    # A programming error, not a rejection: no re-roll.
    assert len(fake.requests) == 1


@pytest.mark.parametrize("verdict", [True, False])
async def test_a_predicate_hook_raises_type_error_on_the_text_lane(verdict: bool) -> None:
    with install(_answer("hello")) as fake, pytest.raises(TypeError, match="bool"):
        _ = await text_llm_call(
            "q", feature="f", provider=_PROVIDER, on_result=_as_hook(lambda _text: verdict)
        )
    assert len(fake.requests) == 1


async def test_an_async_predicate_hook_raises_type_error() -> None:
    async def _predicate(_result: _N) -> bool:
        return True

    with install(_answer('{"n": 1}')) as fake, pytest.raises(TypeError, match="bool"):
        _ = await structured_llm_call(
            "q", _N, feature="f", provider=_PROVIDER, on_result=_as_hook(_predicate)
        )
    assert len(fake.requests) == 1


@pytest.mark.parametrize("verdict", [True, False])
async def test_a_predicate_hook_raises_type_error_on_a_cache_hit(
    verdict: bool, caplog: pytest.LogCaptureFixture
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(_answer('{"n": 1}')) as fake:
        _ = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        _ = await text_llm_call("q", feature="f", provider=_PROVIDER)
        with pytest.raises(TypeError, match="bool"):
            _ = await structured_llm_call(
                "q",
                _N,
                feature="f",
                provider=_PROVIDER,
                on_result=_as_hook(lambda _result: verdict),
            )
        with pytest.raises(TypeError, match="bool"):
            _ = await text_llm_call(
                "q", feature="f", provider=_PROVIDER, on_result=_as_hook(lambda _text: verdict)
            )
    # Both raised on the hit: the two paid calls are the only requests.
    assert len(fake.requests) == 2
    assert _cache_warnings(caplog) == []


def test_the_sync_forms_await_an_async_hook_and_reject_a_predicate() -> None:
    async def _reject_one(result: _N) -> None:
        if result.n == 1:
            raise ResultValidationError("too low")

    async def _reject_text(text: str) -> None:
        if text == "one":
            raise ResultValidationError("too short")

    structured_answers = iter(['{"n": 1}', '{"n": 2}'])
    with install(lambda _request: Reply.text(next(structured_answers))) as fake:
        result = structured_llm_call_sync(
            "q", _N, feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject_one
        )
    assert result == _N(n=2)
    assert len(fake.requests) == 2

    text_answers = iter(["one", "two"])
    with install(lambda _request: Reply.text(next(text_answers))) as fake:
        text = text_llm_call_sync(
            "q", feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject_text
        )
    assert text == "two"
    assert len(fake.requests) == 2

    with install(_answer('{"n": 1}')) as fake:
        with pytest.raises(TypeError, match="bool"):
            _ = structured_llm_call_sync(
                "q", _N, feature="f", provider=_PROVIDER, on_result=_as_hook(lambda _r: False)
            )
        with pytest.raises(TypeError, match="bool"):
            _ = text_llm_call_sync(
                "q", feature="f", provider=_PROVIDER, on_result=_as_hook(lambda _t: True)
            )
    assert len(fake.requests) == 2


# --- (c) the cache hit and the caller's hook ---------------------------------


async def test_a_hook_error_on_a_hit_propagates_without_a_paid_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_llm_cache(InMemoryLLMCache())

    def _buggy(_result: _N) -> None:
        raise KeyError("a bug in the host's hook")

    with install(_answer('{"n": 1}')) as fake:
        _ = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        with pytest.raises(KeyError, match="a bug in the host's hook"):
            _ = await structured_llm_call(
                "q", _N, feature="f", provider=_PROVIDER, on_result=_buggy
            )
    assert len(fake.requests) == 1
    # Not mislabelled as a broken store.
    assert _cache_warnings(caplog) == []


def _pydantic_error() -> Exception:
    try:
        _ = _N.model_validate({"n": "not a number"})
    except Exception as exc:
        return exc
    raise AssertionError("expected a ValidationError")


@pytest.mark.parametrize(
    "make_error",
    [lambda: ResultValidationError("reject"), _pydantic_error],
    ids=["result-validation-error", "pydantic-validation-error"],
)
async def test_a_hook_error_the_call_would_reroll_makes_a_hit_a_miss(
    make_error: Callable[[], Exception], caplog: pytest.LogCaptureFixture
) -> None:
    """On a paid answer the call re-rolls on an error in its validation budget,
    so on a hit the same error makes a miss and the call pays for a new answer."""
    configure_llm_cache(InMemoryLLMCache())
    replies = iter(['{"n": 1}', '{"n": 2}'])

    def _reject_one(result: _N) -> None:
        if result.n == 1:
            raise make_error()

    with install(lambda _request: Reply.text(next(replies))) as fake:
        _ = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        result = await structured_llm_call(
            "q", _N, feature="f", provider=_PROVIDER, on_result=_reject_one
        )
    assert result == _N(n=2)
    assert len(fake.requests) == 2
    assert _cache_warnings(caplog) == []


async def test_a_hook_that_mutates_the_paid_answer_does_not_change_the_stored_one() -> None:
    configure_llm_cache(InMemoryLLMCache())

    def _mutate(result: _N) -> None:
        result.n = 999

    with install(_answer('{"n": 1}')) as fake:
        paid = await structured_llm_call(
            "q", _N, feature="f", provider=_PROVIDER, on_result=_mutate
        )
        hit = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER, on_result=_mutate)
        later = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
    # The caller's own instance carries its mutation; the store never does.
    assert paid.n == 999
    assert hit.n == 999
    assert later == _N(n=1)
    assert len(fake.requests) == 1


async def test_an_async_hook_that_rejects_a_hit_on_the_structured_lane_pays_for_a_reroll(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    seen: list[int] = []

    async def _reject_one(result: _N) -> None:
        await asyncio.sleep(0)
        seen.append(result.n)
        if result.n == 1:
            raise ResultValidationError("too low")

    answers = iter(['{"n": 1}', '{"n": 2}'])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        seeded = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        # The hit decodes n=1; the awaited hook rejects it, so this caller pays.
        result = await structured_llm_call(
            "q", _N, feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject_one
        )
    assert seeded == _N(n=1)
    assert result == _N(n=2)
    assert seen == [1, 2]
    assert len(fake.requests) == 2
    assert _cache_warnings(caplog) == []


async def test_an_async_hook_that_rejects_a_hit_on_the_text_lane_pays_for_a_reroll(
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_llm_cache(InMemoryLLMCache())
    seen: list[str] = []

    async def _reject_one(text: str) -> None:
        await asyncio.sleep(0)
        seen.append(text)
        if text == "one":
            raise ResultValidationError("too short")

    answers = iter(["one", "two"])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        seeded = await text_llm_call("q", feature="f", provider=_PROVIDER)
        result = await text_llm_call(
            "q", feature="f", retry=_NO_BACKOFF, provider=_PROVIDER, on_result=_reject_one
        )
    assert seeded == "one"
    assert result == "two"
    assert seen == ["one", "two"]
    assert len(fake.requests) == 2
    assert _cache_warnings(caplog) == []


async def test_the_cache_stores_the_providers_exact_answer_text() -> None:
    """Not a re-serialization: the spacing a dump would normalize survives."""
    cache = _SpyCache()
    configure_llm_cache(cache)
    with install(_answer('{"n":   1}')) as fake:
        paid = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
    assert paid == _N(n=1)
    assert len(fake.requests) == 1
    [key] = cache.keys
    entry = await cache.get(key)
    assert entry is not None
    assert entry.response == '{"n":   1}'


class _ToolsPinnedOllama(OllamaProvider):
    """A host-defined provider pinned to ``Mode.TOOLS``: the answer is a tool call."""

    _mode: ClassVar[instructor.Mode] = instructor.Mode.TOOLS


async def test_a_tools_mode_answer_is_returned_but_never_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The answer arrives as tool-call arguments with empty message text, so
    there is nothing a hit could validate: each call pays, nothing is stored,
    and no entry is ever read back as a broken one."""
    cache = _SpyCache()
    configure_llm_cache(cache)
    provider = _ToolsPinnedOllama()
    reply = Reply(tool_calls=[FakeToolCall("_N", {"n": 1})])
    with install(lambda _request: reply) as fake:
        first = await structured_llm_call("q", _N, feature="f", retry=NO_RETRY, provider=provider)
        second = await structured_llm_call("q", _N, feature="f", retry=NO_RETRY, provider=provider)
    assert first == _N(n=1)
    assert second == _N(n=1)
    assert len(fake.requests) == 2
    # Both calls looked; neither stored.
    assert cache.gets == 2
    assert cache.keys == []
    # Skipping an answer with no text is deliberate, so it warns about nothing.
    assert _cache_warnings(caplog) == []


async def test_a_corrupted_stored_entry_is_still_a_warned_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = _SpyCache()
    configure_llm_cache(cache)
    answers = iter(['{"n": 1}', '{"n": 2}'])
    with install(lambda _request: Reply.text(next(answers))) as fake:
        _ = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        await _replace_stored(cache, "not json at all")
        repaired = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        hit = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
    assert repaired == _N(n=2)
    # The paid replacement was stored over the corrupted entry.
    assert hit == _N(n=2)
    assert len(fake.requests) == 2
    [warning] = _cache_warnings(caplog)
    assert "decode" in warning.getMessage()


# --- (d) the schema guard ----------------------------------------------------


@dataclasses.dataclass
class _Dataclass:
    n: int


class _Typed(TypedDict):
    n: int


class _AsyncField(BaseModel):
    n: int

    @async_field_validator("n")
    async def check_n(self, value: int) -> None:
        del value


class _AsyncModel(BaseModel):
    n: int

    @async_model_validator()
    async def check_whole(self) -> None:
        return None


class _InheritsAsync(_AsyncField):
    label: str = ""


class _NestsAsyncInList(BaseModel):
    items: list[_AsyncModel]


class _NestsAsyncOptional(BaseModel):
    inner: _AsyncField | None = None


class _AsyncInTypedDict(TypedDict):
    inner: _AsyncField


@dataclasses.dataclass
class _AsyncInDataclass:
    inner: _AsyncModel


class _NestsAsyncInTypedDict(BaseModel):
    x: _AsyncInTypedDict


class _NestsAsyncInDataclass(BaseModel):
    x: _AsyncInDataclass


_UNSUPPORTED: list[object] = [
    dict,
    _Dataclass,
    _Typed,
    _AsyncField,
    _AsyncModel,
    _InheritsAsync,
    _NestsAsyncInList,
    _NestsAsyncOptional,
    _NestsAsyncInTypedDict,
    _NestsAsyncInDataclass,
]
_UNSUPPORTED_IDS = [
    "dict",
    "dataclass",
    "typeddict",
    "async-field",
    "async-model",
    "inherited",
    "nested-list",
    "nested-optional",
    "nested-in-typeddict",
    "nested-in-dataclass",
]


def test_the_guard_error_is_a_permanent_llmkit_type_error() -> None:
    assert issubclass(UnsupportedSchemaError, LlmkitError)
    assert issubclass(UnsupportedSchemaError, TypeError)
    assert not issubclass(UnsupportedSchemaError, LLM_RECOVERABLE_ERRORS)


@pytest.mark.parametrize("schema", _UNSUPPORTED, ids=_UNSUPPORTED_IDS)
async def test_the_guard_refuses_before_any_request_or_cache_read(schema: object) -> None:
    cache = _SpyCache()
    configure_llm_cache(cache)
    output_schema = cast("type[BaseModel]", schema)
    tool = ToolDefinition.from_model("count", _N, "Count")
    with install(_answer('{"n": 1}')) as fake:
        with pytest.raises(UnsupportedSchemaError):
            _ = await structured_llm_call("q", output_schema, feature="f", provider=_PROVIDER)
        with pytest.raises(UnsupportedSchemaError):
            _ = await tool_llm_call(
                "q", [tool], output_schema=output_schema, feature="f", provider=_PROVIDER
            )
    assert fake.requests == []
    assert cache.gets == 0


@pytest.mark.parametrize("schema", _UNSUPPORTED, ids=_UNSUPPORTED_IDS)
def test_the_sync_forms_refuse_before_any_request_or_cache_read(schema: object) -> None:
    cache = _SpyCache()
    configure_llm_cache(cache)
    output_schema = cast("type[BaseModel]", schema)
    tool = ToolDefinition.from_model("count", _N, "Count")
    with install(_answer('{"n": 1}')) as fake:
        with pytest.raises(UnsupportedSchemaError):
            _ = structured_llm_call_sync("q", output_schema, feature="f", provider=_PROVIDER)
        with pytest.raises(UnsupportedSchemaError):
            _ = tool_llm_call_sync(
                "q", [tool], output_schema=output_schema, feature="f", provider=_PROVIDER
            )
    assert fake.requests == []
    assert cache.gets == 0


async def test_a_supported_schema_reaches_the_cache_and_the_fake() -> None:
    """The control for the two tests above: their counters can see a positive."""
    cache = _SpyCache()
    configure_llm_cache(cache)
    with install(_answer('{"n": 1}')) as fake:
        result = await structured_llm_call("q", _N, feature="f", provider=_PROVIDER)
        composed = await tool_llm_call("q", [], output_schema=_N, feature="f", provider=_PROVIDER)
    assert result == _N(n=1)
    assert composed.parsed == _N(n=1)
    assert len(fake.requests) == 2
    assert cache.gets == 1


@pytest.mark.parametrize(
    "schema",
    [IterableModel(_N), Partial[_N]],
    ids=["iterable-model", "partial"],
)
async def test_an_instructor_dsl_class_is_refused_before_any_request(
    schema: type[BaseModel],
) -> None:
    """instructor returns a list for ``IterableModel`` and validates an
    incomplete answer for ``Partial``, so neither lane may run one."""
    with install(_answer('{"tasks": [{"n": 1}]}')) as fake:
        with pytest.raises(UnsupportedSchemaError, match="DSL class"):
            _ = await structured_llm_call(
                "q", schema, feature="f", retry=NO_RETRY, provider=_PROVIDER
            )
        with pytest.raises(UnsupportedSchemaError, match="DSL class"):
            _ = await tool_llm_call(
                "q", [], output_schema=schema, feature="f", retry=NO_RETRY, provider=_PROVIDER
            )
    assert fake.requests == []


# --- (e) the compose lane's log dump is best-effort --------------------------


class _UndumpableAnswer(BaseModel):
    n: int

    @model_serializer
    def refuse(self) -> dict[str, object]:
        raise RuntimeError("this model cannot be dumped")


async def test_a_raising_serializer_still_returns_the_validated_compose_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with install(_answer('{"n": 1}')) as fake, capture_llm_records() as records:
        result = await tool_llm_call(
            "q", [], output_schema=_UndumpableAnswer, feature="f", provider=_PROVIDER
        )
    assert type(result.parsed) is _UndumpableAnswer
    assert result.parsed.n == 1
    assert len(fake.requests) == 1
    [record] = records
    assert record.error is None
    assert cast("object", record.response) is None
    assert [
        warning.getMessage()
        for warning in caplog.records
        if warning.name == "llmkit.calls" and warning.levelno == logging.WARNING
    ] == ["Could not dump the f/None compose result for the log; recording response as null"]


# --- (f) a reply with no choices ---------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"content": "x"},
        {"tool_calls": [FakeToolCall("count")]},
        {"chunks": ["x"]},
        {"native_finish_reason": "ERROR"},
        {"service_tier": "default"},
    ],
    ids=["content", "tool_calls", "chunks", "native_finish_reason", "service_tier"],
)
def test_no_choice_carries_nothing_else(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="no_choice"):
        _ = Reply(no_choice=True, **fields)  # pyright: ignore[reportArgumentType]  # test-helper — kwargs splat


@pytest.mark.parametrize("schema_repairs", [0, 1, 2])
async def test_no_choice_is_a_structured_output_error_re_asked_on_the_validation_budget(
    schema_repairs: int,
) -> None:
    # Whatever the repair allowance: instructor's re-ask handler indexes the
    # empty choice list before its retry loop decides whether to stop.
    with (
        install(lambda _request: Reply(no_choice=True)) as fake,
        capture_llm_records() as records,
        pytest.raises(StructuredOutputError) as caught,
    ):
        _ = await structured_llm_call(
            "q",
            _N,
            feature="f",
            retry=_NO_BACKOFF,
            provider=_PROVIDER,
            schema_repairs=schema_repairs,
        )
    assert caught.value.raw_text is None
    assert caught.value.errors is None
    assert "no choices" in str(caught.value)
    assert not isinstance(caught.value.__cause__, IndexError)
    # One request per attempt (instructor's repair never gets to send), two
    # attempts on the validation budget.
    assert [(request.attempt, request.repair) for request in fake.requests] == [
        (1, False),
        (2, False),
    ]
    assert caught.value.attempt == 2
    assert [record.error is not None for record in records] == [True, True]


async def test_no_choice_is_an_empty_answer_on_the_text_lane() -> None:
    with install(lambda _request: Reply(no_choice=True)) as fake:
        text = await text_llm_call("q", feature="f", provider=_PROVIDER)
    assert text == ""
    assert len(fake.requests) == 1


async def test_no_choice_is_an_empty_result_on_the_tool_lane() -> None:
    tool = ToolDefinition.from_model("count", _N, "Count")
    with install(lambda _request: Reply(no_choice=True)) as fake:
        result = await tool_llm_call("q", [tool], feature="f", provider=_PROVIDER)
    assert result.text is None
    assert result.tool_calls == []
    assert result.invalid_calls == []
    assert len(fake.requests) == 1


@pytest.mark.parametrize("missing", ["id", "name", "arguments"])
async def test_tool_reask_handles_missing_provider_fields(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llmkit.testing as testing
    from llmkit._transport import WireFunction, WireToolCall

    original = testing._reply_tool_calls
    calls = 0

    def malformed_first(reply: Reply) -> tuple[WireToolCall, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                WireToolCall(
                    None if missing == "id" else "bad",
                    WireFunction(
                        None if missing == "name" else "count",
                        None if missing == "arguments" else '{"n": "1"}',
                    ),
                ),
            )
        return original(reply)

    monkeypatch.setattr(testing, "_reply_tool_calls", malformed_first)
    with install(lambda _request: Reply(tool_calls=[FakeToolCall("count", '{"n": 1}')])) as fake:
        _ = await tool_llm_call(
            "q",
            [ToolDefinition.from_model("count", _N, "Count")],
            feature="f",
            provider=_PROVIDER,
            retry=_NO_BACKOFF,
        )
    first, second = fake.requests
    if missing in ("id", "name"):
        assert second.messages == first.messages
        assert not second.repair
    else:
        assert second.repair
        _, assistant, error, _ = second.messages
        assistant = cast("AssistantToolMessage", assistant)
        assert (
            cast("dict[str, object]", assistant["tool_calls"][0]["function"])["arguments"] == "{}"
        )
        assert "No arguments arrived" in cast("ToolResultMessage", error)["content"]


async def test_tool_repair_tracks_latest_round_and_survives_transport_failure() -> None:
    replies = iter(
        [
            Reply(tool_calls=[FakeToolCall("count", '{"n": "1"}', id="first")]),
            TimeoutError("no response"),
            Reply(tool_calls=[FakeToolCall("count", '{"n": "2"}', id="latest")]),
            Reply.text('{"n": "3"}'),
            Reply.text('{"n": 4}'),
        ]
    )

    def answer(_request: TransportRequest) -> Reply:
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    with install(answer) as fake, capture_llm_records() as records:
        result = await tool_llm_call(
            "q",
            [ToolDefinition.from_model("count", _N, "Count")],
            output_schema=_N,
            feature="f",
            provider=_PROVIDER,
            # Three validation failures plus a transport failure need five total passes.
            retry=RetryPolicy(max_attempts=5, validation_max_attempts=4, backoff_base_seconds=0.0),
        )
    assert result.parsed == _N(n=4)
    assert [request.repair for request in fake.requests] == [False, True, True, True, False]
    assert fake.requests[1].messages == fake.requests[2].messages
    assert len(fake.requests[3].messages) == 4
    latest = fake.requests[3].messages[1]
    latest = cast("AssistantToolMessage", latest)
    assert latest["tool_calls"][0]["id"] == "latest"
    assert fake.requests[4].messages == fake.requests[0].messages
    assert [record.prompt for record in records][1] == records[2].prompt


def test_whole_round_error_carries_distinct_individual_errors() -> None:
    from llmkit._transport import WireFunction, WireToolCall
    from llmkit.calls._shared import parse_tool_calls

    calls = [WireToolCall(name, WireFunction("count", '{"n": "1"}')) for name in ("a", "b")]
    with pytest.raises(ToolArgumentError) as caught:
        _ = parse_tool_calls(list[object](calls), {"count": ToolDefinition.from_model("count", _N)})
    error = caught.value
    assert [item.call_id for item in error.round_errors] == ["a", "b"]
    assert all(item is not error and item.round_errors == () for item in error.round_errors)
