"""A structured call returns the caller's own schema class.

instructor parses into a response-model subclass it generates from the
schema. The call rebuilds that answer as the schema itself before returning,
so a paid answer, its cache hit and the ``on_result`` hook all see the class
the caller passed, and ``==`` against a freshly built instance holds. A schema
that cannot survive its own JSON round trip keeps the parser's instance. These
tests run the real parser through an :mod:`llmkit.testing` fake.
"""

from __future__ import annotations

import logging
from typing import ClassVar, override

import pytest
from pydantic import BaseModel, ConfigDict, Field, model_serializer

from llmkit import (
    InMemoryLLMCache,
    configure_llm_cache,
    structured_llm_call,
    structured_llm_call_sync,
)
from llmkit.testing import Reply, install


class _Plain(BaseModel):
    n: int


class _FrozenAliased(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, populate_by_name=False)
    score: int = Field(alias="Score")


class _CustomEq(BaseModel):
    name: str

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is _CustomEq and other.name.casefold() == self.name.casefold()

    @override
    def __hash__(self) -> int:
        return hash(self.name.casefold())


_CASES: list[BaseModel] = [_Plain(n=1), _FrozenAliased(Score=3), _CustomEq(name="Ada")]


def _schema_name(instance: BaseModel) -> str:
    return type(instance).__name__


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", _CASES, ids=_schema_name)
async def test_a_paid_answer_is_the_schema_class(expected: BaseModel) -> None:
    schema = type(expected)
    with install(lambda _request: Reply.json(expected)):
        result = await structured_llm_call("q", schema, feature="f")

    assert type(result) is schema
    assert result == expected


@pytest.mark.parametrize("expected", _CASES, ids=_schema_name)
def test_a_sync_paid_answer_is_the_schema_class(expected: BaseModel) -> None:
    schema = type(expected)
    with install(lambda _request: Reply.json(expected)):
        result = structured_llm_call_sync("q", schema, feature="f")

    assert type(result) is schema
    assert result == expected


@pytest.mark.asyncio
async def test_a_paid_answer_equals_its_cache_hit() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.json(_Plain(n=7))) as installed:
        paid = await structured_llm_call("q", _Plain, feature="f")
        hit = await structured_llm_call("q", _Plain, feature="f")

    assert len(installed.requests) == 1
    assert type(paid) is type(hit) is _Plain
    assert paid == hit


def test_a_sync_paid_answer_equals_its_cache_hit() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.json(_Plain(n=7))) as installed:
        paid = structured_llm_call_sync("q", _Plain, feature="f")
        hit = structured_llm_call_sync("q", _Plain, feature="f")

    assert len(installed.requests) == 1
    assert paid == hit


@pytest.mark.asyncio
async def test_on_result_sees_the_schema_class_paid_and_cached() -> None:
    configure_llm_cache(InMemoryLLMCache())
    seen: list[type] = []

    def _record(result: _Plain) -> None:
        seen.append(type(result))

    with install(lambda _request: Reply.json(_Plain(n=7))) as installed:
        _ = await structured_llm_call("q", _Plain, feature="f", on_result=_record)
        _ = await structured_llm_call("q", _Plain, feature="f", on_result=_record)

    assert len(installed.requests) == 1
    assert seen == [_Plain, _Plain]


class _ExcludedRequired(BaseModel):
    """Drops a required field from its own JSON, so it cannot round-trip."""

    secret: int = Field(exclude=True)
    shown: int = 0


class _RaisingSerializer(BaseModel):
    ok: bool

    @model_serializer
    def _boom(self) -> dict[str, object]:
        raise ValueError("serializer exploded")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "raw"),
    [(_ExcludedRequired, '{"secret": 1, "shown": 2}'), (_RaisingSerializer, '{"ok": true}')],
    ids=["excluded-required", "raising-serializer"],
)
async def test_a_schema_that_cannot_round_trip_still_returns_one_paid_answer(
    schema: type[BaseModel], raw: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The rebuild is skipped with a warning rather than failing the call,
    which would re-roll a paid request on the validation budget."""
    with (
        caplog.at_level(logging.WARNING, logger="llmkit._litellm"),
        install(lambda _request: Reply.text(raw)) as installed,
    ):
        result = await structured_llm_call("q", schema, feature="f")

    assert isinstance(result, schema)
    assert len(installed.requests) == 1
    assert any("failed its JSON round trip" in m for m in caplog.messages)
