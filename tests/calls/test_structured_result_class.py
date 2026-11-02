"""A structured call returns the caller's own schema class, validated once.

instructor validates into a response-model subclass it generates from the
schema. The call moves that answer into the schema's own class by copying its
validated state, with no second serialization or validation, so a paid
answer, its cache hit and the ``on_result`` hook all see the class the caller
passed, with the values the model sent validated exactly once. A cache stores
the answer text and validates a hit from it with the same strict rule, so a
hit equals its paid answer. These tests run the real parser through an
:mod:`llmkit.testing` fake.
"""

from __future__ import annotations

from typing import ClassVar, override

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    SecretStr,
    field_serializer,
    field_validator,
    model_serializer,
)

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


class _SerializationAlias(BaseModel):
    name: str = Field(validation_alias="user_name", serialization_alias="userName")


_ROUND_TRIP_HOSTILE: list[tuple[type[BaseModel], str]] = [
    (_ExcludedRequired, '{"secret": 1, "shown": 2}'),
    (_RaisingSerializer, '{"ok": true}'),
    (_SerializationAlias, '{"user_name": "ada"}'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "raw"),
    _ROUND_TRIP_HOSTILE,
    ids=["excluded-required", "raising-serializer", "serialization-alias"],
)
async def test_a_schema_that_cannot_round_trip_returns_its_own_class_and_caches(
    schema: type[BaseModel], raw: str
) -> None:
    """Nothing is re-serialized, so a schema whose dump cannot be read back
    still returns exactly its class, and its cache hit is the same answer."""
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.text(raw)) as installed:
        paid = await structured_llm_call("q", schema, feature="f")
        hit = await structured_llm_call("q", schema, feature="f")

    assert type(paid) is type(hit) is schema
    assert len(installed.requests) == 1
    assert paid.__dict__ == hit.__dict__


class _Hundredfold(BaseModel):
    price: int

    @field_validator("price")
    @classmethod
    def _cents(cls, value: int) -> int:
        return value * 100


@pytest.mark.asyncio
async def test_a_transforming_validator_runs_once_paid_and_cached() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.text('{"price": 5}')) as installed:
        paid = await structured_llm_call("q", _Hundredfold, feature="f")
        hit = await structured_llm_call("q", _Hundredfold, feature="f")

    assert len(installed.requests) == 1
    assert paid.price == hit.price == 500


class _Secret(BaseModel):
    token: SecretStr


class _Rounded(BaseModel):
    ratio: float

    @field_serializer("ratio")
    def _one_place(self, value: float) -> float:
        return round(value, 1)


@pytest.mark.asyncio
async def test_a_lossy_serializer_never_touches_the_returned_value() -> None:
    with install(lambda _request: Reply.text('{"token": "s3cr3t"}')):
        secret = await structured_llm_call("q", _Secret, feature="f")
    with install(lambda _request: Reply.text('{"ratio": 0.123}')):
        rounded = await structured_llm_call("q", _Rounded, feature="f")

    assert secret.token.get_secret_value() == "s3cr3t"
    assert rounded.ratio == 0.123


class _Defaults(BaseModel):
    title: str = "untitled"
    tags: list[str] = Field(default_factory=list)


class _WithState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)
    n: int
    _seen: int = PrivateAttr(default=0)

    @override
    def model_post_init(self, context: object, /) -> None:
        self._seen = self.n + 1


@pytest.mark.asyncio
async def test_fields_set_extras_and_private_state_survive() -> None:
    with install(lambda _request: Reply.text('{"tags": ["a"]}')):
        defaults = await structured_llm_call("q", _Defaults, feature="f")
    with install(lambda _request: Reply.text('{"n": 1, "note": "x"}')):
        stateful = await structured_llm_call("q", _WithState, feature="f")

    assert defaults.model_fields_set == {"tags"}
    assert type(stateful) is _WithState
    assert stateful.model_extra == {"note": "x"}
    assert stateful._seen == 2
    assert hash(stateful) == hash(stateful)


class _Ints(RootModel[list[int]]):
    pass


@pytest.mark.asyncio
async def test_a_root_model_answer_is_the_schema_class() -> None:
    with install(lambda _request: Reply.text("[1, 2, 3]")):
        result = await structured_llm_call("q", _Ints, feature="f")

    assert type(result) is _Ints
    assert result.root == [1, 2, 3]
