"""The returned instance is built by the caller's own schema, from the answer text.

instructor validates the answer into a subclass it generates, to drive its
repair re-ask. The call then validates the same answer text once into the
caller's class, the step a cache hit takes. So whatever the schema's
validators and ``model_post_init`` build (a subclass instance, a reference
back to the instance, slots, attributes set outside the fields) is built for
the object the caller receives, nothing of instructor's wrapper leaks into
it, and the paid answer equals its cache hit. Every test answers through the
public :mod:`llmkit.testing` fake.
"""

from __future__ import annotations

from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from llmkit import InMemoryLLMCache, configure_llm_cache, structured_llm_call
from llmkit.testing import Reply, install


class _Shape(BaseModel):
    kind: str

    @model_validator(mode="wrap")
    @classmethod
    def _specialise(cls, data: object, handler: object) -> _Shape:
        shape = handler(data)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]  # pydantic's wrap handler
        if cls is _Shape and isinstance(shape, _Shape) and shape.kind == "circle":
            return _Circle(kind="circle", round=True)
        return shape  # pyright: ignore[reportUnknownVariableType]  # pydantic's wrap handler


class _Circle(_Shape):
    round: bool = False


async def test_a_validator_that_returns_a_subclass_is_kept_paid_and_cached() -> None:
    configure_llm_cache(InMemoryLLMCache())
    with install(lambda _request: Reply.text('{"kind": "circle"}')) as fake:
        paid = await structured_llm_call("q", _Shape, feature="f")
        hit = await structured_llm_call("q", _Shape, feature="f")

    assert len(fake.requests) == 1
    assert type(paid) is type(hit) is _Circle
    assert paid == hit


class _Node(BaseModel):
    name: str
    _parent: _Tree | None = PrivateAttr(default=None)


class _Tree(BaseModel):
    children: list[_Node]
    _self: _Tree | None = PrivateAttr(default=None)

    @override
    def model_post_init(self, context: object, /) -> None:
        self._self = self
        for child in self.children:
            child._parent = self


async def test_state_model_post_init_ties_to_the_instance_points_at_the_returned_one() -> None:
    with install(lambda _request: Reply.text('{"children": [{"name": "a"}]}')):
        tree = await structured_llm_call("q", _Tree, feature="f")

    assert type(tree) is _Tree
    assert tree._self is tree
    assert tree.children[0]._parent is tree
    assert "_raw_response" not in vars(tree)


class _Annotated(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    n: int

    @model_validator(mode="after")
    def _note(self) -> _Annotated:
        object.__setattr__(self, "note", f"seen {self.n}")
        return self


async def test_an_attribute_a_validator_sets_outside_the_fields_survives() -> None:
    with install(lambda _request: Reply.text('{"n": 4}')):
        result = await structured_llm_call("q", _Annotated, feature="f")

    assert getattr(result, "note", None) == "seen 4"
