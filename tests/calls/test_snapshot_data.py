"""Plain-data ownership rejects hidden behavior instead of invoking copy hooks."""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator
from typing import Never, cast, override

import pytest

from llmkit import ChatMessage, ToolDefinition
from llmkit._snapshot import snapshot_data, snapshot_prompt, snapshot_tools


class Opaque:
    @override
    def __repr__(self) -> str:
        raise AssertionError("repr must not run")

    def __deepcopy__(self, memo: object) -> object:
        raise AssertionError("copy hook must not run")

    @override
    def __reduce__(self) -> Never:
        raise AssertionError("pickle hook must not run")


class CustomDict(dict[str, object]):
    @override
    def __iter__(self) -> Never:
        raise AssertionError("nested iteration hook must not run")


class CustomString(str):
    @override
    def __repr__(self) -> str:
        raise AssertionError("key repr must not run")


@pytest.mark.parametrize(
    "value",
    [
        Opaque(),
        b"image bytes",
        UserDict({"x": 1}),
        UserList([1]),
        CustomDict(x=1),
        {1},
        CustomString("x"),
    ],
    ids=["opaque", "bytes", "mapping", "sequence", "dict-subclass", "set", "str-subclass"],
)
def test_rejects_unsupported_nested_value_without_hooks(value: object) -> None:
    with pytest.raises(TypeError, match=r"prompt\[0\]\['content'\]\[0\]\['payload'\]"):
        _ = snapshot_prompt(
            [{"role": "user", "content": [{"type": "provider-part", "payload": value}]}]
        )


@pytest.mark.parametrize("key", [1, CustomString("x"), Opaque()], ids=["int", "subclass", "opaque"])
def test_dictionary_keys_are_plain_strings_without_rendering_bad_key(key: object) -> None:
    with pytest.raises(TypeError, match="dictionary keys must be plain strings"):
        _ = snapshot_data({key: 1})


def test_cycle_is_rejected_but_repeated_acyclic_values_are_owned_independently() -> None:
    child: dict[str, object] = {"x": [None, True, 3, 4.5, "five", (1, 2)]}
    original = [child, child]
    copied = snapshot_data(original)
    assert copied == original
    assert copied[0] is not child
    assert copied[0] is not copied[1]
    copied[0]["x"] = "changed"
    assert copied[1] == child
    assert original == [child, child]
    child["self"] = child
    with pytest.raises(TypeError, match=r"input\[0\]\['self'\]: cyclic"):
        _ = snapshot_data(original)


def test_outer_container_subclasses_are_materialized_once() -> None:
    class Messages(list[ChatMessage]):
        iterations: int = 0

        @override
        def __iter__(self) -> Iterator[ChatMessage]:
            self.iterations += 1
            return super().__iter__()

    source = Messages([{"role": "user", "content": "hello"}])
    result = snapshot_prompt(source)
    assert result == source
    assert type(result) is list
    assert source.iterations == 1
    assert result is not source
    assert result[0] is not source[0]


def test_tool_schema_owned_and_model_borrowed() -> None:
    from pydantic import BaseModel

    class Arguments(BaseModel):
        city: str

    tool = ToolDefinition.from_model("weather", Arguments)
    copied = snapshot_tools([tool])
    assert copied == [tool]
    assert copied[0] is not tool
    assert copied[0].model is Arguments
    props = cast("dict[str, object]", copied[0].parameters["properties"])
    props.clear()
    assert tool.parameters["properties"]
    cyclic: dict[str, object] = {}
    cyclic["items"] = cyclic
    with pytest.raises(TypeError, match=r"tools\[0\].parameters\['items'\]: cyclic"):
        _ = snapshot_tools([ToolDefinition("bad", "", cyclic)])


def test_excessive_nesting_fails_with_an_input_location() -> None:
    nested: object = None
    for _ in range(2000):
        nested = [nested]
    with pytest.raises(TypeError, match="input: nesting exceeds"):
        _ = snapshot_data(nested)


def test_metaclass_equality_cannot_impersonate_a_plain_type() -> None:
    class PretendsPlain(type):
        @override
        def __eq__(cls, other: object) -> bool:
            raise AssertionError("type equality hooks must not run")

    class Pretender(metaclass=PretendsPlain):
        pass

    with pytest.raises(TypeError, match="expected plain data"):
        _ = snapshot_data({"value": Pretender()})
