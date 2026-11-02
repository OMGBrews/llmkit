"""Which schemas the OpenRouter strict upgrade may send with ``"strict": true``.

``llmkit._litellm._stricten_json_schema_response_format`` adds ``"strict": true``
and closes the root of a ``json_schema`` ``response_format`` for providers that
opt in (OpenRouter). OpenAI's strict validator, which OpenAI- and Azure-served
OpenRouter endpoints apply, answers HTTP 400 when any nested object is open or
any object leaves a property out of ``required`` (measured 2026-11-02), so the
upgrade must leave those schemas non-strict and untouched. The inputs are
pydantic's own ``model_json_schema()`` output wherever a model expresses the
shape, so a change in how pydantic renders a model shows up here. The live half
is ``test_openrouter_openai_served_*_live`` in ``tests/integration``.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import ClassVar, cast
from unittest.mock import patch

import litellm
from pydantic import BaseModel, ConfigDict

from llmkit import _litellm, model_from_json_schema
from llmkit.providers import OpenRouterProvider
from llmkit.tools import ToolDefinition


def _upgrade(schema: dict[str, object]) -> dict[str, object]:
    """Run the upgrade on *schema* and return the ``json_schema`` document it edited."""
    document: dict[str, object] = {"name": "Answer", "schema": schema}
    _litellm._stricten_json_schema_response_format(
        {"response_format": {"type": "json_schema", "json_schema": document}}
    )
    return document


def _assert_left_non_strict(schema: dict[str, object]) -> None:
    original = deepcopy(schema)
    document = _upgrade(schema)
    assert "strict" not in document
    assert schema == original


def _assert_made_strict(schema: dict[str, object]) -> None:
    """The upgrade adds ``strict`` and closes the root, and changes nothing else."""
    original = deepcopy(schema)
    document = _upgrade(schema)
    closed = {**original, "additionalProperties": False}
    assert document == {"name": "Answer", "schema": closed, "strict": True}
    assert schema == closed


class _Residence(BaseModel):
    city: str


class _OpenNested(BaseModel):
    residence: _Residence


class _ClosedResidence(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    city: str


class _ClosedRootOpenDef(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    residence: _Residence


class _ClosedThroughout(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    residence: _ClosedResidence


class _ClosedDefaultedResidence(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    city: str
    postcode: str | None = None


class _ClosedRootClosedDefaultedDef(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    residence: _ClosedDefaultedResidence


class _Defaulted(BaseModel):
    name: str
    nickname: str | None = None


class _RequiredOptional(BaseModel):
    name: str
    nickname: str | None


class _Counts(BaseModel):
    counts: dict[str, int]


class _Closed(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    name: str


_GENERATED_CLOSED: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer"},
        "assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"confident": {"type": "boolean"}},
            "required": ["confident"],
        },
    },
    "required": ["score", "assessment"],
}


def test_nested_open_object_is_left_non_strict() -> None:
    schema = _OpenNested.model_json_schema()
    defs = cast("dict[str, dict[str, object]]", schema["$defs"])
    assert "additionalProperties" not in defs["_Residence"]
    _assert_left_non_strict(schema)


def test_defaulted_field_is_left_non_strict() -> None:
    schema = _Defaulted.model_json_schema()
    assert schema["required"] == ["name"]
    _assert_left_non_strict(schema)


def test_required_optional_field_is_made_strict_and_root_closed() -> None:
    schema = _RequiredOptional.model_json_schema()
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    assert "anyOf" in properties["nickname"]
    assert schema["required"] == ["name", "nickname"]
    assert "additionalProperties" not in schema
    _assert_made_strict(schema)


def test_generated_closed_all_required_schema_is_made_strict() -> None:
    schema = model_from_json_schema(
        deepcopy(_GENERATED_CLOSED), name="judgment"
    ).model_json_schema()
    assert schema == _GENERATED_CLOSED
    _assert_made_strict(schema)


def test_root_already_closed_is_made_strict() -> None:
    schema = _Closed.model_json_schema()
    assert schema["additionalProperties"] is False
    _assert_made_strict(schema)


def test_nested_closed_model_is_made_strict() -> None:
    """A nested ``$ref`` model passes when it is closed and all-required too."""
    _assert_made_strict(_ClosedThroughout.model_json_schema())


def test_nested_additional_properties_dict_is_left_non_strict() -> None:
    schema = _Counts.model_json_schema()
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    assert properties["counts"]["additionalProperties"] == {"type": "integer"}
    _assert_left_non_strict(schema)


def test_open_object_reached_only_through_defs_is_left_non_strict() -> None:
    """The root is closed and every property is a ``$ref``; only ``$defs`` is open."""
    schema = _ClosedRootOpenDef.model_json_schema()
    assert schema["additionalProperties"] is False
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    assert set(properties["residence"]) == {"$ref"}
    _assert_left_non_strict(schema)


def test_nested_closed_model_with_defaulted_field_is_left_non_strict() -> None:
    """Every object is closed, but the nested one leaves a defaulted field out of ``required``."""
    schema = _ClosedRootClosedDefaultedDef.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["residence"]
    defs = cast("dict[str, dict[str, object]]", schema["$defs"])
    assert defs["_ClosedDefaultedResidence"]["additionalProperties"] is False
    assert defs["_ClosedDefaultedResidence"]["required"] == ["city"]
    _assert_left_non_strict(schema)


_OPEN_ROW: dict[str, object] = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}


def test_open_inline_object_under_items_is_left_non_strict() -> None:
    _assert_left_non_strict(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"rows": {"type": "array", "items": deepcopy(_OPEN_ROW)}},
            "required": ["rows"],
        }
    )


def test_open_inline_object_under_any_of_with_null_is_left_non_strict() -> None:
    _assert_left_non_strict(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"meta": {"anyOf": [deepcopy(_OPEN_ROW), {"type": "null"}]}},
            "required": ["meta"],
        }
    )


def test_open_nullable_object_type_list_is_left_non_strict() -> None:
    """``"type": ["object", "null"]`` with no ``additionalProperties`` is an open object."""
    schema = model_from_json_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"meta": {"type": ["object", "null"]}},
            "required": ["meta"],
        },
        name="x",
    ).model_json_schema()
    properties = cast("dict[str, dict[str, object]]", schema["properties"])
    assert properties["meta"] == {"type": ["object", "null"]}
    _assert_left_non_strict(schema)


def test_root_all_of_is_left_unchanged() -> None:
    _assert_left_non_strict(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "allOf": [{"required": ["name"]}],
        }
    )


def _wire_json_schema(output_schema: type[BaseModel]) -> dict[str, object]:
    """The ``json_schema`` document an OpenRouter structured call sends through real instructor."""
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"residence": {"city": "London"}}',
                    },
                }
            ]
        )

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        _ = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                output_schema,
                temperature=0.0,
                model=None,
                provider=OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini"),
            )
        )
    response_format = cast("dict[str, object]", captured["response_format"])
    return cast("dict[str, object]", response_format["json_schema"])


def test_open_nested_model_reaches_the_openrouter_wire_non_strict() -> None:
    """instructor's request for an idiomatic model leaves the guard's hands non-strict."""
    json_schema = _wire_json_schema(_OpenNested)
    assert "strict" not in json_schema
    assert "additionalProperties" not in cast("dict[str, object]", json_schema["schema"])


def test_closed_nested_model_reaches_the_openrouter_wire_strict() -> None:
    json_schema = _wire_json_schema(_ClosedThroughout)
    assert json_schema["strict"] is True
    assert cast("dict[str, object]", json_schema["schema"])["additionalProperties"] is False


def _compose_wire_json_schema(output_schema: type[BaseModel]) -> dict[str, object]:
    """The ``json_schema`` document the compose transport sends to OpenRouter.

    ``tool_llm_call(..., output_schema=...)`` refuses OpenRouter before any
    request (``ComposeUnsupportedError``: it does not declare
    ``compose_tools_schema``), and OpenRouter is the only shipped provider with
    the strict upgrade, so the public compose lane cannot reach the upgrade
    today. This drives the transport below it, ``acompletion_tools``, with the
    ``response_format`` the compose lane builds, so the shared guard stays
    pinned on the tool lane too.
    """
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{}"},
                }
            ]
        )

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        _ = asyncio.run(
            _litellm.acompletion_tools(
                "hi",
                [ToolDefinition("lookup", "", {"type": "object", "properties": {}})],
                tool_choice=None,
                temperature=None,
                model=None,
                provider=OpenRouterProvider(api_key="k", model="openai/gpt-4o-mini"),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_schema.__name__,
                        "schema": output_schema.model_json_schema(),
                    },
                },
                output_schema=output_schema,
            )
        )
    response_format = cast("dict[str, object]", captured["response_format"])
    return cast("dict[str, object]", response_format["json_schema"])


def test_open_nested_model_reaches_the_compose_wire_non_strict() -> None:
    json_schema = _compose_wire_json_schema(_OpenNested)
    assert "strict" not in json_schema
    assert json_schema["schema"] == _OpenNested.model_json_schema()


def test_closed_nested_model_reaches_the_compose_wire_strict() -> None:
    """The counterpart: the compose transport does run the upgrade for a qualifying schema."""
    json_schema = _compose_wire_json_schema(_ClosedThroughout)
    assert json_schema["strict"] is True
    assert json_schema["schema"] == _ClosedThroughout.model_json_schema()
