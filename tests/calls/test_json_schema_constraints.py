"""Original-schema authority across keywords, locations, and valid alternatives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from llmkit import model_from_json_schema

_RULES: list[tuple[dict[str, object], object, object]] = [
    ({"type": "string", "pattern": "^[0-9]+$"}, "123", "abc"),
    ({"type": "number", "multipleOf": 3}, 9, 8),
    ({"const": {"a": 1}}, {"a": 1}, {"a": 2}),
    ({"type": "array", "uniqueItems": True}, [1, 2], [1, 1]),
    ({"allOf": [{"type": "integer"}, {"minimum": 2}]}, 3, 1),
    ({"oneOf": [{"type": "integer"}, {"type": "string"}]}, "a", None),
    ({"anyOf": [{"type": "null"}, {"pattern": "^x", "type": "string"}]}, None, "bad"),
    ({"type": "string", "minLength": 2, "maxLength": 3}, "ab", "a"),
    ({"type": "integer", "exclusiveMinimum": 2}, 3, 2),
    ({"type": "array", "contains": {"const": 1}}, [1, 2], [2]),
    ({"not": {"const": 1}}, 2, 1),
]


@pytest.mark.parametrize(("rule", "valid", "invalid"), _RULES)
@pytest.mark.parametrize("location", ["property", "nested", "items", "definition"])
def test_keyword_and_location(
    rule: dict[str, object], valid: object, invalid: object, location: str
) -> None:
    schema: dict[str, object] = {"type": "object"}
    if location == "property":
        schema["properties"] = {"x": rule}
        good, bad = {"x": valid}, {"x": invalid}
    elif location == "nested":
        schema["properties"] = {"outer": {"type": "object", "properties": {"x": rule}}}
        good, bad = {"outer": {"x": valid}}, {"outer": {"x": invalid}}
    elif location == "items":
        schema["properties"] = {"x": {"type": "array", "items": rule}}
        good, bad = {"x": [valid]}, {"x": [invalid]}
    else:
        schema["$defs"] = {"Value": rule}
        schema["properties"] = {"x": {"$ref": "#/$defs/Value"}}
        good, bad = {"x": valid}, {"x": invalid}
    model = model_from_json_schema(schema)
    assert model.model_validate(good).model_dump() == good
    with pytest.raises(ValidationError):
        _ = model.model_validate(bad)


def test_ref_sibling_is_conjunction_not_outer_wins() -> None:
    model = model_from_json_schema(
        {
            "type": "object",
            "$defs": {"n": {"type": "integer", "minimum": 5}},
            "properties": {"x": {"$ref": "#/$defs/n", "minimum": 1}},
        }
    )
    assert model.model_validate({"x": 5}).root == {"x": 5}
    with pytest.raises(ValidationError):
        _ = model.model_validate({"x": 2})


def test_draft7_reference_siblings_follow_draft7() -> None:
    model = model_from_json_schema(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "definitions": {"n": {"type": "integer"}},
            "properties": {"x": {"$ref": "#/definitions/n", "minimum": 5}},
        }
    )
    assert model.model_validate({"x": 2}).root == {"x": 2}


@pytest.mark.parametrize(
    ("schema", "good", "bad"),
    [
        ({"type": "object", "minProperties": 1}, {"x": None}, {}),
        ({"type": "object", "additionalProperties": {"type": "integer"}}, {"x": 1}, {"x": "1"}),
        ({"type": "object", "dependentRequired": {"a": ["b"]}}, {"a": 1, "b": None}, {"a": 1}),
        (
            {"type": "object", "if": {"required": ["a"]}, "then": {"required": ["b"]}},
            {"a": 1, "b": 2},
            {"a": 1},
        ),
        (
            {"type": "object", "oneOf": [{"required": ["a"]}, {"required": ["b"]}]},
            {"a": 1},
            {"a": 1, "b": 2},
        ),
        (
            {
                "type": "object",
                "allOf": [{"properties": {"x": {"type": "integer"}}}],
                "unevaluatedProperties": False,
            },
            {"x": 1},
            {"x": 1, "y": 2},
        ),
    ],
)
def test_root_applicators(
    schema: dict[str, object], good: dict[str, object], bad: dict[str, object]
) -> None:
    model = model_from_json_schema(schema)
    assert model.model_validate(good).root == good
    with pytest.raises(ValidationError):
        _ = model.model_validate(bad)


def test_irrelevant_type_bounds_do_not_reject_valid_values() -> None:
    model = model_from_json_schema(
        {
            "type": "object",
            "properties": {
                "s": {"type": "string", "minItems": 3},
                "n": {"type": "number", "minLength": 4},
            },
        }
    )
    assert model.model_validate({"s": "a", "n": 1}).root == {"s": "a", "n": 1}


def test_instance_errors_include_locations() -> None:
    model = model_from_json_schema(
        {
            "type": "object",
            "properties": {"outer": {"type": "object", "properties": {"x": {"pattern": "^x"}}}},
        }
    )
    with pytest.raises(ValidationError) as caught:
        _ = model.model_validate({"outer": {"x": "bad"}})
    assert "outer" in str(caught.value)
    assert "pattern" in str(caught.value)


@pytest.mark.parametrize(
    "keyword", ["pattern", "const", "uniqueItems", "multipleOf", "allOf", "oneOf"]
)
def test_standards_suite_cases(keyword: str) -> None:
    """Upstream MIT cases wrapped in an object property to exercise llmkit's boundary."""
    path = Path(__file__).parents[1] / "data" / "json_schema" / f"{keyword}.json"
    groups = cast("list[dict[str, object]]", json.loads(path.read_text()))
    for group in groups:
        wrapped = {"type": "object", "properties": {"value": group["schema"]}}
        if group["description"] == "pattern with Unicode property escape requires unicode mode":
            # llmkit explicitly rejects ECMA-only regex syntax at construction.
            with pytest.raises(ValueError, match="pattern"):
                _ = model_from_json_schema(wrapped)
            continue
        model = model_from_json_schema(wrapped)
        for case in cast("list[dict[str, object]]", group["tests"]):
            value = {"value": case["data"]}
            if case["valid"]:
                assert model.model_validate(value).model_dump() == value, case["description"]
            else:
                with pytest.raises(ValidationError):
                    _ = model.model_validate(value)
