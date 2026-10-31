"""Construction policy and lossless representation of schema-backed models."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from llmkit import model_from_json_schema


@pytest.mark.parametrize(
    "dialect",
    [
        None,
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2019-09/schema",
        "https://json-schema.org/draft/2020-12/schema",
    ],
)
def test_supported_dialects(dialect: str | None) -> None:
    schema: dict[str, object] = {"type": "object", "properties": {"n": {"const": 1}}}
    if dialect is not None:
        schema["$schema"] = dialect
    model = model_from_json_schema(schema)
    assert model.model_validate({"n": 1}).root == {"n": 1}
    with pytest.raises(ValidationError):
        _ = model.model_validate({"n": 2})


@pytest.mark.parametrize(
    "schema", [{}, {"type": "array"}, {"type": "string"}, {"type": ["object", "null"]}]
)
def test_nonobject_roots_fail_at_construction(schema: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="top level must be an object"):
        _ = model_from_json_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "$schema": "https://example.com/dialect"},
        {"type": "object", "required": "x"},
        {"type": "object", "properties": {"x": {"pattern": "["}}},
        {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
        {"type": "object", "properties": {"x": {"$ref": "https://example.com/missing"}}},
        {"type": "object", "properties": {"x": {"$ref": "#"}}},
    ],
)
def test_invalid_or_unavailable_semantics_fail_early(schema: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ = model_from_json_schema(schema)


def test_root_reference_to_object() -> None:
    model = model_from_json_schema(
        {"$ref": "#/$defs/Root", "$defs": {"Root": {"type": "object", "required": ["x"]}}}
    )
    assert model.model_validate({"x": None}).root == {"x": None}


def test_original_snapshot_and_guidance_are_independent() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"x": {"pattern": "^[0-9]+$"}},
        "required": ["x"],
    }
    original = copy.deepcopy(schema)
    model = model_from_json_schema(schema)
    schema.clear()
    guidance = model.model_json_schema()
    guidance.clear()
    assert model.model_json_schema() == original
    with pytest.raises(ValidationError, match="pattern"):
        _ = model.model_validate({"x": "letters"})


def test_annotations_are_not_subschemas_or_defaults() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "default": {"x": 2},
        "properties": {"x": {"type": "integer", "default": 1}},
        "x-vendor": {
            "$ref": "https://unavailable.example/schema",
            "format": "not-a-format",
            "pattern": "[",
        },
        "examples": [{"$ref": "https://unavailable.example/schema"}],
    }
    model = model_from_json_schema(schema, assert_formats=True)
    assert model.model_validate({}).root == {}
    assert model.model_json_schema()["x-vendor"] == schema["x-vendor"]


def test_format_annotation_and_opt_in_assertion() -> None:
    schema = {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}}
    assert model_from_json_schema(schema).model_validate({"email": "bad"}).root == {"email": "bad"}
    strict = model_from_json_schema(schema, assert_formats=True)
    assert strict.model_validate({"email": "a@example.com"}).root == {"email": "a@example.com"}
    with pytest.raises(ValidationError):
        _ = strict.model_validate({"email": "bad"})
    with pytest.raises(ValueError, match="format"):
        _ = model_from_json_schema(
            {"type": "object", "properties": {"x": {"format": "unknown-format"}}},
            assert_formats=True,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"x": None},
        {"extra": {"nested": [None, 9007199254740993]}},
        {},
        {"x": None, "extra": 9007199254740993},
    ],
)
def test_default_serialization_preserves_json(value: dict[str, object]) -> None:
    model = model_from_json_schema(
        {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
    )
    result = model.model_validate(value)
    assert result.root == value
    assert result.model_dump() == value
    assert json.loads(result.model_dump_json()) == value
    assert model.model_validate_json(result.model_dump_json()).root == value
    assert set(model.model_fields) == {"root"}


def test_large_integer_number_const_does_not_coerce_to_float() -> None:
    model = model_from_json_schema(
        {
            "type": "object",
            "properties": {"x": {"type": "number", "const": 9007199254740993}},
            "required": ["x"],
        }
    )
    result = model.model_validate_json('{"x":9007199254740993}')
    assert result.root["x"] == 9007199254740993
    assert type(result.root["x"]) is int
    assert json.loads(result.model_dump_json())["x"] == 9007199254740993


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}'])
def test_nonfinite_numbers_are_rejected(raw: str) -> None:
    model = model_from_json_schema({"type": "object"})
    with pytest.raises(ValidationError):
        _ = model.model_validate_json(raw)


def test_existing_instance_revalidation() -> None:
    model = model_from_json_schema({"type": "object", "properties": {"x": {"const": 1}}})
    instance = model.model_validate({"x": 1})
    assert model.model_validate(instance).root == {"x": 1}
    instance.root["x"] = 2
    with pytest.raises(ValidationError):
        _ = model.model_validate(instance)


def test_required_unknown_vocabulary_fails_at_its_location() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        _ = model_from_json_schema(
            {"type": "object", "$vocabulary": {"https://example.com/required": True}}
        )
    model = model_from_json_schema(
        {"type": "object", "$vocabulary": {"https://example.com/optional": False}}
    )
    assert model.model_validate({}).root == {}


def test_scoped_ids_and_anchors_resolve_offline() -> None:
    model = model_from_json_schema(
        {
            "$id": "https://example.com/root",
            "type": "object",
            "$defs": {
                "scope": {
                    "$id": "child",
                    "$defs": {"n": {"$anchor": "number", "type": "integer"}},
                    "type": "object",
                    "properties": {"n": {"$ref": "#number"}},
                }
            },
            "properties": {"child": {"$ref": "child"}},
        }
    )
    assert model.model_validate({"child": {"n": 1}}).root == {"child": {"n": 1}}
    with pytest.raises(ValidationError):
        _ = model.model_validate({"child": {"n": "1"}})


def test_escaped_json_pointer_names() -> None:
    model = model_from_json_schema(
        {
            "type": "object",
            "$defs": {"a/b~c": {"type": "integer"}},
            "properties": {"x/y~z": {"$ref": "#/$defs/a~1b~0c"}},
        }
    )
    assert model.model_validate({"x/y~z": 1}).root == {"x/y~z": 1}
    assert model.model_json_schema()["properties"] == {"x/y~z": {"type": "integer"}}
    with pytest.raises(ValidationError):
        _ = model.model_validate({"x/y~z": "1"})


@pytest.mark.parametrize("target", [True, False])
def test_root_boolean_reference_has_clear_object_root_error(target: bool) -> None:
    with pytest.raises(ValueError, match="top level must be an object"):
        _ = model_from_json_schema({"$ref": "#/$defs/root", "$defs": {"root": target}})


def test_default_json_is_validated_after_serialization() -> None:
    from pydantic import model_serializer

    base = model_from_json_schema(
        {"type": "object", "required": ["x"], "properties": {"x": {"type": "null"}}}
    )

    from llmkit.json_schema.emitted import JsonSchemaModel

    class LosingNull(JsonSchemaModel):
        @model_serializer
        def lose_null(self) -> dict[str, object]:
            return {key: value for key, value in self.root.items() if value is not None}

    LosingNull.__llmkit_validate__ = staticmethod(base.__llmkit_validate__)
    # Input satisfies the schema. A lossy serializer must not escape merely
    # because the input was valid; the actual default JSON bytes are checked.
    with pytest.raises(ValidationError):
        _ = LosingNull.model_validate({"x": None})


def test_invalid_input_is_checked_before_field_coercion() -> None:
    model = model_from_json_schema({"type": "object", "properties": {"x": {"type": "integer"}}})
    with pytest.raises(ValidationError):
        _ = model.model_validate({"x": "1"})


@pytest.mark.parametrize("keyword", ["items", "dependentSchemas"])
def test_2019_subschema_references_are_checked_at_construction(keyword: str) -> None:
    node: object = (
        [{"$ref": "https://example.com/unavailable"}]
        if keyword == "items"
        else {"x": {"$ref": "https://example.com/unavailable"}}
    )
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "type": "object",
        "properties": {"array": {keyword: node}},
    }
    with pytest.raises(ValueError, match=r"Unresolvable.*properties/array"):
        _ = model_from_json_schema(schema)


def test_2019_prefixitems_payload_is_an_annotation() -> None:
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "type": "object",
        "prefixItems": [{"$ref": "https://example.com/unavailable", "pattern": "["}],
    }
    model = model_from_json_schema(schema)
    assert model.model_validate({}).root == {}
    assert model.model_json_schema()["prefixItems"] == schema["prefixItems"]


def test_draft7_dynamicref_is_an_annotation() -> None:
    model = model_from_json_schema(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "$dynamicRef": "https://example.com/unavailable",
        }
    )
    assert model.model_validate({}).root == {}
    assert model.model_json_schema()["$dynamicRef"] == "https://example.com/unavailable"


def test_nested_different_dialect_is_rejected_with_location() -> None:
    with pytest.raises(ValueError, match=r"properties/x.*\$schema"):
        _ = model_from_json_schema(
            {
                "type": "object",
                "properties": {
                    "x": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                    }
                },
            }
        )


def test_draft7_vocabulary_is_an_unknown_annotation() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "$vocabulary": {"https://example.test/custom": True},
    }
    model = model_from_json_schema(schema)
    assert model.model_validate({}).root == {}
    assert model.model_json_schema()["$vocabulary"] == schema["$vocabulary"]


def test_relative_root_id_is_not_applied_twice() -> None:
    model = model_from_json_schema(
        {
            "$id": "schemas/root.json",
            "type": "object",
            "$defs": {"value": {"type": "integer"}},
            "properties": {"x": {"$ref": "#/$defs/value"}},
        }
    )
    assert model.model_validate({"x": 1}).root == {"x": 1}
    assert model.model_json_schema()["properties"] == {"x": {"type": "integer"}}


def test_reference_to_nonschema_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="not a schema"):
        _ = model_from_json_schema(
            {
                "type": "object",
                "x-annotation": 3,
                "properties": {"x": {"$ref": "#/x-annotation"}},
            }
        )


def test_shared_annotation_values_are_never_rewritten_as_schemas() -> None:
    child = {"$ref": "#/$defs/value"}
    model = model_from_json_schema(
        {
            "type": "object",
            "$defs": {"value": {"type": "integer"}},
            "properties": {"x": child},
            "x-annotation": {"payload": child},
        }
    )
    assert model.model_json_schema()["x-annotation"] == {"payload": child}


def test_draft7_mixed_dependencies_check_every_schema() -> None:
    with pytest.raises(ValueError, match="Unresolvable"):
        _ = model_from_json_schema(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "dependencies": {"a": ["b"], "x": {"$ref": "https://unavailable.example/schema"}},
            }
        )
