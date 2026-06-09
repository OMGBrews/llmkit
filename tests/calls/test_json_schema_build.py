"""Tests for ``model_from_json_schema`` — building a model from a schema dict.

Covers the conversion subset (nested objects, ``$defs``/``$ref``, arrays,
enums, required/optional fields), the two CaCL footguns (omitted optionals
serialising as ``null``; title-less schemas), clear failures on unsupported
constructs (multi-branch unions, recursion, bad ``$ref``), and a full
round-trip through ``structured_llm_call`` against a faked transport (patching
``llmkit._litellm.acompletion_structured``, the same seam the other offline
tests use). Per-field constraint handling lives in
``test_json_schema_constraints.py``.
"""

from __future__ import annotations

import asyncio
import warnings
from enum import Enum

import pytest
from pydantic import BaseModel, ValidationError

from llmkit import model_from_json_schema, structured_output
from tests._support import model_attr as _attr

# A representative schema: nested objects (inline + via $defs/$ref), an array
# of objects, an enum, and a mix of required and optional fields.
_INVOICE_SCHEMA: dict[str, object] = {
    "title": "Invoice",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "total": {"type": "number"},
        "paid": {"type": "boolean"},
        "note": {"type": ["string", "null"]},
        "status": {"enum": ["open", "closed", "void"]},
        "customer": {"$ref": "#/$defs/Party"},
        "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "total", "status", "customer", "lines"],
    "$defs": {
        "Party": {
            "title": "Party",
            "type": "object",
            "properties": {"name": {"type": "string"}, "vip": {"type": "boolean"}},
            "required": ["name"],
        },
        "Line": {
            "title": "Line",
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku"],
        },
    },
}


def test_builds_named_model_for_representative_schema() -> None:
    model = model_from_json_schema(_INVOICE_SCHEMA)
    assert issubclass(model, BaseModel)
    assert model.__name__ == "Invoice"
    # Required field is required; optional field has a default.
    assert model.model_fields["id"].is_required()
    assert not model.model_fields["note"].is_required()


def test_validates_and_rejects_per_schema() -> None:
    model = model_from_json_schema(_INVOICE_SCHEMA)
    inst = model(
        id="INV-1",
        total=42.0,
        status="open",
        customer={"name": "Acme"},
        lines=[{"sku": "A", "qty": 2}, {"sku": "B"}],
    )
    dumped = inst.model_dump()
    assert dumped["id"] == "INV-1"
    assert dumped["status"] == "open"
    assert dumped["lines"][0] == {"sku": "A", "qty": 2}
    # A missing required field raises.
    with pytest.raises(ValidationError):
        _ = model(total=1.0, status="open", customer={"name": "X"}, lines=[])
    # An out-of-enum value raises.
    with pytest.raises(ValidationError):
        _ = model(
            id="x",
            total=1.0,
            status="nonsense",
            customer={"name": "X"},
            lines=[],
        )


def test_forbids_unexpected_extra_field() -> None:
    """Generated models set ``extra='forbid'`` — a key not in the schema is
    rejected (deliberately stricter than JSON Schema's permissive default, so a
    hallucinated extra field fails loudly rather than passing silently)."""
    model = model_from_json_schema(
        {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
    )
    assert model(id="ok").model_dump() == {"id": "ok"}
    with pytest.raises(ValidationError):
        _ = model(id="ok", hallucinated="nope")


def test_multi_branch_union_raises_clear_value_error() -> None:
    """A genuine multi-type union (not just ``<type>`` + null) is unsupported
    and fails loudly rather than mis-converting."""
    with pytest.raises(ValueError, match="Unsupported union type"):
        _ = model_from_json_schema(
            {
                "type": "object",
                "properties": {"x": {"type": ["string", "integer"]}},
                "required": ["x"],
            }
        )
    with pytest.raises(ValueError, match=r"Unsupported anyOf with \d+ non-null branches"):
        _ = model_from_json_schema(
            {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                "required": ["x"],
            }
        )
    with pytest.raises(ValueError, match=r"Unsupported oneOf with \d+ non-null branches"):
        _ = model_from_json_schema(
            {
                "type": "object",
                "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
                "required": ["x"],
            }
        )


def test_non_mapping_schema_raises_clear_value_error() -> None:
    """A non-mapping argument fails loudly with a clear message."""
    with pytest.raises(ValueError, match="mapping schema"):
        _ = model_from_json_schema([{"type": "object"}])  # pyright: ignore[reportArgumentType]


def test_ref_to_same_def_yields_one_class() -> None:
    """A ``$def`` referenced from two places resolves to the *same* class —
    build-once, not a fresh model per reference site."""
    schema: dict[str, object] = {
        "title": "Pair",
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Point"}, "b": {"$ref": "#/$defs/Point"}},
        "required": ["a", "b"],
        "$defs": {
            "Point": {
                "title": "Point",
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        },
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["a"].annotation is model.model_fields["b"].annotation


# --- Footgun #1: omitted optional must not serialise as null ---------------


def test_omitted_optional_is_absent_not_null() -> None:
    """An omitted optional must NOT round-trip as ``"field": null`` and fail
    downstream re-validation against the same JSON schema."""
    schema: dict[str, object] = {
        "title": "Record",
        "type": "object",
        "properties": {"required_field": {"type": "string"}, "optional_field": {"type": "string"}},
        "required": ["required_field"],
    }
    model = model_from_json_schema(schema)
    inst = model(required_field="present")

    # model_dump / model_dump_json default to exclude_none: the optional is
    # absent, not present-and-null.
    assert inst.model_dump() == {"required_field": "present"}
    assert "optional_field" not in inst.model_dump()
    assert "null" not in inst.model_dump_json()

    # The dumped payload re-validates cleanly against the same model (the
    # downstream re-validation that the null would have broken).
    revalidated = model.model_validate(inst.model_dump())
    assert revalidated.model_dump() == {"required_field": "present"}

    # Opt back in to the nulls explicitly when actually wanted.
    assert inst.model_dump(exclude_none=False) == {
        "required_field": "present",
        "optional_field": None,
    }


# --- Footgun #2: title-less / empty-title schema --------------------------


def test_titleless_schema_gets_safe_default_name() -> None:
    model = model_from_json_schema({"type": "object", "properties": {"x": {"type": "string"}}})
    assert model.__name__ == "JsonSchemaModel"
    assert model.__name__  # non-empty — instructor/create_model need a real name


def test_empty_title_gets_safe_default_name() -> None:
    model = model_from_json_schema(
        {"title": "   ", "type": "object", "properties": {"x": {"type": "string"}}}
    )
    assert model.__name__ == "JsonSchemaModel"


def test_explicit_name_overrides_title() -> None:
    model = model_from_json_schema(_INVOICE_SCHEMA, name="MyInvoice")
    assert model.__name__ == "MyInvoice"


# --- Unsupported constructs fail loudly -----------------------------------


def test_unsupported_type_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported JSON-schema type 'foobar'"):
        _ = model_from_json_schema(
            {"title": "Bad", "type": "object", "properties": {"x": {"type": "foobar"}}}
        )


def test_unresolvable_ref_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="Unresolvable \\$ref"):
        _ = model_from_json_schema(
            {
                "title": "Bad",
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/Missing"}},
                "required": ["x"],
            }
        )


def test_non_object_root_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="top level must be an object"):
        _ = model_from_json_schema({"type": "array", "items": {"type": "string"}})


def test_typeless_enum_root_raises_clear_value_error() -> None:
    """A bare ``{"enum": [...]}`` root must fail loudly naming the construct —
    pre-fix it silently built a zero-field ``extra='forbid'`` model that
    rejected every real response (or validated ``{}``, losing the data)."""
    with pytest.raises(ValueError, match="top level must be an object, got 'enum'"):
        _ = model_from_json_schema({"enum": ["a", "b", "c"]})


def test_typeless_anyof_root_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="top level must be an object, got 'anyOf'"):
        _ = model_from_json_schema({"anyOf": [{"type": "string"}, {"type": "null"}]})


def test_typeless_oneof_root_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="top level must be an object, got 'oneOf'"):
        _ = model_from_json_schema({"oneOf": [{"type": "string"}, {"type": "integer"}]})


def test_empty_root_raises_clear_value_error() -> None:
    """A root with neither ``type`` nor ``properties`` is not recognisably an
    object — fail clearly rather than build an empty everything-rejecting model."""
    with pytest.raises(ValueError, match="top level must be an object"):
        _ = model_from_json_schema({})
    with pytest.raises(ValueError, match="top level must be an object"):
        _ = model_from_json_schema({"title": "JustATitle"})


def test_ref_root_to_enum_def_raises_clear_value_error() -> None:
    """A root ``$ref`` resolving to an enum def is the same non-object root —
    the typeless-root gate applies to the resolved target too."""
    with pytest.raises(ValueError, match="top level must be an object, got 'enum'"):
        _ = model_from_json_schema(
            {"$ref": "#/$defs/Status", "$defs": {"Status": {"enum": ["open", "closed"]}}}
        )


def test_bare_properties_root_still_builds() -> None:
    """Regression: a typeless root that carries ``properties`` is recognisably
    an object and must keep building (the shape the typeless-root gate exists
    to preserve)."""
    model = model_from_json_schema({"properties": {"x": {"type": "string"}}, "required": ["x"]})
    assert _attr(model(x="hi"), "x") == "hi"
    with pytest.raises(ValidationError):
        _ = model()


def test_propertyless_object_root_raises_clear_value_error() -> None:
    """An explicit ``{"type": "object"}`` with no ``properties`` must fail loud
    (suggesting ``additionalProperties: true`` for free-form intent) — pre-fix
    it silently built a zero-field ``extra='forbid'`` model that validated only
    ``{}`` and rejected every real response."""
    with pytest.raises(
        ValueError, match=r"Unsupported object at '\$'.*'additionalProperties': true"
    ):
        _ = model_from_json_schema({"type": "object"})


def test_propertyless_nested_object_raises_naming_field_path() -> None:
    """The same propertyless-object failure nested as a field names the field's
    path, not the root."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"meta": {"type": "object"}},
        "required": ["meta"],
    }
    with pytest.raises(ValueError, match=r"Unsupported object at '\$\.meta'.*no 'properties'"):
        _ = model_from_json_schema(schema)


def test_propertyless_object_with_additional_properties_true_builds_open() -> None:
    """``additionalProperties: true`` is the one shape that makes a propertyless
    object meaningful (free-form): it builds ``extra='allow'`` and keeps
    whatever keys arrive, at the root and nested."""
    root = model_from_json_schema({"type": "object", "additionalProperties": True})
    assert root.model_validate({"anything": 1}).model_dump() == {"anything": 1}

    nested = model_from_json_schema(
        {
            "type": "object",
            "properties": {"meta": {"type": "object", "additionalProperties": True}},
            "required": ["meta"],
        }
    )
    inst = nested.model_validate({"meta": {"free": "form"}})
    assert inst.model_dump() == {"meta": {"free": "form"}}


# --- Malformed ``required`` fails loudly ------------------------------------


def test_non_list_required_raises_clear_value_error() -> None:
    """``"required": "a"`` (a bare string, not a list) must fail loud — pre-fix
    it was silently ignored, so every field built optional and the model
    validated payloads missing fields the author marked required."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": "a",
    }
    with pytest.raises(ValueError, match=r"'required' must be a list of property-name strings"):
        _ = model_from_json_schema(schema)


def test_non_string_required_entry_raises_clear_value_error() -> None:
    """A non-string entry inside ``required`` is the same malformed-`required`
    failure and names the offending entry."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", 1],
    }
    with pytest.raises(ValueError, match=r"'required' entries must be property-name strings"):
        _ = model_from_json_schema(schema)


def test_required_name_missing_from_properties_raises_clear_value_error() -> None:
    """A ``required`` name with no matching property (a schema typo) must fail
    loud naming the missing property — pre-fix it degraded silently in BOTH
    directions: payloads missing 'ghost' were accepted, and payloads carrying
    it were rejected as an extra under ``extra='forbid'``."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "ghost"],
    }
    with pytest.raises(ValueError, match=r"'required' names \['ghost'\] not present"):
        _ = model_from_json_schema(schema)


# --- Required-but-nullable fields accept null ------------------------------


def test_required_nullable_list_form_accepts_null() -> None:
    """A REQUIRED field typed ``["string", "null"]`` must accept the provider's
    ``null`` (not be mis-converted to a plain non-nullable ``str``)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"a": {"type": ["string", "null"]}},
        "required": ["a"],
    }
    model = model_from_json_schema(schema)
    # Required: must be present...
    assert model.model_fields["a"].is_required()
    with pytest.raises(ValidationError):
        _ = model()
    # ...but null is a valid value.
    assert _attr(model(a=None), "a") is None
    assert _attr(model(a="x"), "a") == "x"

    # An explicitly-null REQUIRED field must SURVIVE the default dump: the
    # exclude-none scoping drops only unset *optionals*, never a required null
    # (dropping it would break the re-validation this module exists to protect).
    assert model(a=None).model_dump() == {"a": None}
    assert "null" in model(a=None).model_dump_json()
    revalidated = model.model_validate(model(a=None).model_dump())
    assert _attr(revalidated, "a") is None


def test_required_nullable_anyof_form_accepts_null() -> None:
    """Same, via the ``anyOf`` + null-branch shape."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["a"],
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["a"].is_required()
    assert _attr(model(a=None), "a") is None
    assert _attr(model(a="x"), "a") == "x"

    # Required null survives the default dump and re-validates cleanly.
    assert model(a=None).model_dump() == {"a": None}
    assert model.model_validate(model(a=None).model_dump()) is not None


def test_required_null_and_unset_optional_dump_independently() -> None:
    """A model mixing a required-nullable null with an unset optional drops only
    the optional — the required null stays, so the payload re-validates."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "req": {"type": ["string", "null"]},
            "opt": {"type": "string"},
        },
        "required": ["req"],
    }
    model = model_from_json_schema(schema)
    inst = model(req=None)
    # Required null kept; unset optional dropped.
    assert inst.model_dump() == {"req": None}
    # Escape hatch still surfaces every null.
    assert inst.model_dump(exclude_none=False) == {"req": None, "opt": None}
    # Native "drop all nulls" remains available when explicitly requested.
    assert inst.model_dump(exclude_none=True) == {}
    # The default payload re-validates against the same model.
    assert model.model_validate(inst.model_dump()).model_dump() == {"req": None}


# --- anyOf/oneOf nullable shape (optional) ---------------------------------


def test_optional_nullable_anyof_excluded_when_unset() -> None:
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
    }
    model = model_from_json_schema(schema)
    assert not model.model_fields["a"].is_required()
    inst = model()
    assert _attr(inst, "a") is None
    assert inst.model_dump() == {}
    assert _attr(model(a=5), "a") == 5


def test_optional_nullable_oneof_excluded_when_unset() -> None:
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"a": {"oneOf": [{"type": "string"}, {"type": "null"}]}},
    }
    model = model_from_json_schema(schema)
    assert not model.model_fields["a"].is_required()
    assert model().model_dump() == {}
    assert _attr(model(a="hi"), "a") == "hi"


# --- Integer enums ---------------------------------------------------------


def test_integer_enum_accepts_rejects_and_dumps_raw_int() -> None:
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"level": {"enum": [1, 2, 3]}},
        "required": ["level"],
    }
    model = model_from_json_schema(schema)
    inst = model(level=2)
    # Dumps as a raw int (int-mixin enum), not ``Level.VALUE``.
    dumped = inst.model_dump()
    assert dumped["level"] == 2
    assert isinstance(dumped["level"], int)
    assert "2" in model(level=2).model_dump_json()
    # Genuinely the raw scalar, not the Enum member (use_enum_values).
    assert not isinstance(dumped["level"], Enum)
    # Out-of-enum value rejected.
    with pytest.raises(ValidationError):
        _ = model(level=9)


def test_signed_integer_enum_builds_and_dumps_raw() -> None:
    """A non-contiguous integer enum with a negative sentinel (FiW's eval-judge
    schema) must build without raising and round-trip as raw ints.

    Regression: ``-1`` and ``1`` both reduced to the member key ``"1"`` (the
    sign was stripped), so the collision suffix bumped one to ``_1_`` — a
    reserved ``_sunder_`` name Python's ``Enum`` rejects."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"score": {"type": "integer", "enum": [-1, 1, 2, 3, 4, 5]}},
        "required": ["score"],
    }
    model = model_from_json_schema(schema)  # must not raise
    # Every member is accepted and the non-contiguous gaps are rejected.
    for value in (-1, 1, 2, 3, 4, 5):
        dumped: dict[str, object] = model(score=value).model_dump()
        assert dumped == {"score": value}
        assert type(dumped["score"]) is int  # raw scalar, not an Enum member
    for bad in (0, -2, 6):
        with pytest.raises(ValidationError):
            _ = model(score=bad)


def test_enum_member_names_are_never_reserved() -> None:
    """Generated enum member names must never be ``_sunder_`` / ``__dunder__``
    (Python's ``Enum`` reserves them), whatever digit-led or colliding values
    the schema carries."""
    model = model_from_json_schema(
        {
            "type": "object",
            "properties": {"v": {"type": "integer", "enum": [-1, 0, 1, -2, 2]}},
            "required": ["v"],
        }
    )
    annotation = model.model_fields["v"].annotation
    # The field is a required integer enum, so its annotation is the generated
    # ``Enum`` subclass — narrow to it so ``__members__`` is statically known.
    assert annotation is not None
    assert issubclass(annotation, Enum)
    names = list(annotation.__members__)
    assert len(names) == 5  # no collisions collapsed members
    for name in names:
        assert not (name.startswith("_") and name.endswith("_")), name  # not _sunder_/__dunder__


# --- Canonical nullable enum (null as an enum member) ----------------------


def test_nullable_enum_list_form_accepts_members_and_null() -> None:
    """The canonical nullable-enum spelling — ``type: ["string", "null"]`` with
    ``null`` as an enum member — must build a nullable enum that accepts every
    string member AND an actual ``null`` (JSON Schema requires ``null`` in the
    ``enum`` list for it to validate)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"choice": {"type": ["string", "null"], "enum": ["a", "b", None]}},
        "required": ["choice"],
    }
    model = model_from_json_schema(schema)
    # Required field, but null is one of its valid values.
    assert model.model_fields["choice"].is_required()
    for value in ("a", "b"):
        assert _attr(model(choice=value), "choice") == value
    assert _attr(model(choice=None), "choice") is None
    # An out-of-enum string is still rejected — null didn't widen the field.
    with pytest.raises(ValidationError):
        _ = model(choice="c")
    # A real null round-trips: it survives the default dump and re-validates.
    assert model(choice=None).model_dump() == {"choice": None}
    revalidated = model.model_validate(model(choice=None).model_dump())
    assert _attr(revalidated, "choice") is None
    assert _attr(model.model_validate({"choice": "a"}), "choice") == "a"


def test_nullable_enum_anyof_form_accepts_members_and_null() -> None:
    """Same canonical spelling via the ``anyOf`` + null-branch shape: the
    non-null branch carries a ``null``-bearing enum, handled consistently."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "choice": {"anyOf": [{"type": "string", "enum": ["a", "b", None]}, {"type": "null"}]}
        },
        "required": ["choice"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(choice="a"), "choice") == "a"
    assert _attr(model(choice=None), "choice") is None
    with pytest.raises(ValidationError):
        _ = model(choice="c")
    assert model(choice=None).model_dump() == {"choice": None}


def test_non_nullable_enum_with_null_member_raises() -> None:
    """A ``null`` enum member on a field whose type is NOT nullable is a
    contradiction (an enum that permits ``null`` but a type that forbids it).
    It must fail loud naming the field path — not silently drop the ``null``."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["a", None]}},
        "required": ["choice"],
    }
    with pytest.raises(ValueError, match=r"choice"):
        _ = model_from_json_schema(schema)


def test_nullable_enum_with_only_null_member_raises() -> None:
    """A nullable enum whose only member is ``null`` collapses to an empty value
    list once ``null`` is dropped — it must fail loud (no empty ``Enum``) with a
    message naming the ACTUAL defect (only-null members), not the misleading
    "must be a non-empty list" the post-strip emptiness used to produce."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"choice": {"type": ["string", "null"], "enum": [None]}},
        "required": ["choice"],
    }
    with pytest.raises(ValueError, match=r"non-null member.*only null"):
        _ = model_from_json_schema(schema)


# --- $defs sharing a title must NOT collapse to one class ------------------


def test_distinct_defs_with_same_title_stay_distinct() -> None:
    """Two structurally-different $defs that happen to share a ``title`` must
    resolve to DIFFERENT classes — not silently collapse to one (a wrong
    model the task forbids)."""
    schema: dict[str, object] = {
        "title": "Pair",
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/$defs/B"}},
        "required": ["a", "b"],
        "$defs": {
            "A": {
                "title": "Item",
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "B": {
                "title": "Item",
                "type": "object",
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            },
        },
    }
    model = model_from_json_schema(schema)
    # Distinct classes despite the shared title.
    assert model.model_fields["a"].annotation is not model.model_fields["b"].annotation
    # And each validates against its OWN shape.
    inst = model(a={"x": "hello"}, b={"y": 5})
    assert inst.model_dump() == {"a": {"x": "hello"}, "b": {"y": 5}}
    # b validated against A's shape would have passed wrongly; assert it fails.
    with pytest.raises(ValidationError):
        _ = model(a={"x": "hello"}, b={"x": "wrong"})


# --- Recursive / self-referential schema fails loud ------------------------


def test_self_referential_schema_raises_clear_value_error() -> None:
    """A self-referential (tree/Node) schema must raise a clear ValueError
    naming the recursive $ref — never a RecursionError."""
    schema: dict[str, object] = {
        "title": "Node",
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
        },
        "required": ["value"],
        "$defs": {
            "Node": {
                "title": "Node",
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
                "required": ["value"],
            }
        },
    }
    with pytest.raises(ValueError, match="recursive"):
        _ = model_from_json_schema(schema)


def test_mutually_recursive_schema_raises_clear_value_error() -> None:
    schema: dict[str, object] = {
        "title": "A",
        "type": "object",
        "properties": {"b": {"$ref": "#/$defs/B"}},
        "required": ["b"],
        "$defs": {
            "A": {
                "title": "A",
                "type": "object",
                "properties": {"b": {"$ref": "#/$defs/B"}},
                "required": ["b"],
            },
            "B": {
                "title": "B",
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "required": ["a"],
            },
        },
    }
    with pytest.raises(ValueError, match="recursive"):
        _ = model_from_json_schema(schema)


def test_deep_ref_chain_resolves() -> None:
    """A ``$ref`` chain deeper than two hops resolves to the final scalar
    rather than falling through to the self-contradictory "no 'type'" error."""
    schema: dict[str, object] = {
        "title": "Deep",
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/A"}},
        "required": ["x"],
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/C"},
            "C": {"type": "integer"},
        },
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["x"].annotation is int


def test_pure_ref_cycle_raises_clear_value_error() -> None:
    """A cycle made only of ``$ref``s (no object in between) fails with the
    recursive-schema error, never an infinite loop or the "no 'type'" message."""
    schema: dict[str, object] = {
        "title": "Loop",
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/A"}},
        "required": ["x"],
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/A"},
        },
    }
    with pytest.raises(ValueError, match="recursive"):
        _ = model_from_json_schema(schema)


@pytest.mark.parametrize("reserved", ["_id", "model_config", "model_dump", "model_fields"])
def test_reserved_property_name_raises_clear_value_error(reserved: str) -> None:
    """A genuinely-breaking property name — leading underscore (silently dropped
    as a private attribute) or the ``model_`` protected namespace (TypeError /
    conflicts-with-member errors, or shadowed model machinery) — fails with a
    clear ValueError naming the property and its path."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {reserved: {"type": "string"}},
        "required": [reserved],
    }
    with pytest.raises(ValueError, match=rf"Unsupported property '{reserved}' at"):
        _ = model_from_json_schema(schema)


def test_v1_shim_property_names_build_and_round_trip() -> None:
    """Names that shadow pydantic's deprecated v1 shims (``schema``, ``json``,
    ``copy``, ...) are perfectly good fields — the reserved-name guard lets them
    through, and the shadow ``UserWarning`` pydantic emits for them is
    suppressed so a legitimate schema builds without warning spam."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"schema": {"type": "string"}, "json": {"type": "string"}},
        "required": ["schema", "json"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any leaked UserWarning fails the build
        model = model_from_json_schema(schema)
    inst = model.model_validate({"schema": "s", "json": "j"})
    assert inst.model_dump() == {"schema": "s", "json": "j"}
    # And the dump re-validates — a full round-trip through the shim names.
    assert model.model_validate(inst.model_dump()).model_dump() == {"schema": "s", "json": "j"}


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_non_dict_union_branch_raises_clear_value_error(keyword: str) -> None:
    """A non-dict branch in ``anyOf``/``oneOf`` (e.g. the bare string ``"string"``)
    fails with a clear ValueError naming the keyword and path, not an
    ``AttributeError`` from calling ``.get`` on a ``str``."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"x": {keyword: ["string", {"type": "null"}]}},
        "required": ["x"],
    }
    with pytest.raises(ValueError, match=rf"Unsupported {keyword} branch at"):
        _ = model_from_json_schema(schema)


# --- Round-trip through structured_llm_call against a faked transport -------


def test_round_trip_through_structured_call() -> None:
    """A representative schema drives ``structured_llm_call`` against a faked
    provider and returns the validated instance — the build-once model is
    passed straight in as ``output_schema``."""
    model = model_from_json_schema(_INVOICE_SCHEMA)

    async def _fake_transport(
        _prompt: object, output_schema: type[BaseModel], **_kwargs: object
    ) -> tuple[BaseModel, float | None]:
        # The seam receives exactly the generated model and parses into it,
        # the way instructor would from the model-derived JSON schema.
        assert output_schema is model
        parsed = output_schema.model_validate(
            {
                "id": "INV-9",
                "total": 99.5,
                "status": "closed",
                "customer": {"name": "Globex", "vip": True},
                "lines": [{"sku": "Z", "qty": 3}],
            }
        )
        return parsed, 0.001

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("llmkit._litellm.acompletion_structured", _fake_transport)
        result = asyncio.run(
            structured_output.structured_llm_call("Extract the invoice.", model, feature="billing")
        )

    assert _attr(result, "id") == "INV-9"
    assert _attr(result, "status") == "closed"
    # Omitted optionals stay absent on the way back out.
    dumped = result.model_dump()
    assert "note" not in dumped
    assert dumped["customer"] == {"name": "Globex", "vip": True}


def test_build_once_model_reused_across_calls() -> None:
    """The same generated model object can be reused across multiple calls
    (build-once-reuse pattern) — no per-call rebuild required."""
    model = model_from_json_schema(_INVOICE_SCHEMA)
    seen: list[type[BaseModel]] = []

    async def _fake_transport(
        _prompt: object, output_schema: type[BaseModel], **_kwargs: object
    ) -> tuple[BaseModel, float | None]:
        seen.append(output_schema)
        parsed = output_schema.model_validate(
            {
                "id": "X",
                "total": 1.0,
                "status": "open",
                "customer": {"name": "N"},
                "lines": [],
            }
        )
        return parsed, None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("llmkit._litellm.acompletion_structured", _fake_transport)
        for _ in range(3):
            _ = asyncio.run(structured_output.structured_llm_call("go", model, feature="billing"))

    assert len(seen) == 3
    assert all(s is model for s in seen)  # the very same object each time
