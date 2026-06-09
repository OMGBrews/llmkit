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
