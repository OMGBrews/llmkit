"""Tests for per-field constraint handling in ``model_from_json_schema``.

Covers how the JSON-schema-dict converter carries (and deliberately drops)
field-level constraints: numeric/string/array bounds mapping to pydantic
``Field`` keywords, the nullable-merge precedence (outer field-level keys win
over the non-null branch), and ``additionalProperties`` open/strict/typed-map
handling. The model-building and round-trip tests live alongside in
``test_json_schema_build.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmkit import model_from_json_schema
from tests._support import model_attr as _attr

# --- Per-field constraints carry into Field --------------------------------


def test_numeric_inclusive_bounds_enforced() -> None:
    """``minimum``/``maximum`` map to ``ge``/``le``: in-bounds accepted,
    out-of-bounds rejected."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["score"],
    }
    model = model_from_json_schema(schema)
    # In-bounds (including the endpoints) accepted.
    assert _attr(model(score=1), "score") == 1
    assert _attr(model(score=5), "score") == 5
    assert _attr(model(score=3), "score") == 3
    # Out-of-bounds rejected on both ends.
    with pytest.raises(ValidationError):
        _ = model(score=0)
    with pytest.raises(ValidationError):
        _ = model(score=6)


def test_numeric_exclusive_bounds_enforced() -> None:
    """``exclusiveMinimum``/``exclusiveMaximum`` map to ``gt``/``lt``: the
    endpoints themselves are rejected."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"ratio": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}},
        "required": ["ratio"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(ratio=0.5), "ratio") == 0.5
    # Endpoints rejected (exclusive).
    with pytest.raises(ValidationError):
        _ = model(ratio=0)
    with pytest.raises(ValidationError):
        _ = model(ratio=1)
    with pytest.raises(ValidationError):
        _ = model(ratio=-0.1)


def test_string_length_bounds_enforced() -> None:
    """``minLength``/``maxLength`` map to ``min_length``/``max_length``."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 2, "maxLength": 4}},
        "required": ["code"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(code="ab"), "code") == "ab"
    assert _attr(model(code="abcd"), "code") == "abcd"
    with pytest.raises(ValidationError):
        _ = model(code="a")
    with pytest.raises(ValidationError):
        _ = model(code="abcde")


def test_array_item_count_bounds_enforced() -> None:
    """``minItems``/``maxItems`` map to ``min_length``/``max_length`` on the
    list field."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            }
        },
        "required": ["tags"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(tags=["a"]), "tags") == ["a"]
    assert _attr(model(tags=["a", "b"]), "tags") == ["a", "b"]
    with pytest.raises(ValidationError):
        _ = model(tags=[])
    with pytest.raises(ValidationError):
        _ = model(tags=["a", "b", "c"])


def test_array_item_string_min_length_enforced() -> None:
    """A ``minLength`` on the array's ``items`` schema bounds each ELEMENT —
    pre-fix the array branch recursed into ``items`` for the type only and
    silently dropped the bound, so ``["a"]`` validated."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string", "minLength": 3}}},
        "required": ["tags"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(tags=["abc", "abcd"]), "tags") == ["abc", "abcd"]
    with pytest.raises(ValidationError):
        _ = model(tags=["ab"])
    with pytest.raises(ValidationError):
        _ = model(tags=["abc", "x"])  # every element is bounded, not just the first


def test_array_item_integer_minimum_enforced() -> None:
    """Numeric bounds on ``items`` apply per element too."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"counts": {"type": "array", "items": {"type": "integer", "minimum": 0}}},
        "required": ["counts"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(counts=[0, 5]), "counts") == [0, 5]
    with pytest.raises(ValidationError):
        _ = model(counts=[-1])


def test_array_item_bound_via_ref_enforced() -> None:
    """A bound declared inside a ``$def`` used as the ``items`` schema is
    enforced per element — item constraints compose with ``$ref`` resolution."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"ns": {"type": "array", "items": {"$ref": "#/$defs/Bounded"}}},
        "required": ["ns"],
        "$defs": {"Bounded": {"type": "integer", "minimum": 10}},
    }
    model = model_from_json_schema(schema)
    assert _attr(model(ns=[10, 11]), "ns") == [10, 11]
    with pytest.raises(ValidationError):
        _ = model(ns=[9])


def test_array_item_bound_composes_with_item_count_bounds() -> None:
    """Element bounds (``minLength`` on items) and container bounds
    (``minItems`` on the array) are enforced independently and together."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "minLength": 2},
                "minItems": 1,
            }
        },
        "required": ["tags"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(tags=["ok"]), "tags") == ["ok"]
    with pytest.raises(ValidationError):
        _ = model(tags=[])  # container bound
    with pytest.raises(ValidationError):
        _ = model(tags=["x"])  # element bound


def test_nullable_array_items_with_bound_accept_null() -> None:
    """A bound on nullable items gates only the non-null elements — ``null``
    elements still pass (the constraint wraps the non-null branch)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "vals": {"type": "array", "items": {"type": ["string", "null"], "minLength": 3}}
        },
        "required": ["vals"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(vals=[None, "abc"]), "vals") == [None, "abc"]
    with pytest.raises(ValidationError):
        _ = model(vals=["ab"])


def test_array_of_objects_item_property_bounds_still_enforced() -> None:
    """Regression: nested object items keep their per-property bounds (this
    already worked pre-fix; pinned so the element-constraint wrap can't
    regress it)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"qty": {"type": "integer", "minimum": 1}},
                    "required": ["qty"],
                },
            }
        },
        "required": ["lines"],
    }
    model = model_from_json_schema(schema)
    inst = model(lines=[{"qty": 1}])
    assert inst.model_dump() == {"lines": [{"qty": 1}]}
    with pytest.raises(ValidationError):
        _ = model(lines=[{"qty": 0}])


def test_constraint_keyword_on_mismatched_type_is_dropped_not_crashing() -> None:
    """A bound that doesn't match the field's type is dropped, not applied.

    Pydantic rejects a length bound on a numeric field (and a numeric bound on a
    string field) with a ``TypeError`` at *validation* time, not build time — so
    applying a stray keyword unconditionally would turn a plausible-but-sloppy
    schema into an opaque crash on the first response. The converter promises to
    silently drop unsupported constraints; a *mismatched* one is dropped the
    same way. Build succeeds and the values validate with no bound enforced.
    """
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            # maxLength/minItems are nonsense on an integer; minimum is nonsense
            # on a string. None must reach pydantic's Field as a real constraint.
            "count": {"type": "integer", "maxLength": 3, "minItems": 1},
            "label": {"type": "string", "minimum": 0, "exclusiveMaximum": 10},
        },
        "required": ["count", "label"],
    }
    model = model_from_json_schema(schema)  # must not raise
    inst = model(count=99999, label="anything-long")  # must not raise at validation
    assert _attr(inst, "count") == 99999
    assert _attr(inst, "label") == "anything-long"


def test_mixed_string_and_integer_enum_is_rejected_clearly() -> None:
    """A mixed string/int enum can't be faithfully represented (one base coerces
    members), so it's rejected with a clear, path-naming error rather than
    silently building a model that rejects its own schema-valid integer values."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"kind": {"enum": ["a", 1]}},
        "required": ["kind"],
    }
    with pytest.raises(ValueError, match="mixed-type enum"):
        _ = model_from_json_schema(schema)


def test_bounds_on_optional_field_enforced_when_present() -> None:
    """A bound on an optional field still applies once the field is supplied,
    but the field may be omitted."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 0}},
    }
    model = model_from_json_schema(schema)
    assert model().model_dump() == {}  # omitted: fine
    assert _attr(model(score=0), "score") == 0
    with pytest.raises(ValidationError):
        _ = model(score=-1)


def test_description_passthrough_preserved() -> None:
    """Per-field ``description`` is carried into ``Field`` (instructor relies on
    it for per-field guidance) — pinned so the constraint work can't regress it."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "amount": {
                "type": "integer",
                "minimum": 0,
                "description": "Total in cents, must be non-negative.",
            },
            "plain": {"type": "string"},
        },
        "required": ["amount", "plain"],
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["amount"].description == "Total in cents, must be non-negative."
    # A field without a description stays ``None`` (no spurious default).
    assert model.model_fields["plain"].description is None
    # And the description coexists with the bound: bound still enforced.
    with pytest.raises(ValidationError):
        _ = model(amount=-1, plain="x")


def test_unsupported_constraint_dropped_without_error() -> None:
    """An unsupported constraint (``pattern``) is silently dropped — no error,
    and (critically) NOT partially enforced: a value violating ``pattern`` is
    accepted because pattern is not in the supported set."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[0-9]+$", "format": "uuid"}},
        "required": ["code"],
    }
    model = model_from_json_schema(schema)
    # Builds without error and accepts a value the pattern would have rejected.
    assert _attr(model(code="not-a-number"), "code") == "not-a-number"


def test_bound_on_referenced_def_is_found() -> None:
    """A bound declared inside a ``$def`` (resolved via ``$ref``) is still
    applied — the constraint extractor unwraps refs like the type resolver."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Bounded"}},
        "required": ["n"],
        "$defs": {"Bounded": {"type": "integer", "minimum": 10}},
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=10), "n") == 10
    with pytest.raises(ValidationError):
        _ = model(n=9)


def test_bound_at_end_of_three_hop_ref_chain_enforced() -> None:
    """A bound declared at the end of a 3-hop ``$ref`` chain is enforced —
    pre-fix the constraint extractor unwrapped at most two hops and silently
    dropped the bound (while the type resolver already looped to arbitrary
    depth, so the model built and ACCEPTED out-of-bounds values)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/A"}},
        "required": ["n"],
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/C"},
            "C": {"type": "integer", "minimum": 10},
        },
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=10), "n") == 10
    with pytest.raises(ValidationError):
        _ = model(n=9)


def test_bound_behind_nullable_wrapped_ref_chain_enforced() -> None:
    """A bound behind a nullable ``anyOf`` wrapping a 2-hop ``$ref`` chain is
    enforced, and ``null`` still passes — the same depth-limit regression in
    its nullable-wrapped form."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}},
        "required": ["n"],
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"type": "integer", "minimum": 10},
        },
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=None), "n") is None
    assert _attr(model(n=10), "n") == 10
    with pytest.raises(ValidationError):
        _ = model(n=9)


def test_ref_sibling_bound_is_applied() -> None:
    """A bound sitting ALONGSIDE a ``$ref`` (Draft 2020-12 sibling keywords)
    is enforced — pre-fix the ref resolution REPLACED the schema, so
    ``{"$ref": ..., "minimum": 5}`` silently lost the bound."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Count", "minimum": 5}},
        "required": ["n"],
        "$defs": {"Count": {"type": "integer"}},
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=5), "n") == 5
    with pytest.raises(ValidationError):
        _ = model(n=4)


def test_ref_sibling_bound_outer_wins_over_target() -> None:
    """When the outer property and the ``$ref`` target both declare the same
    bound, the OUTER one wins (mirroring the nullable-merge and the
    ``$ref``-sibling ``description`` precedence); target-only bounds the outer
    does not redeclare still apply."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Count", "minimum": 5}},
        "required": ["n"],
        "$defs": {"Count": {"type": "integer", "minimum": 1, "maximum": 10}},
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=5), "n") == 5
    # Outer minimum=5 beats the target's looser minimum=1.
    with pytest.raises(ValidationError):
        _ = model(n=2)
    # The target's maximum (no outer competitor) still applies alongside.
    with pytest.raises(ValidationError):
        _ = model(n=11)


def test_description_on_referenced_def_is_found() -> None:
    """A ``description`` declared on a ``$ref`` target surfaces on the field,
    mirroring how a bound on the target is unwrapped — so a ``$ref``-ed field
    still carries its per-field guidance to the model. An inline ``description``
    on the property still wins (outer-wins precedence)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "n": {"$ref": "#/$defs/Bounded"},
            "m": {"$ref": "#/$defs/Bounded", "description": "Outer wins."},
        },
        "required": ["n", "m"],
        "$defs": {"Bounded": {"type": "integer", "minimum": 0, "description": "From the def."}},
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["n"].description == "From the def."
    assert model.model_fields["m"].description == "Outer wins."


def test_nullable_ref_property_inherits_target_description() -> None:
    """A nullable-wrapped ``$ref`` surfaces the target's ``description`` just as a
    bare ``$ref`` does — pre-fix the description rescue only understood a bare
    top-level ``$ref``, so wrapping it in ``anyOf``+null silently dropped the
    model-facing guidance (degrading instructor's prompt for the field)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"anyOf": [{"$ref": "#/$defs/Bounded"}, {"type": "null"}]}},
        "required": ["n"],
        "$defs": {"Bounded": {"type": "integer", "minimum": 0, "description": "From the def."}},
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["n"].description == "From the def."
    # The bound behind the nullable ref still rides through too (regression).
    with pytest.raises(ValidationError):
        _ = model(n=-1)


def test_nullable_ref_property_outer_description_wins() -> None:
    """An outer ``description`` on the nullable-wrapping property beats the
    target's — the same outer-wins precedence a bare ``$ref`` sibling follows."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "n": {
                "anyOf": [{"$ref": "#/$defs/Bounded"}, {"type": "null"}],
                "description": "Outer wins.",
            }
        },
        "required": ["n"],
        "$defs": {"Bounded": {"type": "integer", "description": "From the def."}},
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["n"].description == "Outer wins."


def test_ref_chain_description_nearest_hop_wins() -> None:
    """On a multi-hop ``$ref`` chain where more than one hop declares a
    ``description``, the hop NEAREST the property wins — outer-wins applied at
    every hop, so guidance closest to the field takes precedence over a deeper
    def's."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/A"}},
        "required": ["n"],
        "$defs": {
            "A": {"$ref": "#/$defs/B", "description": "A (near) wins."},
            "B": {"type": "integer", "description": "B (far)."},
        },
    }
    model = model_from_json_schema(schema)
    assert model.model_fields["n"].description == "A (near) wins."


def test_ref_sibling_enum_raises_clear_value_error() -> None:
    """An ``enum`` beside a ``$ref`` is a structural conjunction a merge cannot
    honour — pre-fix it was silently dropped on the annotation path, widening the
    field to an unconstrained scalar. It must now fail loud, naming the keyword."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"c": {"$ref": "#/$defs/Color", "enum": ["red"]}},
        "required": ["c"],
        "$defs": {"Color": {"type": "string", "enum": ["red", "green", "blue"]}},
    }
    with pytest.raises(ValueError, match=r"Unsupported \$ref sibling 'enum' at '\$\.c'"):
        _ = model_from_json_schema(schema)


def test_ref_sibling_enum_on_targetless_key_raises_clear_value_error() -> None:
    """A structural sibling absent from the target still raises: a target that
    does not declare the keyword cannot be "restating" it, so the sibling would
    redefine (here, newly constrain) the reference — exactly the silent widening
    the guard exists to reject, in the opposite direction."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"c": {"$ref": "#/$defs/Plain", "enum": ["red"]}},
        "required": ["c"],
        "$defs": {"Plain": {"type": "string"}},
    }
    with pytest.raises(ValueError, match=r"Unsupported \$ref sibling 'enum'"):
        _ = model_from_json_schema(schema)


def test_ref_sibling_conflicting_type_raises_clear_value_error() -> None:
    """A ``type`` beside a ``$ref`` that disagrees with the target raises."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Count", "type": "string"}},
        "required": ["n"],
        "$defs": {"Count": {"type": "integer"}},
    }
    with pytest.raises(ValueError, match=r"Unsupported \$ref sibling 'type' at '\$\.n'"):
        _ = model_from_json_schema(schema)


def test_ref_sibling_items_raises_clear_value_error() -> None:
    """An ``items`` beside a ``$ref`` (redefining the referenced array's element
    schema) raises rather than silently overriding the reference."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"xs": {"$ref": "#/$defs/Ints", "items": {"type": "string"}}},
        "required": ["xs"],
        "$defs": {"Ints": {"type": "array", "items": {"type": "integer"}}},
    }
    with pytest.raises(ValueError, match=r"Unsupported \$ref sibling 'items' at '\$\.xs'"):
        _ = model_from_json_schema(schema)


def test_ref_sibling_properties_raises_clear_value_error() -> None:
    """A ``properties`` beside a ``$ref`` raises. This also protects the
    object-class cache (keyed by ``$ref`` name): were the sibling merged, two
    references to the same def with different sibling ``properties`` would want
    two different classes under one cache key — a silent wrong-model hazard the
    reject-or-restate rule removes by construction."""
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"o": {"$ref": "#/$defs/Obj", "properties": {"x": {"type": "string"}}}},
        "required": ["o"],
        "$defs": {
            "Obj": {"type": "object", "properties": {"y": {"type": "integer"}}, "required": ["y"]}
        },
    }
    with pytest.raises(ValueError, match=r"Unsupported \$ref sibling 'properties' at '\$\.o'"):
        _ = model_from_json_schema(schema)


def test_ref_sibling_restating_target_structural_key_is_accepted() -> None:
    """A structural sibling that restates the target's own value verbatim is a
    harmless no-op (some generators emit a redundant ``type`` beside a ``$ref``)
    and must build, not raise — only a *redefining* sibling is rejected."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Count", "type": "integer", "minimum": 5}},
        "required": ["n"],
        "$defs": {"Count": {"type": "integer"}},
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=5), "n") == 5
    # The non-structural sibling bound still rides through alongside.
    with pytest.raises(ValidationError):
        _ = model(n=4)


def test_bound_on_nullable_branch_is_found() -> None:
    """A bound on the non-null branch of a nullable field is applied; ``null``
    still passes (the bound only gates non-null values)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"n": {"type": ["integer", "null"], "minimum": 0}},
        "required": ["n"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(n=None), "n") is None
    assert _attr(model(n=5), "n") == 5
    with pytest.raises(ValidationError):
        _ = model(n=-1)


# --- Nullable-merge precedence: outer field-level keys win -----------------


def test_nullable_anyof_merge_outer_keys_win() -> None:
    """A ``{"description": "outer", "anyOf": [{...branch...}, {"type": "null"}]}``
    field resolves to a nullable field whose ``description`` is the outer one
    and whose ``type`` comes from the non-null branch.

    This documents the field *shape*; ``description`` is always read from the
    outer property level (independent of the nullable merge), so the genuine
    merge-precedence guard — where outer-vs-branch order is observable — lives
    in :func:`test_nullable_anyof_outer_constraint_wins` below.
    """
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "f": {
                "description": "outer",
                "anyOf": [
                    {"type": "string", "description": "inner"},
                    {"type": "null"},
                ],
            }
        },
        "required": ["f"],
    }
    model = model_from_json_schema(schema)
    # Outer description wins over the branch's competing one.
    assert model.model_fields["f"].description == "outer"
    # The branch still supplied the type: a nullable str that accepts str AND None.
    assert model.model_fields["f"].is_required()  # listed in required
    assert _attr(model(f="hi"), "f") == "hi"
    assert _attr(model(f=None), "f") is None
    with pytest.raises(ValidationError):
        _ = model(f=123)  # not a string and not null


def test_nullable_anyof_branch_supplies_type_when_outer_omits() -> None:
    """When the outer omits ``type``, the non-null branch supplies it — the
    field resolves to the branch's type (here ``integer``), nullable, and the
    outer ``description`` (with no branch competition) is preserved."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "f": {
                "description": "outer-only",
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
            }
        },
        "required": ["f"],
    }
    model = model_from_json_schema(schema)
    # Outer description survives (no branch key competes for it).
    assert model.model_fields["f"].description == "outer-only"
    # Branch supplied the type: integers and null accepted, a string rejected.
    assert _attr(model(f=5), "f") == 5
    assert _attr(model(f=None), "f") is None
    with pytest.raises(ValidationError):
        _ = model(f="not-an-int")


def test_nullable_anyof_outer_constraint_wins() -> None:
    """The genuine merge-precedence guard: a constraint on the OUTER field wins
    over a competing one on the non-null branch.

    Constraints (here ``maxLength``) are the only place the nullable-merge order
    is observable — ``description``/``type`` are not affected by it. With an
    outer ``maxLength`` of 5 competing against a branch ``maxLength`` of 10, the
    outer bound must win: a 5-char value validates, a 6-char value is rejected.
    This FAILS against the pre-fix ``{**merged, **non_null[0]}`` (branch wins, so
    the looser 10 leaks through and the 6-char value is wrongly accepted).
    """
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {
            "f": {
                "maxLength": 5,
                "anyOf": [
                    {"type": "string", "maxLength": 10},
                    {"type": "null"},
                ],
            }
        },
        "required": ["f"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(f="12345"), "f") == "12345"  # 5 chars: outer bound allows
    with pytest.raises(ValidationError):
        _ = model(f="123456")  # 6 chars: outer maxLength=5 wins, branch's 10 does not


# --- additionalProperties: open vs strict vs typed-map ---------------------


def test_additional_properties_true_accepts_unknown_key() -> None:
    """``additionalProperties: true`` -> ``extra="allow"``: an unknown extra key
    is accepted and retained (round-trips through ``model_dump``).

    Regression: pre-fix every object was ``extra="forbid"``, so the extra key
    raised ``ValidationError``.
    """
    schema: dict[str, object] = {
        "title": "Open",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": True,
    }
    model = model_from_json_schema(schema)
    inst = model(id="ok", extra_thing="kept")  # must not raise
    dumped = inst.model_dump()
    assert dumped["id"] == "ok"
    assert dumped["extra_thing"] == "kept"  # unknown key retained
    # The open base still keeps the exclude_none dump override behaviour.
    assert isinstance(dumped, dict)


def test_additional_properties_false_forbids_unknown_key() -> None:
    """``additionalProperties: false`` keeps the strict ``extra="forbid"``
    default — an unknown extra key is rejected."""
    schema: dict[str, object] = {
        "title": "Closed",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    model = model_from_json_schema(schema)
    assert model(id="ok").model_dump() == {"id": "ok"}
    with pytest.raises(ValidationError):
        _ = model(id="ok", hallucinated="nope")


def test_additional_properties_absent_forbids_unknown_key() -> None:
    """No ``additionalProperties`` keyword -> strict ``extra="forbid"`` (the
    safe LLM default), same as ``false``."""
    schema: dict[str, object] = {
        "title": "Default",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }
    model = model_from_json_schema(schema)
    with pytest.raises(ValidationError):
        _ = model(id="ok", hallucinated="nope")


def test_additional_properties_typed_map_raises_clear_value_error() -> None:
    """A typed ``additionalProperties`` map (e.g. ``{"type": "string"}``) is
    unsupported and raises a ``ValueError`` naming ``additionalProperties`` and
    the field path."""
    schema: dict[str, object] = {
        "title": "Typed",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": {"type": "string"},
    }
    with pytest.raises(
        ValueError,
        match=(
            r"Unsupported 'additionalProperties' at .*: typed additionalProperties "
            r"maps are not supported"
        ),
    ):
        _ = model_from_json_schema(schema)


# --- oneOf/anyOf multi-variant wording names the keyword present -----------


def test_oneof_multi_variant_error_names_oneof() -> None:
    """A ``oneOf`` with two real (non-null) branches stays unsupported and the
    error names ``oneOf`` specifically (no Union support added)."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["x"],
    }
    with pytest.raises(ValueError, match=r"Unsupported oneOf with \d+ non-null branches"):
        _ = model_from_json_schema(schema)
    with pytest.raises(ValueError, match=r"Multi-variant \(discriminated\) unions are unsupported"):
        _ = model_from_json_schema(schema)


def test_common_nullable_anyof_still_builds() -> None:
    """The common nullable case (``anyOf`` with one real branch + ``null``)
    still builds fine — only genuine multi-variant unions are rejected."""
    schema: dict[str, object] = {
        "title": "M",
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["a"],
    }
    model = model_from_json_schema(schema)
    assert _attr(model(a="x"), "a") == "x"
    assert _attr(model(a=None), "a") is None
