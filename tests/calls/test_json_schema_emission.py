"""Emitted-schema shape tests for generated models — OpenAI strict compatibility.

instructor serialises the generated model class with a **zero-argument**
``model_json_schema()`` call, and OpenAI's strict structured-outputs
validator (``response_format`` with ``"strict": true``) rejects any node
where ``$ref`` carries sibling keywords (``$ref cannot have keywords
{'description'}``) as well as ``allOf`` wholesale. Measured live 2026-07-24
against gpt-4o-mini: the pre-fix shapes 400, the post-fix shapes are
accepted. These tests pin the exact call instructor makes, so every
assertion here runs against the document a provider actually receives.

Validation behaviour of the generated models (what they accept and dump)
lives in ``test_json_schema_build.py`` / ``test_json_schema_constraints.py``;
this file is only about the *emitted document*.
"""

from __future__ import annotations

from typing import cast

import pytest

from llmkit import model_from_json_schema

# The downstream schema that surfaced the bug (found-in-words' eval-judge
# rubric): an enum property WITH a description — the single most common
# LLM-judge shape. On 0.7.0 the enum became a ``$defs`` entry and the
# description landed beside the ``$ref``: the exact node OpenAI 400s.
_FIW_JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recognizability_score": {
            "type": "integer",
            "enum": [-1, 1, 2, 3, 4, 5],
            "description": "How recognizable the phrase is.",
        }
    },
    "required": ["recognizability_score"],
}


def _assert_strict_ref_hygiene(doc: object, path: str = "$") -> None:
    """Assert no node anywhere carries ``$ref`` with siblings, nor any ``allOf``.

    The two shapes OpenAI's strict validator rejects. Recursive over the whole
    document, so a violation inside ``$defs``, ``items``, or an ``anyOf``
    branch fails just as loudly as one on a top-level property.
    """
    if isinstance(doc, dict):
        node = cast("dict[str, object]", doc)
        if "$ref" in node:
            assert set(node) == {"$ref"}, f"$ref with sibling keys at {path}: {sorted(node)}"
        assert "allOf" not in node, f"allOf at {path}"
        for key, value in node.items():
            _assert_strict_ref_hygiene(value, f"{path}.{key}")
    elif isinstance(doc, list):
        for index, item in enumerate(cast("list[object]", doc)):
            _assert_strict_ref_hygiene(item, f"{path}[{index}]")


# Every schema family the converter supports that historically produced (or
# could produce) a ``$ref`` — each must emit a strict-safe document. Battery
# entries deliberately put a ``description`` everywhere one can sit, since the
# description sibling is what turns a legal bare ``$ref`` into the rejected
# shape.
_STRICT_BATTERY: dict[str, dict[str, object]] = {
    "string-enum-described": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["a", "b"], "description": "d"}},
        "required": ["kind"],
    },
    "int-enum-described": _FIW_JUDGE_SCHEMA,
    "nullable-enum-described": {
        "type": "object",
        "properties": {
            "choice": {"type": ["string", "null"], "enum": ["a", "b", None], "description": "d"}
        },
        "required": ["choice"],
    },
    "array-of-enum-described": {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string", "enum": ["x", "y"]},
                "description": "d",
            }
        },
        "required": ["tags"],
    },
    "shared-enum-def-two-descriptions": {
        "type": "object",
        "$defs": {"Score": {"type": "integer", "enum": [1, 2, 3]}},
        "properties": {
            "clarity": {"$ref": "#/$defs/Score", "description": "How clear."},
            "novelty": {"$ref": "#/$defs/Score", "description": "How novel."},
        },
        "required": ["clarity", "novelty"],
    },
    "described-inline-object": {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "description": "A nested object with guidance.",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        },
        "required": ["inner"],
    },
    "shared-object-def-bare-and-described": {
        "type": "object",
        "$defs": {
            "Party": {
                "title": "Party",
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
        "properties": {
            "a": {"$ref": "#/$defs/Party"},
            "b": {"$ref": "#/$defs/Party", "description": "The counterparty."},
        },
        "required": ["a", "b"],
    },
    "array-of-described-objects": {
        "type": "object",
        "properties": {
            "items_field": {
                "type": "array",
                "description": "d",
                "items": {
                    "type": "object",
                    "description": "One entry.",
                    "properties": {"v": {"type": "integer"}},
                    "required": ["v"],
                },
            }
        },
        "required": ["items_field"],
    },
    "def-containing-described-members": {
        # A def whose OWN properties carry described enum and object members —
        # the violation would sit inside ``$defs``, not on a root property.
        "type": "object",
        "$defs": {
            "Rubric": {
                "title": "Rubric",
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "enum": [1, 2], "description": "d"},
                    "detail": {
                        "type": "object",
                        "description": "Nested guidance.",
                        "properties": {"note": {"type": "string"}},
                        "required": ["note"],
                    },
                },
                "required": ["score", "detail"],
            }
        },
        "properties": {
            "first": {"$ref": "#/$defs/Rubric"},
            "second": {"$ref": "#/$defs/Rubric", "description": "Second pass."},
        },
        "required": ["first", "second"],
    },
}


@pytest.mark.parametrize("name", list(_STRICT_BATTERY))
def test_emitted_schema_has_no_ref_siblings_and_no_allof(name: str) -> None:
    model = model_from_json_schema(_STRICT_BATTERY[name], name="Battery")
    _assert_strict_ref_hygiene(model.model_json_schema())


def test_fiw_judge_schema_emits_inline_enum_with_description() -> None:
    """The exact downstream regression: the enum stays inline beside its
    description (the shape OpenAI's own documentation shows), and an
    enum-only schema emits no ``$defs`` at all."""
    model = model_from_json_schema(_FIW_JUDGE_SCHEMA, name="phrase_judgment")
    doc = model.model_json_schema()
    prop = cast("dict[str, dict[str, object]]", doc["properties"])["recognizability_score"]
    assert prop["type"] == "integer"
    assert prop["enum"] == [-1, 1, 2, 3, 4, 5]
    assert prop["description"] == "How recognizable the phrase is."
    assert "$ref" not in prop
    assert "$defs" not in doc


def test_shared_enum_def_lands_each_description_on_its_own_property() -> None:
    """Two properties referencing the same enum def with different descriptions
    each get their own inline enum + their own description — per-field guidance
    survives, which no shared ``$defs`` entry could carry."""
    model = model_from_json_schema(
        _STRICT_BATTERY["shared-enum-def-two-descriptions"], name="Rubric"
    )
    doc = model.model_json_schema()
    props = cast("dict[str, dict[str, object]]", doc["properties"])
    for field, description in (("clarity", "How clear."), ("novelty", "How novel.")):
        assert props[field]["enum"] == [1, 2, 3]
        assert props[field]["description"] == description
    assert "$defs" not in doc


def test_described_object_ref_inlines_def_and_bare_ref_stays_shared() -> None:
    """A described reference to an object def is inlined at the use site (its
    own copy of the def's shape, description merged on top — outer wins), while
    a bare reference to the same def keeps the strict-legal shared ``$ref``,
    and the def survives for it."""
    model = model_from_json_schema(
        _STRICT_BATTERY["shared-object-def-bare-and-described"], name="Deal"
    )
    doc = model.model_json_schema()
    props = cast("dict[str, dict[str, object]]", doc["properties"])
    assert props["a"] == {"$ref": "#/$defs/Party"}
    inlined = props["b"]
    assert inlined["type"] == "object"
    assert inlined["description"] == "The counterparty."
    assert "name" in cast("dict[str, object]", inlined["properties"])
    assert "$ref" not in inlined
    assert "Party" in cast("dict[str, object]", doc["$defs"])


def test_fully_inlined_document_drops_defs_entirely() -> None:
    """When inlining leaves a def with no remaining referrer it is pruned, and
    an empty ``$defs`` never survives to the wire."""
    model = model_from_json_schema(_STRICT_BATTERY["described-inline-object"], name="Wrap")
    doc = model.model_json_schema()
    assert "$defs" not in doc
    inner = cast("dict[str, dict[str, object]]", doc["properties"])["inner"]
    assert inner["type"] == "object"
    assert inner["description"] == "A nested object with guidance."


def test_single_value_enum_emits_const_shape() -> None:
    """A one-member enum serialises as pydantic's ``const`` form — documented
    as supported by OpenAI strict mode — and still validates as the enum."""
    model = model_from_json_schema(
        {
            "type": "object",
            "properties": {"only": {"type": "string", "enum": ["fixed"], "description": "d"}},
            "required": ["only"],
        },
        name="Single",
    )
    doc = model.model_json_schema()
    prop = cast("dict[str, dict[str, object]]", doc["properties"])["only"]
    assert prop["const"] == "fixed"
    assert prop["type"] == "string"
    _assert_strict_ref_hygiene(doc)
    assert model(only="fixed").model_dump() == {"only": "fixed"}


def test_nullable_enum_emits_inline_anyof_branch() -> None:
    """A nullable enum emits ``anyOf: [{inline enum}, {"type": "null"}]`` with
    the description a sibling of ``anyOf`` — legal under the strict validator,
    with no ``$ref`` in either branch."""
    model = model_from_json_schema(_STRICT_BATTERY["nullable-enum-described"], name="Nullable")
    doc = model.model_json_schema()
    prop = cast("dict[str, dict[str, object]]", doc["properties"])["choice"]
    branches = cast("list[dict[str, object]]", prop["anyOf"])
    assert {"type": "null"} in branches
    enum_branch = next(b for b in branches if "enum" in b)
    assert enum_branch["enum"] == ["a", "b"]
    assert prop["description"] == "d"
    assert "$defs" not in doc


def test_emission_is_deterministic_across_calls() -> None:
    """Repeated zero-argument calls emit equal documents — the in-place
    post-processing operates on a fresh dict each time, never accumulating."""
    model = model_from_json_schema(
        _STRICT_BATTERY["def-containing-described-members"], name="Repeat"
    )
    assert model.model_json_schema() == model.model_json_schema()


def test_generated_models_carry_a_pinned_module_so_colliding_defs_stay_stable() -> None:
    """The emitted ``$defs`` keys must not depend on where the converter lives.

    ``create_model`` reads ``__module__`` off its calling frame, and pydantic
    disambiguates two ``$defs`` that share a ``title`` using the module path. So
    an unpinned ``__module__`` makes the document sent to the provider depend on
    which file inside llmkit happens to call ``create_model`` — a wire-format
    change from a pure refactor, invisible to every other test here because it
    only shows up when two definitions collide on ``title``.
    """
    model = model_from_json_schema({"type": "object", "properties": {"a": {"type": "string"}}})
    assert model.__module__ == "llmkit.json_schema"

    colliding = {
        "type": "object",
        "properties": {
            "left": {"$ref": "#/$defs/L"},
            "right": {"$ref": "#/$defs/R"},
        },
        "$defs": {
            "L": {"title": "Shared", "type": "object", "properties": {"x": {"type": "string"}}},
            "R": {"title": "Shared", "type": "object", "properties": {"y": {"type": "integer"}}},
        },
    }
    emitted = cast("dict[str, object]", model_from_json_schema(colliding).model_json_schema())
    defs = cast("dict[str, object]", emitted.get("$defs", {}))
    # Both definitions survive under distinct keys, and neither key carries a
    # module path that would move with the converter.
    assert len(defs) == 2, defs
    assert not any("convert" in key for key in defs), sorted(defs)
