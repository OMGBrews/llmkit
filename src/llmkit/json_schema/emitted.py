"""What a generated model does once it exists: dump, and describe itself.

The *output* side of :func:`~llmkit.model_from_json_schema`, and a strict
import leaf — not one symbol here calls into the converter. The split follows
a real boundary rather than a line count: this half's failure mode is a
provider 400 or a bad wire payload at *call* time, where the converter's is a
``ValueError`` at *build* time; its trigger is a caller, instructor or pydantic
touching a model the converter has long since finished producing; and it has
its own test file (``tests/calls/test_json_schema_emission.py``), whose
docstring says it is "only about the emitted document".

Ten of this module's fourteen pyright suppressions live here, which is the
other half of the same fact: this is where the untyped pydantic boundary
actually is.

One contract crosses back the other way and does so *by attribute name*, not
by import: the converter stamps ``__llmkit_optional_fields__`` onto each model
it builds, and :meth:`JsonSchemaModel._drop_none` reads it. Nothing enforces
that the two spellings agree — keep the comment on both sides.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal, Self, TypedDict, Unpack, cast, override

from pydantic import BaseModel, ConfigDict, JsonValue, SerializationInfo, model_serializer
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode

# A JSON-schema dict: string keys to arbitrary JSON values. Modelled with
# pydantic's ``JsonValue`` so the schema data carries a precise type rather
# than ``Any`` everywhere it is read.
type JsonDict = dict[str, JsonValue]

# pydantic's ``IncEx`` (the ``include`` / ``exclude`` selector type) is a
# private alias; mirror its public shape here so the dump-kwargs TypedDict
# below stays a real type rather than ``Any``.
type _IncEx = set[int] | set[str] | Mapping[int, "_IncEx | bool"] | Mapping[str, "_IncEx | bool"]


class _DumpKwargs(TypedDict, total=False):
    """The exact keyword surface of ``BaseModel.model_dump`` minus
    ``exclude_none`` (which the overrides below name explicitly).

    Typing ``**kwargs`` with ``Unpack[_DumpKwargs]`` instead of ``Any`` lets the
    overrides forward to pydantic's dump methods with real types, so no argument
    arrives as ``Any``. Mirrors ``pydantic`` 2.13's signature — the *current*
    pydantic, not the ``>=2.5`` floor: newer keys (e.g. ``fallback``,
    ``polymorphic_serialization``) type-check here but would raise on an older
    pydantic. Safe because these overrides are internal and only ever called
    with ``exclude_none``, never the wider surface.
    """

    mode: Literal["json", "python"] | str
    include: _IncEx | None
    exclude: _IncEx | None
    context: Any | None  # pyright: ignore[reportExplicitAny]  # raw-pydantic — pydantic types ``context`` as ``Any | None``
    by_alias: bool | None
    exclude_unset: bool
    exclude_defaults: bool
    exclude_computed_fields: bool
    round_trip: bool
    warnings: bool | Literal["none", "warn", "error"]
    fallback: Callable[[Any], Any] | None  # pyright: ignore[reportExplicitAny]  # raw-pydantic — pydantic types ``fallback`` as ``Callable[[Any], Any]``
    serialize_as_any: bool
    polymorphic_serialization: bool | None


class _DumpJsonKwargs(TypedDict, total=False):
    """The keyword surface of ``BaseModel.model_dump_json`` minus
    ``exclude_none`` — same as ``_DumpKwargs`` but with ``indent`` /
    ``ensure_ascii`` in place of ``mode``. Mirrors ``pydantic`` 2.13.
    """

    indent: int | None
    ensure_ascii: bool
    include: _IncEx | None
    exclude: _IncEx | None
    context: Any | None  # pyright: ignore[reportExplicitAny]  # raw-pydantic — pydantic types ``context`` as ``Any | None``
    by_alias: bool | None
    exclude_unset: bool
    exclude_defaults: bool
    exclude_computed_fields: bool
    round_trip: bool
    warnings: bool | Literal["none", "warn", "error"]
    fallback: Callable[[Any], Any] | None  # pyright: ignore[reportExplicitAny]  # raw-pydantic — pydantic types ``fallback`` as ``Callable[[Any], Any]``
    serialize_as_any: bool
    polymorphic_serialization: bool | None


class _ModelJsonSchemaExtra(TypedDict, total=False):
    """``model_json_schema`` keywords newer than the pydantic floor (2.8).

    The ``model_json_schema`` override below names the four parameters every
    supported pydantic has and takes the rest through ``Unpack`` of this
    TypedDict, so a newer keyword (``union_format``, pydantic 2.13) is
    forwarded only when a caller actually passes it — forwarding it eagerly
    with its default would crash the lowest-versions resolution.
    """

    union_format: Literal["any_of", "primitive_type_array"]


# Serialization-context key carrying the None-dropping directive from the dump
# overrides down into the ``model_serializer`` — the one place that can drop a
# field per-class across arbitrary nesting (pydantic drives nested models
# through the core serializer, not the Python ``model_dump`` method, so a
# top-level post-filter could never reach them). Values:
#   "optional" — drop ``None`` only for fields *not* in the schema's ``required``
#                array (the default; keeps an explicitly-null required field).
#   "all"      — drop every ``None`` (native ``exclude_none=True`` semantics).
#   "keep"     — drop nothing (the ``exclude_none=False`` escape hatch).
_DROP_DIRECTIVE_KEY = "__llmkit_none_directive__"


def _none_directive(exclude_none: bool | None) -> str:
    """Map the ``exclude_none`` tri-state to a serializer directive."""
    if exclude_none is None:
        return "optional"
    return "all" if exclude_none else "keep"


def _with_directive(context: object, exclude_none: bool | None) -> object:
    """Thread the None-dropping directive into the serialization ``context``.

    Merges into a caller-supplied ``dict`` context (preserving their keys) or
    starts a fresh one. A non-dict context is left untouched — our generated
    models carry no field serializer that consumes a custom context, so the
    serializer simply falls back to the default ``"optional"`` directive.
    """
    directive = _none_directive(exclude_none)
    if context is None:
        return {_DROP_DIRECTIVE_KEY: directive}
    if isinstance(context, dict):
        merged = dict(cast("dict[str, object]", context))
        merged[_DROP_DIRECTIVE_KEY] = directive
        return merged
    return context


def _read_directive(context: object) -> str:
    """Recover the directive a dump override stowed in the context."""
    if isinstance(context, dict):
        value = cast("dict[str, object]", context).get(_DROP_DIRECTIVE_KEY)
        if isinstance(value, str):
            return value
    return "optional"


# The one ``$ref`` spelling pydantic's default ``ref_template`` produces, and
# the only one the emission walkers below rewrite: a custom template yields
# refs they cannot resolve, which are left untouched rather than guessed at.
_REF_PREFIX = "#/$defs/"


def _inline_ref_siblings(root: JsonDict) -> None:
    """Rewrite every ``$ref``-with-siblings node into an inline schema, in place.

    Pydantic factors a named model annotation into ``$defs`` and references it;
    a per-field ``description`` then lands **beside** the ``$ref`` (pydantic
    ≥2.9) or wraps it in a single-branch ``allOf`` (pydantic <2.9). OpenAI's
    strict structured-outputs validator rejects both shapes — ``$ref cannot
    have keywords {...}``, and ``allOf`` is unsupported outright — so each
    offending use site gets its own copy of the referenced def with the
    sibling keys merged on top (outer wins, the same precedence
    ``_resolve_field`` applies on the input side). Bare ``$ref`` nodes, which
    the validator allows, are left shared.

    Termination: generated schemas cannot contain a ``$ref`` cycle (recursive
    input schemas are rejected at build time), so both the inline loop and the
    walk over freshly-inlined content bottom out.
    """
    defs_raw = root.get("$defs")
    defs: JsonDict = cast("JsonDict", defs_raw) if isinstance(defs_raw, dict) else {}

    def rewrite(node: JsonDict) -> None:
        # Normalise the pre-2.9 spelling — ``{"allOf": [{"$ref": ...}],
        # "description": ...}`` — to the sibling form, so one inline path
        # below covers both pydantic eras.
        all_of = node.get("allOf")
        if (
            "$ref" not in node
            and isinstance(all_of, list)
            and len(cast("list[JsonValue]", all_of)) == 1
            and isinstance(cast("list[JsonValue]", all_of)[0], dict)
            and set(cast("JsonDict", cast("list[JsonValue]", all_of)[0])) == {"$ref"}
        ):
            node["$ref"] = cast("JsonDict", cast("list[JsonValue]", all_of)[0])["$ref"]
            del node["allOf"]
        # Inline while the node is a resolvable ``$ref`` with siblings — the
        # loop covers a def that is itself a bare ``$ref`` alias.
        while True:
            ref = node.get("$ref")
            if not isinstance(ref, str) or len(node) == 1 or not ref.startswith(_REF_PREFIX):
                return
            target = defs.get(ref.removeprefix(_REF_PREFIX))
            if not isinstance(target, dict):
                return
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            node.clear()
            node.update(copy.deepcopy(cast("JsonDict", target)))
            node.update(siblings)

    def walk(value: JsonValue) -> None:
        if isinstance(value, dict):
            node = cast("JsonDict", value)
            rewrite(node)
            for child in node.values():
                walk(child)
        elif isinstance(value, list):
            for item in cast("list[JsonValue]", value):
                walk(item)

    walk(root)


def _prune_unreferenced_defs(root: JsonDict) -> None:
    """Drop ``$defs`` entries nothing references any more, in place.

    Inlining ``$ref``-with-siblings sites can leave a def with no remaining
    referrer. Reachability is computed transitively from the non-``$defs``
    part of the document, so a def referenced only by another dead def goes
    too; when nothing survives the ``$defs`` key is removed entirely.
    """
    defs_raw = root.get("$defs")
    if not isinstance(defs_raw, dict):
        return
    defs = cast("JsonDict", defs_raw)

    def refs_in(value: JsonValue, acc: set[str]) -> None:
        if isinstance(value, dict):
            node = cast("JsonDict", value)
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
                acc.add(ref.removeprefix(_REF_PREFIX))
            for child in node.values():
                refs_in(child, acc)
        elif isinstance(value, list):
            for item in cast("list[JsonValue]", value):
                refs_in(item, acc)

    needed: set[str] = set()
    for key, value in root.items():
        if key != "$defs":
            refs_in(value, needed)
    frontier = list(needed)
    while frontier:
        target = defs.get(frontier.pop())
        if target is None:
            continue
        found: set[str] = set()
        refs_in(target, found)
        for name in found - needed:
            needed.add(name)
            frontier.append(name)
    for name in [n for n in defs if n not in needed]:
        del defs[name]
    if not defs:
        del root["$defs"]


class JsonSchemaModel(BaseModel):
    """Base for every generated model: drops *optional* ``None`` on dump.

    A non-required JSON-schema field becomes an optional Pydantic field
    defaulting to ``None``. Dumping drops such an unset optional rather than
    serialising it as ``"field": null`` (footgun #1 — that null then fails
    downstream re-validation against the same JSON schema).

    Crucially the drop is *scoped to optional fields*: a field that is in the
    schema's ``required`` array but typed nullable (``["string", "null"]`` / an
    ``anyOf`` null branch) and legitimately set to ``None`` is **kept**, because
    dropping it would itself break re-validation (``'field' is a required
    property``) — the very failure this class exists to prevent. Pass
    ``exclude_none=False`` to keep every null, or ``exclude_none=True`` for the
    native "drop all nulls" behaviour.
    """

    # ``use_enum_values`` is inert since enum fields became inline ``Literal``
    # annotations (a validated instance always holds the raw scalar; there is
    # no Enum member to unwrap). Kept for one release as a belt while that
    # change beds in, then removable.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", use_enum_values=True)

    # Field names OPTIONAL in the source schema (absent from its ``required``
    # array) — the only fields whose ``None`` is safe to drop. The builder
    # stamps the real set onto each generated subclass; the empty default makes
    # the base classes themselves drop nothing. A dunder name so pydantic skips
    # it as a field and it is exempt from name mangling.
    __llmkit_optional_fields__: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _drop_none(
        self,
        handler: Callable[[Self], dict[str, Any]],  # pyright: ignore[reportExplicitAny]  # raw-pydantic — wrap-serializer handler returns the dynamic dump dict
        info: SerializationInfo,
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — mirrors model_dump's dict[str, Any] return
        data = handler(self)
        directive = _read_directive(info.context)
        if directive == "keep":
            return data
        if directive == "all":
            return {k: v for k, v in data.items() if v is not None}  # pyright: ignore[reportAny]  # raw-pydantic — dump values are dict[str, Any]
        # "optional": keep an explicitly-null *required* field (dropping it
        # would break re-validation); drop only unset optionals.
        droppable = type(self).__llmkit_optional_fields__
        return {k: v for k, v in data.items() if v is not None or k not in droppable}  # pyright: ignore[reportAny]  # raw-pydantic — dump values are dict[str, Any]

    @override
    def model_dump(
        self,
        *,
        exclude_none: bool | None = None,
        **kwargs: Unpack[_DumpKwargs],
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — mirrors model_dump's dict[str, Any] return
        # Do not forward ``exclude_none`` to pydantic: the serializer above is
        # the single source of truth for None-dropping (so nested models obey
        # the same rule). Translate it into the context directive instead.
        kwargs["context"] = _with_directive(kwargs.get("context"), exclude_none)
        return super().model_dump(**kwargs)

    @override
    def model_dump_json(
        self,
        *,
        exclude_none: bool | None = None,
        **kwargs: Unpack[_DumpJsonKwargs],
    ) -> str:
        kwargs["context"] = _with_directive(kwargs.get("context"), exclude_none)
        return super().model_dump_json(**kwargs)

    @classmethod
    @override
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        **kwargs: Unpack[_ModelJsonSchemaExtra],
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — mirrors model_json_schema's dict[str, Any] return
        """Pydantic's schema, post-processed to survive OpenAI's strict validator.

        instructor serialises the model with exactly this call, zero-argument,
        so this override is the one seam where the emitted document can be
        made strict-safe: a ``$ref`` that pydantic gave sibling keywords (a
        described object-typed property) is inlined at the use site, and defs
        nothing references any more are pruned. Rewrites only apply to the
        default ``ref_template``'s ``#/$defs/`` refs — a custom template's
        refs pass through untouched.
        """
        schema = cast(
            "JsonDict",
            super().model_json_schema(
                by_alias=by_alias,
                ref_template=ref_template,
                schema_generator=schema_generator,
                mode=mode,
                **kwargs,
            ),
        )
        _inline_ref_siblings(schema)
        _prune_unreferenced_defs(schema)
        return schema


class JsonSchemaModelAllow(JsonSchemaModel):
    """Open-ended variant for objects whose schema sets ``additionalProperties: true``.

    Subclasses the strict base so the child ``model_config`` merges with the
    parent's: it keeps the parent's config keys and the ``exclude_none`` dump
    and ``model_json_schema`` overrides while flipping ``extra`` from
    ``"forbid"`` to ``"allow"``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
