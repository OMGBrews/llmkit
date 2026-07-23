"""Runtime JSON-schema-dict → Pydantic-model conversion.

Consumers that declare their structured-output contracts as **JSON-schema
dicts** — typically because the same contract is shared across a Node
backend, a frontend, and Python — would otherwise have to hand-write a
converter to a Pydantic model before they can call
:func:`~llmkit.structured_llm_call` (which is, and stays,
Pydantic-model-only). :func:`model_from_json_schema` is that converter,
centralised and tested once so each consumer doesn't re-discover the same
footguns.

The intended pattern is **build once, reuse**::

    from llmkit import model_from_json_schema, structured_llm_call

    Invoice = model_from_json_schema(invoice_schema)  # once, at import
    result = await structured_llm_call(prompt, Invoice, feature="billing")

Supported JSON-schema subset (day one)
--------------------------------------
The subset is deliberately the one real dict-based consumers use; anything
outside it raises a clear :class:`ValueError` naming the unsupported
construct, rather than silently producing a wrong model:

* ``object`` with ``properties``; ``required`` vs optional via the
  ``required`` array
* scalar types: ``string``, ``integer``, ``number``, ``boolean`` (and
  ``null`` / nullable via ``["string", "null"]`` or ``anyOf`` with a null
  branch)
* ``array`` (``items``), including arrays of objects
* ``enum`` (on a scalar field)
* nested objects, inline or via local ``$defs`` / ``$ref`` references
  (``#/$defs/Name`` or the legacy ``#/definitions/Name``). A ``$ref`` may carry
  siblings: metadata and value bounds (``description``, ``default``, the
  numeric/length bounds) merge over the target with the outer value winning,
  so ``{"$ref": "#/$defs/Count", "minimum": 5}`` keeps the bound and a
  nullable-wrapped ``$ref`` inherits the target's ``description``. A *structural*
  sibling — a type/shape keyword (``type`` / ``enum`` / ``items`` /
  ``properties`` / ``required`` / ``additionalProperties`` / ``title``) or any
  subschema applicator (``anyOf`` / ``oneOf`` / ``allOf`` / ``not`` / ``if`` /
  ``then`` / ``else`` / …) — is a JSON-Schema conjunction a merge cannot express,
  so it is rejected unless it restates the target's own value: a ``$ref``-sibling
  ``enum`` or ``allOf`` is a clear error, never a silently-widened field
* subschema *applicators* — ``allOf`` / ``not`` / ``if`` / ``then`` / ``else`` /
  ``dependentSchemas`` / ``dependentRequired`` / ``propertyNames`` /
  ``patternProperties`` / ``prefixItems`` / ``contains`` /
  ``unevaluatedProperties`` / ``unevaluatedItems`` — are **rejected at every
  site**, not only beside a ``$ref``. Each constrains by composition and a
  generated field is one annotation plus a fixed set of ``Field`` bounds, so an
  applicator has nowhere to land; dropping one is wrong in both directions (a
  dropped ``allOf`` bound accepts what the schema forbids; a dropped
  ``prefixItems`` re-reads the sibling ``items`` as "every element" and rejects
  what the schema permits). ``anyOf`` / ``oneOf`` are exempt — they are the
  nullable spelling, consumed below
* ``object`` with ``properties``; a propertyless object (``properties`` absent
  *or* an explicit empty ``{}``) is rejected unless it opts into open-ended keys
  with ``additionalProperties: true`` — otherwise it would build a zero-field
  model that rejects every real response
* ``additionalProperties``: ``true`` (an open object — extra keys are accepted
  and kept) or ``false`` / absent (strict ``extra="forbid"``, the default); a
  *typed* ``additionalProperties`` map is rejected

Per-field constraints
----------------------
A small, fixed set of per-field constraints is carried through to the
generated Pydantic ``Field`` so the model validates *value bounds*, not just
shape. The supported set is **exactly**:

* numeric: ``minimum`` → ``ge``, ``maximum`` → ``le``,
  ``exclusiveMinimum`` → ``gt``, ``exclusiveMaximum`` → ``lt`` — the **numeric**
  (Draft 2020-12) form only. The Draft-4 *boolean* form
  (``"exclusiveMinimum": true`` qualifying a sibling ``minimum``) is not
  recognised and is dropped, so such a bound is treated as inclusive.
* string: ``minLength`` → ``min_length``, ``maxLength`` → ``max_length``
* array: ``minItems`` → ``min_length``, ``maxItems`` → ``max_length``
* ``description`` → ``Field(description=...)`` (instructor surfaces this as
  per-field guidance to the model)

Any other *leaf* constraint keyword (``pattern``, ``format``, ``multipleOf``,
``uniqueItems``, ``const``, …) is **silently dropped** — deliberately, to
avoid partial enforcement that looks complete. Nothing outside the list above
is enforced; if a schema relies on one of those, validate it elsewhere.

The silent drop is scoped to those per-value keywords. A *structural* construct
outside the supported subset — a subschema applicator, a multi-variant union, a
typed ``additionalProperties`` map — raises instead, because losing one changes
the shape the model validates rather than leaving a single value unchecked.

A schema-level ``default`` on a non-required field is likewise **not** carried
into the model: the field becomes optional with a ``None`` default and, via the
``exclude_none`` dump contract below, is simply omitted when unset. Supply
defaults after parsing if you need them.

Serialization contract
-----------------------
The generated model maps a **non-required** JSON-schema field to an
*optional* Pydantic field whose default is ``None``. To keep an omitted
optional from round-tripping back out as ``"field": null`` — which fails
re-validation against a JSON schema that lists the field as a non-nullable
optional — the generated model's :meth:`~pydantic.BaseModel.model_dump`
and :meth:`~pydantic.BaseModel.model_dump_json` drop such ``None`` values by
default. An optional the model never set is therefore *absent* from the dump,
not present-and-null.

The drop is **scoped to optional fields**: a field that is in the schema's
``required`` array but typed nullable (``["string", "null"]`` or an ``anyOf``
null branch) and legitimately set to ``None`` is **kept**, because dropping a
required field would itself break re-validation. Callers can pass
``exclude_none=False`` to keep every null, or ``exclude_none=True`` for the
native "drop all nulls" behaviour.
"""

from __future__ import annotations

import keyword
import re
import warnings
from collections.abc import Callable, Mapping
from enum import Enum
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Self,
    TypedDict,
    Unpack,
    cast,
    override,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializationInfo,
    create_model,
    model_serializer,
)

__all__ = ["model_from_json_schema"]

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


# A JSON-schema dict: string keys to arbitrary JSON values. Modelled with
# pydantic's ``JsonValue`` so the schema data carries a precise type rather
# than ``Any`` everywhere it is read.
type JsonDict = dict[str, JsonValue]

_DEFAULT_MODEL_NAME = "JsonSchemaModel"

# JSON-schema scalar type name -> Python type.
_SCALAR_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

# The ONLY per-field constraint keywords carried into Pydantic ``Field``.
# Each JSON-schema keyword maps to the ``Field`` keyword that enforces it.
# Anything outside this set is silently dropped (no partial enforcement):
#   numeric: minimum -> ge, maximum -> le, exclusiveMinimum -> gt,
#            exclusiveMaximum -> lt
#   string : minLength -> min_length, maxLength -> max_length
#   array  : minItems -> min_length, maxItems -> max_length (on list fields)
# ``pattern`` / ``format`` / ``multipleOf`` / ``uniqueItems`` / ``const`` /
# etc. are intentionally NOT supported — see the module docstring.


class _FieldConstraints(NamedTuple):
    """Resolved Pydantic ``Field`` bounds for one property (``None`` = unset).

    A precise carrier so the bounds reach ``Field`` as typed keyword arguments
    rather than an untyped splat. ``min_length`` / ``max_length`` serve both
    string length (``minLength`` / ``maxLength``) and array item count
    (``minItems`` / ``maxItems``) — the keyword Pydantic uses is the same.
    """

    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None


# Subschema applicators the converter never consumes, at any site. Each
# constrains the instance by *composition* — a conjunction (``allOf``), a
# negation (``not``), a condition (``if`` / ``then`` / ``else`` /
# ``dependentSchemas`` / ``dependentRequired``), or a positional / name-keyed
# rule (``prefixItems`` / ``contains`` / ``patternProperties`` /
# ``propertyNames`` / ``unevaluated*``). A generated field carries exactly one
# annotation plus a fixed set of ``Field`` bounds, so an applicator has nowhere
# to land: it would simply vanish, and the model would then validate something
# the schema never described. Hence it fails loud at EVERY site (see
# :meth:`_Converter._reject_unsupported_applicators`), not only beside a ``$ref``.
_UNSUPPORTED_APPLICATORS: frozenset[str] = frozenset(
    {
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
        "dependentRequired",
        "propertyNames",
        "patternProperties",
        "prefixItems",
        "contains",
        "unevaluatedProperties",
        "unevaluatedItems",
    }
)

# The type/shape keywords the converter dispatches on. Away from a ``$ref``
# these are read and honoured; beside one they are structural (below).
# ``title`` belongs here because the object-class cache is keyed by the ``$ref``
# name and names the class from ``title``, so a per-site sibling ``title`` would
# rename the *shared* referenced class for every other reference site. It
# belongs on the ``$def``, not a use site.
_SHAPE_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "enum",
        "items",
        "properties",
        "required",
        "additionalProperties",
        "title",
    }
)

# JSON-schema keywords that define a schema's *structure* (its type and shape),
# as opposed to metadata or value bounds. When one of these sits beside a
# ``$ref``, it cannot be folded into the referenced target without changing what
# the target validates: Draft 2020-12 applies such a sibling as a *conjunction*
# (an intersection with the target), which a last-writer-wins merge cannot
# express. So a structural sibling is honoured only when it restates the
# target's own value verbatim (a redundant no-op some generators emit) and
# otherwise fails loud — never silently widening or replacing the reference.
# Everything else beside a ``$ref`` (``description``, ``default``, the
# numeric/length bounds, benign annotations, unknown keywords) is non-structural
# and merges over the target with the outer, property-level value winning.
#
# Derived from the two sets above so they cannot drift, plus the union spellings:
#   - :data:`_SHAPE_KEYWORDS` — dispatched on elsewhere, a conjunction here;
#   - :data:`_UNSUPPORTED_APPLICATORS` — rejected here *and* everywhere else. A
#     bare applicator fails loud only when no sibling ``type`` / ``enum`` sits
#     with it (``{"allOf": [...]}`` alone has nothing to dispatch on); with one,
#     dispatch succeeds and the applicator would be silently dropped, which is
#     why the applicator guard is not scoped to ``$ref`` sites at all;
#   - ``anyOf`` / ``oneOf``, which are the converter's *nullable* spelling and
#     are consumed by :meth:`_Converter._unwrap_nullable` — structural beside a
#     ``$ref``, but deliberately absent from the applicator set above, since
#     rejecting them everywhere would reject nullability itself.
_STRUCTURAL_REF_SIBLINGS: frozenset[str] = (
    _SHAPE_KEYWORDS | _UNSUPPORTED_APPLICATORS | frozenset({"anyOf", "oneOf"})
)


class _ResolvedField(NamedTuple):
    """A property schema with its ``$ref`` chain and nullable wrappers stripped.

    ``schema`` is the effective node: every ``$ref`` on the chain resolved and
    its siblings folded in (outer-wins for metadata and bounds; structural
    siblings validated, never merged — see :data:`_STRUCTURAL_REF_SIBLINGS`),
    and every nullable wrapper unwrapped. ``nullable`` is the nullability
    accumulated across the chain, and ``ref_name`` is the last ``$ref`` name
    resolved — the object-class cache identity — or ``None`` for a schema that
    never went through a ``$ref``.

    Resolving once into this carrier is what lets the annotation, the
    description, and the value bounds all read a single, consistently-resolved
    node instead of each re-walking the chain with its own sibling rule (the
    divergence that silently dropped a ``$ref``-sibling ``enum`` while merging a
    ``$ref``-sibling bound).
    """

    schema: JsonDict
    nullable: bool
    ref_name: str | None


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


class _JsonSchemaModel(BaseModel):
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


class _JsonSchemaModelAllow(_JsonSchemaModel):
    """Open-ended variant for objects whose schema sets ``additionalProperties: true``.

    Subclasses the strict base so the child ``model_config`` merges with the
    parent's: it keeps ``use_enum_values`` and the ``exclude_none`` dump
    override while flipping ``extra`` from ``"forbid"`` to ``"allow"``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


def _safe_model_name(raw: object) -> str:
    """Turn a schema ``title`` into a valid, non-empty Python class name.

    A title-less or empty-titled schema (footgun #2) must still yield a
    validly-named model — ``pydantic.create_model`` and ``instructor`` both
    need a real, non-empty name. Falls back to ``JsonSchemaModel`` and
    sanitises any title into a safe identifier.
    """
    if not isinstance(raw, str):
        return _DEFAULT_MODEL_NAME
    cleaned = re.sub(r"\W+", "", raw.strip().replace(" ", "_").replace("-", "_"))
    if not cleaned:
        return _DEFAULT_MODEL_NAME
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned or _DEFAULT_MODEL_NAME


def _nullable(annotation: object) -> object:
    """Union a runtime-built field annotation with ``None``.

    The annotation is a runtime type-like object (a ``type``, a parametrised
    generic such as ``list[str]``, or an existing ``X | None`` union), so the
    ``|`` operator runs against an ``object`` whose concrete ``__or__`` only
    exists at runtime. Confining the union here keeps the one unavoidable
    type-level cast in a single named place rather than at every call site.
    """
    return cast("type", annotation) | None


def _as_list(element: object) -> object:
    """Build the ``list[...]`` annotation for an array field's element type.

    Like :func:`_nullable`, the subscript targets a runtime-built element
    annotation, so the type-level cast is confined here.
    """
    return list[cast("type", element)]


def _with_constraints(annotation: object, c: _FieldConstraints) -> object:
    """Wrap a runtime-built annotation with resolved bounds via ``Annotated``.

    Used where the bounds cannot ride on the property's own ``Field`` — array
    *element* constraints (``minLength`` on the items schema, say) must attach
    to the element annotation, not the list field. A no-op when every bound is
    unset. Like :func:`_nullable`, the type-level cast is confined here.
    """
    if all(bound is None for bound in c):
        return annotation
    return Annotated[
        cast("type", annotation),
        Field(
            ge=c.ge,
            le=c.le,
            gt=c.gt,
            lt=c.lt,
            min_length=c.min_length,
            max_length=c.max_length,
        ),
    ]


class _Converter:
    """One conversion pass over a single root schema and its ``$defs``.

    Holds the root's ``$defs`` / ``definitions`` so ``$ref`` resolution and a
    name cache (so a ``$def`` referenced twice yields the *same* model class)
    are scoped to one :func:`model_from_json_schema` call.
    """

    def __init__(self, root: JsonDict) -> None:
        self._root: JsonDict = root
        defs = root.get("$defs")
        if not isinstance(defs, dict):
            defs = root.get("definitions")
        self._defs: JsonDict = cast("JsonDict", defs) if isinstance(defs, dict) else {}
        # Object-class cache keyed by $defs identity (the resolved ``$ref``
        # name), NOT by title text: two distinct $defs that happen to share a
        # title must NOT collapse into one class. Keyed by the *name* string,
        # but only $defs are cached — anonymous inline objects are never cached.
        self._built: dict[str, type[BaseModel]] = {}
        # $ref names currently being resolved on the active build path, so a
        # self-referential schema fails loud (clear ValueError naming the
        # recursive $ref) rather than blowing the Python stack with a
        # RecursionError.
        self._in_progress: set[str] = set()
        self._counter: int = 0

    def _ref_name(self, ref: str) -> str:
        prefix_defs = "#/$defs/"
        prefix_definitions = "#/definitions/"
        if ref.startswith(prefix_defs):
            return ref[len(prefix_defs) :]
        if ref.startswith(prefix_definitions):
            return ref[len(prefix_definitions) :]
        raise ValueError(
            f"Unsupported $ref {ref!r}: only local references into "
            + "'#/$defs/' or '#/definitions/' are supported."
        )

    def _resolve_ref(self, ref: str) -> tuple[str, JsonDict]:
        name = self._ref_name(ref)
        target = self._defs.get(name)
        if not isinstance(target, dict):
            raise ValueError(f"Unresolvable $ref {ref!r}: no '{name}' in $defs/definitions.")
        return name, cast("JsonDict", target)

    def _unwrap_nullable(self, schema: JsonDict, field_path: str) -> tuple[JsonDict, bool]:
        """Split a possibly-nullable schema into (inner schema, is_nullable).

        Handles the two shapes real consumers emit: ``type: ["string",
        "null"]`` and ``anyOf: [{...}, {"type": "null"}]``.

        Nullable-merge precedence: for the ``anyOf``/``oneOf`` shape, the OUTER
        field-level keys win on conflict — they are the field's declared intent,
        while the union merely expresses nullability — and the non-null branch
        supplies ``type`` plus any keys the outer does not define. Example: an
        outer ``description`` beats a branch ``description``, while the branch
        supplies the ``type``.

        Multi-variant (discriminated) unions are unsupported: an ``anyOf`` or
        ``oneOf`` with more than one non-null branch is rejected loudly.
        """
        type_field = schema.get("type")
        if isinstance(type_field, list):
            non_null = [t for t in type_field if t != "null"]
            nullable = "null" in type_field
            if len(non_null) != 1:
                raise ValueError(
                    f"Unsupported union type {type_field!r}: only a single non-null "
                    + "type (optionally with 'null') is supported."
                )
            return {**schema, "type": non_null[0]}, nullable

        if schema.get("anyOf") and schema.get("oneOf"):
            raise ValueError(
                "A field declares both 'anyOf' and 'oneOf'; only one is supported "
                + "at a time. Combine them into a single union or drop one."
            )
        keyword = "anyOf" if schema.get("anyOf") else "oneOf" if schema.get("oneOf") else None
        any_of = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(any_of, list):
            branches: list[JsonDict] = []
            for branch in cast("list[JsonValue]", any_of):
                if not isinstance(branch, dict):
                    raise ValueError(
                        f"Unsupported {keyword} branch at {field_path!r}: each branch must "
                        + f"be a schema object, got {type(branch).__name__}."
                    )
                branches.append(cast("JsonDict", branch))
            non_null = [b for b in branches if b.get("type") != "null"]
            nullable = any(b.get("type") == "null" for b in branches)
            if len(non_null) != 1:
                raise ValueError(
                    f"Unsupported {keyword} with {len(non_null)} non-null branches: "
                    + "only a single non-null branch (optionally with a 'null' branch) "
                    + "is supported. Multi-variant (discriminated) unions are unsupported."
                )
            merged = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            # Outer field-level keys win on conflict; the non-null branch
            # supplies ``type`` and any keys the outer does not define.
            return {**non_null[0], **merged}, nullable

        return schema, False

    def _resolve_field(self, schema: JsonDict, field_path: str) -> _ResolvedField:
        """Resolve a property schema's ``$ref`` chain and nullable wrappers once.

        A property may chain ``$ref`` -> nullable-wrapper -> ``$ref`` -> ... to
        arbitrary (acyclic) depth. This walks that chain to a fixed point and
        returns the effective node (with siblings folded in and nullability
        accumulated), so the annotation, description, and value-bound consumers
        all read one consistently-resolved schema instead of each re-walking the
        chain with a divergent sibling rule.

        ``$ref`` sibling handling follows Draft 2020-12: keywords beside a
        ``$ref`` apply *together* with the target's. Non-structural keywords
        (metadata, value bounds) merge with the outer, property-level value
        winning on conflict — so ``{"$ref": "#/$defs/Count", "minimum": 5}``
        carries the bound, and a nullable-wrapped ``$ref`` inherits the target's
        ``description``. A structural sibling is a conjunction a merge cannot
        honour and is validated instead (see :meth:`_reject_structural_ref_siblings`).

        A ``$ref`` name seen twice on one chain is a pure-``$ref`` cycle: fail
        loud naming it (object-level recursion is caught separately by
        ``_in_progress`` in :meth:`_build_object`).
        """
        ref_name: str | None = None
        nullable = False
        seen_refs: set[str] = set()
        inner = schema
        while True:
            if "$ref" in inner:
                resolved_name, target = self._resolve_ref(cast("str", inner["$ref"]))
                if resolved_name in seen_refs:
                    raise ValueError(
                        f"Unsupported recursive schema at {field_path!r}: $ref "
                        + f"'#/$defs/{resolved_name}' forms a reference cycle "
                        + "(self-referential / cyclic schemas are not supported)."
                    )
                seen_refs.add(resolved_name)
                siblings = {k: v for k, v in inner.items() if k != "$ref"}
                self._reject_structural_ref_siblings(siblings, target, field_path)
                # Draft 2020-12: siblings apply together with the target's
                # keywords; outer (property-level) keys win on conflict, matching
                # the nullable-merge precedence in ``_unwrap_nullable``.
                ref_name, inner = resolved_name, {**target, **siblings}
                continue
            unwrapped, inner_nullable = self._unwrap_nullable(inner, field_path)
            nullable = nullable or inner_nullable
            if unwrapped is inner:
                break
            inner = unwrapped
        # Check the FULLY resolved node — after the ``$ref`` merge and after
        # ``_unwrap_nullable`` — so one call covers the property node, the array
        # item node, a ``$def`` body reached through a bare ``$ref``, and the
        # non-null branch of a nullable union, each reported at its use-site path.
        # The ``$ref``-sibling guard above already ran inside the loop, so a
        # sibling applicator still gets the more specific "Unsupported $ref
        # sibling" message rather than this one.
        self._reject_unsupported_applicators(inner, field_path)
        return _ResolvedField(inner, nullable, ref_name)

    def _reject_structural_ref_siblings(
        self, siblings: JsonDict, target: JsonDict, field_path: str
    ) -> None:
        """Fail loud if a structural keyword beside a ``$ref`` would redefine the target.

        A structural sibling (``type`` / ``enum`` / ``items`` / ``properties`` /
        ...) is a Draft 2020-12 *conjunction* with the referenced schema, not an
        override: folding it in last-writer-wins would silently *replace* the
        target's structure — most visibly, a ``$ref``-sibling ``enum`` would
        widen the field to an unconstrained scalar by discarding the referenced
        member set. That silent widening is the defect this rejects. The sibling
        is allowed only when it restates the target's own value for that keyword
        (a redundant no-op some generators emit); anything else must move into
        the referenced ``$def`` or inline the schema.
        """
        for key, value in siblings.items():
            if key in _STRUCTURAL_REF_SIBLINGS and value != target.get(key):
                raise ValueError(
                    f"Unsupported $ref sibling {key!r} at {field_path!r}: a {key!r} "
                    + "keyword beside a '$ref' would redefine the referenced schema's "
                    + "structure, which is not supported (JSON Schema applies it as an "
                    + "intersection, not an override, so merging it would silently drop "
                    + f"or widen the reference). Move {key!r} into the referenced $def, "
                    + "or inline the schema instead of referencing it."
                )

    def _reject_unsupported_applicators(self, schema: JsonDict, field_path: str) -> None:
        """Fail loud on a subschema applicator the converter cannot honour.

        A generated field is one annotation plus a fixed set of ``Field`` bounds,
        so an applicator (see :data:`_UNSUPPORTED_APPLICATORS`) has nowhere to
        land. Dropping one is wrong in BOTH directions: a dropped ``allOf`` bound
        makes the model accept a value the schema forbids, and a dropped
        ``prefixItems`` makes the sibling ``items`` mean "every element" instead
        of "every element after the prefix", so the model *rejects* a response the
        schema permits. Unlike a dropped leaf constraint (``pattern`` / ``format``
        — documented as unenforced), neither loss is visible to the caller.

        Distinct from :meth:`_reject_structural_ref_siblings`, which fires only
        beside a ``$ref`` and only for a sibling that redefines the target. This
        runs at every site — including a bare ``$ref`` whose *target body* carries
        an applicator, which the sibling guard structurally cannot see.
        """
        offender = next((k for k in sorted(schema) if k in _UNSUPPORTED_APPLICATORS), None)
        if offender is None:
            return
        raise ValueError(
            f"Unsupported keyword {offender!r} at {field_path!r}: subschema applicators "
            + "(allOf / not / if / then / else / contains / prefixItems / patternProperties "
            + "/ propertyNames / dependentSchemas / dependentRequired / unevaluated*) "
            + "constrain by composition and cannot be carried into a generated field, so "
            + "they would be silently dropped — changing what the model accepts. Express "
            + f"{offender!r} with a supported keyword, or validate it outside the model."
        )

    def _annotation_from_resolved(self, resolved: _ResolvedField, field_path: str) -> object:
        """Build a field's Python annotation from an already-resolved schema node.

        The ``$ref`` / nullable resolution happened once in :meth:`_resolve_field`;
        this dispatches on the effective node's ``enum`` / ``type``. Nullability
        rides on :class:`_ResolvedField` so a REQUIRED-but-nullable field (a
        ``{"type": ["string", "null"]}`` also listed in ``required``) is unioned
        with ``None`` by the caller rather than rejecting the provider's ``null``.
        """
        inner = resolved.schema
        if "enum" in inner:
            # The canonical nullable-enum spelling carries ``null`` as an enum
            # member (``{"type": ["string", "null"], "enum": ["a", null]}``) —
            # JSON Schema requires it there for an actual ``null`` to validate.
            # Nullability is already resolved into the ``X | None`` union, so
            # drop the ``null`` member before building the Enum. A ``null``
            # member on a NON-nullable field is left in place so ``_build_enum``
            # still rejects the contradiction loudly.
            if resolved.nullable and isinstance(inner["enum"], list):
                non_null_members: list[JsonValue] = [v for v in inner["enum"] if v is not None]
                if not non_null_members:
                    # Stripping would leave a member-less Enum; name the actual
                    # defect (only-null members), not the post-strip emptiness.
                    raise ValueError(
                        f"Unsupported enum at {field_path!r}: a nullable enum must have "
                        + "at least one non-null member — 'enum' contains only null."
                    )
                inner = {**inner, "enum": cast("JsonValue", non_null_members)}
            return self._build_enum(inner, field_path)

        jtype = inner.get("type")
        if jtype is None:
            raise ValueError(
                f"Unsupported schema at {field_path!r}: no 'type', 'enum', or '$ref' — "
                + f"got keys {sorted(inner)}."
            )

        if jtype == "object":
            return self._build_object(inner, field_path, ref_name=resolved.ref_name)
        if jtype == "array":
            items = inner.get("items")
            if not isinstance(items, dict):
                raise ValueError(
                    f"Unsupported array at {field_path!r}: 'items' must be a single "
                    + "schema object (tuple/heterogeneous arrays are not supported)."
                )
            item_resolved = self._resolve_field(cast("JsonDict", items), f"{field_path}[]")
            element = self._annotation_from_resolved(item_resolved, f"{field_path}[]")
            # Per-element bounds (e.g. ``minLength`` on the items schema) ride
            # on the element annotation — the list field's own ``Field`` only
            # carries ``minItems``/``maxItems``. Wrap BEFORE the nullable
            # union so a ``null`` element still passes unbounded.
            element = _with_constraints(
                element, self._constraints_from_resolved(item_resolved.schema)
            )
            if item_resolved.nullable:
                element = _nullable(element)
            return _as_list(element)
        if isinstance(jtype, str) and jtype in _SCALAR_TYPES:
            return _SCALAR_TYPES[jtype]
        raise ValueError(f"Unsupported JSON-schema type {jtype!r} at {field_path!r}.")

    def _constraints_from_resolved(self, resolved: JsonDict) -> _FieldConstraints:
        """Pull the supported per-field bounds off an already-resolved schema node.

        The ``$ref`` / nullable resolution happens once in :meth:`_resolve_field`,
        so this is a pure extractor over the effective node — a bound declared on
        the non-null branch of an ``anyOf`` or inside a referenced ``$def`` has
        already been folded in. Returns a :class:`_FieldConstraints` carrying the
        resolved Pydantic ``Field`` bounds (``ge`` / ``le`` / ``gt`` / ``lt`` /
        ``min_length`` / ``max_length``).

        Constraints outside the supported set (see the module docstring and the
        ``_FieldConstraints`` fields) are silently dropped — no partial
        enforcement. ``minLength`` (strings) and ``minItems`` (arrays) never
        co-occur on one field, so mapping both onto ``min_length`` is safe.

        A bound is also dropped when it does **not match the field's resolved
        JSON type** — a numeric bound (``minimum`` …) on a non-numeric field, or
        a length bound (``minLength`` / ``minItems`` …) on a non-string/array
        field. Pydantic rejects such a mismatched constraint with a ``TypeError``
        at *validation* time (not build time), so applying it unconditionally
        would turn a stray keyword in an otherwise-valid schema into an opaque
        crash on the first response. Gating by type keeps the drop-the-
        unsupported promise instead of crashing.
        """
        raw_type = resolved.get("type")
        types: set[str] = (
            {raw_type}
            if isinstance(raw_type, str)
            else {t for t in raw_type if isinstance(t, str)}
            if isinstance(raw_type, list)
            else set()
        )
        # Numeric bounds (ge/le/gt/lt) apply only to integer/number; length
        # bounds (min_length/max_length) only to string/array. A field whose
        # type is absent or anything else gets no bounds — drop, never crash.
        numeric_field = bool(types & {"integer", "number"})
        sized_field = bool(types & {"string", "array"})

        def _number(key: str) -> float | None:
            value = resolved.get(key)
            # ``bool`` is an ``int`` subclass — exclude it. A non-numeric (or
            # missing) bound silently drops, matching the drop-the-unsupported
            # contract rather than erroring.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            return None

        def _length(key: str) -> int | None:
            value = resolved.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None

        return _FieldConstraints(
            ge=_number("minimum") if numeric_field else None,
            le=_number("maximum") if numeric_field else None,
            gt=_number("exclusiveMinimum") if numeric_field else None,
            lt=_number("exclusiveMaximum") if numeric_field else None,
            # minLength (string) and minItems (array) both map to min_length;
            # at most one is present for a given field, so the ``or`` picks the
            # one that applies without conflict.
            min_length=(
                (_length("minLength") if "minLength" in resolved else _length("minItems"))
                if sized_field
                else None
            ),
            max_length=(
                (_length("maxLength") if "maxLength" in resolved else _length("maxItems"))
                if sized_field
                else None
            ),
        )

    def _build_enum(self, schema: JsonDict, field_path: str) -> type[Enum]:
        values = schema.get("enum")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Unsupported enum at {field_path!r}: 'enum' must be a non-empty list."
            )
        members: dict[str, JsonValue] = {}
        all_int = True
        any_int = False
        for value in cast("list[JsonValue]", values):
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(
                    f"Unsupported enum value {value!r} at {field_path!r}: only string and "
                    + "integer enum members are supported."
                )
            if isinstance(value, int):
                any_int = True
            else:
                all_int = False
            raw = str(value)
            key = re.sub(r"\W+", "_", raw).strip("_").upper() or "VALUE"
            # Prefix with a LETTER, never a leading underscore. Stripping ``_``
            # above drops a leading sign, so ``-1`` and ``1`` both reduce to
            # ``"1"`` — re-encode the sign here so they stay distinct, and keep
            # the name letter-led so the collision suffix below can never form a
            # reserved ``_sunder_`` / ``__dunder__`` name (Python's Enum rejects
            # those: ``_1_`` from a digit-led ``_1`` was the original crash).
            if raw.lstrip().startswith("-"):
                key = f"NEG_{key}"
            elif key[0].isdigit():
                key = f"N_{key}"
            while key in members:
                key = f"{key}_"
            members[key] = value
        # A mixed string/integer enum has no faithful single base: with
        # ``use_enum_values=True`` an ``int`` base + str member (or vice versa)
        # coerces members to one type, so the generated model would reject its
        # own schema-valid values (e.g. integer ``1`` stored as ``"1"``). The
        # supported subset is a homogeneous string *or* integer enum; reject the
        # mix loudly, naming the construct, rather than silently misbuilding.
        if any_int and not all_int:
            raise ValueError(
                f"Unsupported mixed-type enum at {field_path!r}: enum members must be all "
                + "strings or all integers, not a mix of both."
            )
        title = schema.get("title")
        name = (
            _safe_model_name(title) if isinstance(title, str) and title else self._anon_name("Enum")
        )
        # Mix in str/int so members compare equal to their raw value and stay
        # JSON-clean. The generated models also set ``use_enum_values=True``
        # (see ``_JsonSchemaModel``), so a validated instance stores the raw
        # scalar — ``model_dump()`` yields ``2``, not ``<Enum._2: 2>`` — which
        # is what dict consumers (those who ``model_dump()`` the result) and
        # the provider JSON both expect.
        base: type = int if all_int else str
        # The functional ``Enum(...)`` call returns a new enum *class* at runtime,
        # but the stubs type it as an ``Enum`` instance — hence the narrow ignore.
        return Enum(name, members, type=base)  # pyright: ignore[reportReturnType]  # runtime str/int-mixin enum class

    def _build_object(
        self, schema: JsonDict, field_path: str, *, ref_name: str | None = None
    ) -> type[BaseModel]:
        # Cache by $defs IDENTITY (the resolved ``$ref`` name), not title text,
        # so a def referenced twice is one class while two distinct defs that
        # merely share a title stay distinct. Anonymous inline objects (no
        # ``ref_name``) are never cached and always get a fresh class.
        if ref_name is not None and ref_name in self._built:
            return self._built[ref_name]

        # A $ref back to a def still being built is a recursion cycle. Pydantic
        # would need deferred forward-refs to model that; we instead fail loud,
        # naming the recursive $ref, rather than dying with a RecursionError.
        if ref_name is not None:
            if ref_name in self._in_progress:
                raise ValueError(
                    f"Unsupported recursive schema at {field_path!r}: $ref "
                    + f"'#/$defs/{ref_name}' refers back to an object still being "
                    + "built (self-referential / cyclic schemas are not supported)."
                )
            self._in_progress.add(ref_name)
        try:
            return self._build_object_uncached(schema, field_path, ref_name)
        finally:
            if ref_name is not None:
                self._in_progress.discard(ref_name)

    def _build_object_uncached(
        self, schema: JsonDict, field_path: str, ref_name: str | None
    ) -> type[BaseModel]:
        title = schema.get("title")
        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ValueError(
                f"Unsupported object at {field_path!r}: 'properties' must be a mapping."
            )
        # ``additionalProperties`` picks the model's extra policy:
        #   absent / false -> strict ``extra="forbid"`` (the good LLM default);
        #   true           -> ``extra="allow"`` (author wants open-ended keys);
        #   a typed dict    -> unsupported, raise.
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            raise ValueError(
                f"Unsupported 'additionalProperties' at {field_path!r}: typed "
                + "additionalProperties maps are not supported (only true/false or absent)."
            )
        props: JsonDict = cast("JsonDict", properties) if isinstance(properties, dict) else {}
        # A propertyless ``type: "object"`` would build a zero-field
        # ``extra="forbid"`` model that validates only ``{}`` and rejects every
        # real response — the silent-wrong-model failure this module refuses.
        # Test emptiness, not just a *missing* key: an explicit ``properties: {}``
        # is equally propertyless and must fail the same way rather than slipping
        # past to build the reject-everything model. Only ``additionalProperties:
        # true`` (an intentionally free-form object, the ``extra="allow"`` base)
        # makes a propertyless object meaningful, so everything else fails loud.
        if not props and additional is not True:
            raise ValueError(
                f"Unsupported object at {field_path!r}: 'type' is 'object' but there are "
                + "no properties (a missing or empty 'properties') — this would build a "
                + "zero-field model that rejects every real response. Declare 'properties', "
                + "or set 'additionalProperties': true for an intentionally free-form object."
            )
        required_raw = schema.get("required")
        if required_raw is None:
            required_list: list[JsonValue] = []
        elif isinstance(required_raw, list):
            required_list = cast("list[JsonValue]", required_raw)
        else:
            # A non-list ``required`` (e.g. ``"required": "id"``) silently made
            # every field optional pre-fix — a wrong model, so fail loud.
            raise ValueError(
                f"Unsupported object at {field_path!r}: 'required' must be a list of "
                + f"property-name strings, got {type(required_raw).__name__}."
            )
        required: set[str] = set()
        for entry in required_list:
            if not isinstance(entry, str):
                raise ValueError(
                    f"Unsupported object at {field_path!r}: 'required' entries must be "
                    + f"property-name strings, got {entry!r}."
                )
            required.add(entry)
        # A ``required`` name with no matching property is a schema typo that
        # would otherwise degrade silently in BOTH directions: the model accepts
        # payloads missing the field and (under ``extra="forbid"``) rejects
        # payloads that legitimately carry it.
        missing = required - set(props)
        if missing:
            raise ValueError(
                f"Unsupported object at {field_path!r}: 'required' names "
                + f"{sorted(missing)} not present in 'properties'."
            )

        # ``fields`` feeds ``create_model``'s dynamic ``**field_definitions``
        # splat (each value a ``(annotation, FieldInfo)`` tuple). The annotation
        # is built at runtime and pydantic's factory is untyped here, so this
        # one dict carries ``Any`` deliberately.
        fields: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]  # raw-pydantic — create_model **field_definitions splat (runtime annotations)
        # Optional field names (absent from ``required``) — stamped onto the
        # generated class so its dump serializer drops only *these* fields'
        # ``None``, never an explicitly-null required field.
        optional_fields: set[str] = set()
        for prop_name, prop_schema in props.items():
            # Reject names pydantic cannot carry as public fields BEFORE they
            # reach ``create_model``, which otherwise fails opaquely deep inside
            # pydantic (or worse: silently). A leading underscore is treated as a
            # private attribute and the field is *silently dropped*; the
            # ``model_`` prefix is pydantic's protected namespace and collides
            # with real machinery (``model_config`` -> ``TypeError``,
            # ``model_dump`` -> a conflicts-with-member ``ValueError``). Names
            # that merely shadow pydantic's *deprecated v1 shims* (``schema``,
            # ``json``, ``copy``, ``dict``, ...) work fine as fields, so they
            # pass through — their shadow warning is suppressed at
            # ``create_model`` below.
            if prop_name.startswith("_"):
                raise ValueError(
                    f"Unsupported property {prop_name!r} at {field_path!r}: property names "
                    + "may not start with an underscore — pydantic treats them as private "
                    + "attributes and would silently drop the field."
                )
            if prop_name.startswith("model_"):
                raise ValueError(
                    f"Unsupported property {prop_name!r} at {field_path!r}: the 'model_' "
                    + "prefix is pydantic's protected namespace ('model_config', "
                    + "'model_dump', ...) and cannot be used as a field; rename the "
                    + "property."
                )
            if not isinstance(prop_schema, dict):
                raise ValueError(
                    f"Unsupported property {prop_name!r} at {field_path!r}: "
                    + "must be a schema object."
                )
            prop = cast("JsonDict", prop_schema)
            prop_path = f"{field_path}.{prop_name}"
            # Resolve the property's ``$ref`` chain and nullable wrappers ONCE;
            # the annotation, description, and value bounds all read the same
            # effective node. A ``$ref`` target's ``description`` is folded in by
            # the resolution (outer-wins), so a bare ``$ref``, a nullable-wrapped
            # ``$ref``, and a multi-hop chain all surface the target's guidance —
            # no separate description rescue, and no double resolution.
            resolved = self._resolve_field(prop, prop_path)
            annotation = self._annotation_from_resolved(resolved, prop_path)
            is_nullable = resolved.nullable
            description = resolved.schema.get("description")
            desc = description if isinstance(description, str) else None
            # Per-field value bounds (ge/le/gt/lt/min_length/max_length). Only
            # the supported keywords cross over; everything else is dropped.
            c = self._constraints_from_resolved(resolved.schema)
            optional = prop_name not in required
            if optional:
                optional_fields.add(prop_name)
            # Union with None when the field is nullable OR optional — the two
            # are independent: a REQUIRED nullable field must still accept the
            # provider's ``null``.
            if is_nullable or optional:
                annotation = _nullable(annotation)
            default = None if optional else ...
            field_info = Field(
                default,
                description=desc,
                ge=c.ge,
                le=c.le,
                gt=c.gt,
                lt=c.lt,
                min_length=c.min_length,
                max_length=c.max_length,
            )
            fields[prop_name] = (annotation, field_info)

        base: type[_JsonSchemaModel] = (
            _JsonSchemaModelAllow if additional is True else _JsonSchemaModel
        )

        model_name = (
            _safe_model_name(title)
            if isinstance(title, str) and title
            else self._anon_name("Object")
        )
        # ``**fields`` is the one unavoidable ``Any`` boundary: pydantic's
        # ``create_model`` is a dynamic factory whose ``**field_definitions`` is
        # typed ``Any | tuple[Any, Any]`` in the stubs, so splatting the runtime
        # field map lands every keyword argument on ``Any``. Confined and
        # documented here rather than scattered. The deprecated v1-shim names
        # allowed through above (``schema`` / ``json`` / ...) are valid fields
        # but trip pydantic's shadows-an-attribute ``UserWarning``; suppress
        # exactly that one so a legitimate schema builds without warning spam.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r'Field name ".*" in .* shadows an attribute in parent',
                category=UserWarning,
            )
            model = create_model(model_name, __base__=base, **fields)  # pyright: ignore[reportAny]  # raw-pydantic — create_model dynamic **field_definitions splat
        model.__llmkit_optional_fields__ = frozenset(optional_fields)
        if ref_name is not None:
            self._built[ref_name] = model
        return model

    def _anon_name(self, kind: str) -> str:
        self._counter += 1
        return f"{_DEFAULT_MODEL_NAME}{kind}{self._counter}"

    def convert(self, name: str | None) -> type[BaseModel]:
        root = self._root
        root_ref: str | None = None
        # Resolve a top-level $ref so the root can be a bare reference. Siblings
        # on the root $ref get the same treatment as anywhere else: a structural
        # one is rejected (it cannot redefine a referenced — and possibly shared,
        # cached — schema), and the rest merge over the target, outer-wins. Only
        # ``$ref`` is dropped from the siblings; ``$defs`` and other metadata ride
        # along harmlessly.
        if "$ref" in root:
            siblings = {k: v for k, v in root.items() if k != "$ref"}
            root_ref, target = self._resolve_ref(cast("str", root["$ref"]))
            self._reject_structural_ref_siblings(siblings, target, "$")
            root = {**target, **siblings}
        # Outside the ``$ref`` branch on purpose: the root reaches
        # ``_build_object`` directly and never passes through ``_resolve_field``,
        # so without this call a root-level ``if`` / ``allOf`` stays silent.
        self._reject_unsupported_applicators(root, "$")
        jtype = root.get("type")
        if jtype not in (None, "object"):
            raise ValueError(
                f"Unsupported root schema: top level must be an object, got type {jtype!r}."
            )
        if jtype is None:
            # A typeless root is accepted only when it is recognisably an
            # object (bare ``properties``). An ``enum`` / ``anyOf`` / ``oneOf``
            # root — or an empty schema — would otherwise fall through to a
            # zero-field ``extra="forbid"`` model that rejects every real
            # response (or silently validates ``{}``).
            offender = next((k for k in ("enum", "anyOf", "oneOf") if k in root), None)
            if offender is not None:
                raise ValueError(
                    f"Unsupported root schema: top level must be an object, got {offender!r} "
                    + "with no 'type'. Wrap it in an object property instead."
                )
            if "properties" not in root:
                raise ValueError(
                    "Unsupported root schema: top level must be an object with 'properties' "
                    + f'(or "type": "object") — got keys {sorted(root)}.'
                )
        model = self._build_object(root, "$", ref_name=root_ref)
        chosen = name if name else root.get("title")
        # Rename so an explicit name / title wins, and a title-less root
        # still gets a real, non-empty name (footgun #2).
        model.__name__ = _safe_model_name(chosen)
        model.__qualname__ = model.__name__
        return model


def model_from_json_schema(
    schema: Mapping[str, object], *, name: str | None = None
) -> type[BaseModel]:
    """Convert a JSON-schema dict into a Pydantic model class at runtime.

    Built on ``pydantic.create_model`` (no new third-party dependency),
    recursively, for the subset of JSON Schema that real dict-based
    consumers use. Pass the result straight to
    :func:`~llmkit.structured_llm_call` as ``output_schema`` — ideally
    **built once and reused** rather than rebuilt per call.

    Args:
        schema: The JSON-schema dict. The root must be an object (or a
            ``$ref`` to one). Local ``$defs`` / ``definitions`` are resolved.
        name: Optional explicit class name. When omitted, the schema's
            ``title`` is used; a title-less schema falls back to a safe
            default (``"JsonSchemaModel"``) so the generated class is always
            validly named.

    Supported subset (anything else raises a clear ``ValueError``):

    * ``object`` with ``properties``; ``required`` vs optional via ``required``
    * scalars: ``string`` / ``integer`` / ``number`` / ``boolean``, plus
      ``null`` / nullable (``["string", "null"]`` or ``anyOf`` + null branch)
    * ``array`` (``items``), including arrays of objects
    * ``enum`` (string or integer members)
    * nested objects, inline or via local ``$ref`` (``#/$defs/...``)
    * ``additionalProperties``: ``true`` (open object, extra keys kept) or
      ``false`` / absent (strict); a typed map raises ``ValueError``

    Subschema applicators (``allOf`` / ``not`` / ``if`` / ``then`` / ``else`` /
    ``dependentSchemas`` / ``dependentRequired`` / ``propertyNames`` /
    ``patternProperties`` / ``prefixItems`` / ``contains`` / ``unevaluated*``)
    constrain by composition and cannot ride on a generated field, so each
    raises ``ValueError`` naming the keyword and its path — at *every* site, not
    only beside a ``$ref``. Nullable ``anyOf`` / ``oneOf`` are unaffected.

    Per-field constraints (carried into ``Field``; other leaf keywords dropped):

    * numeric ``minimum``/``maximum`` → ``ge``/``le``,
      ``exclusiveMinimum``/``exclusiveMaximum`` → ``gt``/``lt``
    * ``minLength``/``maxLength`` → ``min_length``/``max_length`` (strings)
    * ``minItems``/``maxItems`` → ``min_length``/``max_length`` (arrays)
    * ``description`` → per-field ``Field`` description (instructor guidance)

    Any *leaf* constraint outside that set (``pattern``, ``format``,
    ``multipleOf``, …) is **silently dropped** — no partial enforcement. A
    structural construct outside the subset raises instead of vanishing.

    Serialization contract:
        A non-required field becomes an optional Pydantic field defaulting to
        ``None``, and the generated model's ``model_dump`` /
        ``model_dump_json`` drop unset *optional* fields by default — so an
        omitted optional is *absent* from the dump, not ``"field": null``
        (which would fail downstream re-validation). The drop is scoped: a
        required-but-nullable field set to ``None`` is kept. Pass
        ``exclude_none=False`` to keep every null, or ``exclude_none=True``
        to drop them all.

    Strictness:
        Generated models default to ``extra="forbid"``, so a response carrying
        a key not in the schema is *rejected*. This is deliberately stricter
        than JSON Schema's permissive ``additionalProperties`` default — for an
        LLM output contract you want a hallucinated extra field to fail loudly,
        not pass silently. An object whose schema sets ``additionalProperties:
        true`` opts into ``extra="allow"`` instead (extra keys are accepted and
        kept); ``false`` or absent keeps the strict default.

    Returns:
        A ``type[BaseModel]`` subclass ready to use as ``output_schema``.

    Raises:
        ValueError: On any construct outside the supported subset, naming the
            offending construct and its path in the schema.
    """
    # Deliberate runtime guard: the annotation says ``Mapping`` but this is the
    # library's public boundary, so a mistyped (untyped) caller still gets a
    # clear ValueError rather than an obscure failure deeper in conversion.
    if not isinstance(schema, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard at public boundary
        raise ValueError(  # pyright: ignore[reportUnreachable]  # reachable from untyped callers
            f"model_from_json_schema expects a mapping schema, got {type(schema).__name__}."
        )
    return _Converter(cast("JsonDict", dict(schema))).convert(name)
