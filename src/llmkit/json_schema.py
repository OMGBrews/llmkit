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
  (``#/$defs/Name`` or the legacy ``#/definitions/Name``)

Per-field constraints
----------------------
A small, fixed set of per-field constraints is carried through to the
generated Pydantic ``Field`` so the model validates *value bounds*, not just
shape. The supported set is **exactly**:

* numeric: ``minimum`` → ``ge``, ``maximum`` → ``le``,
  ``exclusiveMinimum`` → ``gt``, ``exclusiveMaximum`` → ``lt``
* string: ``minLength`` → ``min_length``, ``maxLength`` → ``max_length``
* array: ``minItems`` → ``min_length``, ``maxItems`` → ``max_length``
* ``description`` → ``Field(description=...)`` (instructor surfaces this as
  per-field guidance to the model)

Any other constraint keyword (``pattern``, ``format``, ``multipleOf``,
``uniqueItems``, ``const``, …) is **silently dropped** — deliberately, to
avoid partial enforcement that looks complete. Nothing outside the list above
is enforced; if a schema relies on one of those, validate it elsewhere.

Serialization contract
-----------------------
The generated model maps a **non-required** JSON-schema field to an
*optional* Pydantic field whose default is ``None``. To keep an omitted
optional from round-tripping back out as ``"field": null`` — which fails
re-validation against a JSON schema that lists the field as a non-nullable
optional — the generated model's :meth:`~pydantic.BaseModel.model_dump`
and :meth:`~pydantic.BaseModel.model_dump_json` default to
``exclude_none=True``. An optional the model never set is therefore
*absent* from the dump, not present-and-null. Callers who genuinely want
the nulls back can pass ``exclude_none=False`` explicitly.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, ClassVar, Literal, NamedTuple, TypedDict, Unpack, cast, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model

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


class _JsonSchemaModel(BaseModel):
    """Base for every generated model: dumps exclude ``None`` by default.

    A non-required JSON-schema field becomes an optional Pydantic field
    defaulting to ``None``. Defaulting dumps to ``exclude_none=True`` keeps
    an unset optional *absent* from the output rather than serialising as
    ``"field": null`` (footgun #1 — that null then fails downstream
    re-validation against the same JSON schema). Pass ``exclude_none=False``
    to opt back in to the nulls.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", use_enum_values=True)

    @override
    def model_dump(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Unpack[_DumpKwargs],
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — mirrors model_dump's dict[str, Any] return
        return super().model_dump(exclude_none=exclude_none, **kwargs)

    @override
    def model_dump_json(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Unpack[_DumpJsonKwargs],
    ) -> str:
        return super().model_dump_json(exclude_none=exclude_none, **kwargs)


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


def _as_dict(value: JsonValue) -> JsonDict:
    """Narrow a JSON value to a schema dict (``object``)."""
    return cast("JsonDict", value)


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

    def _unwrap_nullable(self, schema: JsonDict) -> tuple[JsonDict, bool]:
        """Split a possibly-nullable schema into (inner schema, is_nullable).

        Handles the two shapes real consumers emit: ``type: ["string",
        "null"]`` and ``anyOf: [{...}, {"type": "null"}]``.
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

        any_of = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(any_of, list):
            branches = [_as_dict(b) for b in cast("list[JsonValue]", any_of)]
            non_null = [b for b in branches if b.get("type") != "null"]
            nullable = any(b.get("type") == "null" for b in branches)
            if len(non_null) != 1:
                raise ValueError(
                    f"Unsupported anyOf/oneOf with {len(non_null)} non-null branches: "
                    + "only a single non-null branch (optionally with a 'null' branch) "
                    + "is supported."
                )
            merged = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            return {**merged, **non_null[0]}, nullable

        return schema, False

    def _field_type(self, schema: JsonDict, field_path: str) -> tuple[object, bool]:
        """Resolve (annotation, is_nullable) for one property schema.

        ``is_nullable`` is threaded back out so the caller can union the
        annotation with ``None`` for a REQUIRED-but-nullable field too — a
        ``{"type": ["string", "null"]}`` (or ``anyOf`` + null branch) that is
        also listed in ``required`` must accept the provider's ``null``, not
        reject it.
        """
        ref_name: str | None = None
        if "$ref" in schema:
            ref_name, schema = self._resolve_ref(cast("str", schema["$ref"]))

        inner, nullable = self._unwrap_nullable(schema)

        if "$ref" in inner:
            ref_name, inner = self._resolve_ref(cast("str", inner["$ref"]))
            inner, inner_nullable = self._unwrap_nullable(inner)
            nullable = nullable or inner_nullable

        if "enum" in inner:
            return self._build_enum(inner, field_path), nullable

        jtype = inner.get("type")
        if jtype is None:
            raise ValueError(
                f"Unsupported schema at {field_path!r}: no 'type', 'enum', or '$ref' — "
                + f"got keys {sorted(inner)}."
            )

        if jtype == "object":
            return self._build_object(inner, field_path, ref_name=ref_name), nullable
        if jtype == "array":
            items = inner.get("items")
            if not isinstance(items, dict):
                raise ValueError(
                    f"Unsupported array at {field_path!r}: 'items' must be a single "
                    + "schema object (tuple/heterogeneous arrays are not supported)."
                )
            element, element_nullable = self._field_type(cast("JsonDict", items), f"{field_path}[]")
            if element_nullable:
                element = _nullable(element)
            return _as_list(element), nullable
        if isinstance(jtype, str) and jtype in _SCALAR_TYPES:
            return _SCALAR_TYPES[jtype], nullable
        raise ValueError(f"Unsupported JSON-schema type {jtype!r} at {field_path!r}.")

    def _field_constraints(self, schema: JsonDict) -> _FieldConstraints:
        """Pull the supported per-field bounds off one property schema.

        Mirrors :meth:`_field_type`'s ``$ref`` / nullable unwrapping so a
        bound declared on the inner schema (e.g. on the non-null branch of an
        ``anyOf``, or inside a referenced ``$def``) is still found. Returns a
        :class:`_FieldConstraints` carrying the resolved Pydantic ``Field``
        bounds (``ge`` / ``le`` / ``gt`` / ``lt`` / ``min_length`` /
        ``max_length``).

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
        if "$ref" in schema:
            _, schema = self._resolve_ref(cast("str", schema["$ref"]))
        inner, _ = self._unwrap_nullable(schema)
        if "$ref" in inner:
            _, inner = self._resolve_ref(cast("str", inner["$ref"]))
            inner, _ = self._unwrap_nullable(inner)

        raw_type = inner.get("type")
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
            value = inner.get(key)
            # ``bool`` is an ``int`` subclass — exclude it. A non-numeric (or
            # missing) bound silently drops, matching the drop-the-unsupported
            # contract rather than erroring.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            return None

        def _length(key: str) -> int | None:
            value = inner.get(key)
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
                (_length("minLength") if "minLength" in inner else _length("minItems"))
                if sized_field
                else None
            ),
            max_length=(
                (_length("maxLength") if "maxLength" in inner else _length("maxItems"))
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
        props: JsonDict = cast("JsonDict", properties) if isinstance(properties, dict) else {}
        required_raw = schema.get("required")
        required: set[str] = (
            {str(r) for r in cast("list[JsonValue]", required_raw)}
            if isinstance(required_raw, list)
            else set()
        )

        # ``fields`` feeds ``create_model``'s dynamic ``**field_definitions``
        # splat (each value a ``(annotation, FieldInfo)`` tuple). The annotation
        # is built at runtime and pydantic's factory is untyped here, so this
        # one dict carries ``Any`` deliberately.
        fields: dict[str, Any] = {}  # pyright: ignore[reportExplicitAny]  # raw-pydantic — create_model **field_definitions splat (runtime annotations)
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                raise ValueError(
                    f"Unsupported property {prop_name!r} at {field_path!r}: "
                    + "must be a schema object."
                )
            prop = cast("JsonDict", prop_schema)
            annotation, is_nullable = self._field_type(prop, f"{field_path}.{prop_name}")
            description = prop.get("description")
            desc = description if isinstance(description, str) else None
            # Per-field value bounds (ge/le/gt/lt/min_length/max_length). Only
            # the supported keywords cross over; everything else is dropped.
            c = self._field_constraints(prop)
            optional = prop_name not in required
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

        model_name = (
            _safe_model_name(title)
            if isinstance(title, str) and title
            else self._anon_name("Object")
        )
        # ``**fields`` is the one unavoidable ``Any`` boundary: pydantic's
        # ``create_model`` is a dynamic factory whose ``**field_definitions`` is
        # typed ``Any | tuple[Any, Any]`` in the stubs, so splatting the runtime
        # field map lands every keyword argument on ``Any``. Confined and
        # documented here rather than scattered.
        model = create_model(model_name, __base__=_JsonSchemaModel, **fields)  # pyright: ignore[reportAny]  # raw-pydantic — create_model dynamic **field_definitions splat
        if ref_name is not None:
            self._built[ref_name] = model
        return model

    def _anon_name(self, kind: str) -> str:
        self._counter += 1
        return f"{_DEFAULT_MODEL_NAME}{kind}{self._counter}"

    def convert(self, name: str | None) -> type[BaseModel]:
        root = self._root
        root_ref: str | None = None
        # Resolve a top-level $ref so the root can be a bare reference.
        if "$ref" in root:
            root_ref, root = self._resolve_ref(cast("str", root["$ref"]))
        jtype = root.get("type")
        if jtype not in (None, "object"):
            raise ValueError(
                f"Unsupported root schema: top level must be an object, got type {jtype!r}."
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

    Per-field constraints (carried into ``Field``; everything else dropped):

    * numeric ``minimum``/``maximum`` → ``ge``/``le``,
      ``exclusiveMinimum``/``exclusiveMaximum`` → ``gt``/``lt``
    * ``minLength``/``maxLength`` → ``min_length``/``max_length`` (strings)
    * ``minItems``/``maxItems`` → ``min_length``/``max_length`` (arrays)
    * ``description`` → per-field ``Field`` description (instructor guidance)

    Any constraint outside that set (``pattern``, ``format``, ``multipleOf``,
    …) is **silently dropped** — no partial enforcement.

    Serialization contract:
        A non-required field becomes an optional Pydantic field defaulting to
        ``None``, and the generated model's ``model_dump`` /
        ``model_dump_json`` default to ``exclude_none=True`` — so an omitted
        optional is *absent* from the dump, not ``"field": null`` (which
        would fail downstream re-validation). Pass ``exclude_none=False`` to
        keep the nulls.

    Strictness:
        Generated models set ``extra="forbid"``, so a response carrying a key
        not in the schema is *rejected*. This is deliberately stricter than
        JSON Schema's permissive ``additionalProperties`` default — for an LLM
        output contract you want a hallucinated extra field to fail loudly, not
        pass silently.

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
