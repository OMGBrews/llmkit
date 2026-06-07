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
from collections.abc import Mapping
from enum import Enum
from typing import Any, NamedTuple, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model

__all__ = ["model_from_json_schema"]

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

    model_config = ConfigDict(extra="forbid")

    def model_dump(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]  # raw-pydantic — forwards to model_dump's exhaustive keyword surface
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — mirrors model_dump's dict[str, Any] return
        return super().model_dump(exclude_none=exclude_none, **kwargs)

    def model_dump_json(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]  # raw-pydantic — forwards to model_dump_json's exhaustive keyword surface
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


class _Converter:
    """One conversion pass over a single root schema and its ``$defs``.

    Holds the root's ``$defs`` / ``definitions`` so ``$ref`` resolution and a
    name cache (so a ``$def`` referenced twice yields the *same* model class)
    are scoped to one :func:`model_from_json_schema` call.
    """

    def __init__(self, root: JsonDict) -> None:
        self._root = root
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
        self._counter = 0

    def _ref_name(self, ref: str) -> str:
        prefix_defs = "#/$defs/"
        prefix_definitions = "#/definitions/"
        if ref.startswith(prefix_defs):
            return ref[len(prefix_defs) :]
        if ref.startswith(prefix_definitions):
            return ref[len(prefix_definitions) :]
        raise ValueError(
            f"Unsupported $ref {ref!r}: only local references into "
            f"'#/$defs/' or '#/definitions/' are supported."
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
                    f"type (optionally with 'null') is supported."
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
                    f"only a single non-null branch (optionally with a 'null' branch) "
                    f"is supported."
                )
            merged = {k: v for k, v in schema.items() if k not in ("anyOf", "oneOf")}
            return {**merged, **non_null[0]}, nullable

        return schema, False

    def _field_type(self, schema: JsonDict, field_path: str) -> tuple[Any, bool]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — returns a runtime-built model annotation (a type, a generic, or a `X | None` union)
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
                f"got keys {sorted(inner)}."
            )

        if jtype == "object":
            return self._build_object(inner, field_path, ref_name=ref_name), nullable
        if jtype == "array":
            items = inner.get("items")
            if not isinstance(items, dict):
                raise ValueError(
                    f"Unsupported array at {field_path!r}: 'items' must be a single "
                    f"schema object (tuple/heterogeneous arrays are not supported)."
                )
            element, element_nullable = self._field_type(cast("JsonDict", items), f"{field_path}[]")
            if element_nullable:
                element = element | None
            return list[element], nullable  # type: ignore[valid-type]  # runtime-built element type
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
        """
        if "$ref" in schema:
            _, schema = self._resolve_ref(cast("str", schema["$ref"]))
        inner, _ = self._unwrap_nullable(schema)
        if "$ref" in inner:
            _, inner = self._resolve_ref(cast("str", inner["$ref"]))
            inner, _ = self._unwrap_nullable(inner)

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
            ge=_number("minimum"),
            le=_number("maximum"),
            gt=_number("exclusiveMinimum"),
            lt=_number("exclusiveMaximum"),
            # minLength (string) and minItems (array) both map to min_length;
            # at most one is present for a given field, so the ``or`` picks the
            # one that applies without conflict.
            min_length=_length("minLength") if "minLength" in inner else _length("minItems"),
            max_length=_length("maxLength") if "maxLength" in inner else _length("maxItems"),
        )

    def _build_enum(self, schema: JsonDict, field_path: str) -> type[Enum]:
        values = schema.get("enum")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Unsupported enum at {field_path!r}: 'enum' must be a non-empty list."
            )
        members: dict[str, JsonValue] = {}
        all_int = True
        for value in cast("list[JsonValue]", values):
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(
                    f"Unsupported enum value {value!r} at {field_path!r}: only string and "
                    f"integer enum members are supported."
                )
            if not isinstance(value, int):
                all_int = False
            key = re.sub(r"\W+", "_", str(value)).strip("_").upper() or "VALUE"
            if key[0].isdigit():
                key = f"_{key}"
            while key in members:
                key = f"{key}_"
            members[key] = value
        title = schema.get("title")
        name = (
            _safe_model_name(title) if isinstance(title, str) and title else self._anon_name("Enum")
        )
        # Mix in str/int so members compare equal to and serialise as their raw
        # value (e.g. ``"open"``, not ``Status.OPEN``) — matching the JSON the
        # provider returns and keeping ``model_dump`` JSON-clean.
        base: type = int if all_int else str
        return Enum(name, members, type=base)  # type: ignore[return-value]  # runtime str/int-mixin enum

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
                    f"'#/$defs/{ref_name}' refers back to an object still being "
                    f"built (self-referential / cyclic schemas are not supported)."
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
        required = (
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
                    f"must be a schema object."
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
                annotation = annotation | None
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
        model = create_model(model_name, __base__=_JsonSchemaModel, **fields)
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
    if not isinstance(schema, Mapping):
        raise ValueError(
            f"model_from_json_schema expects a mapping schema, got {type(schema).__name__}."
        )
    return _Converter(cast("JsonDict", dict(schema))).convert(name)
