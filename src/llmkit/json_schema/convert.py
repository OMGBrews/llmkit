"""Build schema-backed object models; import the standards engine only on use."""

from __future__ import annotations

import copy
import keyword
import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import JsonValue

from llmkit.json_schema.emitted import JsonDict, JsonSchemaModel

if TYPE_CHECKING:
    from referencing import Resource


class _Resolved(Protocol):
    @property
    def contents(self) -> object: ...
    @property
    def resolver(self) -> _Resolver: ...


class _Resolver(Protocol):
    def lookup(self, ref: str) -> _Resolved: ...
    def in_subresource(self, subresource: Resource[JsonValue]) -> _Resolver: ...


def _safe_model_name(raw: object) -> str:
    if not isinstance(raw, str):
        return "JsonSchemaModel"
    name = re.sub(r"\W+", "", raw.strip().replace(" ", "_").replace("-", "_"))
    if not name:
        return "JsonSchemaModel"
    if name[0].isdigit():
        name = "_" + name
    return name + "_" if keyword.iskeyword(name) else name


def _children(node: JsonDict, dialect: str) -> Iterator[tuple[tuple[str, ...], JsonDict]]:
    """Visit the dialect's schema positions, never annotation payloads."""
    draft7 = dialect == "http://json-schema.org/draft-07/schema#"
    draft2020 = dialect == "https://json-schema.org/draft/2020-12/schema"
    maps = {"properties", "patternProperties", "definitions"}
    singles = {
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "additionalProperties",
        "items",
    }
    arrays = {"allOf", "anyOf", "oneOf"}
    if not draft7:
        maps.update({"$defs", "dependentSchemas"})
        singles.update({"unevaluatedItems", "unevaluatedProperties", "contentSchema"})
    if draft2020:
        arrays.add("prefixItems")
    else:
        singles.add("additionalItems")
    if draft7:
        maps.add("dependencies")
    for key, value in node.items():
        if key in maps and isinstance(value, dict):
            for name, child in value.items():
                if isinstance(child, dict):
                    yield (key, name), child
        elif key in singles and isinstance(value, dict):
            yield (key,), value
        elif (key in arrays or (key == "items" and not draft2020)) and isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    yield (key, str(index)), child


def model_from_json_schema(
    schema: Mapping[str, object], *, name: str | None = None, assert_formats: bool = False
) -> type[JsonSchemaModel]:
    """Build once, reuse: validate JSON objects and read values via ``.root``.

    Draft-07, 2019-09 and 2020-12 are supported (default: 2020-12).
    References must resolve offline and be acyclic. Formats are annotations
    unless ``assert_formats=True``; unavailable assertion formats then fail.
    """
    from jsonschema import (
        Draft7Validator,
        Draft201909Validator,
        Draft202012Validator,
        FormatChecker,
    )
    from jsonschema.exceptions import SchemaError, ValidationError
    from referencing import Registry, Resource
    from referencing.exceptions import Unresolvable
    from referencing.jsonschema import DRAFT7, DRAFT201909, DRAFT202012, SchemaRegistry

    from llmkit.exceptions import SchemaEngineError

    snapshot = cast("JsonDict", copy.deepcopy(dict(schema)))
    dialect = snapshot.get("$schema", "https://json-schema.org/draft/2020-12/schema")
    choices = {
        "http://json-schema.org/draft-07/schema#": (Draft7Validator, DRAFT7),
        "https://json-schema.org/draft/2019-09/schema": (Draft201909Validator, DRAFT201909),
        "https://json-schema.org/draft/2020-12/schema": (Draft202012Validator, DRAFT202012),
    }
    if not isinstance(dialect, str) or dialect not in choices:
        raise ValueError(f"Unsupported $schema at /$schema; supported: {', '.join(choices)}")
    validator_type, specification = choices[dialect]
    try:
        validator_type.check_schema(snapshot)
    except SchemaError as exc:
        raise ValueError(
            f"Invalid schema at /{'/'.join(map(str, exc.absolute_path))}: {exc.message}"
        ) from exc
    checker = FormatChecker() if assert_formats else None
    resource: Resource[JsonValue] = Resource.from_contents(
        snapshot, default_specification=specification
    )
    base = resource.id() or "urn:llmkit:root"
    registry = Registry[JsonValue]().with_resource(base, resource).crawl()
    resolver = cast("_Resolver", registry.resolver(base))

    refkeys = (
        ("$ref", "$dynamicRef")
        if specification is DRAFT202012
        else ("$ref", "$recursiveRef")
        if specification is DRAFT201909
        else ("$ref",)
    )

    def inspect(
        node: JsonDict, current: _Resolver, path: str, active: frozenset[int], *, enter: bool = True
    ) -> None:
        # Resolver's generic type varies with schema contents; keep its public
        # methods typed through the root resolver rather than private imports.
        resolved = current
        if id(node) in active:
            raise ValueError(f"Recursive reference at {path or '/'}")
        active = active | {id(node)}
        try:
            validator_type.check_schema(node)
        except SchemaError as exc:
            raise ValueError(
                f"Invalid schema at {path}/{'/'.join(map(str, exc.absolute_path))}: {exc.message}"
            ) from exc
        if "$schema" in node and node["$schema"] != dialect:
            raise ValueError(
                f"Unsupported dialect change at {path}/$schema; root dialect is {dialect}"
            )
        subresource = Resource.from_contents(node, default_specification=specification)
        resolved = resolved.in_subresource(subresource) if enter else resolved
        vocab = node.get("$vocabulary")
        if specification is not DRAFT7 and isinstance(vocab, dict):
            known = cast("dict[str, object]", validator_type.META_SCHEMA).get("$vocabulary", {})
            known = cast("dict[str, object]", known)
            for uri, required in vocab.items():
                if required is True and uri not in known:
                    raise ValueError(f"Unsupported required vocabulary at {path}/$vocabulary/{uri}")
        if (
            checker is not None
            and "format" in node
            and cast("str", node["format"]) not in checker.checkers
        ):
            raise ValueError(f"Unsupported format at {path}/format: {node['format']}")
        for refkey in refkeys:
            ref = node.get(refkey)
            if isinstance(ref, str):
                try:
                    target = resolved.lookup(ref)
                except Unresolvable as exc:
                    raise ValueError(f"Unresolvable reference at {path}/{refkey}: {ref}") from exc
                contents = cast("JsonValue", target.contents)
                if not isinstance(contents, (dict, bool)):
                    raise ValueError(f"Reference target is not a schema at {path}/{refkey}: {ref}")
                if isinstance(contents, dict):
                    inspect(
                        contents,
                        target.resolver,
                        f"{path}/{refkey}",
                        active,
                        enter=False,
                    )
        for suffix, child in _children(node, dialect):
            inspect(
                child,
                resolved,
                path + "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in suffix),
                active,
            )

    inspect(snapshot, resolver, "", frozenset(), enter=False)
    root = snapshot
    root_resolver = resolver
    while isinstance(root.get("$ref"), str):
        target = root_resolver.lookup(cast("str", root["$ref"]))
        if not isinstance(target.contents, dict):
            raise ValueError("Unsupported root schema: top level must be an object")
        root = cast("JsonDict", target.contents)
        root_resolver = target.resolver
    if root.get("type") != "object":
        raise ValueError("Unsupported root schema: top level must be an object")
    validator = validator_type(
        snapshot, registry=cast("SchemaRegistry", registry), format_checker=checker
    )

    def validate(value: object) -> None:
        try:
            validator.validate(value)  # pyright: ignore[reportUnknownMemberType]  # raw-jsonschema — typeshed omits validate parameter types
        except ValidationError as exc:
            raise ValueError(
                f"{exc.json_path}: {exc.message} (schema /{'/'.join(map(str, exc.absolute_schema_path))})"
            ) from exc
        except Exception as exc:
            raise SchemaEngineError(f"JSON Schema engine failed: {exc}") from exc

    def guidance(node: JsonDict, current: _Resolver, *, enter: bool = True) -> JsonDict:
        resolved = (
            current.in_subresource(
                Resource.from_contents(node, default_specification=specification)
            )
            if enter
            else current
        )
        result = copy.deepcopy(node)
        # Adapt only actual schema positions. Resolve each child with its own
        # base before merging referenced guidance, preserving scoped references.
        for suffix, child in _children(node, dialect):
            parts = suffix
            container: JsonValue = result
            for part in parts[:-1]:
                if isinstance(container, list):
                    container = cast("list[JsonValue]", container)[int(part)]
                else:
                    container = cast("JsonDict", container)[part]
            adapted = guidance(child, resolved)
            if isinstance(container, list):
                cast("list[JsonValue]", container)[int(parts[-1])] = adapted
            else:
                cast("JsonDict", container)[parts[-1]] = adapted
        for refkey in refkeys:
            ref = result.get(refkey)
            if isinstance(ref, str):
                target = resolved.lookup(ref)
                if isinstance(target.contents, dict):
                    referenced = guidance(
                        cast("JsonDict", target.contents), target.resolver, enter=False
                    )
                    referenced.update(
                        {
                            k: v
                            for k, v in result.items()
                            if k != refkey
                            and (
                                specification is not DRAFT7
                                or k
                                in {
                                    "title",
                                    "description",
                                    "$comment",
                                    "examples",
                                    "default",
                                    "deprecated",
                                    "readOnly",
                                    "writeOnly",
                                }
                            )
                        }
                    )
                    result = referenced
                elif isinstance(target.contents, bool):
                    # A false reference admits no instances; retain its semantics
                    # in guidance without leaving an unresolved reference behind.
                    _ = result.pop(refkey, None)
                    if not target.contents:
                        result["not"] = {}
        for key in ("$schema", "$defs", "definitions"):
            _ = result.pop(key, None)
        return result

    model = type(
        _safe_model_name(name if name is not None else snapshot.get("title")),
        (JsonSchemaModel,),
        {"__module__": __name__},
    )
    model.__llmkit_validate__ = staticmethod(validate)
    model.__llmkit_guidance__ = guidance(snapshot, resolver, enter=False)
    model.__llmkit_identity__ = {
        "schema": snapshot,
        "dialect": dialect,
        "assert_formats": assert_formats,
    }
    return model
