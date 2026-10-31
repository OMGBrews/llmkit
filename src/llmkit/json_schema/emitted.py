"""Lossless JSON values with independent validation and generation guidance."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Self, TypedDict, Unpack, cast, override

from pydantic import JsonValue, ModelWrapValidatorHandler, RootModel, model_validator
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode

type JsonDict = dict[str, JsonValue]


class _ModelJsonSchemaExtra(TypedDict, total=False):
    union_format: Literal["any_of", "primitive_type_array"]


def _finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers are not JSON values")
    if isinstance(value, dict):
        for child in cast("dict[str, object]", value).values():
            _finite(child)
    elif isinstance(value, list):
        for child in cast("list[object]", value):
            _finite(child)


def _unconfigured(_value: object) -> None:
    raise TypeError("Construct schema-backed models with model_from_json_schema")


class JsonSchemaModel(RootModel[dict[str, JsonValue]]):
    """An object value validated against its original JSON Schema snapshot.

    Access values through ``root``; default serialization preserves JSON values.
    Provider guidance is a separate, fresh document on every request.
    """

    __llmkit_validate__: ClassVar[Callable[[object], None]] = staticmethod(_unconfigured)
    __llmkit_guidance__: ClassVar[JsonDict] = {}
    __llmkit_identity__: ClassVar[JsonDict] = {}

    @model_validator(mode="wrap")
    @classmethod
    def _validate_contract(cls, value: object, handler: ModelWrapValidatorHandler[Self]) -> Self:
        raw = value.root if isinstance(value, JsonSchemaModel) else value
        _finite(raw)
        cls.__llmkit_validate__(raw)
        result = handler(value)
        cls.__llmkit_validate__(cast("object", json.loads(result.model_dump_json())))
        return result

    @classmethod
    def __llmkit_contract_identity__(cls) -> JsonDict:
        return copy.deepcopy(cls.__llmkit_identity__)

    @classmethod
    @override
    def model_json_schema(
        cls,
        by_alias: bool | None = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        **kwargs: Unpack[_ModelJsonSchemaExtra],
    ) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]  # raw-pydantic — matches the public Pydantic schema signature
        return copy.deepcopy(cls.__llmkit_guidance__)
