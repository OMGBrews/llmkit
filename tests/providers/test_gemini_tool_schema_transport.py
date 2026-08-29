"""The tool schemas llmkit emits survive LiteLLM's Gemini transform intact.

``ToolDefinition.to_litellm()`` sends ``parameters`` verbatim, so a schema built
by ``from_model()`` still carries ``$defs``, ``$ref`` and ``additionalProperties``
when it leaves llmkit — none of which Gemini's ``Schema`` subset accepts. The
transport closes that gap: on both Gemini routes (``vertex_ai/`` and
``gemini/``) LiteLLM maps every function through
``VertexGeminiConfig._map_function``, which strips ``additionalProperties`` and
runs each ``parameters`` object through ``_build_vertex_schema`` (popping
``$defs``, inlining ``$ref`` targets, filtering unsupported keywords).

That means **a consumer needs no schema pre-processing of their own** — the
guarantee stated in the README's tool-calling section. This module is its
tripwire: LiteLLM is a floor-pinned dependency (``litellm>=1.95.0``) and CI
resolves both the floor and the newest release, so a transport version that
stopped normalising would otherwise reach a consumer as provider-side 400s with
nothing in our suite going red.

**These tests import LiteLLM private API on purpose.**
``VertexGeminiConfig._map_function`` is the real tool path and has no public
equivalent. If a LiteLLM upgrade *renames* it, that is a rename to follow here —
an ``ImportError`` or ``AttributeError`` from this module is not evidence that
the guarantee broke. A test that runs and *fails* its assertions is.

The complementary live proof is ``test_vertex_tool_schema_roundtrip_live`` in
``tests/integration/test_live_providers.py``: this module pins the transform,
that one pins that the transform's output is what Vertex actually accepts.
"""

from __future__ import annotations

import enum
from typing import cast

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from pydantic import BaseModel

from llmkit import ToolDefinition


class _Unit(enum.StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class _Location(BaseModel):
    city: str
    country: str


class _ForecastArgs(BaseModel):
    """A nested model plus a string enum — the shape that carries $defs/$ref."""

    where: _Location
    unit: _Unit
    days: int


def _map_to_gemini(*tools: ToolDefinition) -> list[dict[str, object]]:
    """Return the ``function_declarations`` LiteLLM would put on the wire."""
    # raw-llm: `_map_function` is the real tool path and has no public
    # equivalent; its signature is loosely typed (`List[dict]` in, a TypedDict
    # with no required keys out), so the boundary is cast once, here. A rename
    # upstream is a rename to follow — see the module docstring.
    mapped = cast(
        "list[dict[str, object]]",
        VertexGeminiConfig()._map_function(  # pyright: ignore[reportUnknownMemberType]
            [tool.to_litellm() for tool in tools], {}
        ),
    )
    assert len(mapped) == 1, mapped
    return cast("list[dict[str, object]]", mapped[0]["function_declarations"])


def _walk(node: object) -> list[dict[str, object]]:
    """Every mapping nested anywhere inside *node*, including *node* itself."""
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        return [mapping, *[found for value in mapping.values() for found in _walk(value)]]
    if isinstance(node, list):
        return [found for item in cast("list[object]", node) for found in _walk(item)]
    return []


def test_from_model_schema_leaves_llmkit_with_refs_and_defs() -> None:
    """The premise: llmkit itself does not normalise, so the transport must."""
    definition = ToolDefinition.from_model("forecast", _ForecastArgs, "Get a forecast")
    assert "$defs" in definition.parameters
    properties = cast("dict[str, object]", definition.parameters["properties"])
    assert properties["where"] == {"$ref": "#/$defs/_Location"}


def test_nested_model_and_enum_reach_gemini_in_its_accepted_subset() -> None:
    """``$defs`` popped, ``$ref`` inlined, a string enum preserved in place."""
    definition = ToolDefinition.from_model("forecast", _ForecastArgs, "Get a forecast")

    (declaration,) = _map_to_gemini(definition)
    parameters = cast("dict[str, object]", declaration["parameters"])
    properties = cast("dict[str, object]", parameters["properties"])

    assert declaration["name"] == "forecast"
    assert all("$defs" not in mapping for mapping in _walk(parameters))
    assert all("$ref" not in mapping for mapping in _walk(parameters))

    where = cast("dict[str, object]", properties["where"])
    assert where["type"] == "object"
    assert set(cast("dict[str, object]", where["properties"])) == {"city", "country"}

    unit = cast("dict[str, object]", properties["unit"])
    assert unit["type"] == "string"
    assert unit["enum"] == ["celsius", "fahrenheit"]


def test_additional_properties_is_stripped_from_the_wire_schema() -> None:
    """``additionalProperties`` is a hard 400 on Gemini; nothing may carry it."""
    definition = ToolDefinition(
        "strict",
        "A hand-written strict schema",
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    (declaration,) = _map_to_gemini(definition)

    assert all("additionalProperties" not in mapping for mapping in _walk(declaration))


def test_no_argument_tool_is_sent_as_a_bare_object_declaration() -> None:
    """``{"type": "object", "properties": {}}`` is the portable no-arg schema.

    Documented in the README so a consumer does not have to guess: LiteLLM
    reduces the empty ``properties`` map to a bare ``OBJECT`` declaration, which
    is what Gemini accepts for a tool that takes no arguments.
    """
    definition = ToolDefinition("ping", "Takes no arguments", {"type": "object", "properties": {}})

    (declaration,) = _map_to_gemini(definition)

    assert declaration["parameters"] == {"type": "object"}


def test_a_tool_batch_maps_as_one_declaration_list() -> None:
    """Both shapes travel together, in order, in a single tool entry."""
    forecast = ToolDefinition.from_model("forecast", _ForecastArgs, "Get a forecast")
    ping = ToolDefinition("ping", "Takes no arguments", {"type": "object", "properties": {}})

    declarations = _map_to_gemini(forecast, ping)

    assert [declaration["name"] for declaration in declarations] == ["forecast", "ping"]
