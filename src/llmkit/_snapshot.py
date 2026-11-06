"""Strict ownership copies for call inputs, independent of log delivery.

Only plain data is copied. In particular, this never calls user copy hooks or
falls back to sharing opaque objects. Outer sequences are materialized by the
public-input helpers; nested containers must be exact built-ins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from llmkit._types import ChatMessage
from llmkit.tools import ToolDefinition


def snapshot_data[T](value: T, path: str = "input") -> T:
    """Copy a plain-data tree, rejecting unsupported values and cycles by path.

    Repeated references are copied independently. The active ancestor set
    distinguishes them from cycles; it is not an alias-preserving memo.
    Callers must keep their data stable during this synchronous traversal.
    """
    active: set[int] = set()

    def visit(item: object, location: str) -> object:
        kind = type(item)
        if item is None or kind is str or kind is int or kind is float or kind is bool:
            return item
        if kind is not dict and kind is not list and kind is not tuple:
            raise TypeError(f"{location}: expected plain data (dict, list, tuple or scalar)")
        identity = id(item)
        if identity in active:
            raise TypeError(f"{location}: cyclic input is not supported")
        active.add(identity)
        try:
            if kind is dict:
                result: dict[str, object] = {}
                for key, child in cast("dict[object, object]", item).items():
                    if type(key) is not str:
                        raise TypeError(f"{location}: dictionary keys must be plain strings")
                    result[key] = visit(child, f"{location}[{key!r}]")
                return result
            children = [
                visit(child, f"{location}[{index}]")
                for index, child in enumerate(cast("list[object] | tuple[object, ...]", item))
            ]
            return tuple(children) if kind is tuple else children
        finally:
            active.remove(identity)

    try:
        return cast("T", visit(value, path))
    except RecursionError:
        raise TypeError(f"{path}: nesting exceeds the Python recursion limit") from None


def snapshot_prompt(prompt: str | Sequence[ChatMessage]) -> str | list[ChatMessage]:
    """Own a prompt, materializing its outer message sequence exactly once."""
    if isinstance(prompt, str):
        return snapshot_data(prompt, "prompt")
    return [snapshot_data(message, f"prompt[{i}]") for i, message in enumerate(prompt)]


def snapshot_tools(tools: Sequence[ToolDefinition]) -> list[ToolDefinition]:
    """Own offered definitions and schemas; the validating model stays borrowed."""
    return [
        ToolDefinition(
            name=snapshot_data(tool.name, f"tools[{i}].name"),
            description=snapshot_data(tool.description, f"tools[{i}].description"),
            parameters=snapshot_data(tool.parameters, f"tools[{i}].parameters"),
            model=tool.model,
        )
        for i, tool in enumerate(tools)
    ]
