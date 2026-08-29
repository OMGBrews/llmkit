"""Public, provider-neutral types for one tool-enabled completion turn.

The two *wire shapes* a tool loop appends to a prompt —
:class:`~llmkit.AssistantToolMessage` and :class:`~llmkit.ToolResultMessage` —
are defined in :mod:`llmkit._types` beside :data:`~llmkit._types.ChatMessage`,
which names them, and imported back here (they remain importable from either
module, and from the package top level). Everything else a tool turn needs —
the definition, the parsed call, the result — lives here, because it is
pydantic-backed and the leaf module deliberately is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, override

from pydantic import BaseModel

from llmkit._types import AssistantToolMessage, ToolResultMessage


@dataclass(frozen=True)
class ToolName:
    """Select one named tool, distinct from the ``\"required\"`` mode."""

    value: str


type ToolChoice = Literal["auto", "none", "required"] | ToolName


@dataclass(frozen=True)
class ToolDefinition:
    """A callable function offered to a model, described with JSON Schema."""

    name: str
    description: str
    parameters: dict[str, object]
    model: type[BaseModel] | None = None

    @classmethod
    def from_model(
        cls, name: str, model: type[BaseModel], description: str | None = None
    ) -> ToolDefinition:
        """Build a definition whose arguments are validated by *model*."""
        return cls(
            name, description or (model.__doc__ or "").strip(), model.model_json_schema(), model
        )

    def to_litellm(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """One requested invocation returned by the model."""

    id: str
    name: str
    arguments_raw: str
    arguments: dict[str, object]
    validated: BaseModel | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments_raw},
        }


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class ToolCallResult:
    """The text and/or requested calls from a single completion turn."""

    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str | None
    usage: TokenUsage

    def to_message(self) -> AssistantToolMessage:
        return {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [call.to_wire() for call in self.tool_calls],
        }

    def to_log_dict(self) -> dict[str, object]:
        return {"text": self.text, "tool_calls": [call.to_wire() for call in self.tool_calls]}


@dataclass(frozen=True)
class ToolComposeResult[T: BaseModel](ToolCallResult):
    """A tool turn that may instead contain a validated final answer.

    ``parsed`` is ``None`` when the model requested tools.  When no tools were
    requested, it is the caller's validated ``output_schema`` instance.
    """

    parsed: T | None

    @override
    def to_log_dict(self) -> dict[str, object]:
        return {**super().to_log_dict(), "parsed": self.parsed}


def tool_result_message(tool_call_id: str, content: str) -> ToolResultMessage:
    """Construct a history message containing an application's tool result."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
