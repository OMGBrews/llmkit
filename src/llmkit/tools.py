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

from dataclasses import dataclass, field
from typing import Literal, override

from pydantic import BaseModel

from llmkit._types import AssistantToolMessage, ToolResultMessage
from llmkit.exceptions import ToolArgumentError


@dataclass(frozen=True)
class ToolName:
    """Select one named tool, distinct from the ``\"required\"`` mode."""

    value: str


type ToolChoice = Literal["auto", "none", "required"] | ToolName


@dataclass(frozen=True)
class ToolDefinition:
    """A callable function offered to a model, described with JSON Schema.

    ``parameters`` is forwarded to the transport **verbatim** — llmkit does not
    rewrite it — and it does not need to be pre-processed for any supported
    provider. LiteLLM normalises per provider on the way to the wire: on the
    Gemini routes (``vertex_ai/``, ``gemini/``) it strips ``additionalProperties``,
    pops ``$defs``, inlines ``$ref`` targets, and filters keywords outside
    Gemini's ``Schema`` subset, so schemas that API rejects when posted raw are
    accepted through llmkit. Writing a normalisation layer on top of this
    duplicates it and will drift from it.

    A tool that takes **no arguments** is ``{"type": "object", "properties": {}}``
    — the portable spelling, sent to Gemini as a bare ``OBJECT``. No dummy
    parameter is needed.

    The guarantee is delivery in the provider's accepted subset, not lossless
    translation: a construct the subset cannot express is *dropped, not errored*.
    Notably Gemini accepts ``enum`` only on string-typed fields, so an enum on an
    integer field is silently discarded and the model sees an unconstrained
    number — :meth:`from_model` still validates the returned arguments locally.
    ``tests/providers/test_gemini_tool_schema_transport.py`` pins the transform;
    ``test_vertex_tool_schema_roundtrip_live`` pins that Vertex accepts its output.
    """

    name: str
    description: str
    parameters: dict[str, object]
    model: type[BaseModel] | None = None

    @classmethod
    def from_model(
        cls, name: str, model: type[BaseModel], description: str | None = None
    ) -> ToolDefinition:
        """Build a definition whose arguments are validated by *model*.

        A nested model or an enum renders ``$defs``/``$ref`` here, which is
        correct: the transport inlines them per provider (see the class
        docstring). Hand the result straight to ``tool_llm_call``.
        """
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
    """The text and/or requested calls from a single completion turn.

    ``tool_calls`` holds every call that parsed and validated. When a turn
    requests several calls in parallel and only *some* of them are malformed,
    the well-formed ones survive here and each failure is reported on
    ``invalid_calls`` as the :class:`~llmkit.ToolArgumentError` it raised —
    nothing has executed yet, so discarding the whole round would only be a
    lossy re-ask. A turn in which **every** requested call is malformed still
    raises ``ToolArgumentError``, keeping the whole-round re-ask contract
    (``RetryPolicy.validation_max_attempts``) exactly as it was.

    ``to_message()`` is built from ``tool_calls`` alone, so the assistant turn
    it produces names only the calls you can actually answer: feed a
    :func:`~llmkit.tool_result_message` for each and the history stays
    consistent, with no dangling ``tool_call_id`` for the provider to reject.
    The dropped calls are yours to log, count, or re-prompt about in your own
    words; they are also recorded in the call's log entry.
    """

    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str | None
    usage: TokenUsage
    invalid_calls: list[ToolArgumentError] = field(default_factory=list)

    def to_message(self) -> AssistantToolMessage:
        return {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [call.to_wire() for call in self.tool_calls],
        }

    def to_log_dict(self) -> dict[str, object]:
        logged: dict[str, object] = {
            "text": self.text,
            "tool_calls": [call.to_wire() for call in self.tool_calls],
        }
        # Only present when something was actually dropped, so the log shape of
        # an ordinary turn is unchanged — and a turn that silently lost a call
        # can never look like a clean one in the record.
        if self.invalid_calls:
            logged["invalid_calls"] = [
                {
                    "name": error.tool_name,
                    "id": error.call_id,
                    "arguments": error.arguments_raw,
                    "reason": str(error),
                }
                for error in self.invalid_calls
            ]
        return logged


@dataclass(frozen=True)
class ToolComposeResult[T: BaseModel](ToolCallResult):
    """A tool turn that may instead contain a validated final answer.

    ``parsed`` is ``None`` when the model requested tools.  When no tools were
    requested, it is the caller's validated ``output_schema`` instance.
    """

    parsed: T | None = None

    @override
    def to_log_dict(self) -> dict[str, object]:
        return {**super().to_log_dict(), "parsed": self.parsed}


def tool_result_message(tool_call_id: str, content: str) -> ToolResultMessage:
    """Construct a history message containing an application's tool result."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
