"""Shared public call-surface types: the message shapes and the effort levels.

A true leaf module — it imports nothing from ``llmkit`` and nothing outside the
standard library — so every layer that names these types can do so without an
import cycle: the providers (:class:`~llmkit.LLMClientConfig`, the provider
protocol), the logging record (:class:`~llmkit.LLMCallRecord`), the transport,
the tool types in :mod:`llmkit.tools`, and the public call functions. The names
are re-exported from the package top level (see :mod:`llmkit.__init__`), so a
caller can annotate their own prompt builders and effort constants against
``llmkit.Message`` / ``llmkit.ReasoningEffort``.

The two tool *wire shapes* live here rather than beside the richer tool types
in :mod:`llmkit.tools` for one structural reason: :data:`ChatMessage` — the
prompt type every layer above depends on — names them, so hosting them in
``tools`` would make this module import ``tools`` and drag pydantic into every
importer of the leaf. The dependency runs the other way instead: ``tools``
imports these two back for :meth:`~llmkit.ToolCallResult.to_message` and
:func:`~llmkit.tool_result_message`, and both names stay importable from
``llmkit.tools`` as well as from the package top level.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class AssistantToolMessage(TypedDict):
    """An assistant turn that requested one or more tool calls."""

    role: Literal["assistant"]
    tool_calls: list[dict[str, object]]
    content: str | None


class ToolResultMessage(TypedDict):
    """The application-supplied result of one requested tool call."""

    role: Literal["tool"]
    tool_call_id: str
    content: str


class Message(TypedDict):
    """One chat message in a ``prompt``: a ``role`` and its ``content``.

    The list form of a ``prompt`` argument is a list of these, replacing the
    raw ``dict[str, str]`` wire shape that used to leak through the call
    surface. ``role`` is the closed set of prompting roles, so an unknown key
    (``{"roel": ...}``) or a mistyped role is a type error — the loose-in-keys
    half of the old type. ``content`` is either a plain string or the
    multimodal content-parts form LiteLLM accepts — a list of part dicts
    (``{"type": "text", ...}``, ``{"type": "image_url", ...}``) — which the old
    ``str``-only content type wrongly rejected. The parts stay
    ``dict[str, object]`` rather than a modelled union: fully typing the
    provider content-part taxonomy would re-import the very wire format this
    type exists to hide, and llmkit forwards the parts verbatim to LiteLLM.
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, object]]


#: The reasoning/thinking effort levels, mirroring LiteLLM's ``reasoning_effort``
#: parameter: ``"disable"`` turns provider thinking off; ``"low"`` / ``"medium"``
#: / ``"high"`` scale it. The trailing ``| str`` is a deliberate escape hatch —
#: providers accept values outside this set (e.g. OpenAI's ``"minimal"``), and
#: llmkit forwards the value without validating it, translating only where a
#: provider needs a native request control (for example OpenRouter).
#:
#: This alias is **advisory, not enforcing**: under the type checker
#: ``Literal[...] | str`` widens to ``str``, so it names and documents the
#: canonical set and drives editor autocomplete but does *not* statically reject
#: a typo. A closed ``Literal`` (no ``| str``) would reject a typo, but it would
#: also reject the provider-specific values the non-validating design
#: intentionally permits — so the open tail is the right trade for a thin,
#: non-validating wrapper.
type ReasoningEffort = Literal["disable", "low", "medium", "high"] | str

# ``Message`` deliberately stays closed for existing callers. Tool loops use
# this union, which retains ordinary messages and adds the OpenAI-normalized
# assistant-call / tool-result wire shapes.
type ChatMessage = Message | AssistantToolMessage | ToolResultMessage
