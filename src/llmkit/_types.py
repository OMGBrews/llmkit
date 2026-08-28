"""Shared public call-surface types: the message shape and the effort levels.

A leaf module — it imports nothing from ``llmkit`` — so every layer that names
these types can do so without an import cycle: the providers
(:class:`~llmkit.LLMClientConfig`, the provider protocol), the logging record
(:class:`~llmkit.LLMCallRecord`), the transport, and the public call functions.
Both names are re-exported from the package top level (see
:mod:`llmkit.__init__`), so a caller can annotate their own prompt builders and
effort constants against ``llmkit.Message`` / ``llmkit.ReasoningEffort``.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from llmkit.tools import AssistantToolMessage, ToolResultMessage


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
