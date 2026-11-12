"""Shared call-surface types: messages, effort levels and transport usage.

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

from dataclasses import dataclass
from typing import Literal, TypedDict


class AssistantToolMessage(TypedDict):
    """An assistant turn that requested one or more tool calls."""

    role: Literal["assistant"]
    tool_calls: list[dict[str, object]]
    content: str | None


class _ToolResultMessageFields(TypedDict):
    """The keys every tool-result message carries; see :class:`ToolResultMessage`."""

    role: Literal["tool"]
    tool_call_id: str
    content: str


class ToolResultMessage(_ToolResultMessageFields, total=False):
    """The application-supplied result of one requested tool call.

    ``is_error`` is the optional fourth key: set it when the tool failed, so
    the model is told the difference between a result and a failure instead of
    inferring it from prose. Build the message with
    :func:`~llmkit.tool_result_message` rather than by hand — the key is
    omitted entirely when the result is fine, so an ordinary tool-result
    message is exactly the three-key dict it always was.

    It is llmkit's own field, not a wire field. No provider accepts it on the
    OpenAI-normalized ``{"role": "tool"}`` shape: LiteLLM's Anthropic
    translation builds its ``tool_result`` block from the id and the content
    alone (measured against litellm 1.95.0 —
    ``convert_to_anthropic_tool_result``, whose own comment records that the
    OpenAI shape cannot express the distinction), and an OpenAI-compatible
    route would forward the unknown key to a provider that may reject it. So
    the transport renders it into the content as
    :data:`TOOL_ERROR_PREFIX` and drops the key, which reaches every provider
    identically; the flag itself stays on the message the caller holds, and in
    the ``prompt`` the log records, so the provenance keeps the distinction
    structurally even though the wire cannot.
    """

    is_error: bool


#: What :data:`ToolResultMessage`'s ``is_error`` becomes on the wire: the
#: content is sent prefixed with this marker. One convention, owned by llmkit
#: and identical on every provider, so a host does not invent a private JSON
#: error envelope per application and models do not diverge in how they
#: recover. Exported so a caller can recognise the marker in a transcript or
#: strip it before display.
TOOL_ERROR_PREFIX = "ERROR: "


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
#: parameter, plus two portable values llmkit translates per route:
#:
#: - ``"disable"`` turns thinking off, or raises ReasoningEffortUnsupportedError
#:   before transport on a model that cannot stop thinking. It never substitutes
#:   a lower effort that still reasons.
#: - ``"lowest"`` asks for the least reasoning the route supports: off where the
#:   model can turn thinking off, else its lowest effort.
#:
#: ``"low"`` / ``"medium"`` / ``"high"`` scale thinking. Per route:
#:
#: - Google AI Studio and Vertex Gemini: ``disable`` raises on Gemini 3 and later
#:   and on gemini-2.5-pro, else thinkingBudget 0; ``lowest`` sends LiteLLM's
#:   ``disable`` (lowest thinking level, no thought summaries), or ``minimal``
#:   on gemini-2.5-pro. Vertex partner models forward the value unchanged.
#: - OpenRouter: ``disable`` raises on the same Gemini models, else ``none``;
#:   ``lowest`` sends ``minimal`` there, else ``none``.
#: - Anthropic and Bedrock: ``disable`` sends ``none`` and raises on models LiteLLM
#:   flags as always-on (unknown metadata permits it); ``lowest`` sends ``low``
#:   there, else ``none``.
#: - OpenAI: ``disable`` sends ``none`` where capabilities establish support,
#:   omits the parameter for known non-reasoning models, and raises otherwise;
#:   ``lowest`` follows the same branches but sends ``minimal`` or ``low`` where
#:   the metadata flags one and off is unsupported.
#: - Cerebras: ``disable`` raises on gpt-oss, else ``none``; ``lowest`` sends
#:   ``low`` on gpt-oss, else ``none``.
#: - DeepSeek: both send ``none``, the one value LiteLLM turns into thinking off.
#: - Ollama: ``disable`` raises on gpt-oss, else sends the server's off control,
#:   which a model that cannot stop thinking may ignore; ``lowest`` sends ``low``
#:   on gpt-oss, else the off control.
#:
#: The trailing ``| str`` is a deliberate escape hatch — providers accept values
#: outside this set (e.g. OpenAI's ``"minimal"``), and llmkit forwards them
#: without validating.
#:
#: This alias is **advisory, not enforcing**: under the type checker
#: ``Literal[...] | str`` widens to ``str``, so it names and documents the
#: canonical set and drives editor autocomplete but does *not* statically reject
#: a typo. A closed ``Literal`` (no ``| str``) would reject a typo, but it would
#: also reject the provider-specific values the non-validating design
#: intentionally permits — so the open tail is the right trade for a thin,
#: non-validating wrapper.
type ReasoningEffort = Literal["disable", "lowest", "low", "medium", "high"] | str

# ``Message`` deliberately stays closed for existing callers. Tool loops use
# this union, which retains ordinary messages and adds the OpenAI-normalized
# assistant-call / tool-result wire shapes.
type ChatMessage = Message | AssistantToolMessage | ToolResultMessage


@dataclass(frozen=True)
class WireUsage:
    """Token counts as the provider reported them; each ``None`` when absent."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None

    def counts(self) -> tuple[int | None, int | None, int | None]:
        """The ``(prompt, completion, total)`` triple the call log records."""
        return self.prompt_tokens, self.completion_tokens, self.total_tokens

    def debit(self) -> int | None:
        """Tokens for the tokens-per-minute limiter to debit.

        ``total_tokens`` when reported, else the sum of the two halves with a
        missing one counted as 0, else ``None``. Best-effort, like all TPM
        accounting.
        """
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is not None or self.completion_tokens is not None:
            return (self.prompt_tokens or 0) + (self.completion_tokens or 0)
        return None
