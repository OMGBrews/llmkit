"""llmkit's own transport boundary types.

The request, response and stream-chunk shapes that pass between the call
transports in :mod:`llmkit._litellm` and whatever answers them. The LiteLLM
adapter is the only code that builds these from LiteLLM objects. Everything
after it reads only these types. This module imports nothing from LiteLLM,
so the public testing transport (:mod:`llmkit.testing`) can build the same
shapes without loading it.

They are private: :class:`WireRequest` carries the provider-routed request
kwargs, credentials included. The public request a test fake receives is a
separate, credential-free type, mapped one way from :class:`CallContext`.
This module also holds the slot through which :mod:`llmkit.testing`
installs a :class:`TransportDouble`.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.tools import ToolChoice, ToolDefinition

CallKind = Literal["structured", "text", "tool", "text_stream", "tool_stream"]


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


@dataclass(frozen=True)
class WireFunction:
    """A tool call's function half; a field that did not arrive as ``str`` is ``None``."""

    name: str | None
    arguments: str | None


@dataclass(frozen=True)
class WireToolCall:
    """One complete tool call.

    The ``.id`` / ``.function.name`` / ``.function.arguments`` shape is the one
    :func:`~llmkit.calls._shared.parse_tool_calls` reads, so buffered, streamed
    and faked calls share one parser and one set of malformed-call errors.
    """

    id: str | None
    function: WireFunction


@dataclass(frozen=True)
class WireResponse:
    """One buffered completion.

    ``has_choice`` is false for a degenerate response with no choices, which
    the text and tool lanes treat as an empty answer. ``text`` is ``None`` when
    the message carried no text content, distinct from an empty string.
    """

    has_choice: bool = True
    text: str | None = None
    tool_calls: tuple[WireToolCall, ...] = ()
    finish_reason: str | None = None
    usage: WireUsage = field(default_factory=WireUsage)
    cost: float | None = None


@dataclass(frozen=True)
class WireToolFragment:
    """A piece of one streamed tool call, joined to its siblings by ``index``."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass(frozen=True)
class WireChunk:
    """One stream frame.

    ``usage`` is ``None`` when the frame carried no usage at all; a frame that
    carried a usage object reports it even when every count is missing.
    """

    text: str | None = None
    tool_fragments: tuple[WireToolFragment, ...] = ()
    finish_reason: str | None = None
    usage: WireUsage | None = None
    cost: float | None = None


@dataclass(frozen=True)
class WireStreamEnd:
    """What the stream reported as a whole once its frames ran out.

    The last item a wire stream yields. A provider stream can expose usage and
    cost on the stream object itself rather than on any frame; the transports
    fall back to these values.
    """

    usage: WireUsage = field(default_factory=WireUsage)
    cost: float | None = None


@dataclass(frozen=True)
class CallContext:
    """Who is asking, and for what, on one provider round trip.

    Built by each transport from what the call layer passed it, and never
    sent to LiteLLM. It is the source of the public request a test fake
    receives, so it holds no credentials or provider-native kwargs.
    ``feature``, ``label``, ``call_id`` and ``attempt`` are ``None`` when a
    transport is called directly, without the call layer's identity.
    """

    kind: CallKind
    prompt: str | Sequence[ChatMessage]
    messages: Sequence[ChatMessage]
    model: str
    provider: str
    temperature: float | None
    max_tokens: int | None
    reasoning_effort: ReasoningEffort | None
    feature: str | None = None
    label: str | None = None
    call_id: str | None = None
    attempt: int | None = None
    output_schema: type[BaseModel] | None = None
    tools: Sequence[ToolDefinition] = ()
    tool_choice: ToolChoice | None = None
    repair: bool = False
    timeout: float | None = None


@dataclass(frozen=True)
class WireRequest:
    """One provider round trip.

    ``kwargs`` is the complete request as LiteLLM receives it: model, messages,
    tools, sampling settings and the provider's credential and native kwargs.
    ``call`` is the request's identity and inputs, for an installed double.
    """

    kwargs: dict[str, object]
    call: CallContext | None = None


class TransportDouble(Protocol):
    """A stand-in for the provider round trip, installed by :mod:`llmkit.testing`.

    The transports consult it inside the rate-limiter slot, so whatever it
    returns or raises passes through the same limiter outcome, circuit breaker
    and retry handling as a real provider's answer.
    """

    async def respond(self, call: CallContext) -> WireResponse:
        """Answer one buffered round trip."""
        ...

    def stream(self, call: CallContext) -> AsyncIterator[WireChunk | WireStreamEnd]:
        """Answer one streamed round trip, ending with a :class:`WireStreamEnd`."""
        ...


# The installed doubles, innermost last. One process-wide slot by design: a
# ContextVar would miss threads the host starts itself, such as a web server
# under test. The lock guards the list against installs from several threads.
_doubles: list[TransportDouble] = []
_doubles_lock = threading.Lock()


def push_double(double: TransportDouble) -> None:
    """Install *double* as the active one, above any already installed."""
    with _doubles_lock:
        _doubles.append(double)


def remove_double(double: TransportDouble) -> None:
    """Uninstall *double*, wherever it sits; a no-op when it is not installed."""
    with _doubles_lock:
        for position in range(len(_doubles) - 1, -1, -1):
            if _doubles[position] is double:
                del _doubles[position]
                return


def active_double() -> TransportDouble | None:
    """The most recently installed double still in place, or ``None``."""
    with _doubles_lock:
        return _doubles[-1] if _doubles else None
