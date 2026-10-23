"""llmkit's own transport boundary types.

The request, response and stream-chunk shapes that pass between the call
transports in :mod:`llmkit._litellm` and whatever answers them. The LiteLLM
adapter is the only code that builds these from LiteLLM objects. Everything
after it reads only these types. This module imports nothing from LiteLLM,
so the public testing transport (:mod:`llmkit.testing`) can build the same
shapes without loading it.

They are private: :class:`WireRequest` carries the provider-routed request
kwargs, credentials included. The public request a test fake receives is a
separate, credential-free type that is mapped from these one way.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
class WireRequest:
    """One provider round trip.

    ``kwargs`` is the complete request as LiteLLM receives it: model, messages,
    tools, sampling settings and the provider's credential and native kwargs.
    """

    kwargs: dict[str, object]
