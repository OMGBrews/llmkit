"""A supported fake transport for testing code built on llmkit.

Install a fake with :func:`install` and every llmkit call made while it is in
place is answered by the fake instead of a provider::

    from llmkit.testing import Reply, TransportRequest, install


    def fake(request: TransportRequest) -> Reply:
        if request.feature == "summary":
            return Reply.json(Summary(text="short"))
        return Reply.text("hello")


    with install(fake):
        run_the_app()

The fake receives one :class:`TransportRequest` per provider round trip,
carrying the call's identity (``feature``, ``label``, call kind, ``call_id``,
``attempt``), its inputs, and the effective model settings. It returns a
:class:`Reply` or raises; it may be a plain function or a coroutine function.
The helpers :func:`rate_limit_error`, :func:`service_unavailable_error`,
:func:`server_error` and :func:`timeout_error` build the exceptions real
providers raise, so retry, back-off, adaptive concurrency and the circuit
breaker treat them exactly as they treat a real provider's.

The fake replaces only the provider round trip. Everything around it still
runs: option merging, the response cache, the rate limiter and circuit
breaker, retry, schema parsing and in-call repair, tool-argument parsing, and
call logging. Consequences worth knowing:

- **Process-wide.** One slot holds the installed fake, so it reaches the sync
  API and threads the application starts itself. Concurrent tests in one
  process share it. Nested installs stack; leaving a block restores the
  previous fake.
- **The cache sits above the fake.** A cache hit, and an identical request
  that joins one already in flight, never reach it.
- **A configured provider is still required.** The transport resolves the
  provider before the round trip, and a key-based provider raises without a
  key. Configure one that needs none, such as Ollama.
- **No mid-stream failure.** A fake can fail a streaming call only before its
  first chunk.

This module imports neither LiteLLM nor pytest.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import httpx
import openai
from pydantic import BaseModel

from llmkit._transport import (
    CallContext,
    CallKind,
    WireChunk,
    WireFunction,
    WireResponse,
    WireStreamEnd,
    WireToolCall,
    WireToolFragment,
    WireUsage,
    push_double,
    remove_double,
)
from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.exceptions import ServiceUnavailableError
from llmkit.tools import TokenUsage, ToolChoice, ToolDefinition

__all__ = [
    "CallKind",
    "Fake",
    "FakeToolCall",
    "FakeTransport",
    "Reply",
    "TransportRequest",
    "install",
    "rate_limit_error",
    "server_error",
    "service_unavailable_error",
    "timeout_error",
]


@dataclass(frozen=True)
class TransportRequest:
    """One provider round trip, as a fake sees it.

    Attributes:
        kind: Which call family is asking: ``"structured"``, ``"text"``,
            ``"tool"``, ``"text_stream"`` or ``"tool_stream"``.
        feature: The caller's ``feature``.
        label: The caller's ``label``, or ``None``.
        call_id: The logical call's id, the same as the call log record's.
        attempt: The absolute attempt number, the same as the call log
            record's. A retried call reaches the fake once per attempt.
        repair: ``True`` for instructor's in-attempt schema-repair re-ask of a
            structured call, sent after the previous answer failed to parse.
        prompt: The caller's prompt as given: a string, or a tuple of messages.
        messages: The messages sent to the provider. They differ from
            ``prompt`` only for structured calls, whose output mode may add a
            schema instruction and whose repair re-ask adds the failed answer.
        output_schema: The structured output model, or the compose schema of a
            tool call; otherwise ``None``.
        tools: The tool definitions offered (tool kinds only).
        tool_choice: The caller's tool choice (tool kinds only).
        model: The resolved, provider-routed model id.
        provider: The provider's name.
        temperature: The temperature the call requested, or ``None``.
        max_tokens: The output-token cap, or ``None``.
        reasoning_effort: The effective reasoning effort, or ``None``.
    """

    kind: CallKind
    feature: str | None
    label: str | None
    call_id: str | None
    attempt: int | None
    repair: bool
    prompt: str | tuple[ChatMessage, ...]
    messages: tuple[ChatMessage, ...]
    output_schema: type[BaseModel] | None
    tools: tuple[ToolDefinition, ...]
    tool_choice: ToolChoice | None
    model: str
    provider: str
    temperature: float | None
    max_tokens: int | None
    reasoning_effort: ReasoningEffort | None


@dataclass(frozen=True)
class FakeToolCall:
    """A tool call for a :class:`Reply` to request.

    ``arguments`` is a mapping, serialized as JSON, or a raw string passed
    through as the provider would send it, so malformed arguments exercise the
    real parsing. ``id`` defaults to ``call_<n>``, ``n`` being the call's
    position in the reply.
    """

    name: str
    arguments: Mapping[str, object] | str = field(default_factory=dict[str, object])
    id: str | None = None


@dataclass(frozen=True)
class Reply:
    """A fake's answer to one round trip, for every call kind.

    Attributes:
        content: The answer text. A structured call parses it, so raw JSON
            runs the real parsing and schema repair. ``None`` when ``chunks``
            supplies it, or when the answer is tool calls alone.
        tool_calls: Tool calls to request (tool kinds).
        finish_reason: The provider's finish reason. ``None`` means
            ``"tool_calls"`` when tool calls are present, otherwise ``"stop"``.
            ``"length"`` gets each kind's real truncation behavior: a
            structured call, a text call and a text stream raise
            :class:`~llmkit.OutputLimitError` (the stream after delivering
            its chunks), and a tool call reports it as ``stop_reason``.
        usage: Token usage to report, or ``None``. It is debited from the
            tokens-per-minute limiter like a provider's.
        cost: Approximate cost to report, or ``None``.
        delay: Seconds to wait before answering, inside the rate-limiter slot.
        chunks: Explicit text deltas for a streaming call. When omitted,
            ``content`` is split into word-sized deltas. Buffered calls answer
            with their concatenation.
    """

    content: str | None = None
    tool_calls: Sequence[FakeToolCall] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    cost: float | None = None
    delay: float = 0.0
    chunks: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError("Reply.delay must not be negative")
        if (
            self.chunks is not None
            and self.content is not None
            and "".join(self.chunks) != self.content
        ):
            raise ValueError("Reply.chunks must concatenate to Reply.content")

    @classmethod
    def text(
        cls,
        text: str,
        *,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        cost: float | None = None,
        delay: float = 0.0,
    ) -> Reply:
        """A plain-text answer, also the way to send raw JSON to a structured call."""
        return cls(content=text, finish_reason=finish_reason, usage=usage, cost=cost, delay=delay)

    @classmethod
    def json(
        cls,
        instance: BaseModel,
        *,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        cost: float | None = None,
        delay: float = 0.0,
    ) -> Reply:
        """*instance* serialized with ``model_dump_json(by_alias=True)``.

        The structured call parses it back with the real parser. A model whose
        computed fields are serialized but rejected on input (``extra="forbid"``)
        does not round-trip; send its JSON with :meth:`text` instead.
        """
        return cls.text(
            instance.model_dump_json(by_alias=True),
            finish_reason=finish_reason,
            usage=usage,
            cost=cost,
            delay=delay,
        )


def _reply_text(reply: Reply) -> str | None:
    if reply.content is not None:
        return reply.content
    return "".join(reply.chunks) if reply.chunks is not None else None


def _reply_deltas(reply: Reply) -> list[str]:
    if reply.chunks is not None:
        return [chunk for chunk in reply.chunks if chunk]
    # Each word with the whitespace after it, so the deltas join back exactly.
    return re.findall(r"\S+\s*|\s+", reply.content or "")


def _reply_tool_calls(reply: Reply) -> tuple[WireToolCall, ...]:
    return tuple(
        WireToolCall(
            id=call.id if call.id is not None else f"call_{position}",
            function=WireFunction(
                name=call.name,
                arguments=(
                    call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(dict(call.arguments))
                ),
            ),
        )
        for position, call in enumerate(reply.tool_calls)
    )


def _reply_finish_reason(reply: Reply) -> str:
    if reply.finish_reason is not None:
        return reply.finish_reason
    return "tool_calls" if reply.tool_calls else "stop"


def _reply_usage(reply: Reply) -> WireUsage:
    if reply.usage is None:
        return WireUsage()
    return WireUsage(
        prompt_tokens=reply.usage.prompt_tokens,
        completion_tokens=reply.usage.completion_tokens,
        total_tokens=reply.usage.total_tokens,
    )


Fake = Callable[[TransportRequest], Reply | Awaitable[Reply]]
"""What :func:`install` accepts: a function or coroutine function answering one request."""


def _public_request(call: CallContext) -> TransportRequest:
    """Map the transport's private context to the fake's request, copying inputs."""
    return TransportRequest(
        kind=call.kind,
        feature=call.feature,
        label=call.label,
        call_id=call.call_id,
        attempt=call.attempt,
        repair=call.repair,
        prompt=call.prompt if isinstance(call.prompt, str) else tuple(copy.deepcopy(call.prompt)),
        messages=tuple(copy.deepcopy(call.messages)),
        output_schema=call.output_schema,
        tools=tuple(call.tools),
        tool_choice=call.tool_choice,
        model=call.model,
        provider=call.provider,
        temperature=call.temperature,
        max_tokens=call.max_tokens,
        reasoning_effort=call.reasoning_effort,
    )


class _Double:
    """The :class:`~llmkit._transport.TransportDouble` wrapping one installed fake."""

    def __init__(self, fake: Fake) -> None:
        self._fake: Fake = fake
        self._lock: threading.Lock = threading.Lock()
        self._requests: list[TransportRequest] = []

    def snapshot(self) -> list[TransportRequest]:
        with self._lock:
            return list(self._requests)

    async def _reply(self, call: CallContext) -> Reply:
        request = _public_request(call)
        with self._lock:
            self._requests.append(request)
        answer = self._fake(request)
        reply = await answer if inspect.isawaitable(answer) else answer
        if not isinstance(reply, Reply):  # pyright: ignore[reportUnnecessaryIsInstance]  # untyped fakes
            raise TypeError(f"a fake transport must return Reply, not {type(reply).__name__}")
        if reply.delay:
            await asyncio.sleep(reply.delay)
        return reply

    async def respond(self, call: CallContext) -> WireResponse:
        reply = await self._reply(call)
        return WireResponse(
            text=_reply_text(reply),
            tool_calls=_reply_tool_calls(reply),
            finish_reason=_reply_finish_reason(reply),
            usage=_reply_usage(reply),
            cost=reply.cost,
        )

    async def stream(self, call: CallContext) -> AsyncIterator[WireChunk | WireStreamEnd]:
        reply = await self._reply(call)
        for delta in _reply_deltas(reply):
            yield WireChunk(text=delta)
        for position, tool_call in enumerate(_reply_tool_calls(reply)):
            yield WireChunk(
                tool_fragments=(
                    WireToolFragment(
                        index=position,
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )
        yield WireChunk(finish_reason=_reply_finish_reason(reply), cost=reply.cost)
        yield WireStreamEnd(usage=_reply_usage(reply), cost=reply.cost)


class FakeTransport:
    """An installed fake: a context manager for ``with`` and ``async with``.

    Entering installs the fake above any already installed; leaving removes it
    and restores the previous one. :attr:`requests` lists every request the
    fake has received, in arrival order.
    """

    def __init__(self, fake: Fake) -> None:
        self._double: _Double = _Double(fake)

    @property
    def requests(self) -> list[TransportRequest]:
        """A snapshot of the requests this fake has received."""
        return self._double.snapshot()

    def __enter__(self) -> Self:
        push_double(self._double)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        remove_double(self._double)

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, traceback)


def install(fake: Fake) -> FakeTransport:
    """Answer llmkit's provider round trips with *fake* inside a ``with`` block.

    Usable as ``with install(fake):`` or ``async with install(fake):``. With
    no fake installed, the real transport runs.
    """
    return FakeTransport(fake)


# A placeholder request for the simulated responses: httpx and the openai
# error types read the request off a response.
_SIMULATED_REQUEST = httpx.Request("POST", "https://llmkit.testing/simulated")


def _response(status: int, retry_after: float | None) -> httpx.Response:
    headers = {"retry-after": f"{retry_after:g}"} if retry_after is not None else {}
    return httpx.Response(status, headers=headers, request=_SIMULATED_REQUEST)


def rate_limit_error(retry_after: float | None = None) -> openai.RateLimitError:
    """A provider 429, optionally asking the client to wait *retry_after* seconds.

    Retried with back-off (honoring the ``Retry-After`` header), and a
    throttle signal for adaptive concurrency and the circuit breaker.
    """
    return openai.RateLimitError(
        "Rate limit exceeded (simulated by llmkit.testing)",
        response=_response(429, retry_after),
        body=None,
    )


def service_unavailable_error(retry_after: float | None = None) -> ServiceUnavailableError:
    """A provider 503 as llmkit raises it: retried, and a throttle signal."""
    return ServiceUnavailableError(
        "Service unavailable (simulated by llmkit.testing)",
        provider=None,
        model=None,
        response=_response(503, retry_after),
    )


def server_error() -> openai.InternalServerError:
    """A provider 500: retried, but not a throttle signal."""
    return openai.InternalServerError(
        "Internal server error (simulated by llmkit.testing)",
        response=_response(500, None),
        body=None,
    )


def timeout_error() -> openai.APITimeoutError:
    """A request timeout: retried, but not a throttle signal."""
    return openai.APITimeoutError(request=_SIMULATED_REQUEST)
