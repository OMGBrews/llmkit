"""The streaming tool-calling family: prose as it arrives, then the turn.

The tool lane returns a turn only once it is complete, which is correct for a
loop and wrong for a chat panel: assistant prose lands in paragraph-sized
blobs instead of token by token. This family streams that prose and still ends
at the same :class:`~llmkit.ToolCallResult` the buffered lane returns, so a
host's existing tool loop is unchanged from the result onward.

The yield type is a **union**, not a value plus a return: an async generator
cannot ``return`` a value, so the completed turn is the last thing yielded.
:class:`~llmkit.TextDeltaEvent` and :class:`~llmkit.ToolCallResult` are
distinct types precisely so the consumer discriminates on the type rather
than on position.

Two invariants this module owns, both easy to get subtly wrong:

* **The record is honest about how the stream ended.** Abandonment before the
  turn is assembled logs :data:`~llmkit.calls.STREAM_ABANDONED_ERROR`, exactly
  as the text stream does. But a consumer that takes the result and stops has
  *completed* the call, and the naive pattern would log that idiomatic ending
  as an abandonment — an error beside populated ``tool_calls``. A latch set
  before the terminal yield separates the two.
* **The rate-limit slot is released before the terminal yield** (in
  :func:`~llmkit._litellm.astream_tools`), because the consumer parked on the
  result is off running its tools.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import parse_tool_calls, prepare_call, resolve_model_and_provider
from llmkit.calls.stream import STREAM_ABANDONED_ERROR
from llmkit.capture import record_call, record_call_async
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import begin_queue_wait, current_queue_wait_ms
from llmkit.retry import RetryPolicy, with_retries_stream
from llmkit.run_scope import get_run_id
from llmkit.tools import (
    TextDeltaEvent,
    TokenUsage,
    ToolCallResult,
    ToolChoice,
    ToolDefinition,
)

#: The ``schema`` a streamed tool turn records, distinct from the buffered
#: lane's ``"tools"`` and the text stream's ``"stream"`` so one ``grep`` over
#: the logs separates the three surfaces. Nothing branches on the value.
TOOL_STREAM_LOG_SCHEMA = "tools-stream"


async def tool_llm_call_stream(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    feature: str,
    label: str | None = None,
    tool_choice: ToolChoice | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> AsyncGenerator[TextDeltaEvent | ToolCallResult]:
    """Stream one tool-enabled turn: text deltas, then the completed result.

    Yields a :class:`~llmkit.TextDeltaEvent` per chunk of assistant prose as it
    arrives, and finally exactly one :class:`~llmkit.ToolCallResult` — the same
    object :func:`~llmkit.tool_llm_call` returns, carrying ``text``,
    ``tool_calls``, ``invalid_calls``, ``stop_reason`` and ``usage``. The result
    is the **last** item, so a loop that stops there has the complete turn::

        async for event in tool_llm_call_stream(prompt, tools, feature="chat"):
            if isinstance(event, TextDeltaEvent):
                render(event.text)
            else:
                result = event  # the completed turn; the stream is done

    Takes the same ``tools`` / ``tool_choice`` surface as
    :func:`~llmkit.tool_llm_call`, and the same per-call keywords, with two
    deliberate absences:

    * **No ``output_schema``.** Compose is rejected at the signature rather
      than at runtime, so the type checker enforces it. Use the portable
      two-step pattern (tool loop, then :func:`~llmkit.structured_llm_call`).
    * **No sync variant.** :func:`~llmkit.sync.run_sync` bridges coroutines,
      not async generators; :func:`~llmkit.text_llm_call_stream` has no sync
      form either, for the same reason.

    One log record per attempt is handed to the configured sink, under the
    ``"tools-stream"`` schema, carrying the offered ``tools``, the returned
    ``tool_calls`` and the turn's ``usage`` — the streamed transport asks for
    ``stream_options={"include_usage": True}``, so a streamed tool turn keeps
    the usage the buffered lane records rather than logging nulls. A consumer
    that abandons the stream before the result records
    :data:`~llmkit.calls.STREAM_ABANDONED_ERROR` over the partial transcript,
    never a clean ``ok``; taking the result and stopping is a *completed* call
    and records as one.

    ``retry`` is the transient-error budget, and it behaves as it does on the
    text stream: a partially-consumed stream cannot be transparently
    restarted, so only a transient failure **before the first yielded item** is
    retried. One consequence is worth stating plainly — a round in which
    *every* requested tool call is malformed raises
    :class:`~llmkit.ToolArgumentError` (the same whole-round contract the
    buffered lane keeps), but here that raise reaches the caller **unretried**:
    :func:`~llmkit.retry.with_retries_stream` is a transport-only budget with
    no validation re-ask. A round in which only *some* calls are malformed
    keeps the good ones on ``tool_calls`` and reports the rest on
    ``invalid_calls``, exactly as the buffered lane does.
    """
    args, provider, call_id = prepare_call(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    tag = label or feature
    # ``call_id`` reaches ``_stream_tools_once`` as a plain parameter — NEVER a
    # ContextVar, for the reason ``calls/stream.py`` spells out: an async
    # generator's body runs in its *consumer's* context, so the stream's
    # identity would leak into the consumer's own llmkit calls between chunks.

    def _attempt(attempt: int) -> AsyncGenerator[TextDeltaEvent | ToolCallResult]:
        return _stream_tools_once(
            prompt,
            tools,
            feature=feature,
            label=label,
            tool_choice=tool_choice,
            temperature=args.temperature,
            model=args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            provider=provider,
            call_id=call_id,
            attempt=attempt,
        )

    # ``with_retries_stream`` is generic in the yield type, so the union flows
    # through unchanged. ``aclosing`` propagates an abandoning consumer's close
    # down to the attempt generator while these frames are still live, so the
    # record is written deterministically rather than at GC.
    async with aclosing(
        with_retries_stream(
            _attempt,
            policy=args.retry,
            label=tag,
            surface="tool_llm_call_stream",
            # This frame's ``async for`` sits between the double-wrap warning
            # and the consumer, so the warning needs one extra level to name
            # the caller's own line rather than this one.
            warn_stacklevel=3,
        )
    ) as stream:
        async for event in stream:
            yield event


async def _stream_tools_once(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    feature: str,
    label: str | None,
    tool_choice: ToolChoice | None,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
    provider: LLMProviderInterface | None,
    call_id: str,
    attempt: int,
) -> AsyncGenerator[TextDeltaEvent | ToolCallResult]:
    """One streaming tool attempt: yield deltas, assemble the turn, log it.

    The single-attempt core of :func:`tool_llm_call_stream`, so each attempt
    writes its own record (the one-attempt-one-log contract
    ``capture_llm_log_paths`` relies on). ``call_id``/``attempt`` arrive as
    plain parameters for the ContextVar reason the caller documents.
    """
    # Function-local + module-bound; see :mod:`llmkit._litellm`.
    import llmkit._litellm as _litellm

    started_at = datetime.now(UTC)
    start_t = time.monotonic()
    # First ``__anext__`` runs here, in the consumer's context — the same place
    # the transport's acquire stamps the wait, so reset/stamp/read stay
    # coherent. See ``calls/stream.py``.
    begin_queue_wait()
    definitions = {definition.name: definition for definition in tools}
    accumulated: list[str] = []
    result: ToolCallResult | None = None
    cost: float | None = None
    error: str | None = None
    # The latch that separates "abandoned mid-flight" from "took the result and
    # stopped". Set *before* the terminal yield, because that is where an
    # idiomatic consumer's ``break`` throws ``GeneratorExit`` into this frame.
    completed = False
    try:
        async with aclosing(
            _litellm.astream_tools(
                prompt,
                tools,
                tool_choice=tool_choice,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        ) as transport:
            async for item in transport:
                if isinstance(item, str):
                    accumulated.append(item)
                    yield TextDeltaEvent(item)
                    continue
                # The terminal transport item: the turn is assembled here, so
                # a parse failure is raised before anything is recorded as a
                # clean turn.
                parsed_calls, invalid_calls = parse_tool_calls(item.raw_calls, definitions)
                cost = item.approximate_cost
                result = ToolCallResult(
                    "".join(accumulated) or None,
                    parsed_calls,
                    item.stop_reason,
                    TokenUsage(*item.usage),
                    invalid_calls,
                )
                completed = True
                yield result
    except (GeneratorExit, asyncio.CancelledError):
        # The consumer abandoned the stream. Both are BaseExceptions, so
        # without this clause they would bypass the handler below while the
        # ``finally`` still wrote a record with ``error=None`` — a truncated
        # turn indistinguishable from one the model finished. Always re-raise:
        # swallowing GeneratorExit is illegal and swallowing cancellation
        # would break task teardown.
        #
        # The latch is what keeps this honest in *both* directions. A consumer
        # that received the result and closed the generator is parked on the
        # terminal yield with ``completed`` already set, and its close is a
        # completion, not an abandonment; anything earlier is a real one.
        if not completed:
            error = STREAM_ABANDONED_ERROR
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
        record = LLMCallRecord(
            started_at=started_at,
            feature=feature,
            label=label,
            model=resolved_model,
            provider=resolved_provider,
            temperature=temperature,
            duration_ms=(time.monotonic() - start_t) * 1000,
            schema=TOOL_STREAM_LOG_SCHEMA,
            prompt=prompt,
            # The partial transcript on an abandoned turn, the full result
            # otherwise — the same "say what actually happened" rule the text
            # stream follows.
            response=(
                result.to_log_dict()
                if result is not None
                else {"text": "".join(accumulated) or None, "tool_calls": []}
            ),
            error=error,
            approximate_cost=cost,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            call_id=call_id,
            attempt=attempt,
            queue_wait_ms=current_queue_wait_ms(),
            run_id=get_run_id(),
            tools=[definition.to_litellm() for definition in tools],
            tool_calls=[call.to_wire() for call in result.tool_calls]
            if result is not None
            else None,
            usage=(
                None
                if result is None
                else {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }
            ),
        )
        if error == STREAM_ABANDONED_ERROR:
            # Abandonment unwinds via GeneratorExit / a re-deliverable
            # CancelledError: awaiting here risks a second cancellation landing
            # mid-await and losing the one honest witness that the turn is
            # truncated. One blocking write is the safe trade — the same choice
            # ``calls/stream.py`` makes, and for the same reason.
            _ = record_call(record)
        else:
            _ = await record_call_async(record)
