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
import dataclasses
import time
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime

from llmkit._transport import WireUsage
from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    error_text,
    parse_tool_calls,
    prepare_call,
    resolve_model_and_provider,
    tool_repair_messages,
    tool_validation_budget,
    usage_log_dict,
)
from llmkit.calls.stream import STREAM_ABANDONED_ERROR
from llmkit.capture import record_call_async, submit_record_call
from llmkit.exceptions import (
    FinishReasonError,
    OutputLimitDiagnostics,
    OutputLimitError,
    ToolArgumentError,
)
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface, Provider, build_provider
from llmkit.rate_limiting import begin_queue_wait, current_queue_wait_ms
from llmkit.retry import RetryPolicy, with_retries_stream
from llmkit.run_scope import get_run_id
from llmkit.sync import iterate_sync
from llmkit.tools import (
    TextDeltaEvent,
    TokenUsage,
    ToolCallResult,
    ToolChoice,
    ToolDefinition,
    guard_tool_choice,
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
    timeout: float | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | Provider | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> AsyncGenerator[TextDeltaEvent | ToolCallResult]:
    """Stream one tool-enabled turn: text deltas, then the completed result.

    ``timeout`` must be a finite nonnegative number (not bool), or ``None``;
    invalid values raise ``ValueError`` before call admission (at first
    iteration for streams). It is seconds for one provider round trip, a deadline llmkit
    enforces. For a stream it bounds opening the request, the wait for the
    first chunk and each later wait for the next chunk, never the stream's
    whole duration or your time between chunks. A breach raises
    ``TimeoutError``: before anything reached you it is retried as a fresh
    attempt, after that it ends the stream unretried. The value is forwarded
    to LiteLLM too; ``0.0`` is accepted and fires at the first wait. Unset or
    ``None`` sets no deadline and sends no timeout keyword, leaving LiteLLM's
    default in effect.

    Yields a :class:`~llmkit.TextDeltaEvent` per chunk of assistant prose as it
    arrives, and on a complete turn finally one :class:`~llmkit.ToolCallResult` — the same
    object :func:`~llmkit.tool_llm_call` returns, carrying ``text``,
    ``tool_calls``, ``invalid_calls``, ``stop_reason`` and ``usage``. The result
    is the **last** item, so a loop that stops there has the complete turn::

        async for event in tool_llm_call_stream(prompt, tools, feature="chat"):
            if isinstance(event, TextDeltaEvent):
                render(event.text)
            else:
                result = event  # the completed turn; the stream is done

    A truncated, filtered or failed turn raises after any text deltas,
    before parsing tool arguments, and yields no terminal result. Once a
    delta reached the consumer the stream is never replayed by retries.

    Takes the same ``tools`` / ``tool_choice`` surface as
    :func:`~llmkit.tool_llm_call`, and the same per-call keywords, with one
    deliberate absence: **no ``output_schema``.** Compose is rejected at the
    signature rather than at runtime, so the type checker enforces it. Use the
    portable two-step pattern (tool loop, then
    :func:`~llmkit.structured_llm_call`). Blocking code uses
    :func:`~llmkit.tool_llm_call_stream_sync`, which yields the same items.

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
    restarted, so a failure is retried only **before the first yielded item**.
    A round in which *every* requested tool call is malformed raises
    :class:`~llmkit.ToolArgumentError` (the same whole-round contract the
    buffered lane keeps). When the round streamed no prose, nothing has reached
    you yet, so it is re-asked on the validation allowance
    (``validation_max_attempts``, within the ``max_attempts`` ceiling) exactly
    as the buffered lane re-asks it: the retry carries the rejected calls and
    their errors so the model can correct them. ``ToolArgumentError`` is always
    part of that allowance, whatever ``validation_retry_on`` lists; opt out
    with ``validation_max_attempts=1``, ``max_attempts=1`` or ``NO_RETRY``. A
    round that streamed any prose first raises unretried, because those deltas
    were already yielded. A round in which only *some* calls are malformed
    keeps the good ones on ``tool_calls`` and reports the rest on
    ``invalid_calls``, exactly as the buffered lane does. No tool executes in
    either case: llmkit never runs your tools.
    """
    args, provider, call_id, attempt_offset = prepare_call(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
        feature=feature,
        label=label,
    )
    # Before the retry loop, so a route that cannot honour the choice refuses
    # it once, unlogged. With no provider (the registered config would not
    # build) the transport raises that build error inside the attempt instead.
    if provider is not None:
        guard_tool_choice(provider, tool_choice)
    tag = label or feature
    # ``call_id`` and ``attempt_offset`` reach ``_stream_tools_once`` as plain
    # parameters — NEVER a ContextVar, for the reasons ``calls/stream.py``
    # spells out: an async generator's body runs in its *consumer's* context,
    # so the stream's identity would leak into the consumer's own llmkit calls
    # between chunks.
    #
    # The rejected round of an all-malformed attempt, fed to the next attempt
    # as the buffered lane does. Only a received round replaces it: a transport
    # failure before the terminal item leaves the previous round in place.
    repair: list[ChatMessage] = []

    def _set_repair(messages: list[ChatMessage]) -> None:
        nonlocal repair
        repair = messages

    def _attempt(attempt: int) -> AsyncGenerator[TextDeltaEvent | ToolCallResult]:
        return _stream_tools_once(
            prompt,
            tools,
            repair=repair,
            on_repair=_set_repair,
            feature=feature,
            label=label,
            tool_choice=tool_choice,
            temperature=args.temperature,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_effort,
            provider=provider,
            call_id=call_id,
            attempt=attempt_offset + attempt,
        )

    # ``with_retries_stream`` is generic in the yield type, so the union flows
    # through unchanged. ``aclosing`` propagates an abandoning consumer's close
    # down to the attempt generator while these frames are still live, so the
    # record is written deterministically rather than at GC.
    async with aclosing(
        with_retries_stream(
            _attempt,
            # The buffered lane's tool-argument allowance. Only a round that
            # yielded nothing can use it: the loop refuses to retry after any
            # yield, so a round that streamed prose first still raises.
            policy=dataclasses.replace(
                args.retry, validation_retry_on=tool_validation_budget(args.retry)
            ),
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


def tool_llm_call_stream_sync(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    feature: str,
    label: str | None = None,
    tool_choice: ToolChoice | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    timeout: float | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | Provider | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> Generator[TextDeltaEvent | ToolCallResult]:
    """Synchronous form of :func:`tool_llm_call_stream`.

    Same arguments, same items, same records and retry behavior as the async
    form: text deltas one per ``next()``, then the completed
    :class:`~llmkit.ToolCallResult` as the last item. The returned generator
    drives the async stream on llmkit's persistent sync loop (see
    :func:`~llmkit.sync.iterate_sync`), so no item is fetched before it is
    asked for. ``timeout`` bounds each wait for the next chunk, as on the async
    form; the sync bridge adds no whole-stream ceiling. It must be a finite
    nonnegative number (not bool), or ``None``; invalid values raise
    ``ValueError`` on first iteration before call admission.

    The caller's :mod:`contextvars` context is captured when this function is
    called, not when iteration starts. Closing the generator, breaking out of a
    ``for`` loop over it, or dropping it before the result records the
    abandonment exactly as the async stream does; stopping at the result is a
    completed call. Calling or iterating it on the sync loop's own thread raises
    :class:`RuntimeError`; async code should use the async form.
    """
    return iterate_sync(
        tool_llm_call_stream(
            prompt,
            tools,
            feature=feature,
            label=label,
            tool_choice=tool_choice,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            provider=provider,
            retry=retry,
            options=options,
        )
    )


async def _stream_tools_once(
    prompt: str | Sequence[ChatMessage],
    tools: Sequence[ToolDefinition],
    *,
    repair: Sequence[ChatMessage],
    on_repair: Callable[[list[ChatMessage]], None],
    feature: str,
    label: str | None,
    tool_choice: ToolChoice | None,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None,
    timeout: float | None,
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

    *repair* is the previous attempt's rejected round, appended after the
    prompt; *on_repair* receives this attempt's replacement once its terminal
    item arrives (empty unless every call in it was malformed).
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
    # What this attempt sent, for its record: the caller's prompt, or on a
    # repair attempt the prompt followed by the rejected round.
    attempt_prompt: str | Sequence[ChatMessage] = prompt
    if repair:
        original: list[ChatMessage] = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        )
        attempt_prompt = [*original, *repair]
    accumulated: list[str] = []
    result: ToolCallResult | None = None
    cost: float | None = None
    error: str | None = None
    diagnostics: OutputLimitDiagnostics | None = None
    partial_text: str | None = None
    native_finish_reason: str | None = None
    usage: TokenUsage | None = None
    # The latch that separates "abandoned mid-flight" from "took the result and
    # stopped". Set *before* the terminal yield, because that is where an
    # idiomatic consumer's ``break`` throws ``GeneratorExit`` into this frame.
    completed = False
    # What the transport actually sent, copied once, before the first yield:
    # until then every line here runs inside the consumer's first ``__anext__``,
    # the same context the funnel stamped. After a yield the consumer's own
    # llmkit calls may restamp that context, so it is never read again (see
    # :func:`llmkit._litellm.sent_temperature`).
    sent_temperature = temperature
    captured = False
    # When the provider's first wire frame of any kind arrived, reported by the
    # transport: a tool-only round yields nothing here until it is assembled.
    first_chunk_at: float | None = None

    def _first_frame(at: float) -> None:
        nonlocal first_chunk_at
        first_chunk_at = at

    _litellm.begin_sent_temperature()
    # Kept as a local: the adapter fills this instance in (see ``calls/stream.py``).
    tiers = _litellm.begin_service_tiers()
    try:
        async with aclosing(
            _litellm.astream_tools(
                prompt,
                tools,
                repair=repair,
                tool_choice=tool_choice,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                provider=provider,
                feature=feature,
                label=label,
                call_id=call_id,
                attempt=attempt,
                on_first_frame=_first_frame,
            )
        ) as transport:
            async for item in transport:
                if not captured:
                    sent_temperature = _litellm.sent_temperature(temperature)
                    captured = True
                if isinstance(item, str):
                    accumulated.append(item)
                    yield TextDeltaEvent(item)
                    continue
                # The terminal transport item: the turn is assembled here, so
                # a parse failure is raised before anything is recorded as a
                # clean turn. A received round replaces the repair history
                # before parsing, as on the buffered lane; only an
                # all-malformed round sets a new one.
                on_repair([])
                cost = item.approximate_cost
                native_finish_reason = item.native_stop_reason
                usage = TokenUsage(*item.usage)
                _litellm.raise_for_finish(
                    item.stop_reason,
                    native_finish_reason,
                    partial_text="".join(accumulated) or None,
                    model=(provider or build_provider()).litellm_model(model),
                    max_tokens=max_tokens,
                    usage=WireUsage(*item.usage),
                )
                try:
                    parsed_calls, invalid_calls = parse_tool_calls(item.raw_calls, definitions)
                except ToolArgumentError as exc:
                    on_repair(tool_repair_messages(exc.round_errors, list(definitions)))
                    raise
                cost = item.approximate_cost
                result = ToolCallResult(
                    "".join(accumulated) or None,
                    parsed_calls,
                    item.stop_reason,
                    TokenUsage(*item.usage),
                    invalid_calls,
                    native_finish_reason,
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
        error = error_text(exc)
        if isinstance(exc, OutputLimitError):
            diagnostics = exc.diagnostics
        elif isinstance(exc, FinishReasonError):
            partial_text = exc.partial_text
            native_finish_reason = exc.native_finish_reason
            usage = TokenUsage(*exc.usage.counts())
        raise
    finally:
        if not captured:
            # Nothing was yielded, so this is still the funnel's context.
            sent_temperature = _litellm.sent_temperature(temperature)
        resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
        record = LLMCallRecord(
            started_at=started_at,
            feature=feature,
            label=label,
            model=resolved_model,
            provider=resolved_provider,
            temperature=sent_temperature,
            duration_ms=(time.monotonic() - start_t) * 1000,
            schema=TOOL_STREAM_LOG_SCHEMA,
            prompt=attempt_prompt,
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
            usage=usage_log_dict(result.usage if result is not None else usage),
            output_limit=diagnostics.to_log_dict() if diagnostics is not None else None,
            partial_text=diagnostics.partial_text if diagnostics is not None else partial_text,
            native_finish_reason=native_finish_reason,
            requested_service_tier=tiers.requested,
            served_service_tier=tiers.served,
            first_chunk_ms=(
                (first_chunk_at - start_t) * 1000 if first_chunk_at is not None else None
            ),
        )
        if error == STREAM_ABANDONED_ERROR:
            # Capture and submit without awaiting or running sink I/O while
            # GeneratorExit / cancellation unwinds. Delivery may finish later.
            submit_record_call(record)
        else:
            _ = await record_call_async(record)
