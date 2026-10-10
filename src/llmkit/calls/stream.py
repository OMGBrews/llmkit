"""The streaming text family.

Owns what one streaming *attempt* is; the retry loop around it lives in
:func:`llmkit.retry.with_retries_stream`, because retrying a stream is a
different problem from retrying an awaitable and both loops belong together.

The record this family writes is honest about how the stream ended: a provider
error logs that error, and a consumer that abandons the stream mid-flight logs
:data:`STREAM_ABANDONED_ERROR` — never a clean ``ok`` over a truncated
transcript.

It reads the response cache, with the same key as a buffered text call, but
takes no part in the single flight: a stream's answer is complete only when its
consumer pulls the last chunk, so an identical caller waiting on it would wait
on a consumer that may never move. A hit replays the stored text as one chunk
outside the retry loop; a miss stores only a stream that ran to completion.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.cache import LLMCacheEntry
from llmkit.cache.read_through import cached_entry, store_entry
from llmkit.calls._shared import (
    build_text_record,
    cache_and_key,
    prepare_call,
    resolve_model_and_provider,
)
from llmkit.capture import record_call, record_call_async
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import begin_queue_wait
from llmkit.retry import RetryPolicy, with_retries_stream

#: The ``error`` recorded when a stream's consumer abandons it mid-flight
#: (breaks out / closes the generator / cancels the task). The verdict header
#: then reads ``# ERROR`` with this marker, so a truncated transcript is never
#: mistaken for one the model finished (a plain ``# ok`` over partial text).
STREAM_ABANDONED_ERROR = "Abandoned: stream closed by consumer before completion"


async def text_llm_call_stream(
    prompt: str | Sequence[ChatMessage],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    cache: bool | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> AsyncGenerator[str]:
    """Stream raw text from the LLM, logging the full transcript on completion.

    Yields each chunk's textual content as it arrives. Callers parse the
    accumulated text themselves — typically as in-progress JSON, when the
    prompt instructs the model to return JSON. One log record per call is
    handed to the configured log sink (see
    :func:`~llmkit.configure_llm_logging`) after the stream finishes, errors,
    or is abandoned by its consumer — an abandoned stream records the partial
    transcript with :data:`STREAM_ABANDONED_ERROR` as its error, never a clean
    ``ok`` — mirroring :func:`structured_llm_call`'s logging contract so the
    invocation appears in the same per-feature analysis tooling.

    ``max_tokens`` caps the streamed completion length and
    ``reasoning_effort`` controls provider thinking tokens — parity with
    :func:`text_llm_call` and :func:`structured_llm_call`. Each is forwarded
    to the provider only when it resolves to a value (``reasoning_effort``
    resolved against the configured :class:`~llmkit.LLMClientConfig` value
    when ``None``) and is recorded on the call's
    :class:`~llmkit.logging.LLMCallRecord`. ``temperature`` follows the same
    rule: unset resolves to :data:`~llmkit.DEFAULT_TEMPERATURE` (``None``),
    and a resolved ``None`` forwards **no** ``temperature`` kwarg at all (the
    provider's default sampling applies).

    The ``schema`` field in the log is the literal string ``"stream"``
    since there is no Pydantic schema applied here. Streamed responses
    carry no per-call cost (LiteLLM does not stamp ``response_cost`` on
    stream chunks), so ``approximate_cost`` is left ``None``.

    ``retry`` is the transient-error budget (default-on; pass
    :data:`~llmkit.NO_RETRY` to opt out). A partially-consumed stream
    cannot be transparently restarted, so retry applies **only** to a
    transient failure that occurs *before the first chunk is yielded*:
    once any chunk has reached the caller, a mid-stream error propagates
    unretried. Each attempt is its own logged call.

    ``cache`` is response-cache participation (see
    :func:`structured_llm_call`): unset defers to ``options``, then to
    ``True``; ``False`` bypasses lookup and store. With a cache configured, a
    request identical to one answered before — streamed or buffered, the two
    share a key — is replayed from the store as a **single chunk**, with no
    provider call, no rate-limiter slot and no retry loop, and logged with
    ``cache_hit=True`` and ``approximate_cost=0.0``. A stream is stored only
    when it runs to completion: one its consumer abandons, or one that raises,
    stores nothing. A stream has no ``on_result`` hook, so outside an
    enclosing :func:`~llmkit.retry.with_retries` loop ``cache=False`` is how
    to ask for a fresh sample. Identical streams in flight at once each pay
    for their own answer — the single flight is the buffered calls' alone.

    ``options`` is the same opt-in :class:`LLMCallOptions` bundle the other
    call functions accept, with the same **config < options < explicit
    keyword** precedence (see :func:`structured_llm_call`); ``None`` leaves
    the flat-keyword path unchanged.
    """
    prepared = prepare_call(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
        feature=feature,
        label=label,
        cache=cache,
    )
    args, provider, call_id, attempt_offset = prepared
    tag = label or feature
    # The same key a buffered text call computes, so either lane answers the
    # other. A pass re-run by an enclosing ``with_retries`` loop skips the
    # lookup (the host rejected the earlier answer), as ``run_cached`` does.
    participation = cache_and_key(prepared, prompt, output_schema=None)
    if participation is not None and attempt_offset == 0:
        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        entry = await cached_entry(*participation)
        if entry is not None:
            # Inline rather than a helper generator: another generator layer
            # would need its own ``aclosing`` for the abandoned record to be
            # written deterministically. The hit never touches the transport,
            # so it takes no slot and never calls ``begin_queue_wait``.
            error: str | None = None
            try:
                # A stored empty answer yields nothing, as a live empty stream
                # does; its hit is still recorded.
                if entry.response:
                    yield entry.response
            except (GeneratorExit, asyncio.CancelledError):
                error = STREAM_ABANDONED_ERROR
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record = build_text_record(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    prompt=prompt,
                    text=entry.response,
                    start_t=start_t,
                    temperature=args.temperature,
                    model=args.model,
                    provider=provider,
                    error=error,
                    approximate_cost=0.0,
                    schema="stream",
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    call_id=call_id,
                    attempt=attempt_offset + 1,
                    cache_hit=True,
                    source_call_id=entry.call_id,
                )
                # The same record-step split as ``_stream_once``, for the same
                # reason: never suspend while abandonment is unwinding.
                if error == STREAM_ABANDONED_ERROR:
                    _ = record_call(record)
                else:
                    _ = await record_call_async(record)
            return
    # ``call_id`` and ``attempt_offset`` reach ``_stream_once`` as plain
    # parameters — NEVER a ContextVar: an async generator's body runs in its
    # *consumer's* context, so a ContextVar set here would leak the stream's
    # identity into the consumer's own llmkit calls between chunks. Nor is the
    # offset re-read from the retry scope per attempt: an abandoned stream's
    # record is written during cleanup, possibly from another task.

    def _attempt(attempt: int) -> AsyncGenerator[str]:
        return _stream_once(
            prompt,
            feature=feature,
            label=label,
            temperature=args.temperature,
            model=args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            provider=provider,
            call_id=call_id,
            attempt=attempt_offset + attempt,
        )

    accumulated: list[str] = []
    # The retry loop itself lives in :mod:`llmkit.retry` beside the awaitable
    # one, so the two stay reviewable together; this surface owns only what one
    # attempt *is*. ``aclosing`` propagates an abandoning consumer's close down
    # to the attempt generator while these frames are still live, so the
    # abandoned-stream record is written deterministically rather than at GC.
    async with aclosing(
        with_retries_stream(
            _attempt,
            policy=args.retry,
            label=tag,
            surface="text_llm_call_stream",
            # This frame — the ``async for`` below — sits between the warning
            # and the consumer, so the double-wrap warning needs one extra
            # level to name the caller's own line rather than this one.
            warn_stacklevel=3,
        )
    ) as stream:
        async for chunk in stream:
            accumulated.append(chunk)
            yield chunk
    # Reached only when the stream ran out on its own: an abandoning consumer
    # raises ``GeneratorExit`` at the ``yield`` and a provider error
    # propagates, so neither stores. Retries happen only before the first
    # chunk, so *accumulated* holds the one attempt that answered.
    if participation is not None:
        resolved_model, provider_name = resolve_model_and_provider(args.model, provider)
        await store_entry(
            *participation,
            LLMCacheEntry(
                response="".join(accumulated),
                # The entry is a text answer; which lane produced it is the
                # record's concern, so a buffered call can serve it as well.
                schema="text",
                provider=provider_name,
                model=resolved_model,
                call_id=call_id,
            ),
        )


def stream_text_with_log(
    prompt: str | Sequence[ChatMessage],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    cache: bool | Unset = UNSET,
    options: LLMCallOptions | None = None,
) -> AsyncGenerator[str]:
    """Deprecated alias for :func:`text_llm_call_stream`; removed in llmkit 1.0.

    A plain ``def`` (not ``async def``): ``text_llm_call_stream`` is an
    async-generator function, so *calling* it returns the async generator
    without awaiting — a plain ``def`` wrapper therefore preserves
    ``async for chunk in stream_text_with_log(...)`` exactly, and warns
    **eagerly at call time** rather than deferring to first iteration.
    """
    warnings.warn(
        "stream_text_with_log() is deprecated; use text_llm_call_stream(). "
        + "The alias will be removed in llmkit 1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return text_llm_call_stream(
        prompt,
        feature=feature,
        label=label,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        provider=provider,
        retry=retry,
        cache=cache,
        options=options,
    )


async def _stream_once(
    prompt: str | Sequence[ChatMessage],
    *,
    feature: str,
    label: str | None,
    temperature: float | None,
    model: str | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
    provider: LLMProviderInterface | None,
    call_id: str,
    attempt: int,
) -> AsyncGenerator[str]:
    """One streaming attempt: yield each chunk, log the transcript on close.

    The single-attempt core of :func:`text_llm_call_stream` — the retry
    loop there wraps this so each attempt writes its own log record (the
    one-attempt-one-log contract `capture_llm_log_paths` relies on). The
    record is honest about *how* the stream ended: a provider error logs
    that error, and a consumer that abandons the stream mid-flight (close /
    cancellation) logs :data:`STREAM_ABANDONED_ERROR` — never a clean ``ok``
    over a truncated transcript. ``call_id``/``attempt`` arrive as plain
    parameters (this is a module-level generator with no enclosing closure,
    and ContextVars set in a generator body leak into the consumer's
    context — see the caller).
    """
    # Function-local + module-bound; see :mod:`llmkit._litellm`.
    import llmkit._litellm as _litellm

    started_at = datetime.now(UTC)
    start_t = time.monotonic()
    # This runs at first ``__anext__`` — consumer context, like every other
    # line of this generator body — which is exactly where the transport's
    # acquire will stamp it, so reset/stamp/read stay coherent.
    begin_queue_wait()
    accumulated: list[str] = []
    error: str | None = None
    try:
        async for chunk in _litellm.astream_text(
            prompt,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            provider=provider,
        ):
            if chunk:
                accumulated.append(chunk)
                yield chunk
    except (GeneratorExit, asyncio.CancelledError):
        # The *consumer* abandoned the stream (broke out / closed the
        # generator / cancelled the task) — both are BaseExceptions, so
        # without this clause they would bypass the except below and the
        # finally would log the partial transcript with ``error=None``,
        # indistinguishable from a stream the model finished. Mark it
        # honestly and always re-raise: swallowing GeneratorExit is illegal
        # (RuntimeError: generator ignored GeneratorExit) and swallowing
        # cancellation would break task teardown.
        error = STREAM_ABANDONED_ERROR
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = build_text_record(
            started_at=started_at,
            feature=feature,
            label=label,
            prompt=prompt,
            text="".join(accumulated),
            start_t=start_t,
            temperature=temperature,
            model=model,
            provider=provider,
            error=error,
            approximate_cost=None,
            schema="stream",
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            call_id=call_id,
            attempt=attempt,
        )
        if error == STREAM_ABANDONED_ERROR:
            # Abandonment unwinds via GeneratorExit / a re-deliverable
            # CancelledError: suspending again here (an await) risks a second
            # cancellation landing mid-await and losing this record — the one
            # honest witness that the transcript is truncated. Abandonment is
            # rare and terminal, so one blocking write is the safe trade.
            _ = record_call(record)
        else:
            # Clean finish or provider error: normal unwinding, safe to
            # await — and streams carry the largest payloads, so this is the
            # off-loop offload that matters most.
            _ = await record_call_async(record)
