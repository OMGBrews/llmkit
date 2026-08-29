"""The buffered plain-text call family.

The thinnest of the four: option merge, one transport call, one record built by
:func:`~llmkit.calls._shared.build_text_record`, wrapped in the shared retry
pass. A failed attempt logs ``response: None``, distinct from a successful
empty completion's ``""`` — the streaming surface differs on purpose and logs
its partial transcript instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    build_text_record,
    prepare_call,
    result_validation_budget,
    run_with_policy,
)
from llmkit.capture import record_call_async
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import begin_queue_wait
from llmkit.retry import RetryPolicy
from llmkit.sync import run_sync


async def text_llm_call(
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
    on_result: Callable[[str], object] | None = None,
    options: LLMCallOptions | None = None,
) -> str:
    """Call the LLM for a plain-text (non-structured) response.

    The non-streaming, plain-text counterpart to
    :func:`structured_llm_call`: callers that just want text back (and
    parse it themselves, e.g. as JSON) use this.

    Args:
        prompt: Either a plain string or a list of :class:`~llmkit.Message`
            dicts (``[{"role": "system", "content": "..."}, ...]``); a
            message's ``content`` may be a plain string or a list of
            content-part dicts for multimodal input.
        feature: Caller feature name; embedded in the log filename/body.
            Required as a telemetry forcing function (see
            :func:`structured_llm_call`); not part of :class:`LLMCallOptions`.
        label: Optional finer-grained identifier for the log filename.
        temperature: Sampling temperature passed to the provider. Resolves
            to :data:`~llmkit.DEFAULT_TEMPERATURE` (``0.2``) when neither
            this keyword nor ``options`` supplies a value; an explicit
            ``None`` forwards **no** ``temperature`` kwarg at all, leaving
            the provider's default sampling in effect.
        model: Optional model override (provider default when it resolves
            to ``None``). *Dual-homed* with
            :class:`~llmkit.LLMClientConfig`: per-call overrides
            ``options`` overrides config.
        max_tokens: Optional cap on the completion length, forwarded to
            the provider when set (e.g. the readiness healthcheck uses
            ``max_tokens=1`` to keep its ping cheap).
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort; unset defers to ``options``, then to
            the configured :class:`~llmkit.LLMClientConfig` value.
            *Dual-homed* like ``model``.
        provider: Optional provider override for THIS call only (unset
            uses the globally-configured provider).
        retry: Transient-error retry budget (default-on; see
            :func:`structured_llm_call`). Pass :data:`~llmkit.NO_RETRY` to
            opt out. Each attempt is its own logged call.
        on_result: Optional semantic-validation re-roll hook (see
            :func:`structured_llm_call`). Called with the response *text*; raise
            :class:`~llmkit.ResultValidationError` from it to reject a
            structurally-fine-but-wrong answer (e.g. text that fails to parse as
            the JSON you asked for) and re-roll on the validation budget.
        options: Optional :class:`LLMCallOptions` bundle (see
            :func:`structured_llm_call`); explicit keywords here override it,
            it overrides config, and ``None`` leaves the flat path unchanged.

    Returns:
        The model's textual response.

    Raises:
        Any exception from the LLM provider — transient ones
        (``LLM_RECOVERABLE_ERRORS``) are retried per *retry*, others
        propagate immediately. The log is still written on every attempt
        with the error recorded.
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
    attempt_count = 0

    async def _attempt() -> str:
        # Function-local + module-bound; see :mod:`llmkit._litellm`.
        import llmkit._litellm as _litellm

        nonlocal attempt_count
        attempt_count += 1
        attempt = attempt_count
        begin_queue_wait()
        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        # None until the transport returns: a failed attempt logs
        # ``response: None`` (per the LLMCallRecord contract), distinct from
        # a successful empty completion's ``response: ""``. The streaming
        # path differs on purpose — it logs the partial transcript.
        text: str | None = None
        cost: float | None = None
        error: str | None = None
        try:
            text, cost = await _litellm.acompletion_text(
                prompt,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                provider=provider,
            )
            if on_result is not None:
                # A raise (ResultValidationError) rejects this text and re-rolls
                # within the validation budget; the attempt is still logged below.
                _ = on_result(text)
            return text
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _ = await record_call_async(
                build_text_record(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    prompt=prompt,
                    text=text,
                    start_t=start_t,
                    temperature=args.temperature,
                    model=args.model,
                    provider=provider,
                    error=error,
                    approximate_cost=cost,
                    schema="text",
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    call_id=call_id,
                    attempt=attempt,
                )
            )

    return await run_with_policy(
        _attempt,
        policy=args.retry,
        tag=label or feature,
        validation_retry_on=result_validation_budget(args.retry),
    )


def text_llm_call_sync(
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
    on_result: Callable[[str], object] | None = None,
    options: LLMCallOptions | None = None,
) -> str:
    """Synchronous wrapper around :func:`text_llm_call`.

    Same arguments, same logging, same plain-text result as the async version;
    the coroutine is driven to completion via :func:`run_sync`, matching the
    structured sync wrapper — every rate-limit dimension (including the
    per-provider concurrency cap) is enforced inside the async path on the shared
    persistent loop, with no separate calling-thread semaphore (see
    :func:`structured_llm_call_sync`).
    """
    return run_sync(
        text_llm_call(
            prompt,
            feature=feature,
            label=label,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            provider=provider,
            retry=retry,
            on_result=on_result,
            options=options,
        )
    )
