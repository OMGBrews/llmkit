"""The structured-output call family: a validated Pydantic instance back.

The only family that dumps its parsed response for the log, and the only one
whose ``on_result`` rejection is charged to the validation budget as a
schema failure would be.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel, JsonValue

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    logger,
    prepare_call,
    resolve_model_and_provider,
    result_validation_budget,
    run_with_policy,
)
from llmkit.capture import record_call_async
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import begin_queue_wait, current_queue_wait_ms
from llmkit.retry import RetryPolicy
from llmkit.run_scope import get_run_id
from llmkit.sync import run_sync


async def structured_llm_call[T: BaseModel](
    prompt: str | Sequence[ChatMessage],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    on_result: Callable[[T], object] | None = None,
    options: LLMCallOptions | None = None,
) -> T:
    """Call LLM with structured output parsing.

    Args:
        prompt: Either a plain string (sent as-is) or a list of
            :class:`~llmkit.Message` dicts
            (``[{"role": "system", "content": "..."}, ...]``); a message's
            ``content`` may be a plain string or a list of content-part dicts
            for multimodal input.
        output_schema: Pydantic model class the LLM response is parsed
            into, via ``instructor`` pinned to the provider's native
            JSON-schema mode.
        feature: Caller feature name (e.g. ``"summarization"``,
            ``"extraction"``, ``"classification"``, ``"multi_field"``,
            ``"schema"``). Embedded in the log filename and YAML body.
            **Required on purpose** — a telemetry forcing function that
            scopes the per-call logs; it is *not* part of
            :class:`LLMCallOptions` so it cannot be defaulted-away.
        label: Optional finer-grained identifier (e.g. ``"risk_register"``);
            used in the log filename.
        temperature: Sampling temperature passed to the LLM provider.
            Resolves to :data:`~llmkit.DEFAULT_TEMPERATURE` (``0.2``) when
            neither this keyword nor ``options`` supplies a value; an
            explicit ``None`` forwards **no** ``temperature`` kwarg at all,
            leaving the provider's default sampling in effect (the escape
            hatch for providers whose guidance says to omit it — e.g.
            Gemini 3.x).
        model: Optional model override (provider default when it resolves to
            ``None``). *Dual-homed* — also settable on
            :class:`~llmkit.LLMClientConfig`; this per-call value overrides
            the config (with ``options`` sitting between, see ``options``).
        max_tokens: Optional cap on the completion length, forwarded to the
            provider when set (no cap when it resolves to ``None`` —
            byte-identical to the prior request). Parity with
            :func:`text_llm_call`.
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort (``"disable" | "low" | "medium" |
            "high"``; provider-native values are also accepted). Unset defers to ``options``, then to the value
            configured on :class:`~llmkit.LLMClientConfig`; an explicit
            value wins for this call. ``"disable"`` turns Gemini thinking
            off so it doesn't consume the ``max_tokens`` budget.
            *Dual-homed* like ``model``: per-call overrides ``options``
            overrides config.
        provider: Optional provider override for THIS call only. Unset (the
            default) uses the globally-configured provider — so every
            existing caller is unchanged. Pass an explicit provider (e.g. an
            :class:`~llmkit.OpenRouterProvider` built from credentials) to
            route a single call through a different provider family without
            touching the app-wide :func:`~llmkit.configure_llm_client`
            registration. The log records the provider that actually ran.
        retry: Transient-error retry budget applied to this call. Resolves
            to :data:`~llmkit.DEFAULT_RETRY_POLICY` (retry the curated
            ``LLM_RECOVERABLE_ERRORS`` with full-jitter backoff) when
            neither this keyword nor ``options`` supplies one. Pass
            :data:`~llmkit.NO_RETRY` to opt out, or a custom
            :class:`~llmkit.RetryPolicy` to tune the budget. Each attempt is
            its own logged call; this layer stays separate from instructor's
            in-call schema-repair budget.
        on_result: Optional semantic-validation hook. Called with the parsed
            result of each attempt; raise
            :class:`~llmkit.ResultValidationError` from it to **reject** a
            result that parsed cleanly but is semantically wrong (an empty
            register, an unresolved citation, a total that doesn't reconcile)
            and re-roll the call. The re-roll is charged against the *validation*
            budget (``retry.validation_max_attempts``), exactly like a schema
            failure, so a deterministically-bad result can't burn the full
            transport budget; on exhaustion the last
            :class:`~llmkit.ResultValidationError` propagates. ``None`` (the
            default) leaves the call unchanged. Folds an
            LLM-then-validate-then-re-roll loop the caller would otherwise
            hand-roll into the call itself.
        options: Optional :class:`LLMCallOptions` supplying any of
            ``temperature``/``model``/``max_tokens``/``reasoning_effort``/
            ``retry``/``provider`` once for reuse across many calls.
            Precedence is **config < options < explicit keyword**, exactly:
            every mergeable keyword above defaults to :data:`~llmkit.UNSET`,
            so any keyword you pass — including ``None``, including a value
            equal to the documented default — overrides the matching
            ``options`` field. An unset ``options`` field defers to the
            config, and ``options=None`` (the default) leaves the
            flat-keyword path unchanged.

    Returns:
        An instance of *output_schema* populated by the LLM.

    Raises:
        Any exception from the LLM provider or output parser — transient
        ones (``LLM_RECOVERABLE_ERRORS``) are retried per *retry* first,
        then re-raised on exhaustion; non-transient ones propagate
        immediately. The log is still written on every attempt with the
        error recorded.
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

    async def _attempt() -> T:
        # Function-local so ``import llmkit`` never pays litellm's import
        # cost, and module-object-bound so a test patch installed after
        # import is still seen (see :mod:`llmkit._litellm`).
        import llmkit._litellm as _litellm

        nonlocal attempt_count
        attempt_count += 1
        attempt = attempt_count
        begin_queue_wait()
        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        response: T | None = None
        cost: float | None = None
        error: str | None = None
        try:
            response, cost = await _litellm.acompletion_structured(
                prompt,
                output_schema,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                provider=provider,
            )
            if on_result is not None:
                # A raise (ResultValidationError) rejects this result and
                # re-rolls within the validation budget; the attempt is still
                # logged below with both the rejected response and the error.
                _ = on_result(response)
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = (time.monotonic() - start_t) * 1000
            resolved_model, resolved_provider = resolve_model_and_provider(args.model, provider)
            # ``T`` is bounded to ``BaseModel``, so every parsed result dumps;
            # the cast only launders ``model_dump``'s ``dict[str, Any]``.
            response_dump: dict[str, JsonValue] | None = None
            if response is not None:
                try:
                    response_dump = cast("dict[str, JsonValue]", response.model_dump())
                except Exception:
                    # A custom @field_serializer/@model_serializer on the
                    # schema raised. The dump exists only for the log, so it
                    # degrades to None — logging must never break the call
                    # (success path) or mask the real provider error (error
                    # path). Same warn pattern as the sink failures in
                    # llmkit.logging.
                    logger.warning(
                        "Failed to serialize LLM response for log %s/%s; recording response as None",
                        feature,
                        label,
                        exc_info=True,
                    )
            _ = await record_call_async(
                LLMCallRecord(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    model=resolved_model,
                    provider=resolved_provider,
                    temperature=args.temperature,
                    duration_ms=duration_ms,
                    schema=output_schema.__name__,
                    prompt=prompt,
                    response=response_dump,
                    error=error,
                    approximate_cost=cost,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    call_id=call_id,
                    attempt=attempt,
                    queue_wait_ms=current_queue_wait_ms(),
                    run_id=get_run_id(),
                )
            )

    return await run_with_policy(
        _attempt,
        policy=args.retry,
        tag=label or feature,
        validation_retry_on=result_validation_budget(args.retry),
    )


def structured_llm_call_sync[T: BaseModel](
    prompt: str | Sequence[ChatMessage],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | None | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    on_result: Callable[[T], object] | None = None,
    options: LLMCallOptions | None = None,
) -> T:
    """Synchronous wrapper around :func:`structured_llm_call`.

    For the handful of synchronous call sites that cannot ``await``. Same
    arguments, same logging, same output as the async version; the coroutine is
    driven to completion via :func:`run_sync`, which routes it onto llmkit's
    single persistent event loop (handling running-loop detection). All three
    rate-limit dimensions — concurrency, RPM, and TPM — are acquired inside the
    async path: because every sync call shares that one persistent loop, the
    per-(provider, loop) async concurrency semaphore is genuinely shared and
    bounds sync fan-out across threads, with no separate calling-thread
    semaphore. ``max_tokens`` caps the completion length when set (parity with
    :func:`text_llm_call`); unset leaves it uncapped. ``reasoning_effort`` is
    forwarded identically (unset defers to the configured
    :class:`~llmkit.LLMClientConfig` value). ``retry`` is the transient-error
    budget, inherited from the async path (default-on; pass
    :data:`~llmkit.NO_RETRY` to opt out). ``on_result`` is the same
    semantic-validation re-roll hook the async call takes (raise
    :class:`~llmkit.ResultValidationError` to reject a result and re-roll on the
    validation budget). ``options`` is the same opt-in :class:`LLMCallOptions`
    bundle the async call takes, with the same **config < options < explicit
    keyword** precedence; every argument is forwarded untouched.
    """
    return run_sync(
        structured_llm_call(
            prompt,
            output_schema,
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
