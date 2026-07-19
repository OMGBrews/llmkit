"""The structured / plain-text / streaming LLM call functions.

Every call to :func:`structured_llm_call`, :func:`text_llm_call`, and
:func:`stream_text_with_log` builds an
:class:`~llmkit.logging.LLMCallRecord` (prompt, schema, response,
duration, resolved model/provider, approximate cost, any provider error)
and hands it to the configured log sink (see :mod:`llmkit.logging`).
Logging is unconditional — failures to write the log are swallowed so the
LLM call itself never breaks because logging did.

The actual provider transport lives in :mod:`llmkit._litellm`
(LiteLLM, with ``instructor`` for structured output); these functions own
the logging + cost-recording contract around it. Two adjacent concerns are
factored out: the three-layer per-call option merge lives in
:mod:`llmkit.options` (:class:`~llmkit.options.LLMCallOptions`,
:func:`~llmkit.options.resolve_call_args`), and the record sink seam plus
the capture context managers live in :mod:`llmkit.capture`
(:func:`~llmkit.capture.record_call`, :func:`~llmkit.capture.capture_llm_records`,
:func:`~llmkit.capture.capture_llm_log_paths`).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
import warnings
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel, JsonValue

from llmkit.capture import record_call, resolve_model_and_provider
from llmkit.exceptions import ResultValidationError
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset, resolve_call_args
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import (
    _queue_wait_ms,  # pyright: ignore[reportPrivateUsage]  # shared intra-package state
)
from llmkit.retry import (
    RetryPolicy,
    _retry_active,  # pyright: ignore[reportPrivateUsage]  # shared intra-package guard state
    handle_retry_failure,
    with_retries,
)
from llmkit.sync import run_sync

logger = logging.getLogger(__name__)


def _result_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """The validation retry-set augmented with :class:`ResultValidationError`.

    The ``on_result`` re-roll hook charges a rejected-result against the same
    budget as a schema-validation failure (semantically the content is wrong,
    not the transport), so :class:`ResultValidationError` is folded into the
    policy's ``validation_retry_on`` for the call's :func:`with_retries` pass.
    Including it unconditionally is harmless when no ``on_result`` is supplied —
    nothing raises it — and keeps the call functions from branching on the hook.
    """
    return (*retry.validation_retry_on, ResultValidationError)


def _build_call_provider(
    provider: LLMProviderInterface | None,
) -> LLMProviderInterface | None:
    """Resolve the provider for a call exactly once, best-effort.

    A per-call ``provider`` override is returned unchanged; otherwise the
    globally-configured provider is built a single time so the transport and
    the log record share one instance — the default path no longer calls
    :func:`~llmkit.providers.build_provider` twice (once to run the call,
    again only to read ``.model``/``.name`` for the log). A build failure
    degrades to ``None`` rather than breaking the call: the transport then
    re-resolves and surfaces the real configuration error, and the log path
    degrades to ``(model, None)`` via
    :func:`~llmkit.capture.resolve_model_and_provider`. Building is config +
    cached SDK checks (no I/O), so resolving it here before the retry loop is
    safe and happens once per call, not per attempt.
    """
    if provider is not None:
        return provider
    try:
        from llmkit.providers import build_provider

        return build_provider()
    except Exception:
        # The call must not break here; let the transport raise the real
        # error (logged + retried per attempt) and the log degrade to None.
        logger.debug("Could not pre-resolve provider for LLM call", exc_info=True)
        return None


#: The ``error`` recorded when a stream's consumer abandons it mid-flight
#: (breaks out / closes the generator / cancels the task). The verdict header
#: then reads ``# ERROR`` with this marker, so a truncated transcript is never
#: mistaken for one the model finished (a plain ``# ok`` over partial text).
STREAM_ABANDONED_ERROR = "Abandoned: stream closed by consumer before completion"


async def structured_llm_call[T: BaseModel](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: str | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    on_result: Callable[[T], object] | None = None,
    options: LLMCallOptions | None = None,
) -> T:
    """Call LLM with structured output parsing.

    Args:
        prompt: Either a plain string (sent as-is) or a list of message
            dicts (``[{"role": "system", "content": "..."}, ...]``).
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
            neither this keyword nor ``options`` supplies a value.
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
            "high"``). Unset defers to ``options``, then to the value
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
    resolved = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    temperature = resolved.temperature
    model = resolved.model
    max_tokens = resolved.max_tokens
    reasoning_effort = resolved.reasoning_effort
    retry = resolved.retry
    # Build the provider once for this call; the transport reuses this
    # instance (no rebuild) and the per-attempt log reads its model/name
    # from it rather than constructing a second one.
    provider = _build_call_provider(resolved.provider)
    # One id per *logical* call; every retry attempt shares it and numbers
    # itself via the closure counter, so the N records of one call join on
    # ``call_id`` instead of feature + timestamp proximity. Deliberately
    # closure state, not a ContextVar: this coroutine runs entirely inside
    # one task, and closures cannot leak into a consumer's context the way
    # generator-set ContextVars do (see ``_retry_active`` in the stream path).
    call_id = uuid.uuid4().hex
    attempt_count = 0

    async def _attempt() -> T:
        # Deferred import so test patches on ``llmkit._litellm``
        # call functions resolve at call time.
        from llmkit import _litellm

        nonlocal attempt_count
        attempt_count += 1
        attempt = attempt_count
        # Reset before the transport runs: a stale stamp from a previous
        # attempt (or an unrelated earlier call in this context) must never
        # be attributed to an attempt that failed before acquiring.
        _ = _queue_wait_ms.set(None)
        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        response: T | None = None
        cost: float | None = None
        error: str | None = None
        try:
            response, cost = await _litellm.acompletion_structured(
                prompt,
                output_schema,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
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
            resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
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
            _ = record_call(
                LLMCallRecord(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    model=resolved_model,
                    provider=resolved_provider,
                    temperature=temperature,
                    duration_ms=duration_ms,
                    schema=output_schema.__name__,
                    prompt=prompt,
                    response=response_dump,
                    error=error,
                    approximate_cost=cost,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    call_id=call_id,
                    attempt=attempt,
                    queue_wait_ms=_queue_wait_ms.get(),
                )
            )

    return await with_retries(
        _attempt,
        max_attempts=retry.max_attempts,
        label=label or feature,
        backoff_base_seconds=retry.backoff_base_seconds,
        max_backoff_seconds=retry.max_backoff_seconds,
        retry_after_cap=retry.retry_after_cap,
        retry_on=retry.retry_on,
        validation_max_attempts=retry.validation_max_attempts,
        validation_retry_on=_result_validation_budget(retry),
    )


def structured_llm_call_sync[T: BaseModel](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: str | None | Unset = UNSET,
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


async def text_llm_call(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: str | None | Unset = UNSET,
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
        prompt: Either a plain string or a list of message dicts
            (``[{"role": "system", "content": "..."}, ...]``).
        feature: Caller feature name; embedded in the log filename/body.
            Required as a telemetry forcing function (see
            :func:`structured_llm_call`); not part of :class:`LLMCallOptions`.
        label: Optional finer-grained identifier for the log filename.
        temperature: Sampling temperature passed to the provider. Resolves
            to :data:`~llmkit.DEFAULT_TEMPERATURE` (``0.2``) when neither
            this keyword nor ``options`` supplies a value.
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
    resolved = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    temperature = resolved.temperature
    model = resolved.model
    max_tokens = resolved.max_tokens
    reasoning_effort = resolved.reasoning_effort
    retry = resolved.retry
    # Build the provider once for this call; the transport reuses this
    # instance (no rebuild) and the per-attempt log reads its model/name
    # from it rather than constructing a second one.
    provider = _build_call_provider(resolved.provider)
    # Per-logical-call correlation, mirroring ``structured_llm_call`` (see
    # the comment there for why closure state, not a ContextVar).
    call_id = uuid.uuid4().hex
    attempt_count = 0

    async def _attempt() -> str:
        from llmkit import _litellm

        nonlocal attempt_count
        attempt_count += 1
        attempt = attempt_count
        _ = _queue_wait_ms.set(None)
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
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
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
            _log_text_call(
                started_at=started_at,
                feature=feature,
                label=label,
                prompt=prompt,
                text=text,
                start_t=start_t,
                temperature=temperature,
                model=model,
                provider=provider,
                error=error,
                approximate_cost=cost,
                schema="text",
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                call_id=call_id,
                attempt=attempt,
            )

    return await with_retries(
        _attempt,
        max_attempts=retry.max_attempts,
        label=label or feature,
        backoff_base_seconds=retry.backoff_base_seconds,
        max_backoff_seconds=retry.max_backoff_seconds,
        retry_after_cap=retry.retry_after_cap,
        retry_on=retry.retry_on,
        validation_max_attempts=retry.validation_max_attempts,
        validation_retry_on=_result_validation_budget(retry),
    )


def text_llm_call_sync(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: str | None | Unset = UNSET,
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


async def stream_text_with_log(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float | Unset = UNSET,
    model: str | None | Unset = UNSET,
    max_tokens: int | None | Unset = UNSET,
    reasoning_effort: str | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
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
    :class:`~llmkit.logging.LLMCallRecord`.

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

    ``options`` is the same opt-in :class:`LLMCallOptions` bundle the other
    call functions accept, with the same **config < options < explicit
    keyword** precedence (see :func:`structured_llm_call`); ``None`` leaves
    the flat-keyword path unchanged.
    """
    resolved = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    temperature = resolved.temperature
    model = resolved.model
    max_tokens = resolved.max_tokens
    reasoning_effort = resolved.reasoning_effort
    retry = resolved.retry
    # Build the provider once for this call; every streaming attempt reuses
    # this instance for both the transport and the per-attempt log record
    # rather than constructing a second one.
    provider = _build_call_provider(resolved.provider)
    tag = label or feature
    # Per-logical-call correlation id shared by every streaming attempt.
    # Passed to ``_stream_once`` as a plain parameter — NEVER a ContextVar: an
    # async generator's body runs in its *consumer's* context, so a
    # ContextVar set here would leak the stream's identity into the
    # consumer's own llmkit calls between chunks (the exact hazard the
    # ``_retry_active`` reset-around-every-yield dance below exists to solve).
    call_id = uuid.uuid4().hex

    # Nested-retry guard, mirroring ``with_retries``: when an outer llmkit
    # retry loop is already active (the documented composable path — a host
    # wrapping stream consumption, since mid-stream errors propagate
    # unretried), this loop collapses to a single pass so the budgets don't
    # multiply (the 3 x 3 = 9 trap). The accidental double-wrap — a failure
    # this policy *would* have retried — warns; an explicit NO_RETRY inner
    # stays silent.
    if _retry_active.get():
        yielded_any = False
        try:
            # ``aclosing`` so an abandoning consumer's close propagates to the
            # attempt generator *here and now* — its finally then logs the
            # abandoned record deterministically, not whenever GC finalizes
            # the suspended generator.
            async with aclosing(
                _stream_once(
                    prompt,
                    feature=feature,
                    label=label,
                    temperature=temperature,
                    model=model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    provider=provider,
                    call_id=call_id,
                    attempt=1,
                )
            ) as attempt_stream:
                async for chunk in attempt_stream:
                    yielded_any = True
                    yield chunk
            return
        except Exception as exc:
            would_have_retried = (
                retry.max_attempts > 1
                and not yielded_any
                and isinstance(exc, (*retry.retry_on, *retry.validation_retry_on))
            )
            if would_have_retried:
                warnings.warn(
                    f"stream_text_with_log({tag!r}) is nested inside an already-retrying "
                    + "llmkit retry loop; this inner layer ran a single pass to avoid "
                    + "multiplying retry budgets (the outer loop owns the retries). To "
                    + "drive retries from an outer wrapper around a call function, opt "
                    + "the inner call out with retry=NO_RETRY.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise

    # Mark this loop active so a nested llmkit retry layer collapses to a
    # single pass, exactly as ``with_retries`` does. An async generator body
    # runs in its *consumer's* context, so the flag is released around each
    # ``yield`` (and re-armed on resume): holding it across a suspension would
    # leak it into the consumer's own llmkit calls between chunks — and, on an
    # early break, leave it set in their context for good.
    token = _retry_active.set(True)
    try:
        for attempt in range(1, retry.max_attempts + 1):
            yielded_any = False
            try:
                # ``aclosing`` for the same reason as the nested-retry branch
                # above: an abandoning consumer's close must reach the attempt
                # generator before this frame unwinds, so its abandoned-stream
                # log is written deterministically rather than at GC time.
                async with aclosing(
                    _stream_once(
                        prompt,
                        feature=feature,
                        label=label,
                        temperature=temperature,
                        model=model,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        provider=provider,
                        call_id=call_id,
                        attempt=attempt,
                    )
                ) as attempt_stream:
                    async for chunk in attempt_stream:
                        yielded_any = True
                        _retry_active.reset(token)
                        try:
                            yield chunk
                        finally:
                            token = _retry_active.set(True)
                return
            except Exception as exc:
                # A partially-consumed stream can't be restarted, and only the
                # curated transient set is retryable — so a non-transient error or
                # a mid-stream failure (chunks already delivered) propagates as-is.
                # Streaming is plain text (no schema parsing), so it budgets on the
                # transport ``max_attempts`` only; both transient sets are matched
                # for completeness so a stray validation error is still treated as
                # transient rather than escaping unretried.
                if (
                    not isinstance(exc, (*retry.retry_on, *retry.validation_retry_on))
                    or yielded_any
                ):
                    raise
                # Pre-first-chunk transient failure: retry until the budget is
                # spent. The final attempt logs an exhaustion ERROR before
                # re-raising, mirroring ``with_retries`` so an operator greps the
                # streaming and non-streaming surfaces the same way.
                if attempt == retry.max_attempts:
                    logger.error("%s: all %d attempts failed: %s", tag, retry.max_attempts, exc)
                    raise
                await handle_retry_failure(
                    tag=tag,
                    attempt=attempt,
                    max_attempts=retry.max_attempts,
                    error=exc,
                    backoff_base_seconds=retry.backoff_base_seconds,
                    max_backoff_seconds=retry.max_backoff_seconds,
                    retry_after_cap=retry.retry_after_cap,
                )
    finally:
        _retry_active.reset(token)


async def _stream_once(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None,
    temperature: float,
    model: str | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
    provider: LLMProviderInterface | None,
    call_id: str,
    attempt: int,
) -> AsyncGenerator[str]:
    """One streaming attempt: yield each chunk, log the transcript on close.

    The single-attempt core of :func:`stream_text_with_log` — the retry
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
    from llmkit import _litellm

    started_at = datetime.now(UTC)
    start_t = time.monotonic()
    # The limiter stamp for THIS attempt: reset before the transport
    # acquires. This runs at first ``__anext__`` — consumer context, like
    # every other line of this generator body — which is exactly where the
    # transport's acquire will stamp it, so reset/stamp/read stay coherent.
    _ = _queue_wait_ms.set(None)
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
        full_text = "".join(accumulated)
        _log_text_call(
            started_at=started_at,
            feature=feature,
            label=label,
            prompt=prompt,
            text=full_text,
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


def _log_text_call(
    *,
    started_at: datetime,
    feature: str,
    label: str | None,
    prompt: str | list[dict[str, str]],
    text: str | None,
    start_t: float,
    temperature: float,
    model: str | None,
    provider: LLMProviderInterface | None,
    error: str | None,
    approximate_cost: float | None,
    schema: str = "text",
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> None:
    """Build and record an ``LLMCallRecord`` for a plain-text/stream call.

    Shared by :func:`text_llm_call` and :func:`stream_text_with_log`: the
    ``schema`` distinguishes the two surfaces in the log — ``"text"`` for a
    buffered plain-text call, ``"stream"`` for a streamed one — and
    ``response`` is the accumulated text rather than a Pydantic dump
    (``None`` when a buffered attempt failed before any content; the
    streaming surface instead passes the partial transcript). ``model``
    is resolved to
    the effective model (provider default substituted when the caller
    passed ``None``); ``provider`` is the per-call override (``None`` uses
    the globally-configured one) and names the provider the log records.
    ``max_tokens``/``reasoning_effort`` are recorded as on the structured
    path, so the cap and thinking setting appear in the log for these calls
    too (both default ``None`` — absent from the request and unset on the
    record). Routes through :func:`record_call` so both capture primitives
    (records and file paths) see the call, exactly like the structured path.
    """
    duration_ms = (time.monotonic() - start_t) * 1000
    resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
    _ = record_call(
        LLMCallRecord(
            started_at=started_at,
            feature=feature,
            label=label,
            model=resolved_model,
            provider=resolved_provider,
            temperature=temperature,
            duration_ms=duration_ms,
            schema=schema,
            prompt=prompt,
            response=text,
            error=error,
            approximate_cost=approximate_cost,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            call_id=call_id,
            attempt=attempt,
            queue_wait_ms=_queue_wait_ms.get(),
        )
    )
