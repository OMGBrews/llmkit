"""Shared LLM call functions with universal invocation logging.

Every call to :func:`structured_llm_call`, :func:`text_llm_call`, and
:func:`stream_text_with_log` builds an
:class:`~llmkit.logging.LLMCallRecord` (prompt, schema, response,
duration, resolved model/provider, approximate cost, any provider error)
and hands it to the configured log sink (see :mod:`llmkit.logging`).
Logging is unconditional — failures to write the log are swallowed so the
LLM call itself never breaks because logging did.

The actual provider transport lives in :mod:`llmkit._litellm`
(LiteLLM, with ``instructor`` for structured output); these functions own
the logging + cost-recording contract around it.

Callers that need to cross-reference these calls (e.g. a higher-level
orchestrator that writes its own trace spanning several LLM calls) can
install one of two context managers around the call: ``capture_llm_records()``
to receive the per-call :class:`~llmkit.logging.LLMCallRecord` objects
(approximate cost, resolved model/provider, duration — no sink to author),
or ``capture_llm_log_paths()`` to receive the per-call log-file paths.
"""

from __future__ import annotations

import contextvars
import logging
import time
import warnings
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

from llmkit.exceptions import ResultValidationError
from llmkit.logging import LLMCallRecord, write_llm_log
from llmkit.providers import LLMProviderInterface
from llmkit.retry import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    _retry_active,  # pyright: ignore[reportPrivateUsage]  # shared intra-package guard state
    handle_retry_failure,
    with_retries,
)
from llmkit.sync import run_sync

logger = logging.getLogger(__name__)


class _Unset(Enum):
    """Sentinel type for an unset :class:`LLMCallOptions` field.

    A single-member ``Enum`` rather than a bare ``object()`` so the type
    checker can narrow ``field is _UNSET`` precisely: an option left unset
    is typed ``Literal[_Unset.UNSET]`` and a set one is its real type, so
    the merge helper resolves to the right branch with no ``cast``.
    """

    UNSET = "unset"


#: The "this option was not provided" sentinel. An unset option field defers
#: to the per-call kwarg (and, through it, to the configured client) — it
#: never clobbers a config value with a default.
_UNSET = _Unset.UNSET


@dataclass(frozen=True)
class LLMCallOptions:
    """A reusable bundle of the per-call keyword arguments.

    Opt-in ergonomics for the call functions
    (:func:`structured_llm_call`, :func:`structured_llm_call_sync`,
    :func:`text_llm_call`, :func:`text_llm_call_sync`, and
    :func:`stream_text_with_log`): a feature module builds one
    ``LLMCallOptions`` once and passes it as ``options=`` to every call,
    instead of repeating the same nine-keyword block at each site. The flat
    keyword path is untouched — pass no ``options`` and nothing changes.

    Every field is **optional and unset by default** (the private
    :data:`_UNSET` sentinel, not ``None``): an unset field defers to the
    per-call keyword, which in turn defers to the configured
    :class:`~llmkit.LLMClientConfig` for the dual-homed ``model`` /
    ``reasoning_effort``. This is what makes "unset option does not clobber
    config" work — only a field you *set* on the options participates in the
    merge. Precedence, lowest to highest: **config < options < explicit
    per-call keyword.**

    ``feature`` is deliberately **not** part of this bundle: it stays a
    required per-call keyword as a telemetry forcing function (it scopes the
    per-call log filenames and ``index.jsonl`` grouping operators grep), so
    it cannot be defaulted-away into a shared object and forgotten.

    Attributes:
        temperature: Sampling temperature. Unset defers to the per-call
            ``temperature`` keyword (which defaults to ``0.2``).
        model: Model override. Unset defers to the per-call ``model``
            keyword, then to the provider default.
        max_tokens: Completion-length cap. Unset defers to the per-call
            ``max_tokens`` keyword (uncapped when also unset).
        reasoning_effort: Provider reasoning/thinking effort. Unset defers
            to the per-call keyword, then to the configured client value.
        retry: Transient-error retry budget. Unset defers to the per-call
            ``retry`` keyword (:data:`~llmkit.DEFAULT_RETRY_POLICY`).
        provider: Per-call provider override. Unset defers to the per-call
            ``provider`` keyword (the globally-configured provider).
    """

    temperature: float | _Unset = _UNSET
    model: str | None | _Unset = _UNSET
    max_tokens: int | None | _Unset = _UNSET
    reasoning_effort: str | None | _Unset = _UNSET
    retry: RetryPolicy | _Unset = _UNSET
    provider: LLMProviderInterface | None | _Unset = _UNSET


@dataclass(frozen=True, slots=True)
class _ResolvedCallArgs:
    """The per-call arguments after merging ``options`` with the keywords.

    The single shape every call function forwards to the transport once the
    three-layer precedence (config < options < explicit keyword) has been
    collapsed. Internal to the call-function module.
    """

    temperature: float
    model: str | None
    max_tokens: int | None
    reasoning_effort: str | None
    retry: RetryPolicy
    provider: LLMProviderInterface | None


def _resolve_call_args(
    options: LLMCallOptions | None,
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
    retry: RetryPolicy,
    provider: LLMProviderInterface | None,
) -> _ResolvedCallArgs:
    """Merge ``options`` under the explicit per-call keywords.

    The shared seam all the call functions funnel through, so the
    precedence is defined in exactly one place. The keyword arguments are
    the values the call function received with its *own* defaults already
    applied; for each field the rule is:

    * an explicit keyword that differs from the call function's default
      wins outright (highest precedence);
    * otherwise a field that is *set* on ``options`` is used;
    * otherwise the call function's default stands.

    A keyword left at its default (``model=None``, ``temperature=0.2``,
    ``retry=DEFAULT_RETRY_POLICY``, …) is indistinguishable from "not
    passed", so it yields to ``options`` — which is exactly what lets a
    shared ``LLMCallOptions`` supply values while an explicit keyword still
    overrides it. When ``options is None`` the keywords pass straight
    through, so the flat-kwarg path is byte-for-byte unchanged.
    """
    if options is None:
        return _ResolvedCallArgs(
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            retry=retry,
            provider=provider,
        )
    return _ResolvedCallArgs(
        temperature=_pick(temperature, _DEFAULT_TEMPERATURE, options.temperature),
        model=_pick(model, None, options.model),
        max_tokens=_pick(max_tokens, None, options.max_tokens),
        reasoning_effort=_pick(reasoning_effort, None, options.reasoning_effort),
        retry=_pick(retry, DEFAULT_RETRY_POLICY, options.retry),
        provider=_pick(provider, None, options.provider),
    )


def _pick[V](keyword: V, default: V, option: V | _Unset) -> V:
    """Resolve one field: explicit keyword > set option > keyword default.

    A ``keyword`` that differs from this call function's ``default`` was
    passed explicitly and wins. Otherwise a ``option`` that is set (not
    :data:`_UNSET`) is used; an unset option falls through to the
    (default) keyword, so it never clobbers a config-backed value.
    """
    if keyword != default:
        return keyword
    if option is _UNSET:
        return keyword
    return option


#: The shared default sampling temperature for the call functions — the
#: value a per-call ``temperature`` keyword carries when the caller does not
#: override it. Named so :func:`_resolve_call_args` and every call signature
#: stay in agreement on what "unset temperature" means.
_DEFAULT_TEMPERATURE = 0.2


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


_captured_log_paths: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "_llm_captured_log_paths", default=None
)

_captured_records: contextvars.ContextVar[list[LLMCallRecord] | None] = contextvars.ContextVar(
    "_llm_captured_records", default=None
)


def _record_call(record: LLMCallRecord) -> Path | None:
    """Write *record* to the configured sink and feed the active captures.

    The single seam every call path funnels its built
    :class:`~llmkit.logging.LLMCallRecord` through, so the two capture
    primitives stay in lock-step: the record itself is appended to any
    active :func:`capture_llm_records` buffer, and the returned file path
    (when a file sink wrote one) is appended to any active
    :func:`capture_llm_log_paths` buffer. Returns the path so the existing
    path-capture call sites keep working unchanged.
    """
    captured_records = _captured_records.get()
    if captured_records is not None:
        captured_records.append(record)
    path = write_llm_log(record)
    captured_paths = _captured_log_paths.get()
    if captured_paths is not None and path is not None:
        captured_paths.append(path)
    return path


def _resolve_model_and_provider(
    model: str | None, provider: LLMProviderInterface | None = None
) -> tuple[str | None, str | None]:
    """Resolve the *effective* model + provider name for the log record.

    When the caller passes ``model=None`` the provider's configured
    default is what actually ran — record that instead of ``null`` so
    cost attribution is a ``grep | sort | uniq -c`` over the logs, not a
    code trace. An explicit ``provider`` (the per-call override) is used
    as-is so the log names the provider that *actually* ran, not the
    globally-configured one. Best-effort: any failure resolving the
    provider degrades to ``(model, None)`` rather than breaking the log
    write — logging must never break the LLM call.
    """
    try:
        if provider is None:
            from llmkit.providers import build_provider

            provider = build_provider()
        return (model or provider.model, provider.name)
    except Exception:
        # Logging must never break the LLM call; degrade to (model, None).
        logger.debug("Could not resolve provider for LLM log", exc_info=True)
        return (model, None)


@contextmanager
def capture_llm_records() -> Generator[list[LLMCallRecord]]:
    """Capture the :class:`~llmkit.logging.LLMCallRecord` for each call here.

    The records-oriented counterpart to :func:`capture_llm_log_paths`:
    yields a list that is appended to once per LLM call inside the ``with``
    block, giving the host the full record — ``approximate_cost``, resolved
    ``model``/``provider``, ``duration_ms``, ``error`` and the rest —
    without authoring a :class:`~llmkit.logging.LogSink`. Captures every
    call function (:func:`structured_llm_call`, :func:`text_llm_call`,
    :func:`stream_text_with_log`) and works across the ``run_sync`` bridge
    (e.g. :func:`structured_llm_call_sync`), since the record is appended
    inside the async call path the bridge drives.

    Like path-capture, one record is appended *per attempt* — retries each
    produce their own logged record — and capture is independent of the
    configured sink, so it works even when logging is disabled
    (``configure_llm_logging(None)``).
    """
    records: list[LLMCallRecord] = []
    token = _captured_records.set(records)
    try:
        yield records
    finally:
        _captured_records.reset(token)


@contextmanager
def capture_llm_log_paths() -> Generator[list[Path]]:
    """Capture log paths written by the call functions in this scope.

    The returned list is appended to once per LLM call inside the
    ``with`` block — including retries, since ``with_retries`` lives
    outside the call functions and each attempt is its own call. Only the
    configured file sink (:class:`~llmkit.LocalYamlLogSink`) yields a path;
    with a third-party :class:`~llmkit.logging.LogSink` (or logging
    disabled) the list stays empty — use :func:`capture_llm_records` to
    capture cost/metadata regardless of the sink.
    """
    paths: list[Path] = []
    token = _captured_log_paths.set(paths)
    try:
        yield paths
    finally:
        _captured_log_paths.reset(token)


async def structured_llm_call[T](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
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
        model: Optional model override (provider default when ``None``).
            *Dual-homed* — also settable on
            :class:`~llmkit.LLMClientConfig`; this per-call value overrides
            the config (with ``options`` sitting between, see ``options``).
        max_tokens: Optional cap on the completion length, forwarded to the
            provider when set (no cap when ``None`` — byte-identical to the
            prior request). Parity with :func:`text_llm_call`.
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort (``"disable" | "low" | "medium" |
            "high"``). ``None`` (the default) defers to the value configured
            on :class:`~llmkit.LLMClientConfig`; an explicit value wins for
            this call. ``"disable"`` turns Gemini thinking off so it doesn't
            consume the ``max_tokens`` budget. *Dual-homed* like ``model``:
            per-call overrides ``options`` overrides config.
        provider: Optional provider override for THIS call only. ``None``
            (the default) uses the globally-configured provider — so every
            existing caller is unchanged. Pass an explicit provider (e.g. an
            :class:`~llmkit.OpenRouterProvider` built from credentials) to
            route a single call through a different provider family without
            touching the app-wide :func:`~llmkit.configure_llm_client`
            registration. The log records the provider that actually ran.
        retry: Transient-error retry budget applied to this call. Defaults
            to :data:`~llmkit.DEFAULT_RETRY_POLICY` (retry the curated
            ``LLM_RECOVERABLE_ERRORS`` with full-jitter backoff). Pass
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
            Precedence is **config < options < explicit keyword**: a keyword
            you pass here overrides the matching ``options`` field, an
            unset ``options`` field defers to the config, and ``options=None``
            (the default) leaves the flat-keyword path unchanged.

    Returns:
        An instance of *output_schema* populated by the LLM.

    Raises:
        Any exception from the LLM provider or output parser — transient
        ones (``LLM_RECOVERABLE_ERRORS``) are retried per *retry* first,
        then re-raised on exhaustion; non-transient ones propagate
        immediately. The log is still written on every attempt with the
        error recorded.
    """
    resolved = _resolve_call_args(
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
    provider = resolved.provider

    async def _attempt() -> T:
        # Deferred import so test patches on ``llmkit._litellm``
        # call functions resolve at call time.
        from llmkit import _litellm

        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        response: T | None = None
        cost: float | None = None
        error: str | None = None
        try:
            # The public ``T`` is unbounded (frozen call surface); the LiteLLM
            # seam requires ``T: BaseModel``. Every caller passes a Pydantic
            # schema, so this is sound at runtime — suppress the bound mismatch.
            # The suppressed unbounded-T arg below leaves the parsed half of the
            # returned tuple untyped; it is a T by construction, so narrow it.
            parsed, cost = cast(
                "tuple[T, float | None]",
                await _litellm.acompletion_structured(
                    prompt,
                    output_schema,  # pyright: ignore[reportArgumentType]  # raw-model — unbounded public T vs BaseModel-bound seam
                    temperature=temperature,
                    model=model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    provider=provider,
                ),
            )
            response = parsed
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
            resolved_model, resolved_provider = _resolve_model_and_provider(model, provider)
            # The public T is unbounded, but every parsed result is a Pydantic
            # model; narrow to BaseModel to dump it for the log.
            response_dump: dict[str, JsonValue] | None = (
                cast("dict[str, JsonValue]", cast("BaseModel", response).model_dump())
                if response is not None and hasattr(response, "model_dump")
                else None
            )
            _ = _record_call(
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
                )
            )

    return await with_retries(
        _attempt,
        max_attempts=retry.max_attempts,
        label=label or feature,
        backoff_base_seconds=retry.backoff_base_seconds,
        retry_on=retry.retry_on,
        validation_max_attempts=retry.validation_max_attempts,
        validation_retry_on=_result_validation_budget(retry),
    )


def structured_llm_call_sync[T](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
    on_result: Callable[[T], object] | None = None,
    options: LLMCallOptions | None = None,
) -> T:
    """Synchronous wrapper around :func:`structured_llm_call`.

    For the handful of synchronous call sites that cannot ``await``.
    Same arguments, same logging, same
    output as the async version; the coroutine is driven to completion
    via :func:`run_sync`, which handles running-event-loop detection. The
    rate-limit slot is acquired inside the async path, so the sync caller
    inherits it. ``max_tokens`` caps the completion length when set
    (parity with :func:`text_llm_call`); ``None`` leaves it uncapped.
    ``reasoning_effort`` is forwarded identically (``None`` defers to the
    configured :class:`~llmkit.LLMClientConfig` value). ``retry`` is the
    transient-error budget, inherited from the async path (default-on; pass
    :data:`~llmkit.NO_RETRY` to opt out). ``on_result`` is the same
    semantic-validation re-roll hook the async call takes (raise
    :class:`~llmkit.ResultValidationError` to reject a result and re-roll on the
    validation budget). ``options`` is the same opt-in :class:`LLMCallOptions`
    bundle the async call takes, with the same **config < options < explicit
    keyword** precedence; it is forwarded untouched so the merge happens once,
    in :func:`structured_llm_call`.
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
    temperature: float = _DEFAULT_TEMPERATURE,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
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
        temperature: Sampling temperature passed to the provider.
        model: Optional model override (provider default when ``None``).
            *Dual-homed* with :class:`~llmkit.LLMClientConfig`: per-call
            overrides ``options`` overrides config.
        max_tokens: Optional cap on the completion length, forwarded to
            the provider when set (e.g. the readiness healthcheck uses
            ``max_tokens=1`` to keep its ping cheap).
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort; ``None`` defers to the configured
            :class:`~llmkit.LLMClientConfig` value. *Dual-homed* like
            ``model``.
        provider: Optional provider override for THIS call only (``None``
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
    resolved = _resolve_call_args(
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
    provider = resolved.provider

    async def _attempt() -> str:
        from llmkit import _litellm

        started_at = datetime.now(UTC)
        start_t = time.monotonic()
        text = ""
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
            )

    return await with_retries(
        _attempt,
        max_attempts=retry.max_attempts,
        label=label or feature,
        backoff_base_seconds=retry.backoff_base_seconds,
        retry_on=retry.retry_on,
        validation_max_attempts=retry.validation_max_attempts,
        validation_retry_on=_result_validation_budget(retry),
    )


def text_llm_call_sync(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
    on_result: Callable[[str], object] | None = None,
    options: LLMCallOptions | None = None,
) -> str:
    """Synchronous wrapper around :func:`text_llm_call`.

    Same arguments, same logging, same plain-text result as the async version;
    the coroutine is driven to completion via :func:`run_sync`, matching the
    structured sync wrappers.
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
    temperature: float = _DEFAULT_TEMPERATURE,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
    options: LLMCallOptions | None = None,
) -> AsyncIterator[str]:
    """Stream raw text from the LLM, logging the full transcript on completion.

    Yields each chunk's textual content as it arrives. Callers parse the
    accumulated text themselves — typically as in-progress JSON, when the
    prompt instructs the model to return JSON. One YAML log per call is
    written to ``data/llm-logs/`` after the stream finishes (or errors),
    mirroring :func:`structured_llm_call`'s logging contract so the
    invocation appears in the same per-feature analysis tooling.

    ``max_tokens`` caps the streamed completion length and
    ``reasoning_effort`` controls provider thinking tokens — parity with
    :func:`text_llm_call` and :func:`structured_llm_call`. Each is forwarded
    to the provider only when set (``reasoning_effort`` resolved against the
    configured :class:`~llmkit.LLMClientConfig` value when ``None``) and is
    recorded on the call's :class:`~llmkit.logging.LLMCallRecord`.

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
    resolved = _resolve_call_args(
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
    provider = resolved.provider
    tag = label or feature

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
            async for chunk in _stream_once(
                prompt,
                feature=feature,
                label=label,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                provider=provider,
            ):
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
                async for chunk in _stream_once(
                    prompt,
                    feature=feature,
                    label=label,
                    temperature=temperature,
                    model=model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    provider=provider,
                ):
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
) -> AsyncIterator[str]:
    """One streaming attempt: yield each chunk, log the transcript on close.

    The single-attempt core of :func:`stream_text_with_log` — the retry
    loop there wraps this so each attempt writes its own log record (the
    one-attempt-one-log contract `capture_llm_log_paths` relies on).
    """
    from llmkit import _litellm

    started_at = datetime.now(UTC)
    start_t = time.monotonic()
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
        )


def _log_text_call(
    *,
    started_at: datetime,
    feature: str,
    label: str | None,
    prompt: str | list[dict[str, str]],
    text: str,
    start_t: float,
    temperature: float,
    model: str | None,
    provider: LLMProviderInterface | None,
    error: str | None,
    approximate_cost: float | None,
    schema: str = "text",
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Build and record an ``LLMCallRecord`` for a plain-text/stream call.

    Shared by :func:`text_llm_call` and :func:`stream_text_with_log`: the
    ``schema`` distinguishes the two surfaces in the log — ``"text"`` for a
    buffered plain-text call, ``"stream"`` for a streamed one — and
    ``response`` is the accumulated text rather than a Pydantic dump. ``model``
    is resolved to
    the effective model (provider default substituted when the caller
    passed ``None``); ``provider`` is the per-call override (``None`` uses
    the globally-configured one) and names the provider the log records.
    ``max_tokens``/``reasoning_effort`` are recorded as on the structured
    path, so the cap and thinking setting appear in the log for these calls
    too (both default ``None`` — absent from the request and unset on the
    record). Routes through :func:`_record_call` so both capture primitives
    (records and file paths) see the call, exactly like the structured path.
    """
    duration_ms = (time.monotonic() - start_t) * 1000
    resolved_model, resolved_provider = _resolve_model_and_provider(model, provider)
    _ = _record_call(
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
        )
    )
