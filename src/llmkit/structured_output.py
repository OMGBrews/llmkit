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

Callers that need to cross-reference these records (e.g. a higher-level
orchestrator that writes its own trace spanning several LLM calls) can
install a ``capture_llm_log_paths()`` context manager around the call to
receive the per-call log paths.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from llmkit.logging import LLMCallRecord, write_llm_log
from llmkit.providers import LLMProviderInterface
from llmkit.sync import run_sync

logger = logging.getLogger(__name__)


_captured_log_paths: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "_llm_captured_log_paths", default=None
)


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
            from llmkit.providers import get_provider

            provider = get_provider()
        return (model or provider.model, provider.name)
    except Exception:
        # Logging must never break the LLM call; degrade to (model, None).
        logger.debug("Could not resolve provider for LLM log", exc_info=True)
        return (model, None)


@contextmanager
def capture_llm_log_paths() -> Iterator[list[Path]]:
    """Capture log paths written by the call functions in this scope.

    The returned list is appended to once per LLM call inside the
    ``with`` block — including retries, since ``with_retries`` lives
    outside the call functions and each attempt is its own call.
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
    temperature: float = 0.2,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
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
        label: Optional finer-grained identifier (e.g. ``"risk_register"``);
            used in the log filename.
        temperature: Sampling temperature passed to the LLM provider.
        model: Optional model override (provider default when ``None``).
        max_tokens: Optional cap on the completion length, forwarded to the
            provider when set (no cap when ``None`` — byte-identical to the
            prior request). Parity with :func:`text_llm_call`.
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort (``"disable" | "low" | "medium" |
            "high"``). ``None`` (the default) defers to the value configured
            on :class:`~llmkit.LLMClientConfig`; an explicit value wins for
            this call. ``"disable"`` turns Gemini thinking off so it doesn't
            consume the ``max_tokens`` budget.
        provider: Optional provider override for THIS call only. ``None``
            (the default) uses the globally-configured provider — so every
            existing caller is unchanged. Pass an explicit provider (e.g. an
            :class:`~llmkit.OpenRouterProvider` built from credentials) to
            route a single call through a different provider family without
            touching the app-wide :func:`~llmkit.configure_llm_client`
            registration. The log records the provider that actually ran.

    Returns:
        An instance of *output_schema* populated by the LLM.

    Raises:
        Any exception from the LLM provider or output parser — callers
        are responsible for catching ``LLM_RECOVERABLE_ERRORS``. The log
        is still written on exception with the error recorded.
    """
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
        parsed, cost = await _litellm.acompletion_structured(
            prompt,
            output_schema,  # pyright: ignore[reportArgumentType]  # raw-model — unbounded public T vs BaseModel-bound seam
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            provider=provider,
        )
        response = cast("T", parsed)
        return response
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.monotonic() - start_t) * 1000
        resolved_model, resolved_provider = _resolve_model_and_provider(model, provider)
        response_dump = (
            response.model_dump()  # pyright: ignore[reportAttributeAccessIssue]  # raw-llm — Pydantic result dumped for the log
            if response is not None and hasattr(response, "model_dump")
            else None
        )
        path = write_llm_log(
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
        captured = _captured_log_paths.get()
        if captured is not None and path is not None:
            captured.append(path)


def structured_llm_call_sync[T](
    prompt: str | list[dict[str, str]],
    output_schema: type[T],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = 0.2,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
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
    configured :class:`~llmkit.LLMClientConfig` value).
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
        )
    )


async def text_llm_call(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = 0.2,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    provider: LLMProviderInterface | None = None,
) -> str:
    """Call the LLM for a plain-text (non-structured) response.

    The non-streaming, plain-text counterpart to
    :func:`structured_llm_call`: callers that just want text back (and
    parse it themselves, e.g. as JSON) use this.

    Args:
        prompt: Either a plain string or a list of message dicts
            (``[{"role": "system", "content": "..."}, ...]``).
        feature: Caller feature name; embedded in the log filename/body.
        label: Optional finer-grained identifier for the log filename.
        temperature: Sampling temperature passed to the provider.
        model: Optional model override (provider default when ``None``).
        max_tokens: Optional cap on the completion length, forwarded to
            the provider when set (e.g. the readiness healthcheck uses
            ``max_tokens=1`` to keep its ping cheap).
        reasoning_effort: Optional per-call override of the provider's
            reasoning/thinking effort; ``None`` defers to the configured
            :class:`~llmkit.LLMClientConfig` value.
        provider: Optional provider override for THIS call only (``None``
            uses the globally-configured provider).

    Returns:
        The model's textual response.

    Raises:
        Any exception from the LLM provider — callers catch
        ``LLM_RECOVERABLE_ERRORS``. The log is still written on
        exception with the error recorded.
    """
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
        return text
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path = _log_text_call(
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
        )
        captured = _captured_log_paths.get()
        if captured is not None and path is not None:
            captured.append(path)


async def stream_text_with_log(
    prompt: str | list[dict[str, str]],
    *,
    feature: str,
    label: str | None = None,
    temperature: float = 0.2,
    model: str | None = None,
    provider: LLMProviderInterface | None = None,
) -> AsyncIterator[str]:
    """Stream raw text from the LLM, logging the full transcript on completion.

    Yields each chunk's textual content as it arrives. Callers parse the
    accumulated text themselves — typically as in-progress JSON, when the
    prompt instructs the model to return JSON. One YAML log per call is
    written to ``data/llm-logs/`` after the stream finishes (or errors),
    mirroring :func:`structured_llm_call`'s logging contract so the
    invocation appears in the same per-feature analysis tooling.

    The ``schema`` field in the log is the literal string ``"stream"``
    since there is no Pydantic schema applied here. Streamed responses
    carry no per-call cost (LiteLLM does not stamp ``response_cost`` on
    stream chunks), so ``approximate_cost`` is left ``None``.
    """
    from llmkit import _litellm

    started_at = datetime.now(UTC)
    start_t = time.monotonic()
    accumulated: list[str] = []
    error: str | None = None
    try:
        async for chunk in _litellm.astream_text(
            prompt, temperature=temperature, model=model, provider=provider
        ):
            if chunk:
                accumulated.append(chunk)
                yield chunk
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        full_text = "".join(accumulated)
        path = _log_text_call(
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
        )
        captured = _captured_log_paths.get()
        if captured is not None and path is not None:
            captured.append(path)


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
) -> Path | None:
    """Build and write an ``LLMCallRecord`` for a plain-text/stream call.

    Shared by :func:`text_llm_call` and :func:`stream_text_with_log`: the
    ``schema`` is the literal ``"stream"`` and ``response`` is the
    accumulated text rather than a Pydantic dump. ``model`` is resolved to
    the effective model (provider default substituted when the caller
    passed ``None``); ``provider`` is the per-call override (``None`` uses
    the globally-configured one) and names the provider the log records.
    """
    duration_ms = (time.monotonic() - start_t) * 1000
    resolved_model, resolved_provider = _resolve_model_and_provider(model, provider)
    return write_llm_log(
        LLMCallRecord(
            started_at=started_at,
            feature=feature,
            label=label,
            model=resolved_model,
            provider=resolved_provider,
            temperature=temperature,
            duration_ms=duration_ms,
            schema="stream",
            prompt=prompt,
            response=text,
            error=error,
            approximate_cost=approximate_cost,
        )
    )
