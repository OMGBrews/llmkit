"""What every call family does once, in one place.

The five families (structured, text, streamed text, tool, streamed tool) differ
in what one *attempt* is; they agree on everything around it. That agreement
lives here so the next family added does not become another copy:

* :func:`prepare_call` — the per-call resolution every family does before its
  first attempt: merge the options, build the provider exactly once, mint the
  correlation id;
* :func:`run_with_policy` — the :func:`~llmkit.retry.with_retries` invocation,
  which differs between families only in which errors are charged to the
  validation budget;
* :func:`result_validation_budget` / :func:`tool_validation_budget` — the two
  augmented validation sets;
* :func:`build_text_record` and :func:`resolve_model_and_provider` — record
  construction shared by the text and streaming surfaces, and the effective
  model/provider resolution every family records;
* :func:`parse_tool_calls` — narrowing a round's raw provider tool calls,
  shared by the buffered and streaming tool surfaces so the salvage contract
  (well-formed calls survive a partly malformed round) is written once.

The **queue-wait protocol** is a cross-module contract worth stating where the
scaffold lives: every attempt calls
:func:`~llmkit.rate_limiting.begin_queue_wait` before the transport runs and
reads :func:`~llmkit.rate_limiting.current_queue_wait_ms` when it builds the
record. Skipping the reset does not fail loudly — it silently attributes a
previous attempt's queueing to an attempt that never queued.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import NamedTuple, cast

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.exceptions import ResultValidationError, ToolArgumentError
from llmkit.logging import LLMCallRecord
from llmkit.options import LLMCallOptions, ResolvedCallArgs, Unset, resolve_call_args
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import current_queue_wait_ms
from llmkit.retry import RetryPolicy, with_retries
from llmkit.run_scope import get_run_id
from llmkit.tools import ToolCall, ToolDefinition

# One logger for the whole call surface, named explicitly rather than via
# ``__name__`` so every family module shares a single greppable name.
logger = logging.getLogger("llmkit.calls")


def result_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """The validation retry-set augmented with :class:`ResultValidationError`.

    The ``on_result`` re-roll hook charges a rejected-result against the same
    budget as a schema-validation failure (semantically the content is wrong,
    not the transport), so :class:`ResultValidationError` is folded into the
    policy's ``validation_retry_on`` for the call's :func:`with_retries` pass.
    Including it unconditionally is harmless when no ``on_result`` is supplied —
    nothing raises it — and keeps the call functions from branching on the hook.
    """
    return (*retry.validation_retry_on, ResultValidationError)


def tool_validation_budget(retry: RetryPolicy) -> tuple[type[BaseException], ...]:
    """Tool argument errors use the same bounded repair budget as schemas."""
    return (*retry.validation_retry_on, ToolArgumentError)


def build_call_provider(
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
    :func:`resolve_model_and_provider` below. Building is config +
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


class PreparedCall(NamedTuple):
    """What a call resolves once, before its first attempt.

    Attributes:
        args: the merged per-call arguments (config < options < keyword).
        provider: the provider instance this call runs on, built exactly once
            so the transport and every attempt's log record share it.
        call_id: one ``uuid4`` hex per *logical* call. Every retry attempt
            shares it and numbers itself, so the N records a retried call
            produces join on ``call_id`` rather than on feature + timestamp
            proximity, which breaks under concurrent same-feature fan-out.
    """

    args: ResolvedCallArgs
    provider: LLMProviderInterface | None
    call_id: str


def prepare_call(
    options: LLMCallOptions | None,
    *,
    temperature: float | None | Unset,
    model: str | None | Unset,
    max_tokens: int | None | Unset,
    reasoning_effort: ReasoningEffort | None | Unset,
    retry: RetryPolicy | Unset,
    provider: LLMProviderInterface | None | Unset,
) -> PreparedCall:
    """Resolve everything a call decides once, before any attempt runs.

    The preamble every family shares. Each keyword arrives exactly as the
    call function received it — :data:`~llmkit.UNSET` when the caller did not
    pass it — and is merged by :func:`~llmkit.options.resolve_call_args`.

    Building the provider here rather than per attempt is what lets the
    transport and the log record name the same instance; it is config plus
    cached SDK checks with no I/O, so once per call is both correct and cheap.
    """
    args = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    return PreparedCall(args, build_call_provider(args.provider), uuid.uuid4().hex)


async def run_with_policy[T](
    attempt: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    tag: str,
    validation_retry_on: tuple[type[BaseException], ...],
) -> T:
    """Run *attempt* under the call's retry budget.

    The families differ only in which errors are charged to the *validation*
    budget, so that is the one parameter; everything else comes off the policy.
    Note that this is one pass of the retry loop, not one attempt: each attempt
    is its own logged call, because the loop wraps the logging call functions
    rather than living inside them.
    """
    return await with_retries(
        attempt,
        max_attempts=policy.max_attempts,
        label=tag,
        backoff_base_seconds=policy.backoff_base_seconds,
        max_backoff_seconds=policy.max_backoff_seconds,
        retry_after_cap=policy.retry_after_cap,
        retry_on=policy.retry_on,
        validation_max_attempts=policy.validation_max_attempts,
        validation_retry_on=validation_retry_on,
    )


def _tool_call_from_raw(raw: object, definitions: dict[str, ToolDefinition]) -> ToolCall:
    """Narrow LiteLLM's intentionally open tool-call objects at one boundary."""
    call_id = getattr(raw, "id", None)
    function = getattr(raw, "function", None)
    name = getattr(function, "name", None)
    arguments_raw = getattr(function, "arguments", None)
    if (
        not isinstance(call_id, str)
        or not isinstance(name, str)
        or not isinstance(arguments_raw, str)
    ):
        raise ToolArgumentError(
            name if isinstance(name, str) else None,
            call_id if isinstance(call_id, str) else None,
            arguments_raw if isinstance(arguments_raw, str) else None,
            "provider returned a malformed tool call",
        )
    definition = definitions.get(name)
    if definition is None:
        raise ToolArgumentError(
            name, call_id, arguments_raw, f"model requested unknown tool {name!r}"
        )
    try:
        parsed = cast("object", json.loads(arguments_raw))
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(
            name, call_id, arguments_raw, "tool arguments are not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolArgumentError(
            name, call_id, arguments_raw, "tool arguments must be a JSON object"
        )
    try:
        validated = (
            definition.model.model_validate(parsed) if definition.model is not None else None
        )
    except Exception as exc:
        raise ToolArgumentError(
            name, call_id, arguments_raw, f"tool arguments failed validation: {exc}"
        ) from exc
    return ToolCall(call_id, name, arguments_raw, cast("dict[str, object]", parsed), validated)


def parse_tool_calls(
    raw_calls: list[object], definitions: dict[str, ToolDefinition]
) -> tuple[list[ToolCall], list[ToolArgumentError]]:
    """Narrow a round's raw calls per call, not as an all-or-nothing batch.

    Parsing used to be a list comprehension, so the first unparseable call
    raised out of it and took its well-formed siblings with it. Nothing has
    executed at this point, which makes that lossy rather than unsafe — but
    with parallel calls one bad argument string cost the entire round and a
    re-ask. Each call is narrowed on its own instead, and the failures are
    returned beside the survivors for :class:`~llmkit.ToolCallResult` to carry.

    The one case that still raises is a round in which **every** call failed:
    the turn produced nothing actionable, so it stays a whole-round
    :class:`~llmkit.ToolArgumentError` and is re-asked on the validation
    budget exactly as before. A round with no calls at all is not a failure —
    a text-only or compose answer takes that path.

    Public-named inside a private module, the repo's convention for a seam two
    families share: :func:`~llmkit.tool_llm_call` and
    :func:`~llmkit.tool_llm_call_stream` both terminate in a
    :class:`~llmkit.ToolCallResult`, so parsing once here is what keeps the
    buffered and streamed lanes' salvage semantics from drifting apart. The
    streaming lane cannot re-ask on the validation budget (a partially consumed
    stream cannot be restarted), so the all-malformed raise reaches its caller
    unretried — the contract is the same, its enforcement differs.
    """
    parsed: list[ToolCall] = []
    invalid: list[ToolArgumentError] = []
    for raw in raw_calls:
        try:
            parsed.append(_tool_call_from_raw(raw, definitions))
        except ToolArgumentError as exc:
            invalid.append(exc)
    if invalid and not parsed:
        raise invalid[0]
    if invalid:
        logger.warning(
            "Dropped %d malformed tool call(s) from a round of %d: %s",
            len(invalid),
            len(raw_calls),
            "; ".join(f"{error.tool_name!r}: {error}" for error in invalid),
        )
    return parsed, invalid


def resolve_model_and_provider(
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


def build_text_record(
    *,
    started_at: datetime,
    feature: str,
    label: str | None,
    prompt: str | Sequence[ChatMessage],
    text: str | None,
    start_t: float,
    temperature: float | None,
    model: str | None,
    provider: LLMProviderInterface | None,
    error: str | None,
    approximate_cost: float | None,
    schema: str = "text",
    max_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    call_id: str | None = None,
    attempt: int | None = None,
) -> LLMCallRecord:
    """Build the ``LLMCallRecord`` for a plain-text/stream call.

    Shared by :func:`text_llm_call` and :func:`text_llm_call_stream`: the
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
    record).

    Building is separate from recording on purpose: the buffered path hands
    the record to :func:`~llmkit.capture.record_call_async` (off-loop I/O)
    while the abandoned-stream path must record *synchronously* — so the
    shared part is the build, and each caller picks its record step.
    """
    duration_ms = (time.monotonic() - start_t) * 1000
    resolved_model, resolved_provider = resolve_model_and_provider(model, provider)
    return LLMCallRecord(
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
        queue_wait_ms=current_queue_wait_ms(),
        run_id=get_run_id(),
    )
