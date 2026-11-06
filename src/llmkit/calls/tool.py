"""The tool-calling family: one tool-enabled turn, optionally composed.

Deliberately ``tool.py``, not ``tools.py``: :mod:`llmkit.tools` already exists
and this module imports six names from it, so a same-basename sibling would be
a permanent reading hazard even though absolute imports make it legal.

The only family with two result shapes — :class:`~llmkit.ToolCallResult`, or
:class:`~llmkit.ToolComposeResult` when an ``output_schema`` is supplied on a
route measured to support composing — and the only one that records the tool
list and the requested calls.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast, overload

from pydantic import BaseModel, ValidationError

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    compose_validation_budget,
    error_text,
    guard_output_schema,
    logger,
    parse_tool_calls,
    prepare_call,
    resolve_model_and_provider,
    run_with_policy,
    tool_repair_messages,
    tool_validation_budget,
    usage_log_dict,
)
from llmkit.capture import record_call_async
from llmkit.exceptions import (
    ComposeUnsupportedError,
    OutputLimitDiagnostics,
    OutputLimitError,
    SchemaEngineError,
    StructuredOutputError,
    ToolArgumentError,
    validation_failure_summary,
)
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface, Provider
from llmkit.rate_limiting import begin_queue_wait, current_queue_wait_ms
from llmkit.retry import RetryPolicy
from llmkit.run_scope import get_run_id
from llmkit.sync import run_sync
from llmkit.tools import (
    TokenUsage,
    ToolCallResult,
    ToolChoice,
    ToolComposeResult,
    ToolDefinition,
    guard_tool_choice,
)


@overload
async def tool_llm_call(
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
    output_schema: None = None,
) -> ToolCallResult: ...


@overload
async def tool_llm_call[T: BaseModel](
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
    output_schema: type[T],
) -> ToolComposeResult[T]: ...


async def tool_llm_call[T: BaseModel](
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
    output_schema: type[T] | None = None,
) -> ToolCallResult | ToolComposeResult[T]:
    """Run one tool-enabled turn, optionally accepting a schema-validated final answer.

    ``timeout`` is seconds for one provider round trip, a deadline llmkit
    enforces: a round trip still waiting at it is cut off with a
    ``TimeoutError``, which the default retry policy retries with a fresh
    deadline. The value is forwarded to LiteLLM too; ``0.0`` is accepted and
    fires at the first wait. Unset or ``None`` sets no deadline and sends no
    timeout keyword, leaving LiteLLM's default in effect.

    A round in which only *some* requested calls are malformed keeps the
    well-formed ones on ``tool_calls`` and reports the rest on
    ``invalid_calls``; a round in which every call is malformed still raises
    :class:`~llmkit.ToolArgumentError` and is re-asked on the validation
    budget with the rejected calls and validation errors attached when replayable.
    See :class:`~llmkit.ToolCallResult`.

    Tool arguments, and with ``output_schema`` the final answer, are validated
    in pydantic's strict JSON mode. An ``output_schema`` that is not a plain
    pydantic model this lane can validate raises
    :class:`~llmkit.UnsupportedSchemaError` before any request.

    With ``output_schema``, a final answer (a turn that requested no tools)
    that does not validate, or a turn with neither tool calls nor text, raises
    :class:`~llmkit.StructuredOutputError` and is re-asked on the same
    validation budget; a final answer cut off by the output-token limit raises
    :class:`~llmkit.OutputLimitError` and is not re-asked.
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
    if output_schema is not None:
        guard_output_schema(output_schema)
        if provider is None:
            # ``prepare_call`` raised already if nothing is registered, so ``None``
            # here is a registered config that would not build. Build it again so
            # its own error (a missing key, a rejected field, a missing extra)
            # surfaces before any attempt, instead of a compose refusal that
            # blames the provider. Function-local import: tests patch
            # ``llmkit.providers.build_provider`` by module attribute.
            from llmkit.providers import build_provider

            provider = build_provider()
        if not provider.compose_tools_schema:
            raise ComposeUnsupportedError(
                f"{provider.name} does not support combining tools with an output schema; "
                + "use the portable two-step pattern (tool loop, then structured_llm_call)."
            )
        provider.guard_compose_tools_schema(args.model)
    # After the compose block, so a provider it just built is guarded too. With
    # no provider (plain path, unbuildable config) the transport raises that
    # build error inside the attempt instead.
    if provider is not None:
        guard_tool_choice(provider, tool_choice)
    definitions = {definition.name: definition for definition in tools}
    attempt_count = 0
    repair: list[ChatMessage] = []

    async def _attempt() -> ToolCallResult | ToolComposeResult[T]:
        # Function-local + module-bound; see :mod:`llmkit._litellm`.
        import llmkit._litellm as _litellm

        nonlocal attempt_count, repair
        attempt_count += 1
        attempt_prompt: str | Sequence[ChatMessage] = prompt
        if repair:
            original: list[ChatMessage] = (
                [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
            )
            attempt_prompt = [*original, *repair]
        begin_queue_wait()
        started_at, start_t = datetime.now(UTC), time.monotonic()
        result: ToolCallResult | ToolComposeResult[T] | None = None
        cost: float | None = None
        error: str | None = None
        diagnostics: OutputLimitDiagnostics | None = None
        # What the transport actually sent, copied right after its await (see
        # :func:`llmkit._litellm.sent_temperature`); the caller's value until then.
        sent_temperature = args.temperature
        _litellm.begin_sent_temperature()
        try:
            text, raw_calls, stop_reason, counts, cost = await _litellm.acompletion_tools(
                prompt,
                tools,
                repair=repair,
                tool_choice=tool_choice,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                provider=provider,
                response_format=(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_schema.__name__,
                            "schema": output_schema.model_json_schema(),
                        },
                    }
                    if output_schema is not None
                    else None
                ),
                output_schema=output_schema,
                feature=feature,
                label=label,
                call_id=call_id,
                attempt=attempt_offset + attempt_count,
            )
            sent_temperature = _litellm.sent_temperature(args.temperature)
            # Only a received all-malformed round replaces repair. Transport
            # failures above retain it; all other received outcomes clear it.
            repair = []
            try:
                parsed_calls, invalid_calls = parse_tool_calls(raw_calls, definitions)
            except ToolArgumentError as exc:
                repair = tool_repair_messages(exc.round_errors, list(definitions))
                raise
            base = (
                text,
                parsed_calls,
                stop_reason if isinstance(stop_reason, str) else None,
                TokenUsage(*counts),
                invalid_calls,
            )
            if output_schema is None:
                result = ToolCallResult(*base)
            elif parsed_calls:
                result = ToolComposeResult[T](*base, parsed=None)
            else:
                # The routed LiteLLM id, as the structured lane's errors carry.
                # ``provider`` is set: the compose gate above refused ``None``.
                routed_model = (
                    provider.litellm_model(args.model) if provider is not None else args.model
                )
                provider_name = provider.name if provider is not None else None
                if stop_reason == "length":
                    # A final answer cut off by the output-token limit fails
                    # fast like the structured lane's: a re-ask with an
                    # identical budget can only truncate again. A turn that
                    # requested tools is unaffected (its reason rides on
                    # ``stop_reason``).
                    diagnostics = OutputLimitDiagnostics(
                        partial_text=text,
                        finish_reason=stop_reason,
                        prompt_tokens=counts[0],
                        completion_tokens=counts[1],
                        total_tokens=counts[2],
                    )
                    raise OutputLimitError(
                        model=routed_model or "unknown",
                        max_tokens=args.max_tokens,
                        completion_tokens=counts[1],
                        diagnostics=diagnostics,
                    )
                if text is None:
                    raise StructuredOutputError(
                        f"compose response for {output_schema.__name__} had neither "
                        + "tool calls nor text",
                        schema_name=output_schema.__name__,
                        raw_text=None,
                        errors=None,
                        attempt=attempt_offset + attempt_count,
                        model=routed_model,
                        provider=provider_name,
                    )
                try:
                    # Strict JSON mode: the structured lane's rule.
                    parsed = output_schema.model_validate_json(text, strict=True)
                except SchemaEngineError:
                    raise
                except Exception as exc:
                    raise StructuredOutputError(
                        f"compose response for {output_schema.__name__} failed validation "
                        + f"({validation_failure_summary(exc)})",
                        schema_name=output_schema.__name__,
                        raw_text=text,
                        errors=exc.errors(include_url=False)
                        if isinstance(exc, ValidationError)
                        else None,
                        attempt=attempt_offset + attempt_count,
                        model=routed_model,
                        provider=provider_name,
                    ) from exc
                result = ToolComposeResult[T](*base, parsed=parsed)
            return result
        except BaseException as exc:
            # ``BaseException``, not ``Exception``: ``asyncio.CancelledError``
            # is a BaseException, so an interrupted round skipped this handler
            # while the ``finally`` below still wrote its record — producing
            # ``error=None, response=None``, which is byte-identical to a
            # successful round that requested nothing. In a consumer's eval
            # provenance the two were indistinguishable. The round is recorded
            # as cancelled instead (recorded, not skipped: it still consumed
            # queue wait and provider time, and those belong in the log). The
            # exception itself is re-raised untouched, so control flow —
            # including cancellation semantics — is unchanged.
            #
            # The finalizer captures and submits the record, skipping its
            # delivery wait while cancellation unwinds. Accepted delivery may
            # finish later; capture alone is not evidence of persistence.
            #
            # ``str(exc)`` is empty for a bare ``CancelledError``, so the
            # message is dropped rather than logged as ``"CancelledError: "``.
            error = error_text(exc)
            raise
        finally:
            resolved_model, resolved_provider = resolve_model_and_provider(args.model, provider)
            _ = await record_call_async(
                LLMCallRecord(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    model=resolved_model,
                    provider=resolved_provider,
                    temperature=sent_temperature,
                    duration_ms=(time.monotonic() - start_t) * 1000,
                    schema=(
                        "tools" if output_schema is None else f"tools+{output_schema.__name__}"
                    ),
                    prompt=attempt_prompt,
                    response=_dump_for_log(result, feature, label),
                    error=error,
                    approximate_cost=cost,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    call_id=call_id,
                    attempt=attempt_offset + attempt_count,
                    queue_wait_ms=current_queue_wait_ms(),
                    run_id=get_run_id(),
                    tools=[definition.to_litellm() for definition in tools],
                    tool_calls=[call.to_wire() for call in result.tool_calls]
                    if result is not None
                    else None,
                    usage=usage_log_dict(
                        result.usage
                        if result is not None
                        else TokenUsage(
                            diagnostics.prompt_tokens,
                            diagnostics.completion_tokens,
                            diagnostics.total_tokens,
                        )
                        if diagnostics is not None
                        else None
                    ),
                    output_limit=diagnostics.to_log_dict() if diagnostics is not None else None,
                    partial_text=diagnostics.partial_text if diagnostics is not None else None,
                )
            )

    completed = await run_with_policy(
        _attempt,
        policy=args.retry,
        tag=label or feature,
        validation_retry_on=(
            tool_validation_budget(args.retry)
            if output_schema is None
            else compose_validation_budget(args.retry)
        ),
    )
    return completed


def _dump_for_log(
    result: ToolCallResult | None, feature: str, label: str | None
) -> dict[str, object] | None:
    """The record's ``response``, or ``None`` when the caller's schema cannot dump.

    A raising serializer on the output schema must not turn a validated final
    answer into an exception: the dump exists only for the log, as on the
    structured lane.
    """
    if result is None:
        return None
    try:
        return result.to_log_dict()
    except Exception:
        logger.warning(
            "Could not dump the %s/%s compose result for the log; recording response as null",
            feature,
            label,
            exc_info=True,
        )
        return None


@overload
def tool_llm_call_sync(
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
    output_schema: None = None,
) -> ToolCallResult: ...


@overload
def tool_llm_call_sync[T: BaseModel](
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
    output_schema: type[T],
) -> ToolComposeResult[T]: ...


def tool_llm_call_sync[T: BaseModel](
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
    output_schema: type[T] | None = None,
) -> ToolCallResult | ToolComposeResult[T]:
    """Synchronous wrapper around :func:`tool_llm_call`.

    The same call as the async form, driven on llmkit's persistent loop by
    :func:`run_sync`: same arguments, same retries, logging, rate limits and
    result. Every argument is forwarded untouched.

    ``timeout`` is seconds for one provider round trip, enforced by llmkit as
    on the async form; unset or ``None`` sets no deadline. The sync bridge adds
    no whole-call ceiling: as on the async form, the per-request timeout and
    the retry budget bound the call.
    """
    return run_sync(
        tool_llm_call(
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
            # The cast only selects an overload for the type checker; the
            # implementation takes ``None`` and returns a ToolCallResult.
            output_schema=cast("type[T]", output_schema),
        ),
        timeout=None,
    )
