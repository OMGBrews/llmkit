"""The tool-calling family: one tool-enabled turn, optionally composed.

Deliberately ``tool.py``, not ``tools.py``: :mod:`llmkit.tools` already exists
and this module imports six names from it, so a same-basename sibling would be
a permanent reading hazard even though absolute imports make it legal.

The only family with two result shapes — :class:`~llmkit.ToolCallResult`, or
:class:`~llmkit.ToolComposeResult` when an ``output_schema`` is supplied on a
route measured to support composing — and the only one that records the tool
list, the requested calls and the turn's token usage.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import overload

from pydantic import BaseModel

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    parse_tool_calls,
    prepare_call,
    resolve_model_and_provider,
    result_validation_budget,
    run_with_policy,
    tool_validation_budget,
)
from llmkit.capture import record_call_async
from llmkit.exceptions import ComposeUnsupportedError, ResultValidationError
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset
from llmkit.providers import LLMProviderInterface
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
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
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
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
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
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
    output_schema: type[T] | None = None,
) -> ToolCallResult | ToolComposeResult[T]:
    """Run one tool-enabled turn, optionally accepting a schema-validated final answer.

    A round in which only *some* requested calls are malformed keeps the
    well-formed ones on ``tool_calls`` and reports the rest on
    ``invalid_calls``; a round in which every call is malformed still raises
    :class:`~llmkit.ToolArgumentError` and is re-asked on the validation
    budget. See :class:`~llmkit.ToolCallResult`.
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
    if output_schema is not None:
        if provider is None or not provider.compose_tools_schema:
            name = "the configured provider" if provider is None else provider.name
            raise ComposeUnsupportedError(
                f"{name} does not support combining tools with an output schema; use the "
                + "portable two-step pattern (tool loop, then structured_llm_call)."
            )
        provider.guard_compose_tools_schema(args.model)
    definitions = {definition.name: definition for definition in tools}
    attempt_count = 0

    async def _attempt() -> ToolCallResult | ToolComposeResult[T]:
        # Function-local + module-bound; see :mod:`llmkit._litellm`.
        import llmkit._litellm as _litellm

        nonlocal attempt_count
        attempt_count += 1
        begin_queue_wait()
        started_at, start_t = datetime.now(UTC), time.monotonic()
        result: ToolCallResult | ToolComposeResult[T] | None = None
        cost: float | None = None
        error: str | None = None
        try:
            text, raw_calls, stop_reason, counts, cost = await _litellm.acompletion_tools(
                prompt,
                tools,
                tool_choice=tool_choice,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens,
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
            )
            parsed_calls, invalid_calls = parse_tool_calls(raw_calls, definitions)
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
                if text is None:
                    raise ResultValidationError("compose response had neither tool calls nor text")
                try:
                    parsed = output_schema.model_validate_json(text)
                except Exception as exc:
                    raise ResultValidationError(
                        f"compose response failed validation: {exc}"
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
            # The record still lands: ``record_call_async`` offloads the write
            # to a thread that runs to completion, and the single delivered
            # cancellation has already been consumed by the time the
            # ``finally`` awaits.
            #
            # ``str(exc)`` is empty for a bare ``CancelledError``, so the
            # message is dropped rather than logged as ``"CancelledError: "``.
            error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
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
                    temperature=args.temperature,
                    duration_ms=(time.monotonic() - start_t) * 1000,
                    schema=(
                        "tools" if output_schema is None else f"tools+{output_schema.__name__}"
                    ),
                    prompt=prompt,
                    response=result.to_log_dict() if result is not None else None,
                    error=error,
                    approximate_cost=cost,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    call_id=call_id,
                    attempt=attempt_count,
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
            )

    completed = await run_with_policy(
        _attempt,
        policy=args.retry,
        tag=label or feature,
        validation_retry_on=(
            tool_validation_budget(args.retry)
            if output_schema is None
            else result_validation_budget(args.retry)
        ),
    )
    return completed


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
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
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
    reasoning_effort: ReasoningEffort | None | Unset = UNSET,
    provider: LLMProviderInterface | None | Unset = UNSET,
    retry: RetryPolicy | Unset = UNSET,
    options: LLMCallOptions | None = None,
    output_schema: type[T],
) -> ToolComposeResult[T]: ...


def tool_llm_call_sync(
    *args: object, **kwargs: object
) -> ToolCallResult | ToolComposeResult[BaseModel]:
    """Synchronous wrapper around :func:`tool_llm_call`."""
    return run_sync(tool_llm_call(*args, **kwargs))  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownArgumentType]  # dynamic forwarding wrapper
