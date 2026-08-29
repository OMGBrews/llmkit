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

import json
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast, overload

from pydantic import BaseModel

from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.calls._shared import (
    build_call_provider,
    resolve_model_and_provider,
    result_validation_budget,
    tool_validation_budget,
)
from llmkit.capture import record_call_async
from llmkit.exceptions import ComposeUnsupportedError, ResultValidationError, ToolArgumentError
from llmkit.logging import LLMCallRecord
from llmkit.options import UNSET, LLMCallOptions, Unset, resolve_call_args
from llmkit.providers import LLMProviderInterface
from llmkit.rate_limiting import begin_queue_wait, current_queue_wait_ms
from llmkit.retry import RetryPolicy, with_retries
from llmkit.run_scope import get_run_id
from llmkit.sync import run_sync
from llmkit.tools import (
    TokenUsage,
    ToolCall,
    ToolCallResult,
    ToolChoice,
    ToolComposeResult,
    ToolDefinition,
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
    """Run one tool-enabled turn, optionally accepting a schema-validated final answer."""
    resolved = resolve_call_args(
        options,
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        retry=retry,
        provider=provider,
    )
    provider = build_call_provider(resolved.provider)
    if output_schema is not None:
        if provider is None or not provider.compose_tools_schema:
            name = "the configured provider" if provider is None else provider.name
            raise ComposeUnsupportedError(
                f"{name} does not support combining tools with an output schema; use the "
                + "portable two-step pattern (tool loop, then structured_llm_call)."
            )
        provider.guard_compose_tools_schema(resolved.model)
    definitions = {definition.name: definition for definition in tools}
    call_id = uuid.uuid4().hex
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
                temperature=resolved.temperature,
                model=resolved.model,
                max_tokens=resolved.max_tokens,
                reasoning_effort=resolved.reasoning_effort,
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
            parsed_calls = [_tool_call_from_raw(raw, definitions) for raw in raw_calls]
            base = (
                text,
                parsed_calls,
                stop_reason if isinstance(stop_reason, str) else None,
                TokenUsage(*counts),
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
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            resolved_model, resolved_provider = resolve_model_and_provider(resolved.model, provider)
            _ = await record_call_async(
                LLMCallRecord(
                    started_at=started_at,
                    feature=feature,
                    label=label,
                    model=resolved_model,
                    provider=resolved_provider,
                    temperature=resolved.temperature,
                    duration_ms=(time.monotonic() - start_t) * 1000,
                    schema=(
                        "tools" if output_schema is None else f"tools+{output_schema.__name__}"
                    ),
                    prompt=prompt,
                    response=result.to_log_dict() if result is not None else None,
                    error=error,
                    approximate_cost=cost,
                    max_tokens=resolved.max_tokens,
                    reasoning_effort=resolved.reasoning_effort,
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

    completed = await with_retries(
        _attempt,
        max_attempts=resolved.retry.max_attempts,
        label=label or feature,
        backoff_base_seconds=resolved.retry.backoff_base_seconds,
        max_backoff_seconds=resolved.retry.max_backoff_seconds,
        retry_after_cap=resolved.retry.retry_after_cap,
        retry_on=resolved.retry.retry_on,
        validation_max_attempts=resolved.retry.validation_max_attempts,
        validation_retry_on=(
            tool_validation_budget(resolved.retry)
            if output_schema is None
            else result_validation_budget(resolved.retry)
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
