"""Per-request timeouts merge once and survive every transport lane."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from llmkit import UNSET, LLMCallOptions, ToolDefinition, Unset, calls
from llmkit.options import resolve_call_args
from llmkit.providers import OpenAIProvider
from llmkit.testing import Reply, install
from tests._support import OkSchema

_TOOLS = [ToolDefinition("lookup", "", {"type": "object"})]
_LANES = ("text", "structured", "tool", "text_stream", "tool_stream")


def test_signatures_expose_timeout() -> None:
    for surface in (
        calls.structured_llm_call,
        calls.structured_llm_call_sync,
        calls.text_llm_call,
        calls.text_llm_call_sync,
        calls.text_llm_call_stream,
        calls.stream_text_with_log,
        calls.tool_llm_call,
        calls.tool_llm_call_stream,
    ):
        assert "timeout" in inspect.signature(surface).parameters
    # The sync tool implementation forwards dynamically; its overloads are
    # checked by _typecheck_timeout_accepted_on_every_lane.
    assert "kwargs" in inspect.signature(calls.tool_llm_call_sync).parameters


@pytest.mark.parametrize(
    ("keyword", "option", "expected"),
    [(UNSET, UNSET, None), (UNSET, 4.0, 4.0), (1.5, 4.0, 1.5), (None, 4.0, None)],
)
def test_timeout_precedence(
    keyword: float | None | Unset, option: float | None | Unset, expected: float | None
) -> None:
    resolved = resolve_call_args(
        LLMCallOptions(timeout=option),
        timeout=keyword,
        temperature=UNSET,
        model=UNSET,
        max_tokens=UNSET,
        reasoning_effort=UNSET,
        retry=UNSET,
        provider=UNSET,
        cache=UNSET,
    )
    assert resolved.timeout == expected


async def _call(lane: str, timeout: float | None | Unset) -> None:
    provider = OpenAIProvider(api_key="k", model="gpt-4.1-mini")
    if lane == "text":
        assert await calls.text_llm_call("hi", feature="test", timeout=timeout, provider=provider)
    elif lane == "structured":
        parsed = await calls.structured_llm_call(
            "hi", OkSchema, feature="test", timeout=timeout, provider=provider
        )
        assert parsed.ok
    elif lane == "tool":
        _ = await calls.tool_llm_call(
            "hi", _TOOLS, feature="test", timeout=timeout, provider=provider
        )
    elif lane == "text_stream":
        chunks = [
            chunk
            async for chunk in calls.text_llm_call_stream(
                "hi", feature="test", timeout=timeout, provider=provider
            )
        ]
        assert chunks == ["ok"]
    else:
        events = [
            event
            async for event in calls.tool_llm_call_stream(
                "hi", _TOOLS, feature="test", timeout=timeout, provider=provider
            )
        ]
        assert events


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize("timeout", [UNSET, None, 0.0, 1.25])
@pytest.mark.parametrize("provider_timeout", [False, True])
def test_timeout_reaches_litellm_on_every_lane(
    lane: str, timeout: float | None | Unset, provider_timeout: bool
) -> None:
    """Only LiteLLM is stubbed: structured calls use the real instructor client."""
    captured: list[dict[str, object]] = []

    class _Stream:
        def __aiter__(self) -> AsyncIterator[object]:
            async def _iterate() -> AsyncIterator[object]:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="ok", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

            return _iterate()

    async def _fake_acompletion(**kwargs: object) -> object:
        captured.append(kwargs)
        if kwargs.get("stream"):
            return _Stream()
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                }
            ]
        )

    original = OpenAIProvider.completion_kwargs

    def _provider_kwargs(provider: OpenAIProvider) -> dict[str, object]:
        kwargs = original(provider)
        if provider_timeout:
            kwargs["timeout"] = 9.0
        return kwargs

    with (
        patch("llmkit._litellm.litellm.acompletion", _fake_acompletion),
        patch.object(OpenAIProvider, "completion_kwargs", _provider_kwargs),
    ):
        asyncio.run(_call(lane, timeout))
    assert len(captured) == 1
    assert captured[0]["messages"]
    if timeout is UNSET or timeout is None:
        if provider_timeout:
            assert captured[0]["timeout"] == 9.0
        else:
            assert "timeout" not in captured[0]
    else:
        assert captured[0]["timeout"] == timeout


@pytest.mark.parametrize("lane", ["text", "structured", "tool"])
def test_options_timeout_reaches_sync_fake_request(lane: str) -> None:
    with install(lambda _request: Reply.json(OkSchema(ok=True))) as installed:
        opts = LLMCallOptions(timeout=2.5)
        if lane == "text":
            _ = calls.text_llm_call_sync("hi", feature="test", options=opts)
        elif lane == "structured":
            _ = calls.structured_llm_call_sync("hi", OkSchema, feature="test", options=opts)
        else:
            _ = calls.tool_llm_call_sync("hi", _TOOLS, feature="test", options=opts)
    assert [request.timeout for request in installed.requests] == [2.5]


def test_fake_delay_does_not_enforce_provider_timeout() -> None:
    with install(lambda _request: Reply.text("ok", delay=0.01)) as installed:
        assert calls.text_llm_call_sync("hi", feature="test", timeout=0.001) == "ok"
    assert [request.timeout for request in installed.requests] == [0.001]


def test_deprecated_alias_forwards_timeout() -> None:
    async def _drive() -> list[str]:
        return [
            chunk async for chunk in calls.stream_text_with_log("hi", feature="test", timeout=3.0)
        ]

    with install(lambda _request: Reply.text("ok")) as installed:
        with pytest.warns(DeprecationWarning):
            assert asyncio.run(_drive()) == ["ok"]
    assert [request.timeout for request in installed.requests] == [3.0]
