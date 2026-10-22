"""Portable disable on Claude routes, including the real request seam."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import patch

import litellm
import pytest
from pydantic import BaseModel

from llmkit import ReasoningEffortUnsupportedError, _litellm
from llmkit.exceptions import LLM_RECOVERABLE_ERRORS
from llmkit.providers import AnthropicProvider, BedrockProvider, LLMProviderInterface


@pytest.fixture(params=["anthropic", "bedrock"])
def provider(request: pytest.FixtureRequest) -> LLMProviderInterface:
    if cast("str", request.param) == "anthropic":
        return AnthropicProvider(api_key="test", model="claude-haiku-4-5-20251001")
    return BedrockProvider(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")


def test_translate_and_pass_through(provider: LLMProviderInterface) -> None:
    assert provider.reasoning_kwargs("disable", provider.model) == {"reasoning_effort": "none"}
    for effort in ("low", "max", "future-value"):
        with patch("litellm.get_model_info") as lookup:
            assert provider.reasoning_kwargs(effort, provider.model) == {"reasoning_effort": effort}
            lookup.assert_not_called()


def _always_on(provider: LLMProviderInterface) -> str:
    if isinstance(provider, AnthropicProvider):
        return "claude-opus-5-5"
    return "us.anthropic.claude-opus-5-5"


def test_always_on_bundled_model_map(provider: LLMProviderInterface) -> None:
    with pytest.raises(ReasoningEffortUnsupportedError, match=r'cannot turn thinking off.*"low"'):
        _ = provider.reasoning_kwargs("disable", _always_on(provider))
    assert not isinstance(ReasoningEffortUnsupportedError(), LLM_RECOVERABLE_ERRORS)


@pytest.mark.parametrize(
    "metadata", [{}, {"thinking_always_on": None}, {"thinking_always_on": False}]
)
def test_unknown_metadata_is_permissive(
    provider: LLMProviderInterface, metadata: dict[str, object]
) -> None:
    route = "anthropic" if isinstance(provider, AnthropicProvider) else "bedrock"
    with patch("litellm.get_model_info", return_value=metadata) as lookup:
        assert provider.reasoning_kwargs("disable", "future-model") == {"reasoning_effort": "none"}
        lookup.assert_called_once_with("future-model", custom_llm_provider=route)


def test_lookup_failure_is_permissive(provider: LLMProviderInterface) -> None:
    with patch("litellm.get_model_info", side_effect=ValueError("unknown model")):
        assert provider.reasoning_kwargs("disable", "future-model") == {"reasoning_effort": "none"}


class _Answer(BaseModel):
    ok: bool


@pytest.mark.parametrize("configured", [True, False])
def test_structured_config_and_override_reach_transport(
    provider: LLMProviderInterface, configured: bool
) -> None:
    if configured:
        if isinstance(provider, AnthropicProvider):
            provider = AnthropicProvider(
                api_key="test", model=provider.model, reasoning_effort="disable"
            )
        else:
            provider = BedrockProvider(model=provider.model, reasoning_effort="disable")
    captured: dict[str, object] = {}

    async def complete(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                }
            ]
        )

    with patch("llmkit._litellm.litellm.acompletion", complete):
        parsed, _, _ = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                _Answer,
                temperature=None,
                model=None,
                reasoning_effort=None if configured else "disable",
                provider=provider,
            )
        )
    assert parsed.ok
    assert captured["reasoning_effort"] == "none"


def test_model_override_refused_before_transport(provider: LLMProviderInterface) -> None:
    with patch("llmkit._litellm.litellm.acompletion") as transport:
        with pytest.raises(ReasoningEffortUnsupportedError):
            _ = asyncio.run(
                _litellm.acompletion_structured(
                    "hi",
                    _Answer,
                    temperature=None,
                    model=_always_on(provider),
                    reasoning_effort="disable",
                    provider=provider,
                )
            )
        transport.assert_not_called()
