"""OpenAI portable disable: capability decisions and captured request kwargs."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import litellm
import pytest
from pydantic import BaseModel

from llmkit import ReasoningEffortUnsupportedError, _litellm
from llmkit.exceptions import LLM_RECOVERABLE_ERRORS
from llmkit.providers import OpenAIProvider


def test_default_model_bundled_metadata_omits_effort() -> None:
    """The real default lacks a reasoning flag; known-model evidence must suffice."""
    provider = OpenAIProvider(api_key="test")
    assert provider.model == "gpt-4.1-mini"
    assert provider.reasoning_kwargs("disable", provider.model) == {}


def test_bundled_none_support_translates() -> None:
    provider = OpenAIProvider(api_key="test")
    assert provider.reasoning_kwargs("disable", "gpt-5.1") == {"reasoning_effort": "none"}


@pytest.mark.parametrize("model", ["gpt-5", "o3"])
def test_bundled_required_reasoning_refuses(model: str) -> None:
    provider = OpenAIProvider(api_key="test")
    with pytest.raises(ReasoningEffortUnsupportedError) as caught:
        _ = provider.reasoning_kwargs("disable", model)
    assert model in str(caught.value)
    assert "cannot" in str(caught.value).lower()
    assert not isinstance(caught.value, LLM_RECOVERABLE_ERRORS)


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "future-value"])
def test_native_efforts_pass_through_without_lookup(effort: str) -> None:
    provider = OpenAIProvider(api_key="test")
    with (
        patch("litellm.get_model_info") as metadata,
        patch("litellm.get_supported_openai_params") as params,
    ):
        assert provider.reasoning_kwargs(effort, "gateway-alias") == {"reasoning_effort": effort}
    metadata.assert_not_called()
    params.assert_not_called()


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"supports_reasoning": False}, {}),
        ({"supports_none_reasoning_effort": True}, {"reasoning_effort": "none"}),
        (
            {"supports_reasoning": True, "supports_none_reasoning_effort": True},
            {"reasoning_effort": "none"},
        ),
    ],
)
def test_explicit_capabilities(metadata: dict[str, object], expected: dict[str, str]) -> None:
    provider = OpenAIProvider(api_key="test")
    with patch("litellm.get_model_info", return_value=metadata) as lookup:
        assert provider.reasoning_kwargs("disable", "test-model") == expected
    lookup.assert_called_once_with("test-model", custom_llm_provider="openai")


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"supports_reasoning": None},
        {"supports_reasoning": None, "supports_none_reasoning_effort": None},
        {"supports_reasoning": True},
        {"supports_reasoning": True, "supports_none_reasoning_effort": None},
        {"supports_reasoning": True, "supports_none_reasoning_effort": False},
        {"supports_reasoning": False, "supports_none_reasoning_effort": True},
        {"supports_reasoning": False, "supports_minimal_reasoning_effort": True},
        {"supports_none_reasoning_effort": True, "thinking_always_on": True},
        {"supports_reasoning": False, "thinking_always_on": True},
        {
            "supports_reasoning": True,
            "supports_none_reasoning_effort": True,
            "supported_openai_params": ["temperature"],
        },
    ],
)
def test_incomplete_or_contradictory_metadata_refuses(metadata: dict[str, object]) -> None:
    provider = OpenAIProvider(api_key="test")
    with (
        patch("litellm.get_model_info", return_value=metadata),
        patch("litellm.get_supported_openai_params", return_value=["temperature"]),
        pytest.raises(ReasoningEffortUnsupportedError, match="test-model"),
    ):
        _ = provider.reasoning_kwargs("disable", "test-model")


def test_unknown_alias_refuses_even_if_config_excludes_reasoning() -> None:
    provider = OpenAIProvider(api_key="test", base_url="https://gateway.example/v1")
    with (
        patch("litellm.get_supported_openai_params", return_value=["temperature"]),
        pytest.raises(ReasoningEffortUnsupportedError) as caught,
    ):
        _ = provider.reasoning_kwargs("disable", "unregistered-gateway-alias")
    assert "unregistered-gateway-alias" in str(caught.value)
    assert "unknown" in str(caught.value).lower()


def test_metadata_lookup_failure_refuses() -> None:
    provider = OpenAIProvider(api_key="test")
    with (
        patch("litellm.get_model_info", side_effect=ValueError("unavailable metadata")),
        pytest.raises(ReasoningEffortUnsupportedError, match=r"gpt-4\.1-mini"),
    ):
        _ = provider.reasoning_kwargs("disable", provider.model)


@pytest.mark.parametrize("supported", [None, ["temperature", "reasoning_effort"]])
def test_missing_flag_requires_corroborating_config(supported: list[str] | None) -> None:
    provider = OpenAIProvider(api_key="test")
    metadata = dict(litellm.get_model_info(provider.model, custom_llm_provider="openai"))
    metadata["supported_openai_params"] = supported
    with (
        patch("litellm.get_model_info", return_value=metadata),
        pytest.raises(ReasoningEffortUnsupportedError, match=r"gpt-4\.1-mini"),
    ):
        _ = provider.reasoning_kwargs("disable", provider.model)


class _Answer(BaseModel):
    ok: bool


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-5.1"])
def test_structured_configured_and_per_call_disable_reach_transport(
    configured: bool, model: str
) -> None:
    # A required-reasoning default makes ignoring the per-call model observable.
    provider = OpenAIProvider(
        api_key="test", model="gpt-5", reasoning_effort="disable" if configured else "high"
    )
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
                model=model,
                reasoning_effort=None if configured else "disable",
                provider=provider,
            )
        )
    assert parsed.ok
    assert captured["model"] == f"openai/{model}"
    if model == "gpt-5.1":
        assert captured["reasoning_effort"] == "none"
    else:
        assert "reasoning_effort" not in captured


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("model", ["gpt-5", "o3", "unregistered-gateway-alias"])
def test_rejected_model_override_never_calls_transport(configured: bool, model: str) -> None:
    provider = OpenAIProvider(api_key="test", reasoning_effort="disable" if configured else "high")
    with patch("llmkit._litellm.litellm.acompletion") as transport:
        with pytest.raises(ReasoningEffortUnsupportedError, match=model):
            _ = asyncio.run(
                _litellm.acompletion_structured(
                    "hi",
                    _Answer,
                    temperature=None,
                    model=model,
                    reasoning_effort=None if configured else "disable",
                    provider=provider,
                )
            )
        transport.assert_not_called()


def test_native_per_call_effort_overrides_configured_disable() -> None:
    provider = OpenAIProvider(api_key="test", reasoning_effort="disable")
    captured: dict[str, object] = {}

    async def complete(**kwargs: object) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ]
        )

    with (
        patch("llmkit._litellm.litellm.acompletion", complete),
        patch("litellm.get_model_info") as lookup,
    ):
        answer, _, _ = asyncio.run(
            _litellm.acompletion_text(
                "hi",
                temperature=None,
                model="gateway-alias",
                reasoning_effort="future-value",
                provider=provider,
            )
        )
    assert answer == "hello"
    assert captured["reasoning_effort"] == "future-value"
    assert captured["model"] == "openai/gateway-alias"
    lookup.assert_not_called()
