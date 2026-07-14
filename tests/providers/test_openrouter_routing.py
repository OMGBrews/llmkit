"""Tests for OpenRouter schema-honoring routing (``require_parameters``).

OpenRouter exposes ``structured_outputs`` as a *model-level* capability, but
the strict ``response_format`` is enforced by whichever *serving* provider the
request is routed to — so a request can silently land on an endpoint that
ignores the schema and returns free-form JSON, surfacing only as a confusing
downstream validation failure. The :class:`~llmkit.OpenRouterProvider` defends
against this by setting OpenRouter's ``provider.require_parameters`` routing
preference (via ``extra_body``) by default, restricting routing to endpoints
that honor every requested parameter.

Routing is only half the defense: OpenRouter treats a non-strict
``json_schema`` ``response_format`` as *advisory*, and instructor's core
``JSON_SCHEMA`` handler sends neither ``"strict": true`` nor
``additionalProperties: false`` (the removed ``OPENROUTER_STRUCTURED_OUTPUTS``
handler sent both) — measured 2026-07-14: ``mistralai/mistral-nemo``
stochastically echoing the schema itself. The provider therefore opts into
``strict_json_schema``, and ``llmkit._litellm`` upgrades the
``response_format`` at the completion seam.

These tests pin: the preference is on by default, can be opted out, survives
the config-build path, and actually threads through to the LiteLLM call — and
that the strict upgrade reaches the wire for OpenRouter (through the *real*
instructor client) while leaving non-opted-in providers' requests untouched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import litellm
from pydantic import BaseModel

from llmkit import _litellm
from llmkit.providers import (
    LLMClientConfig,
    LLMProviderInterface,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
    build_provider,
)

_ROUTING_PREF = {"provider": {"require_parameters": True}}


def test_require_parameters_on_by_default() -> None:
    """The default OpenRouter provider carries the schema-honoring routing pref."""
    provider = OpenRouterProvider(api_key="k", model="x/y")
    kwargs = provider.completion_kwargs()
    assert kwargs["extra_body"] == _ROUTING_PREF
    assert kwargs["api_key"] == "k"


def test_require_parameters_can_be_opted_out() -> None:
    """``require_parameters=False`` drops the routing pref entirely."""
    provider = OpenRouterProvider(api_key="k", model="x/y", require_parameters=False)
    kwargs = provider.completion_kwargs()
    assert "extra_body" not in kwargs
    assert kwargs == {"api_key": "k", "api_base": "https://openrouter.ai/api/v1"}


def test_config_build_keeps_require_parameters_on() -> None:
    """The config-build path (``build_provider``) keeps routing on by default."""
    provider = build_provider(LLMClientConfig(provider=Provider.OPENROUTER, api_key="k"))
    assert isinstance(provider, OpenRouterProvider)
    assert provider.completion_kwargs()["extra_body"] == _ROUTING_PREF


def test_routing_preference_threads_into_litellm_call() -> None:
    """The ``extra_body`` routing pref reaches the underlying LiteLLM call.

    Proves the preference isn't merely cosmetic: a real call forwards it as a
    completion kwarg, so OpenRouter actually receives the routing instruction.
    """
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    provider = OpenRouterProvider(api_key="k", model="x/y")
    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text("hi", temperature=0.0, model=None, provider=provider)
        )

    assert text == "hi"
    assert captured["extra_body"] == _ROUTING_PREF
    assert captured["api_key"] == "k"


class _OkSchema(BaseModel):
    ok: bool


def _drive_structured_call(provider: object) -> dict[str, object]:
    """Run a structured call through the *real* instructor client and return
    the kwargs the (stubbed) LiteLLM transport received.

    Unlike the shared harness — which patches ``instructor.from_litellm``
    wholesale — this stubs only ``litellm.acompletion``, so instructor's real
    mode handler builds the request and its real parse path validates the
    response. That is the layer the strict upgrade sits under, so this is the
    only kind of offline test that can see it.
    """
    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> litellm.ModelResponse:
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

    with patch("llmkit._litellm.litellm.acompletion", _fake_acompletion):
        parsed, _cost = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                _OkSchema,
                temperature=0.0,
                model=None,
                provider=cast("LLMProviderInterface", provider),
            )
        )
    assert parsed.ok is True
    return captured


def test_strict_json_schema_upgrade_reaches_the_wire() -> None:
    """OpenRouter's structured requests carry ``strict`` + closed properties.

    The exact wire shape the removed ``OPENROUTER_STRUCTURED_OUTPUTS`` handler
    sent, restored at the completion seam.
    """
    captured = _drive_structured_call(OpenRouterProvider(api_key="k", model="x/y"))

    response_format = cast("dict[str, object]", captured["response_format"])
    assert response_format["type"] == "json_schema"
    json_schema = cast("dict[str, object]", response_format["json_schema"])
    assert json_schema["strict"] is True
    schema = cast("dict[str, object]", json_schema["schema"])
    assert schema["additionalProperties"] is False


def test_non_strict_provider_request_is_untouched() -> None:
    """A provider that didn't opt in (OpenAI) keeps instructor's request as-is.

    Guards the trait's scoping: the strict upgrade must never leak into
    providers whose enforcement was measured fine without it.
    """
    captured = _drive_structured_call(OpenAIProvider(api_key="k", model="gpt-4.1-mini"))

    response_format = cast("dict[str, object]", captured["response_format"])
    assert response_format["type"] == "json_schema"
    json_schema = cast("dict[str, object]", response_format["json_schema"])
    assert "strict" not in json_schema
    assert "additionalProperties" not in cast("dict[str, object]", json_schema["schema"])
