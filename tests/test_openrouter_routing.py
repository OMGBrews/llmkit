"""Tests for OpenRouter schema-honoring routing (``require_parameters``).

OpenRouter exposes ``structured_outputs`` as a *model-level* capability, but
the strict ``response_format`` is enforced by whichever *serving* provider the
request is routed to — so a request can silently land on an endpoint that
ignores the schema and returns free-form JSON, surfacing only as a confusing
downstream validation failure. The :class:`~llmkit.OpenRouterProvider` defends
against this by setting OpenRouter's ``provider.require_parameters`` routing
preference (via ``extra_body``) by default, restricting routing to endpoints
that honor every requested parameter.

These tests pin: the preference is on by default, can be opted out, survives
the config-build path, and actually threads through to the LiteLLM call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from llmkit import _litellm
from llmkit.providers import (
    LLMClientConfig,
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
