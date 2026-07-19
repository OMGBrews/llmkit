"""Tests for reasoning/thinking control on the call surface.

``structured_llm_call`` / ``structured_llm_call_sync`` / ``text_llm_call``
gained a keyword-only ``reasoning_effort`` (and ``LLMClientConfig`` a
matching field). These tests pin the seams:

* the public call functions thread ``reasoning_effort`` to the transport,
* the transport only forwards ``reasoning_effort`` to the provider request
  when the resolved value is not ``None`` — so the default is byte-identical
  to the prior request (no ``reasoning_effort`` key),
* the provider's configured value applies when no per-call value is given,
  and an explicit per-call value overrides it, and
* ``LLMClientConfig.reasoning_effort`` flows into the constructed provider.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
    structured_output,
)
from llmkit.providers import GoogleProvider, OllamaProvider
from tests._support import (
    OkSchema,
    capture_structured_provider_kwargs,
    capture_text_provider_kwargs,
    capturing_sink,
)


def test_signatures_expose_reasoning_effort() -> None:
    """Acceptance: all three public call functions carry the parameter."""
    assert "reasoning_effort" in inspect.signature(structured_output.structured_llm_call).parameters
    assert (
        "reasoning_effort"
        in inspect.signature(structured_output.structured_llm_call_sync).parameters
    )
    assert "reasoning_effort" in inspect.signature(structured_output.text_llm_call).parameters


def test_config_carries_reasoning_effort() -> None:
    """``LLMClientConfig`` exposes the field, defaulting to ``None``."""
    assert LLMClientConfig(provider=Provider.GOOGLE, model="m").reasoning_effort is None
    cfg = LLMClientConfig(provider=Provider.GOOGLE, model="m", reasoning_effort="disable")
    assert cfg.reasoning_effort == "disable"


def test_build_provider_wires_reasoning_effort() -> None:
    """The config value reaches the constructed provider's property."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.GOOGLE,
            model="gemini-2.5-flash",
            api_key="k",  # bearer provider: a key is required at build, so pin one for hermeticity
            reasoning_effort="disable",
        )
    )
    assert isinstance(provider, GoogleProvider)
    assert provider.reasoning_effort == "disable"

    # Default stays None (provider default — no behaviour change).
    assert (
        build_provider(LLMClientConfig(provider=Provider.OLLAMA, model="llama3.2")).reasoning_effort
        is None
    )
    assert isinstance(
        build_provider(LLMClientConfig(provider=Provider.OLLAMA, model="llama3.2")), OllamaProvider
    )


def test_sync_call_threads_reasoning_effort_to_transport() -> None:
    """``structured_llm_call_sync(..., reasoning_effort="disable")`` forwards it."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = structured_output.structured_llm_call_sync(
            "hi", OkSchema, feature="test", reasoning_effort="disable"
        )

    assert seen["reasoning_effort"] == "disable"
    assert isinstance(result, OkSchema)
    assert result.ok is True


def test_async_call_threads_reasoning_effort_to_transport() -> None:
    """The async ``structured_llm_call(..., reasoning_effort=...)`` does the same."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[OkSchema, float | None]:
        seen.update(kwargs)
        return OkSchema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = asyncio.run(
            structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", reasoning_effort="low"
            )
        )

    assert seen["reasoning_effort"] == "low"
    assert result.ok is True


def test_transport_omits_reasoning_effort_when_unset() -> None:
    """No per-call value and a provider configured with ``None`` → no kwarg,
    byte-identical to the pre-feature request."""
    seen = capture_structured_provider_kwargs(reasoning_effort=None, provider_effort=None)
    assert "reasoning_effort" not in seen


def test_transport_includes_per_call_reasoning_effort() -> None:
    """A per-call value reaches the provider call kwargs unchanged."""
    seen = capture_structured_provider_kwargs(reasoning_effort="disable", provider_effort=None)
    assert seen["reasoning_effort"] == "disable"


def test_transport_falls_back_to_provider_reasoning_effort() -> None:
    """With no per-call value, the provider's configured value is forwarded."""
    seen = capture_structured_provider_kwargs(reasoning_effort=None, provider_effort="disable")
    assert seen["reasoning_effort"] == "disable"


def test_per_call_reasoning_effort_overrides_provider() -> None:
    """An explicit per-call value wins over the provider's configured one."""
    seen = capture_structured_provider_kwargs(reasoning_effort="high", provider_effort="disable")
    assert seen["reasoning_effort"] == "high"


def test_text_transport_forwards_reasoning_effort() -> None:
    """``acompletion_text`` forwards a resolved reasoning effort to LiteLLM."""
    seen = capture_text_provider_kwargs(provider_effort="disable")
    assert seen["reasoning_effort"] == "disable"


def test_log_record_carries_reasoning_effort() -> None:
    """The ``LLMCallRecord`` built for a structured call records the setting."""

    async def _fake_transport(*_args: object, **_kwargs: object) -> tuple[OkSchema, float | None]:
        return OkSchema(ok=True), None

    with (
        capturing_sink() as captured,
        patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport),
    ):
        _ = asyncio.run(
            structured_output.structured_llm_call(
                "hi", OkSchema, feature="test", reasoning_effort="disable"
            )
        )

    assert len(captured) == 1
    assert captured[0].reasoning_effort == "disable"
