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
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llmkit import (
    LLMCallRecord,
    LLMClientConfig,
    LocalYamlLogSink,
    Provider,
    build_provider,
    configure_llm_logging,
    structured_output,
)
from llmkit.logging import LogSink
from llmkit.providers import GoogleProvider, OllamaProvider


class _Schema(BaseModel):
    ok: bool


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
            provider=Provider.GOOGLE, model="gemini-2.5-flash", reasoning_effort="disable"
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

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        seen.update(kwargs)
        return _Schema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = structured_output.structured_llm_call_sync(
            "hi", _Schema, feature="test", reasoning_effort="disable"
        )

    assert seen["reasoning_effort"] == "disable"
    assert isinstance(result, _Schema)
    assert result.ok is True


def test_async_call_threads_reasoning_effort_to_transport() -> None:
    """The async ``structured_llm_call(..., reasoning_effort=...)`` does the same."""
    seen: dict[str, object] = {}

    async def _fake_transport(*_args: object, **kwargs: object) -> tuple[_Schema, float | None]:
        seen.update(kwargs)
        return _Schema(ok=True), None

    with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
        result = asyncio.run(
            structured_output.structured_llm_call(
                "hi", _Schema, feature="test", reasoning_effort="low"
            )
        )

    assert seen["reasoning_effort"] == "low"
    assert result.ok is True


def test_transport_omits_reasoning_effort_when_unset() -> None:
    """No per-call value and a provider configured with ``None`` → no kwarg,
    byte-identical to the pre-feature request."""
    seen = _capture_provider_kwargs(reasoning_effort=None, provider_effort=None)
    assert "reasoning_effort" not in seen


def test_transport_includes_per_call_reasoning_effort() -> None:
    """A per-call value reaches the provider call kwargs unchanged."""
    seen = _capture_provider_kwargs(reasoning_effort="disable", provider_effort=None)
    assert seen["reasoning_effort"] == "disable"


def test_transport_falls_back_to_provider_reasoning_effort() -> None:
    """With no per-call value, the provider's configured value is forwarded."""
    seen = _capture_provider_kwargs(reasoning_effort=None, provider_effort="disable")
    assert seen["reasoning_effort"] == "disable"


def test_per_call_reasoning_effort_overrides_provider() -> None:
    """An explicit per-call value wins over the provider's configured one."""
    seen = _capture_provider_kwargs(reasoning_effort="high", provider_effort="disable")
    assert seen["reasoning_effort"] == "high"


def test_text_transport_forwards_reasoning_effort() -> None:
    """``acompletion_text`` forwards a resolved reasoning effort to LiteLLM."""
    from llmkit import _litellm

    seen: dict[str, object] = {}

    async def _fake_acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        resp = MagicMock(_hidden_params={})
        resp.choices = [MagicMock(message=MagicMock(content="hello"))]
        return resp

    provider = MagicMock()
    provider.completion_kwargs.return_value = {"api_key": "k", "api_base": None}
    provider.litellm_model.return_value = "fake/model"
    provider.reasoning_effort = "disable"

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text("hi", temperature=0.0, model=None, provider=provider)
        )

    assert text == "hello"
    assert seen["reasoning_effort"] == "disable"


def test_log_record_carries_reasoning_effort(tmp_path: Path) -> None:
    """The ``LLMCallRecord`` built for a structured call records the setting."""
    captured: list[LLMCallRecord] = []

    class _CapturingSink:
        def write(self, record: LLMCallRecord) -> None:
            captured.append(record)

    async def _fake_transport(*_args: object, **_kwargs: object) -> tuple[_Schema, float | None]:
        return _Schema(ok=True), None

    sink: LogSink = _CapturingSink()
    configure_llm_logging(sink)
    try:
        with patch("llmkit._litellm.acompletion_structured", side_effect=_fake_transport):
            asyncio.run(
                structured_output.structured_llm_call(
                    "hi", _Schema, feature="test", reasoning_effort="disable"
                )
            )
    finally:
        configure_llm_logging(LocalYamlLogSink())

    assert len(captured) == 1
    assert captured[0].reasoning_effort == "disable"


def _capture_provider_kwargs(
    *, reasoning_effort: str | None, provider_effort: str | None
) -> dict[str, object]:
    """Drive ``acompletion_structured`` over a faked instructor client and
    return the kwargs the provider call (``create_with_completion``) saw.

    ``provider_effort`` is the provider's configured value; ``reasoning_effort``
    is the per-call override.
    """
    from llmkit import _litellm

    seen: dict[str, object] = {}

    async def _fake_create_with_completion(
        **kwargs: object,
    ) -> tuple[_Schema, MagicMock]:
        seen.update(kwargs)
        return _Schema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat.completions.create_with_completion = _fake_create_with_completion

    provider = MagicMock()
    provider.completion_kwargs.return_value = {"api_key": "k", "api_base": None}
    provider.instructor_mode = "json"
    provider.litellm_model.return_value = "fake/model"
    provider.reasoning_effort = provider_effort

    with patch("llmkit._litellm.instructor.from_litellm", return_value=fake_client):
        parsed, _cost = asyncio.run(
            _litellm.acompletion_structured(
                "hi",
                _Schema,
                temperature=0.0,
                model=None,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        )
    assert parsed.ok is True
    return seen
