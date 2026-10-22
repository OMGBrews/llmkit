"""The funnel opts tool requests out of LiteLLM's proxy MCP gateway check.

``llmkit._litellm._acompletion`` sets LiteLLM's per-call
``_skip_mcp_handler=True`` on every request carrying ``tools`` (see its
docstring and ``tests/packaging/test_litellm_mcp_handler.py``, which runs the
real LiteLLM). These tests pin which requests carry the switch, over a scripted
``litellm.acompletion``.
"""

from __future__ import annotations

from unittest.mock import patch

from litellm import exceptions as litellm_errors
from litellm.types.utils import ModelResponse

from llmkit import _litellm

_TOOLS: list[dict[str, object]] = [
    {"type": "function", "function": {"name": "lookup", "parameters": {}}}
]


async def _send(*errors: Exception, **request: object) -> list[dict[str, object]]:
    """Run *request* through the funnel; return every request LiteLLM received."""
    received: list[dict[str, object]] = []
    pending = list(errors)

    async def fake(**sent: object) -> ModelResponse:
        received.append(sent)
        if pending:
            raise pending.pop(0)
        return ModelResponse()

    with patch("llmkit._litellm.litellm.acompletion", side_effect=fake):
        _ = await _litellm._acompletion(model="openai/m", messages=[], **request)
    return received


async def test_a_tool_request_skips_the_mcp_handler() -> None:
    [sent] = await _send(tools=_TOOLS)

    assert sent["_skip_mcp_handler"] is True


async def test_a_request_without_tools_is_unchanged() -> None:
    requests: list[dict[str, object]] = [{}, {"tools": []}, {"tools": None}]
    for request in requests:
        [sent] = await _send(**request)

        assert "_skip_mcp_handler" not in sent


async def test_the_temperature_reissue_keeps_the_switch() -> None:
    refusal = litellm_errors.UnsupportedParamsError(
        message="gpt-5-mini doesn't support temperature=0.0", llm_provider="openai", model="m"
    )

    refused, reissued = await _send(refusal, tools=_TOOLS, temperature=0.0)

    assert refused["_skip_mcp_handler"] is True
    assert reissued["_skip_mcp_handler"] is True
    assert reissued["additional_drop_params"] == ["temperature"]


async def test_a_caller_supplied_switch_is_kept() -> None:
    [sent] = await _send(tools=_TOOLS, _skip_mcp_handler=False)

    assert sent["_skip_mcp_handler"] is False
