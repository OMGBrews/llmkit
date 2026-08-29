"""A tool result can say "this failed", and what the model actually receives.

``tool_result_message(..., is_error=True)`` is llmkit's one convention for the
distinction between a result and a failure, so a host does not invent a private
JSON error envelope and models do not diverge in how they recover from one
application to the next.

There is no wire field to carry it, which is *why* this module exists rather
than a one-line shape assertion. Three claims, in the order the design rests
on them:

1. the flag is on the message the caller holds (and so in the ``prompt`` the
   log records), and absent entirely when the result is fine;
2. the transport renders it into the content and drops the key, without
   touching the caller's own history;
3. **the measurement behind (2)** — LiteLLM's Anthropic translation discards an
   ``is_error`` key on an OpenAI-shaped tool message, so "let the transport map
   it natively" is not an option that was passed over, it is one that does not
   exist. That test imports LiteLLM private API on purpose, as
   ``tests/providers/test_gemini_tool_schema_transport.py`` does: a rename is a
   rename to follow here, while a version that *started* honouring the key
   fails it and is the signal to revisit the design.
"""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import MagicMock, patch

from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_anthropic_tool_result,
)
from litellm.types.llms.openai import ChatCompletionToolMessage

from llmkit import (
    TOOL_ERROR_PREFIX,
    ChatMessage,
    ToolDefinition,
    tool_result_message,
)
from tests._support import _transport_provider


def _messages_sent(prompt: list[ChatMessage]) -> list[dict[str, object]]:
    """Drive the real tool transport and return the messages LiteLLM was given."""
    from llmkit import _litellm

    seen: dict[str, object] = {}
    response = MagicMock(_hidden_params={})
    response.choices = []

    async def _fake_acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        return response

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        _ = asyncio.run(
            _litellm.acompletion_tools(
                prompt,
                [ToolDefinition("weather", "Look up weather", {"type": "object"})],
                tool_choice=None,
                temperature=None,
                model=None,
                provider=_transport_provider(reasoning_effort=None),
            )
        )
    return cast("list[dict[str, object]]", seen["messages"])


def test_the_error_flag_is_present_only_when_the_tool_actually_failed() -> None:
    """The default has to stay byte-identical to the three-key message this
    returned before the flag existed — a host that never fails a tool must not
    start sending a new key — so both states are asserted as exact dicts."""
    assert tool_result_message("call_1", "sunny") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "sunny",
    }
    assert tool_result_message("call_1", "timed out after 30s", is_error=True) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "timed out after 30s",
        "is_error": True,
    }


def test_the_transport_renders_the_flag_into_the_content_and_drops_the_key() -> None:
    """What the model receives: the marker in the text, and no ``is_error`` key
    for an OpenAI-compatible route to forward to a provider that may reject it.
    The ordinary result beside it is untouched, so the rendering is provably
    keyed on the flag rather than applied to every tool message."""
    history: list[ChatMessage] = [
        {"role": "user", "content": "weather?"},
        tool_result_message("call_ok", "sunny"),
        tool_result_message("call_bad", "the weather service returned HTTP 500", is_error=True),
    ]

    sent = _messages_sent(history)

    assert sent[1] == {"role": "tool", "tool_call_id": "call_ok", "content": "sunny"}
    assert sent[2] == {
        "role": "tool",
        "tool_call_id": "call_bad",
        "content": f"{TOOL_ERROR_PREFIX}the weather service returned HTTP 500",
    }


def test_rendering_does_not_mutate_the_caller_s_own_history() -> None:
    """The caller's list is theirs, is what the log records as ``prompt``, and
    is fed straight back into the next turn. Flattening in place would strip
    the flag from the record and double the prefix on the following call — so
    the message is asserted unchanged *after* a send, not merely copied."""
    failure = tool_result_message("call_bad", "boom", is_error=True)
    history: list[ChatMessage] = [{"role": "user", "content": "weather?"}, failure]

    _ = _messages_sent(history)
    _ = _messages_sent(history)

    assert failure == {
        "role": "tool",
        "tool_call_id": "call_bad",
        "content": "boom",
        "is_error": True,
    }


def test_litellm_discards_an_is_error_key_on_an_openai_shaped_tool_message() -> None:
    """The measurement the whole design rests on. Anthropic's own API *has* an
    ``is_error`` field on a ``tool_result`` block, but LiteLLM's translation
    from the OpenAI shape cannot express it (its own source comment says so),
    so passing the key through would reach Anthropic as nothing at all and
    every other route as an unknown key. Measured against the declared floor,
    litellm>=1.95.0."""
    # Both casts route through ``object``: the key under test is precisely the
    # one neither LiteLLM TypedDict declares, so the two shapes do not overlap
    # and pyright asks for the explicit hop it recommends for that case.
    message = cast(
        "ChatCompletionToolMessage",
        cast(
            "object",
            {
                "role": "tool",
                "tool_call_id": "call_bad",
                "content": "boom",
                "is_error": True,
            },
        ),
    )
    block = cast("dict[str, object]", cast("object", convert_to_anthropic_tool_result(message)))

    assert block["type"] == "tool_result"
    assert block["content"] == "boom"
    assert "is_error" not in block
