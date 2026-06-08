"""Tests for ``text_llm_call``'s list-content coercion.

The plain-text call advertises (README public-API table) that it "coerces
provider list-content blocks" and is annotated to return ``str``. Some
providers, however, return ``message.content`` as a *list* of content
blocks rather than a string. These tests pin two seams:

* ``_coerce_text_content`` joins the text of list-shaped content (dict or
  object blocks), passes strings through, and degrades odd shapes to ``""``;
* the transport (``acompletion_text``) runs that coercion, so a provider
  returning list content yields a single joined string — honouring the
  ``str`` return annotation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llmkit import _litellm


def test_coerce_passes_string_through() -> None:
    assert _litellm._coerce_text_content("already text") == "already text"


def test_coerce_none_becomes_empty_string() -> None:
    assert _litellm._coerce_text_content(None) == ""


def test_coerce_joins_dict_text_blocks() -> None:
    blocks = [
        {"type": "text", "text": "Hello, "},
        {"type": "text", "text": "world."},
    ]
    assert _litellm._coerce_text_content(blocks) == "Hello, world."


def test_coerce_skips_non_text_blocks() -> None:
    """Image / unknown blocks (no string ``text``) are dropped, not stringified."""
    blocks = [
        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
        {"type": "text", "text": "caption"},
    ]
    assert _litellm._coerce_text_content(blocks) == "caption"


def test_coerce_reads_object_blocks_text_attr() -> None:
    blocks = [SimpleNamespace(type="text", text="from "), SimpleNamespace(type="text", text="attr")]
    assert _litellm._coerce_text_content(blocks) == "from attr"


def test_coerce_empty_list_becomes_empty_string() -> None:
    assert _litellm._coerce_text_content([]) == ""


def test_coerce_unexpected_scalar_becomes_empty_string() -> None:
    """A shape that is neither str/list/None (never expected from LiteLLM)
    degrades to ``""`` rather than leaking a non-string return value."""
    assert _litellm._coerce_text_content(123) == ""


def test_acompletion_text_joins_provider_list_content() -> None:
    """End-to-end at the transport seam: a provider returning list-shaped
    content makes ``acompletion_text`` return the joined string (not a list)."""
    fake_resp = MagicMock(_hidden_params={})
    fake_resp.choices = [
        MagicMock(
            message=MagicMock(
                content=[
                    {"type": "text", "text": "Joined "},
                    {"type": "text", "text": "across blocks."},
                ]
            )
        )
    ]

    async def _fake_acompletion(**_kwargs: object) -> MagicMock:
        return fake_resp

    provider = MagicMock()
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", "api_base": None})
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text("hi", temperature=0.0, model=None, provider=provider)
        )

    assert text == "Joined across blocks."
    assert isinstance(text, str)
