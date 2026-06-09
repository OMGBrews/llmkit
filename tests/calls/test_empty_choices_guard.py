"""Guards for degenerate empty-``choices`` responses on the text paths.

LiteLLM preserves an explicitly empty ``choices=[]`` on both its non-streaming
``ModelResponse`` and its streaming ``ModelResponseStream`` (the default
single-choice list is only substituted when ``choices is None``). Such frames
occur in practice — e.g. the Gemini provider emits ``choices=[]`` for
metadata-only / keepalive stream chunks, and a malformed OpenAI-compatible
proxy can return ``{"choices": []}`` for a completion. These tests pin that
neither path indexes ``choices[0]`` blindly (which would raise ``IndexError``,
killing a stream mid-flight or crashing a completion).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from litellm.types.utils import ModelResponse, ModelResponseStream

from llmkit import _litellm


def test_chunk_delta_text_tolerates_empty_choices() -> None:
    """A metadata-only / keepalive chunk (``choices=[]``) yields ``None`` so the
    streaming loop skips it, rather than raising ``IndexError``."""
    chunk = ModelResponseStream(choices=[])
    assert _litellm._chunk_delta_text(chunk) is None


def test_acompletion_text_tolerates_empty_choices() -> None:
    """A degenerate empty-``choices`` completion degrades to ``""`` (via the
    None-coercion) instead of raising ``IndexError``."""
    fake_resp = ModelResponse(choices=[])
    fake_resp._hidden_params = {}  # mirror litellm's cost-lookup attribute

    async def _fake_acompletion(**_kwargs: object) -> ModelResponse:
        return fake_resp

    provider = MagicMock()
    provider.completion_kwargs = MagicMock(return_value={"api_key": "k", "api_base": None})
    provider.litellm_model = MagicMock(return_value="fake/model")
    provider.reasoning_effort = None

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text("hi", temperature=0.0, model=None, provider=provider)
        )

    assert text == ""
