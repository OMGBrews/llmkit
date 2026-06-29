"""End-to-end: a real VertexProvider threads its residency routing to the wire.

The unit tests in ``test_vertex.py`` check ``completion_kwargs()`` in isolation.
These prove the data-residency selection actually *reaches LiteLLM*: a **real**
``VertexProvider`` drives the real ``_litellm`` call seam (only the
LiteLLM/instructor boundary faked), and the captured request carries
``vertex_project`` / ``vertex_location`` (the residency selection) alongside the
``vertex_ai/`` model string. For structured output it additionally asserts
instructor is pinned to ``Mode.JSON_SCHEMA`` — so a regression in how creds splat
into the call, the model prefix, or the mode pin fails here rather than only
against live Vertex. (Parallels ``test_openrouter_routing.py``, which pins that
provider's routing preference reaches the wire.)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import instructor

from llmkit import _litellm
from llmkit.providers import VertexProvider
from tests._support import OkSchema


def _run_text(provider: VertexProvider) -> dict[str, object]:
    """Drive the real ``acompletion_text`` over a faked ``litellm.acompletion``
    and return the kwargs the underlying request saw."""
    seen: dict[str, object] = {}
    fake_resp = MagicMock(_hidden_params={})
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok"))]

    async def _fake_acompletion(**kwargs: object) -> MagicMock:
        seen.update(kwargs)
        return fake_resp

    with patch("llmkit._litellm.litellm.acompletion", side_effect=_fake_acompletion):
        text, _cost = asyncio.run(
            _litellm.acompletion_text("hi", temperature=0.0, model=None, provider=provider)
        )
    assert text == "ok"
    return seen


def test_text_call_forwards_residency_and_vertex_model() -> None:
    """A located VertexProvider sends ``vertex_ai/<model>`` + project + location."""
    provider = VertexProvider(
        model="gemini-2.5-pro",
        vertex_project="proj-123",
        vertex_location="europe-west4",
    )
    seen = _run_text(provider)
    assert seen["model"] == "vertex_ai/gemini-2.5-pro"
    assert seen["vertex_project"] == "proj-123"
    assert seen["vertex_location"] == "europe-west4"


def test_text_call_without_residency_sends_no_vertex_kwargs() -> None:
    """Unset project/location → no ``vertex_*`` keys on the wire (env/ADC resolves).

    The byte-identical-default contract: a residency-less call must not inject an
    explicit ``vertex_location=None`` that would override what the environment or
    ADC would otherwise resolve.
    """
    seen = _run_text(VertexProvider(model="gemini-2.5-flash"))
    assert seen["model"] == "vertex_ai/gemini-2.5-flash"
    assert "vertex_project" not in seen
    assert "vertex_location" not in seen


def test_structured_call_pins_json_schema_and_forwards_residency() -> None:
    """The structured path pins ``Mode.JSON_SCHEMA`` and forwards the residency kwargs."""
    provider = VertexProvider(
        model="gemini-2.5-flash",
        vertex_project="proj-9",
        vertex_location="asia-northeast1",
    )
    seen: dict[str, object] = {}
    seen_mode: dict[str, object] = {}

    async def _fake_create_with_completion(**kwargs: object) -> tuple[OkSchema, MagicMock]:
        seen.update(kwargs)
        return OkSchema(ok=True), MagicMock(_hidden_params={})

    fake_client = MagicMock()
    fake_client.chat = MagicMock(
        completions=MagicMock(create_with_completion=_fake_create_with_completion)
    )

    def _capture_from_litellm(_client_fn: object, mode: object) -> MagicMock:
        seen_mode["mode"] = mode
        return fake_client

    with patch("llmkit._litellm.instructor.from_litellm", side_effect=_capture_from_litellm):
        parsed, _cost = asyncio.run(
            _litellm.acompletion_structured(
                "hi", OkSchema, temperature=0.0, model=None, provider=provider
            )
        )

    assert parsed.ok is True
    # The mode pin is what makes Gemini structured output validate (auto-mode
    # would regress it to TOOLS / silent-empty).
    assert seen_mode["mode"] is instructor.Mode.JSON_SCHEMA
    assert seen["model"] == "vertex_ai/gemini-2.5-flash"
    assert seen["vertex_project"] == "proj-9"
    assert seen["vertex_location"] == "asia-northeast1"
