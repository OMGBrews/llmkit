"""Unit tests for the Google Vertex AI provider.

Parity with the other curated providers, plus the two things Vertex does
differently: it reaches Gemini through Google Cloud (``vertex_ai/<model>``,
not AI Studio's ``gemini/<model>``), and it authenticates through **Application
Default Credentials**, not a bearer ``api_key``. The provider must expose the
right LiteLLM model-prefix string (``vertex_ai/<model>``), pin Gemini's native
JSON-schema mode (``Mode.JSON_SCHEMA`` — the same mode the direct
``GoogleProvider`` uses, since the model is Gemini; instructor's ``VERTEXAI_*``
modes target its ``from_vertexai`` client and don't apply over ``from_litellm``),
and carry only ``vertex_project`` / ``vertex_location`` — and only when set, so
an unset project/location (and the credentials, always) resolve from the ambient
ADC chain / environment. ``vertex_location`` is the data-residency control.
``build_provider`` must map ``Provider.VERTEX`` explicitly and thread model /
project / location / reasoning_effort.

These are mocked-shape assertions; a real round-trip lives in
``tests/integration/test_live_providers.py::test_vertex_live`` (gated on the
``[vertex]`` extra + a GCP account).
"""

from __future__ import annotations

import instructor

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import VertexProvider


def test_model_prefix_and_mode() -> None:
    """LiteLLM ``vertex_ai/`` prefix and Gemini's native JSON-schema mode."""
    provider = VertexProvider(model="gemini-2.5-pro")
    assert provider.name == "Google Vertex AI"
    assert provider.litellm_model() == "vertex_ai/gemini-2.5-pro"
    assert provider.litellm_model("gemini-2.5-flash") == "vertex_ai/gemini-2.5-flash"
    assert provider.instructor_mode is instructor.Mode.JSON_SCHEMA


def test_completion_kwargs_with_project_and_location() -> None:
    """Explicit project + location are forwarded as the Vertex routing kwargs."""
    provider = VertexProvider(vertex_project="my-gcp-project", vertex_location="europe-west4")
    assert provider.completion_kwargs() == {
        "vertex_project": "my-gcp-project",
        "vertex_location": "europe-west4",
    }


def test_completion_kwargs_location_only() -> None:
    """A residency choice can be made without naming a project (project from env/ADC)."""
    provider = VertexProvider(vertex_location="asia-northeast1")
    assert provider.completion_kwargs() == {"vertex_location": "asia-northeast1"}


def test_completion_kwargs_without_project_or_location() -> None:
    """No project/location → empty kwargs; everything resolves from env / ADC."""
    provider = VertexProvider()
    assert provider.completion_kwargs() == {}


def test_default_model() -> None:
    """A current, structured-output-capable default when none is given: Gemini
    2.5 Flash-Lite (parity with the direct Google AI Studio provider)."""
    assert VertexProvider().model == "gemini-2.5-flash-lite"


def test_not_local() -> None:
    """Vertex is a cloud provider — data leaves the host (unlike Ollama)."""
    assert VertexProvider().is_local is False


def test_build_provider_maps_vertex() -> None:
    """``build_provider`` builds a ``VertexProvider`` and threads every field."""
    provider = build_provider(
        LLMClientConfig(
            provider=Provider.VERTEX,
            model="gemini-2.5-flash",
            vertex_project="proj-123",
            vertex_location="us-east4",
            reasoning_effort="low",
        )
    )
    assert isinstance(provider, VertexProvider)
    assert provider.litellm_model() == "vertex_ai/gemini-2.5-flash"
    assert provider.reasoning_effort == "low"
    assert provider.completion_kwargs() == {
        "vertex_project": "proj-123",
        "vertex_location": "us-east4",
    }


def test_build_provider_vertex_defaults() -> None:
    """Without project/location/effort, kwargs are empty (env/ADC) and effort is None."""
    provider = build_provider(LLMClientConfig(provider=Provider.VERTEX, model="gemini-2.5-flash"))
    assert isinstance(provider, VertexProvider)
    assert provider.reasoning_effort is None
    assert provider.completion_kwargs() == {}
