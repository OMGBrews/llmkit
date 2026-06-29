"""google-auth is an opt-in extra for the Vertex provider, not a core dependency.

LiteLLM's ``vertex_ai/`` path mints and refreshes the Google OAuth access token
it signs Gemini-on-Vertex requests with through ``google.auth`` (the Application
Default Credentials chain), so the Vertex provider needs it at call time — but a
host that never routes to Vertex should take on no Google dependency. It
therefore ships in the ``omg-llmkit[vertex]`` extra and is checked *eagerly* at
construction, so a missing ``google-auth`` fails with a fix instead of as a
cryptic ``ModuleNotFound`` deep on the first completion.

``VertexProvider.__init__`` runs one eager check: ``require_google_auth_sdk``.
These tests cover that guard. They simulate ``google-auth``'s absence by stubbing
``importlib.util.find_spec``; the dev/CI environment installs ``google-auth``
(via the ``dev`` extra, which pulls ``[vertex]``) so the rest of the suite runs
against a real, importable ``google.auth``.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec

import pytest

from llmkit import (
    LLMClientConfig,
    Provider,
    build_provider,
)
from llmkit.providers import VertexProvider


@pytest.fixture
def google_auth_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``require_google_auth_sdk`` observe ``google.auth`` as not installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        if name == "google.auth":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.mark.usefixtures("google_auth_absent")
def test_vertex_provider_raises_clearly_without_google_auth() -> None:
    """Constructing Vertex without google-auth names the install fix."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[vertex\]") as excinfo:
        _ = VertexProvider()
    message = str(excinfo.value)
    # The message identifies the missing dep and why it's needed.
    assert "google-auth" in message
    assert "token" in message


@pytest.mark.usefixtures("google_auth_absent")
def test_build_provider_vertex_raises_without_google_auth() -> None:
    """The google-auth error surfaces through the dispatch entrypoint, not just direct construction."""
    with pytest.raises(ModuleNotFoundError, match=r"omg-llmkit\[vertex\]"):
        _ = build_provider(LLMClientConfig(provider=Provider.VERTEX))


def test_vertex_provider_builds_when_google_auth_present() -> None:
    """With google-auth installed (the dev extra), construction succeeds (sanity)."""
    provider = VertexProvider()
    assert provider.name == "Google Vertex AI"
    assert provider.litellm_model() == "vertex_ai/gemini-2.5-flash-lite"
