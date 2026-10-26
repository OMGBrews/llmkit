"""Shared fixtures for the provider tests.

The per-call configuration tests drive the public call functions with the
provider faked through :mod:`llmkit.testing`; the autouse fixture gives every
test a keyless Ollama provider config, no cache, muted logging and the shipped
rate-limiter defaults, restoring the previous state afterwards. Tests that need
no global config unset it on top of this.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """Configure Ollama, no cache, muted logging and default limits."""
    with ollama_call_defaults():
        yield
