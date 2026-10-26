"""Shared fixtures for the public testing-transport tests.

These tests drive the public call functions the way a downstream project
would: a real provider config (Ollama, which needs no key), the fake installed
through :mod:`llmkit.testing`, and nothing patched. The autouse fixture gives
each test that config, no cache, no YAML log, and the shipped rate-limit
defaults, and restores all four afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """An Ollama provider config, no cache, muted logging, default limits."""
    with ollama_call_defaults():
        yield
