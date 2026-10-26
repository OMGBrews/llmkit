"""Shared fixtures for the logging tests.

The disk-facing tests use an explicit ``LocalYamlLogSink(tmp_path)`` and so are
unaffected by the global sink; the call-driven ones fake the provider through
:mod:`llmkit.testing` and should not write a stray log. The autouse fixture
mutes the global sink by default and gives each test a keyless Ollama provider
config, no cache and the shipped rate-limiter defaults, so those calls run the
real limiter, retry and parsing around the fake.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """Mute the global sink and configure Ollama, no cache and default limits."""
    with ollama_call_defaults():
        yield
