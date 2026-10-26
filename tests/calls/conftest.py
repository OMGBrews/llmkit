"""Shared fixtures for the call-surface tests.

Every test here exercises ``llmkit.calls``'s public call functions over a
faked provider; none should write a real YAML log. The autouse fixture gives
each test a keyless Ollama provider config (so an :mod:`llmkit.testing` fake can
answer the round trip), no cache, default rate limits, and muted logging — a
test that wants to inspect the written record opts back in with
:func:`tests._support.capturing_sink`.
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
