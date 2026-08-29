"""Shared fixtures for the logging tests.

The disk-facing tests use an explicit ``LocalYamlLogSink(tmp_path)`` and so are
unaffected by the global sink; the few that drive a real call through
``llmkit.calls`` should not write a stray log. The autouse fixture mutes
the global sink by default.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import quiet_logging


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    """Point the global sink at a no-op so call-driven tests don't touch disk."""
    with quiet_logging():
        yield
