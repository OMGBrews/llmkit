"""Shared fixtures for the call-surface tests.

Every test here exercises ``llmkit.calls``'s public call functions over a
faked transport; none should write a real YAML log. The autouse fixture mutes
logging by default — a test that wants to inspect the written record opts back
in with :func:`tests._support.capturing_sink`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import quiet_logging


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    """Point logging at a no-op sink so call tests don't touch the disk."""
    with quiet_logging():
        yield
