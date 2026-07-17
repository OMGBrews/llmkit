"""``llmkit.__version__`` is exposed and tracks the installed distribution.

The attribute is resolved from installed metadata (the distribution is
``omg-llmkit``; the import name is ``llmkit``), so it can never drift from what
pip installed.
"""

from __future__ import annotations

from importlib.metadata import version

import llmkit


def test_version_is_exposed() -> None:
    """``llmkit.__version__`` is a non-empty string."""
    assert isinstance(llmkit.__version__, str)
    assert llmkit.__version__


def test_version_matches_installed_distribution() -> None:
    """The exposed version equals the installed ``omg-llmkit`` metadata."""
    assert llmkit.__version__ == version("omg-llmkit")
