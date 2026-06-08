"""Import shim: the ``omg-llmkit`` distribution is imported as ``llmkit``.

PyPI publishes this project under the distribution name **``omg-llmkit``** (the
bare ``llmkit`` name was already taken), but the import name is just
``llmkit``. A natural post-install smoke test — ``python -c "import
omg_llmkit"`` — would otherwise fail with a bare
``ModuleNotFoundError: No module named 'omg_llmkit'`` that gives no hint about
the install-vs-import split (the exact papercut a greenfield integration hit).

This tiny shim ships under the install name so that mistaken import surfaces an
actionable, one-line redirect to the real import name instead of a dead end. It
is import-only: it deliberately raises rather than re-exporting ``llmkit``, so
the lesson sticks and call sites converge on the documented ``import llmkit``.
"""

raise ImportError(
    "The 'omg-llmkit' distribution is imported as 'llmkit', not 'omg_llmkit'. "
    + "Replace your import with:\n\n    import llmkit\n\n"
    + "('omg-llmkit' is only the pip/PyPI install name — the bare 'llmkit' name "
    + "was already taken on PyPI.)"
)
