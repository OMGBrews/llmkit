"""Exhaustiveness backstop: every ``Provider`` enum member must be wired.

`get_provider` dispatches on `Provider` with a `match` whose fall-through
calls `typing.assert_never`, so an enum member added without a
corresponding `case` is caught three ways: basedpyright reports a type
error (the member isn't assignable to `Never`), `assert_never` raises an
`AssertionError` at runtime, and this test goes red. This is the
unit-test layer — it converts "added an enum value but forgot to wire it"
into a failing test even for a contributor who skips the type checker
locally, and it guarantees no member silently falls back to another
provider (the old `else: OllamaProvider(...)` behaviour, now removed).
"""

from __future__ import annotations

from llmkit import LLMClientConfig, Provider, get_provider
from llmkit.providers.base import BaseProvider


def test_every_provider_member_is_wired() -> None:
    """`get_provider` constructs a provider for every `Provider` member."""
    for provider in Provider:
        config = LLMClientConfig(provider=provider, model="x", api_key="k")
        result = get_provider(config)
        assert isinstance(result, BaseProvider)
        assert result.name, f"{provider!r} produced a provider with no name"
