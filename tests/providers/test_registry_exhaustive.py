"""Exhaustiveness backstop: every ``Provider`` enum member must be wired.

`build_provider` dispatches on `Provider` with a `match` whose fall-through
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

from llmkit import LLMClientConfig, Provider
from llmkit.providers import build_provider, make_provider
from llmkit.providers.base import BaseProvider


def test_every_provider_member_is_wired() -> None:
    """`build_provider` constructs a provider for every `Provider` member."""
    for provider in Provider:
        config = LLMClientConfig(provider=provider, model="x", api_key="k")
        result = build_provider(config)
        assert isinstance(result, BaseProvider)
        assert result.name, f"{provider!r} produced a provider with no name"


def test_every_member_falsy_model_is_well_formed() -> None:
    """No `Provider` yields a dangling `"<prefix>/"` id when the model is falsy."""
    for provider in Provider:
        built = make_provider(provider, api_key="k", model=None)
        litellm_model = built.litellm_model()
        assert not litellm_model.endswith("/"), (
            f"{provider!r} produced a broken model id {litellm_model!r}"
        )
