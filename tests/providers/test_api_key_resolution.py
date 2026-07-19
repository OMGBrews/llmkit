"""Bearer providers resolve their API key config-first, then env, else raise.

The five key-authenticated providers used to coerce a missing ``api_key`` to
``""`` and let LiteLLM silently fall back to the ambient environment key — an
invisible identity/billing swap. Resolution is now explicit library behavior:
the configured value wins; otherwise the provider's own environment variable is
consulted; otherwise construction fails loud, naming the variable.

These tests control the environment explicitly (every one ``delenv``\\ s the
variable under test first) so their pass/fail never depends on an ambient key —
which matters because importing LiteLLM runs ``load_dotenv()`` and can inject
one into the process.
"""

from __future__ import annotations

import pytest

from llmkit import LLMClientConfig, Provider
from llmkit.providers import build_provider, make_provider

# (provider, env var the litellm route reads, provider name in the error).
_BEARER_PROVIDERS = [
    (Provider.OPENROUTER, "OPENROUTER_API_KEY", "OpenRouter"),
    (Provider.ANTHROPIC, "ANTHROPIC_API_KEY", "Anthropic"),
    (Provider.OPENAI, "OPENAI_API_KEY", "OpenAI"),
    (Provider.DEEPSEEK, "DEEPSEEK_API_KEY", "DeepSeek"),
    (Provider.GOOGLE, "GEMINI_API_KEY", "Google AI Studio"),
]

# The subset the resolution-value tests need — they assert on the resolved key,
# not the error text, so the provider name is irrelevant.
_BEARER_PROVIDER_ENVS = [(provider, env_var) for provider, env_var, _ in _BEARER_PROVIDERS]


@pytest.mark.parametrize(("provider", "env_var", "name"), _BEARER_PROVIDERS)
def test_build_without_key_or_env_raises_naming_env_var(
    provider: Provider, env_var: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No config key and no env var → a loud construction-time error that names
    both remedies, instead of an empty key failing (or silently succeeding off an
    ambient key) at call time."""
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValueError, match=rf"{name} requires an API key.*{env_var}"):
        _ = build_provider(LLMClientConfig(provider=provider))


@pytest.mark.parametrize(("provider", "env_var"), _BEARER_PROVIDER_ENVS)
def test_build_falls_back_to_env_when_config_key_absent(
    provider: Provider, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config key but the env var set, the env value is used — the
    first-party-SDK idiom, now explicit and reaching the wire kwargs."""
    monkeypatch.setenv(env_var, "env-key-value")
    built = build_provider(LLMClientConfig(provider=provider))
    assert built.completion_kwargs()["api_key"] == "env-key-value"


@pytest.mark.parametrize(("provider", "env_var"), _BEARER_PROVIDER_ENVS)
def test_explicit_config_key_wins_over_env(
    provider: Provider, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured key beats the env var — explicit configuration is never
    overridden by the ambient fallback."""
    monkeypatch.setenv(env_var, "env-key-value")
    built = build_provider(LLMClientConfig(provider=provider, api_key="config-key-value"))
    assert built.completion_kwargs()["api_key"] == "config-key-value"


@pytest.mark.parametrize(("provider", "env_var", "name"), _BEARER_PROVIDERS)
def test_empty_string_config_key_is_treated_as_unset(
    provider: Provider, env_var: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty ``api_key=""`` is unset, not a valid empty credential: with no env
    it raises, and with an env var present it falls through to it."""
    monkeypatch.delenv(env_var, raising=False)
    with pytest.raises(ValueError, match=f"{name} requires an API key"):
        _ = build_provider(LLMClientConfig(provider=provider, api_key=""))

    monkeypatch.setenv(env_var, "env-key-value")
    built = build_provider(LLMClientConfig(provider=provider, api_key=""))
    assert built.completion_kwargs()["api_key"] == "env-key-value"


def test_make_provider_bearer_without_key_or_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-call ``make_provider`` seam shares the same resolution and failure."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Anthropic requires an API key"):
        _ = make_provider(Provider.ANTHROPIC)


def test_ollama_builds_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama authenticates with no key at all — key resolution never runs for it,
    so it builds cleanly with every bearer env var cleared."""
    for _provider, env_var, _name in _BEARER_PROVIDERS:
        monkeypatch.delenv(env_var, raising=False)
    built = build_provider(LLMClientConfig(provider=Provider.OLLAMA))
    kwargs = built.completion_kwargs()
    assert "api_key" not in kwargs
    assert kwargs == {"api_base": "http://localhost:11434"}
