"""Shared fixtures for the provider tests.

The per-call configuration tests drive the public call functions with the
provider faked through :mod:`llmkit.testing`; the autouse fixture gives every
test a keyless Ollama provider config, no cache, muted logging and the shipped
rate-limiter defaults, restoring the previous state afterwards. Tests that need
no global config unset it on top of this.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests._support import ollama_call_defaults


@pytest.fixture(autouse=True)
def _ollama_and_defaults() -> Iterator[None]:
    """Configure Ollama, no cache, muted logging and default limits."""
    with ollama_call_defaults():
        yield


#: Moonshot models whose LiteLLM listing differs between the maps a run can load.
#: ``kimi-k3.1-pro`` (the default) is in LiteLLM's remote cost map and in the map
#: bundled from 1.109.1, but not in the maps bundled with 1.104.0 (the floor) or
#: 1.106.6, so with no network or ``LITELLM_LOCAL_MODEL_COST_MAP=True`` it is
#: unlisted there. Models listed the same way in every map (``kimi-k2.6``,
#: ``moonshot-v1-8k``) are left to the real lookup.
_MOONSHOT_DEFAULT = "kimi-k3.1-pro"


@pytest.fixture
def moonshot_default_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make LiteLLM list Moonshot's default as a reasoning model, whichever map loaded.

    The offline suite must not depend on the network: this pins the listing the
    remote map and LiteLLM 1.109.1's bundled map both carry, so the provider's
    metadata-gated reasoning translation is tested the same way everywhere.
    """
    import litellm

    real = litellm.get_model_info

    def _get_model_info(model: str, custom_llm_provider: str | None = None) -> object:
        if custom_llm_provider == "moonshot" and model == _MOONSHOT_DEFAULT:
            return {"key": f"moonshot/{model}", "supports_reasoning": True}
        return real(model, custom_llm_provider=custom_llm_provider)

    monkeypatch.setattr(litellm, "get_model_info", _get_model_info)


@pytest.fixture
def moonshot_default_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make LiteLLM's lookup of Moonshot's default fail, as the floor's bundled map does."""
    import litellm

    real = litellm.get_model_info

    def _get_model_info(model: str, custom_llm_provider: str | None = None) -> object:
        if custom_llm_provider == "moonshot" and model == _MOONSHOT_DEFAULT:
            raise Exception(f"This model isn't mapped yet. model={model}")
        return real(model, custom_llm_provider=custom_llm_provider)

    monkeypatch.setattr(litellm, "get_model_info", _get_model_info)
