"""Gemini 3 and every later generation get ``thinkingLevel``, never ``thinkingBudget``.

Google has announced that its upcoming Gemini models will reject a request
carrying ``thinkingConfig.thinkingBudget`` with ``400 INVALID_ARGUMENT``; Gemini
3 already remaps it to ``thinkingLevel``. llmkit's Google and Vertex providers
send LiteLLM's portable ``reasoning_effort``, and LiteLLM picks the field by
parsing the model name in ``VertexGeminiConfig._is_gemini_3_or_newer``. Before
LiteLLM 1.104.0 that was the substring test ``"gemini-3" in model``, so a
``gemini-4-*`` id fell through to ``thinkingBudget``. 1.104.0 instead excludes
only the 1.x/2.x/exp/legacy names, which is why it is llmkit's floor: the
``gemini-4-flash`` cases below fail on 1.99.0.

Each case drives the real provider's ``reasoning_kwargs`` output through the
route's LiteLLM config (``ProviderConfigManager``), its ``map_openai_params``,
and the shared Gemini request-body builder ``_transform_request_body`` — the
same three steps a real call takes, stopping before any credential or network
use. (``VertexGeminiConfig.transform_request`` is intentionally unimplemented
upstream, so it is not the seam.) A Gemini 2.5 control proves the assertion can
see ``thinkingBudget`` at all.

**These tests import LiteLLM private API on purpose**, as
``test_gemini_tool_schema_transport.py`` does: an ``ImportError`` after an
upgrade is a rename to follow, not evidence that the guarantee broke.

``generationConfig.temperature`` is deliberately not asserted: LiteLLM injects
``1.0`` for Gemini 3 when it is absent, which the follow-up task
``contribute-and-consume-litellm-gemini-3-temperature-fix`` removes upstream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import litellm
import pytest
from litellm.llms.vertex_ai.gemini import transformation as gemini_transformation
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from llmkit import Provider, make_provider
from llmkit._types import ReasoningEffort
from llmkit.providers import LLMProviderInterface

#: raw-llm: the shared Gemini request-body builder, typed at the boundary: its
#: ``dict`` parameters are unparameterized upstream.
_transform_request_body = cast(
    "Callable[..., object]",
    gemini_transformation._transform_request_body,
)

#: The LiteLLM route each llmkit Gemini provider must resolve to.
_EXPECTED_ROUTE: dict[Provider, Literal["gemini", "vertex_ai"]] = {
    Provider.GOOGLE: "gemini",
    Provider.VERTEX: "vertex_ai",
}


def _build(provider_enum: Provider, model: str) -> LLMProviderInterface:
    if provider_enum is Provider.GOOGLE:
        return make_provider(Provider.GOOGLE, api_key="k", model=model)
    return make_provider(
        Provider.VERTEX, model=model, vertex_project="p", vertex_location="us-central1"
    )


def _thinking_config(
    provider_enum: Provider, model: str, effort: ReasoningEffort
) -> dict[str, object]:
    """Return ``generationConfig.thinkingConfig`` as LiteLLM would send it."""
    provider = _build(provider_enum, model)
    # raw-llm: LiteLLM's routing and Gemini transform helpers are loosely typed
    # (several ``Unknown`` slots); the boundary is cast once, here.
    bare_model, route, _key, _base = cast(
        "tuple[str, str, object, object]",
        litellm.get_llm_provider(provider.litellm_model()),
    )
    assert route == _EXPECTED_ROUTE[provider_enum], route
    config = ProviderConfigManager.get_provider_chat_config(
        model=bare_model, provider=LlmProviders(route)
    )
    assert config is not None, f"LiteLLM resolved no chat config for {route}/{bare_model}"
    optional_params = cast(
        "dict[str, object]",
        config.map_openai_params(  # pyright: ignore[reportUnknownMemberType]
            non_default_params=provider.reasoning_kwargs(effort, bare_model),
            optional_params={},
            model=bare_model,
            drop_params=False,
        ),
    )
    body = cast(
        "dict[str, object]",
        _transform_request_body(
            messages=[{"role": "user", "content": "hi"}],
            model=bare_model,
            optional_params=optional_params,
            custom_llm_provider=_EXPECTED_ROUTE[provider_enum],
            litellm_params={},
            cached_content=None,
        ),
    )
    generation_config = cast("dict[str, object]", body["generationConfig"])
    return cast("dict[str, object]", generation_config["thinkingConfig"])


_ROUTES = [Provider.GOOGLE, Provider.VERTEX]
_EFFORTS: list[ReasoningEffort] = ["low", "disable"]


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize("model", ["gemini-3.1-flash-lite", "gemini-4-flash"])
@pytest.mark.parametrize("provider_enum", _ROUTES)
def test_gemini_3_and_later_get_thinking_level(
    provider_enum: Provider, model: str, effort: ReasoningEffort
) -> None:
    thinking = _thinking_config(provider_enum, model, effort)
    assert "thinkingLevel" in thinking, thinking
    assert "thinkingBudget" not in thinking, thinking


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize("provider_enum", _ROUTES)
def test_gemini_2_control_still_gets_thinking_budget(
    provider_enum: Provider, effort: ReasoningEffort
) -> None:
    """The control: the same pipeline on a 2.x id does produce
    ``thinkingBudget``, so the absence asserted above is observed, not assumed."""
    thinking = _thinking_config(provider_enum, "gemini-2.5-flash", effort)
    assert "thinkingBudget" in thinking, thinking
    assert "thinkingLevel" not in thinking, thinking
