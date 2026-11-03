"""Shared Gemini reasoning rules for the Google, Vertex and OpenRouter routes.

Gemini 3 and later cannot turn thinking off: LiteLLM's transform maps
``"disable"`` there to the model's lowest ``thinkingLevel``, which still
reasons. ``gemini-2.5-pro`` cannot either (Google documents it, and LiteLLM's
model map does not flag it), so ``thinkingBudget: 0`` fails at the provider.
The portable ``"disable"`` therefore raises before transport on those models,
and ``"lowest"`` carries the least reasoning each one supports.
"""

import re
from collections.abc import Callable
from typing import cast

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError

#: Gemini ids that predate Gemini 3: 1.x, 2.x, ``exp``, and the legacy bare
#: ``pro``/``flash`` names (``gemini-pro-latest`` and ``gemini-flash-latest``
#: are aliases that now resolve to Gemini 3+, so they are not excluded).
#: Copied from LiteLLM 1.104.0's ``VertexGeminiConfig._is_gemini_3_or_newer``;
#: used only on OpenRouter, where LiteLLM's own test would misclassify every
#: non-Gemini model id.
PRE_GEMINI_3_RE = re.compile(
    r"^gemini-(?:[12](?:\.\d+)?|exp|(?:pro|flash)(?!-(?:lite-)?latest$))(?:-|$)"
)

#: The Gemini model that thinks on every request: its minimum thinking budget is
#: above zero, so it cannot turn thinking off.
ALWAYS_THINKING_PREFIX = "gemini-2.5-pro"


def _bare_name(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower()


def is_always_thinking_gemini(model: str) -> bool:
    """Whether ``model`` names ``gemini-2.5-pro`` (any dated or preview variant)."""
    return _bare_name(model).startswith(ALWAYS_THINKING_PREFIX)


def is_gemini_3_or_newer_direct(model: str) -> bool:
    """Whether LiteLLM's Gemini transform treats ``model`` as Gemini 3 or later.

    Calls LiteLLM's public ``VertexGeminiConfig.is_gemini_3_or_newer`` so the
    decision matches the transform that will run, including on ids such as
    ``learnlm-*`` that it counts as Gemini 3. Only meaningful for a model that
    reaches the Gemini transform (the Google route, and Vertex's Gemini ids).
    LiteLLM 1.104.0, llmkit's floor, names the same static method
    ``_is_gemini_3_or_newer``; it is used only when the public name is absent.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    check = cast(
        "Callable[[str], bool]",
        getattr(VertexGeminiConfig, "is_gemini_3_or_newer", None)
        or getattr(VertexGeminiConfig, "_is_gemini_3_or_newer"),  # noqa: B009
    )
    return check(model)


def is_gemini_3_or_newer_openrouter(model: str) -> bool:
    """Whether an OpenRouter model id names Gemini 3 or any later generation.

    Excludes only the known pre-Gemini-3 names, so a future
    ``google/gemini-4-*`` counts without a code change. The id after the last
    ``/`` must start with ``gemini-``, so ``google/gemma-*`` and non-Google
    models never match; LiteLLM's own test would count them as Gemini 3.
    """
    name = _bare_name(model)
    return name.startswith("gemini-") and PRE_GEMINI_3_RE.match(name) is None


def cannot_disable_error(model: str, *, why: str) -> ReasoningEffortUnsupportedError:
    """The error for a model known to be unable to stop thinking."""
    return ReasoningEffortUnsupportedError(
        f'{model!r} cannot turn thinking off ({why}); reasoning_effort="disable" is not '
        + 'available on it. Send "lowest" for the least reasoning it supports.'
    )


def direct_reasoning_kwargs(effort: ReasoningEffort, model: str) -> dict[str, object]:
    """Translate portable effort for a model served by LiteLLM's Gemini transform.

    ``"disable"`` raises on Gemini 3 and later and on ``gemini-2.5-pro``, and is
    forwarded elsewhere (LiteLLM sends ``thinkingBudget: 0``). ``"lowest"``
    forwards ``"disable"``, LiteLLM's lowest level with thought summaries off,
    except on ``gemini-2.5-pro``, where it sends ``"minimal"`` (the model's
    minimum budget). Other values pass through.
    """
    if effort == "disable":
        if is_always_thinking_gemini(model):
            raise cannot_disable_error(model, why="gemini-2.5-pro always thinks")
        if is_gemini_3_or_newer_direct(model):
            raise cannot_disable_error(model, why="Gemini 3 and later always think")
        return {"reasoning_effort": "disable"}
    if effort == "lowest":
        if is_always_thinking_gemini(model):
            return {"reasoning_effort": "minimal"}
        return {"reasoning_effort": "disable"}
    return {"reasoning_effort": effort}
