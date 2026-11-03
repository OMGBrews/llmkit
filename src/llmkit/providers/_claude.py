"""Shared reasoning translation for the Anthropic and Bedrock routes."""

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError


def _always_on(model: str, route: str) -> bool | None:
    from litellm import get_model_info

    try:
        return get_model_info(model, custom_llm_provider=route).get("thinking_always_on")
    except Exception:
        # Missing or unavailable model metadata cannot establish always-on thinking.
        return None


def reasoning_kwargs(effort: ReasoningEffort, model: str, *, route: str) -> dict[str, object]:
    """Translate portable disable and lowest, refusing models known to require thinking.

    ``"disable"`` becomes ``"none"`` and raises on models LiteLLM flags as
    ``thinking_always_on``. ``"lowest"`` becomes ``"low"`` on those models and
    ``"none"`` elsewhere. Unknown model metadata is deliberately permissive:
    LiteLLM remains the authority for model capabilities and provider-specific
    effort values.
    """
    if effort == "disable":
        if _always_on(model, route) is True:
            raise ReasoningEffortUnsupportedError(
                f'{model!r} cannot turn thinking off; reasoning_effort="disable" is not '
                + 'available on it. Send "lowest" for the least reasoning it supports.'
            )
        return {"reasoning_effort": "none"}
    if effort == "lowest":
        return {"reasoning_effort": "low" if _always_on(model, route) is True else "none"}
    return {"reasoning_effort": effort}
