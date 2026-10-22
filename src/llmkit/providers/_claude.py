"""Shared reasoning translation for the Anthropic and Bedrock routes."""

from llmkit._types import ReasoningEffort
from llmkit.exceptions import ReasoningEffortUnsupportedError


def reasoning_kwargs(effort: ReasoningEffort, model: str, *, route: str) -> dict[str, object]:
    """Translate portable disable, refusing models known to require thinking.

    Unknown model metadata is deliberately permissive: LiteLLM remains the
    authority for model capabilities and provider-specific effort values.
    """
    if effort != "disable":
        return {"reasoning_effort": effort}

    from litellm import get_model_info

    try:
        always_on = get_model_info(model, custom_llm_provider=route).get("thinking_always_on")
    except Exception:
        # Missing or unavailable model metadata cannot establish always-on thinking.
        always_on = None
    if always_on is True:
        raise ReasoningEffortUnsupportedError(
            f'{model!r} cannot turn thinking off; reasoning_effort="disable" is not '
            + 'available on it. Send "low" for the lowest explicit effort.'
        )
    return {"reasoning_effort": "none"}
