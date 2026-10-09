"""The litellm-free half of request building.

What a provider request contains is decided in two places: :mod:`llmkit._litellm`
sends it, and the response cache (:mod:`llmkit.cache`) fingerprints it. The
pieces both need — the message list as it goes on the wire, and the effective
reasoning effort — live here, because :mod:`llmkit._litellm` imports litellm at
module scope and ``import llmkit`` must never load litellm
(``tests/packaging/test_litellm_lazy_import.py``). One definition, so the
fingerprint can never describe a request the transport would not send.
"""

from __future__ import annotations

from collections.abc import Sequence

from llmkit._types import TOOL_ERROR_PREFIX, ChatMessage, ReasoningEffort
from llmkit.providers import LLMProviderInterface


def wire_message(message: ChatMessage) -> ChatMessage:
    """Render llmkit's own message extensions into what LiteLLM can send.

    Today that is exactly one: :class:`~llmkit.ToolResultMessage`'s optional
    ``is_error``. It has no wire equivalent — LiteLLM's Anthropic translation
    builds its ``tool_result`` block from the id and content alone (measured
    against litellm 1.95.0), and an OpenAI-compatible route
    would forward the unknown key to a provider that may reject it — so the
    flag becomes a :data:`~llmkit._types.TOOL_ERROR_PREFIX` prefix on the
    content, which every provider receives identically, and the key is dropped.

    Returns a **new** dict for a message it rewrites: the caller's history is
    theirs, is what the log records as ``prompt``, and is fed back into the
    next turn — flattening it in place would strip the flag from the record and
    double the prefix on the following call.
    """
    if message["role"] != "tool" or not message.get("is_error", False):
        return message
    return {
        "role": "tool",
        "tool_call_id": message["tool_call_id"],
        "content": f"{TOOL_ERROR_PREFIX}{message['content']}",
    }


def wire_messages(prompt: str | Sequence[ChatMessage]) -> list[ChatMessage]:
    """Normalise a prompt into LiteLLM's message-list shape."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [wire_message(message) for message in prompt]


def resolve_reasoning_effort(
    override: ReasoningEffort | None, provider: LLMProviderInterface
) -> ReasoningEffort | None:
    """Resolve the effective reasoning effort for a call.

    A per-call ``override`` wins when set; otherwise the provider's
    configured value (from :class:`~llmkit.LLMClientConfig`) applies. Both
    ``None`` means no reasoning kwarg is forwarded — byte-identical to the
    pre-feature request. ``getattr`` keeps third-party providers that predate
    the ``reasoning_effort`` property working (they degrade to ``None``).
    """
    if override is not None:
        return override
    return getattr(provider, "reasoning_effort", None)
