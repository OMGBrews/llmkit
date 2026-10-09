"""The request fingerprint: what makes two requests the same request.

llmkit owns the key because only llmkit holds the resolved request — the
effective model, the provider's endpoint, its structured-output mode, the
schema JSON. A host building keys from its own call sites would see none of
those.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel

from llmkit._messages import resolve_reasoning_effort, wire_messages
from llmkit._types import ChatMessage, ReasoningEffort
from llmkit.providers import LLMProviderInterface

#: Bumped whenever the canonical form below changes, so a store holding keys
#: from an earlier form misses instead of answering a request it never saw.
#: A pinned-digest test fails on any change that forgets to bump it.
LLM_CACHE_KEY_VERSION: Final = 1

# The provider request kwargs that say *where* a request goes, as opposed to
# who is allowed to send it. An allowlist, so a credential (``api_key``) can
# never reach the key — and so a provider that adds a new kwarg changes no key
# until it is deliberately listed here.
_ENDPOINT_KWARGS: Final = (
    "api_base",
    "extra_body",
    "aws_region_name",
    "vertex_project",
    "vertex_location",
)


def llm_cache_key(
    *,
    provider: LLMProviderInterface,
    prompt: str | Sequence[ChatMessage],
    output_schema: type[BaseModel] | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: ReasoningEffort | None,
) -> str:
    """Fingerprint a buffered request as a sha256 hex digest.

    Takes the call's *resolved* values: ``model`` and ``reasoning_effort`` as
    the call resolved them (``None`` defers to the provider, exactly as the
    transport resolves it), and ``output_schema=None`` for a plain-text call.
    The digest covers, in this fixed order: :data:`LLM_CACHE_KEY_VERSION`; the
    provider's name; the model as routed to LiteLLM; the provider's endpoint
    kwargs (``api_base``, ``extra_body``, ``aws_region_name``,
    ``vertex_project``, ``vertex_location`` — each only when present, and no
    credential); the messages as sent; the output schema's JSON schema, or
    ``"text"``; temperature; max_tokens; the effective reasoning effort; and,
    for a structured call, the provider's structured-output mode and strict
    JSON-schema flag. ``feature`` and ``label`` are telemetry and are not part
    of it.

    The JSON is serialised **without** sorting keys: a schema's property order
    is sent to the provider and steers the order the model generates fields
    in, so two schemas that differ only in field order are different requests.

    Raises whatever the inputs raise — a content part that is not
    JSON-serialisable, a provider missing an attribute. The call functions
    treat any such failure as a miss.
    """
    request_kwargs = provider.completion_kwargs()
    structured = output_schema is not None
    mode: object = provider.instructor_mode if structured else None
    canonical: list[object] = [
        LLM_CACHE_KEY_VERSION,
        provider.name,
        provider.litellm_model(model),
        {name: request_kwargs[name] for name in _ENDPOINT_KWARGS if name in request_kwargs},
        wire_messages(prompt),
        output_schema.model_json_schema() if structured else "text",
        temperature,
        max_tokens,
        resolve_reasoning_effort(reasoning_effort, provider),
        # Real providers hold an ``instructor.Mode`` enum; the value is the
        # stable spelling.
        getattr(mode, "value", mode),
        provider.strict_json_schema if structured else None,
    ]
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
