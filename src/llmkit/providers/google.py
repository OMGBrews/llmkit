"""Google AI Studio LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_gemini_structured_output,
)


class GoogleProvider(BaseProvider):
    """Google AI Studio LLM provider.

    Uses Gemini directly (``gemini/<model>``) for high rate limits and low
    latency. Structured output defaults to Gemini's native JSON-schema mode for
    instructor (``Mode.JSON_SCHEMA``) — the same default the Vertex path uses,
    since the underlying model is Gemini.

    The strategy is host-selectable via the ``structured_output`` argument
    (``LLMClientConfig.gemini_structured_output``), because Gemini's native
    ``JSON_SCHEMA`` path is a measured **repetition-loop trap**. That path is
    grammar-constrained decoding: a token mask enforces the schema, but once the
    model starts to loop the mask blocks exactly the tokens that would break the
    pattern, so the call spins until ``max_tokens`` kills it. PIA Maker measured
    **67-83%** first-attempt runaway under ``JSON_SCHEMA`` versus **0%** under
    ``Mode.JSON`` on real production prompts (interleaved arms, same
    model/schema/prompt bytes; 2026-07-15).

    - ``structured_output="schema"`` (default) keeps ``Mode.JSON_SCHEMA``:
      server-side *schema* enforcement, and the pre-existing wire shape exactly.
    - ``structured_output="json"`` switches to ``Mode.JSON``: the response is
      still server-side guaranteed to be JSON *syntax* (the mime-type
      constraint), but the schema moves into the system prompt and is validated
      client-side, with instructor's single repair re-ask on a mismatch. This
      gives up server-side *schema* enforcement to escape the loop trap. Choose
      it when a non-trivial schema drives runaway loops on real workloads; the
      cost is an occasional repair round-trip. (``DeepSeekProvider`` pins the
      same ``Mode.JSON`` with the same trade.)

    An unrecognized ``structured_output`` raises ``ValueError`` at construction
    rather than silently falling back.
    """

    _provider_name: ClassVar[str] = "Google AI Studio"
    _model_prefix: ClassVar[str] = "gemini/"
    # Default strategy + the contract value ``__init_subclass__`` requires; the
    # effective mode is resolved per instance and returned by ``instructor_mode``
    # (see the Vertex provider for the same pattern and why the ClassVar stays).
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        structured_output: str = "schema",
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key
        # Resolve the host-selected strategy once, loudly rejecting a typo.
        self._instructor_mode: instructor.Mode = resolve_gemini_structured_output(structured_output)

    @property
    @override
    def instructor_mode(self) -> instructor.Mode:
        """The instructor mode pinning structured output, per the selected strategy.

        ``"schema"`` (default) yields ``Mode.JSON_SCHEMA``; ``"json"`` yields
        ``Mode.JSON`` — see the class docstring for the repetition-loop-trap
        rationale behind the choice.
        """
        return self._instructor_mode

    @override
    def completion_kwargs(self) -> dict[str, object]:
        return {"api_key": self._api_key}

    @classmethod
    def build(cls, config: LLMClientConfig) -> GoogleProvider:
        """Construct from an :class:`LLMClientConfig`."""
        return cls(
            api_key=config.api_key or "",
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            structured_output=config.gemini_structured_output,
        )
