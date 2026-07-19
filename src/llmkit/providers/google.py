"""Google AI Studio LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_api_key,
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

    An optional ``base_url`` points the provider at a Gemini-compatible
    gateway (LiteLLM accepts ``api_base`` on the ``gemini/`` route); left
    unset, LiteLLM uses Google AI Studio's default endpoint (so ``api_base``
    is only forwarded when a ``base_url`` is given).
    """

    _provider_name: ClassVar[str] = "Google AI Studio"
    _model_prefix: ClassVar[str] = "gemini/"
    # Default strategy + the contract value ``__init_subclass__`` requires; the
    # effective mode is resolved per instance and returned by ``instructor_mode``
    # (see the Vertex provider for the same pattern and why the ClassVar stays).
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"
    _api_key_env_var: ClassVar[str] = "GEMINI_API_KEY"
    _accepted_config_fields: ClassVar[frozenset[str]] = frozenset(
        {"api_key", "base_url", "gemini_structured_output"}
    )

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        structured_output: str = "schema",
    ):
        super().__init__(model, reasoning_effort)
        self._api_key: str = api_key
        self._base_url: str | None = base_url
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
        kwargs: dict[str, object] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["api_base"] = self._base_url
        return kwargs

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> GoogleProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``GEMINI_API_KEY``
        environment variable, else raises (see :func:`resolve_api_key`);
        ``base_url`` is passed through.
        """
        return cls(
            api_key=resolve_api_key(
                config.api_key, env_var=cls._api_key_env_var, provider_name=cls._provider_name
            ),
            model=config.model,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
            structured_output=config.gemini_structured_output,
        )
