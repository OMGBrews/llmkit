"""Google AI Studio LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    resolve_api_key,
    resolve_base_url,
    resolve_gemini_structured_output,
)

#: Google AI Studio has **no** library-owned default endpoint, and that absence
#: is deliberate — it is the one provider whose base cannot be a constant.
#:
#: The AI Studio base carries an API *version*, and LiteLLM derives that version
#: from the model: ``v1alpha`` for Gemini 3 and newer, ``v1beta`` otherwise
#: (``litellm/llms/vertex_ai/common_utils.py``). It applies that only when
#: ``api_base`` is absent — a pinned base is used verbatim. Measured against
#: litellm 1.92.0 on 2026-07-21: pinning ``…/v1beta`` sends
#: ``gemini-3-pro-preview`` to ``/v1beta`` where it previously went to
#: ``/v1alpha``, and pinning the bare host drops the version segment entirely.
#: Either way a static default silently changes the wire shape for some models,
#: which is exactly what this change promises not to do.
#:
#: So when nothing is configured this provider sends no ``api_base`` and lets
#: LiteLLM pick the version. Computing it here instead would be endpoint
#: derivation — the gateway-shaped work llmkit does not do (``PRINCIPLES.md``),
#: and the same reason Bedrock and Vertex are excluded from endpoint ownership.
#: The cost is scoped and documented: with no ``base_url`` and no
#: ``GEMINI_API_BASE``, this provider leaves the endpoint to LiteLLM's chain —
#: the third of the three cases where llmkit declines to name an endpoint it
#: would have to *compute*, alongside Bedrock and Vertex. Naming either closes
#: it here, which is what makes this case the mildest of the three; the decision
#: covering all three is settled (README, "``BEDROCK`` and ``VERTEX`` do not own
#: their endpoints").
_DEFAULT_BASE_URL: str | None = None


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

    An optional ``base_url`` points the provider at a Gemini-compatible gateway
    (LiteLLM accepts ``api_base`` on the ``gemini/`` route). The endpoint
    resolves from ``base_url`` if set, else ``GEMINI_API_BASE`` — read here
    rather than left to LiteLLM, so it comes from the process environment only
    and is visible in ``completion_kwargs()``.

    **Unlike its three siblings this provider has no library-owned default**, so
    with neither configured it sends no ``api_base`` at all and LiteLLM picks the
    endpoint. That is not an oversight: the AI Studio base carries an API version
    LiteLLM derives *from the model* (``v1alpha`` for Gemini 3 and newer,
    ``v1beta`` otherwise), so any static default would silently move some models
    to the wrong version — see :data:`_DEFAULT_BASE_URL` for the measurement.
    The scoped consequence is that in that one case the endpoint can still come
    from a source llmkit does not read (notably the ``litellm.api_base`` module
    global); naming a ``base_url`` or ``GEMINI_API_BASE`` closes it.
    """

    _provider_name: ClassVar[str] = "Google AI Studio"
    _model_prefix: ClassVar[str] = "gemini/"
    # Default strategy + the contract value ``__init_subclass__`` requires; the
    # effective mode is resolved per instance and returned by ``instructor_mode``
    # (see the Vertex provider for the same pattern and why the ClassVar stays).
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"
    _api_key_env_var: ClassVar[str] = "GEMINI_API_KEY"
    #: The endpoint variable LiteLLM consulted for this route (measured, not read
    #: off the source). Read here so a host that relies on it keeps working —
    #: explicitly, and only here.
    _base_url_env_vars: ClassVar[tuple[str, ...]] = ("GEMINI_API_BASE",)
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
        # The *configured* endpoint, not the resolved one: resolution reads the
        # environment and so must happen at the read point, never here (see
        # ``completion_kwargs``).
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
        # Resolved per call, deliberately — see the note in ``OpenAIProvider``
        # and :func:`resolve_base_url`. The ``api_base`` key is omitted entirely
        # when nothing is configured, so LiteLLM still derives the model's API
        # version (see :data:`_DEFAULT_BASE_URL`).
        kwargs: dict[str, object] = {"api_key": self._api_key}
        endpoint = resolve_base_url(
            self._base_url, env_vars=self._base_url_env_vars, default=_DEFAULT_BASE_URL
        )
        if endpoint:
            kwargs["api_base"] = endpoint
        return kwargs

    @override
    @classmethod
    def build(cls, config: LLMClientConfig) -> GoogleProvider:
        """Construct from an :class:`LLMClientConfig`.

        ``api_key`` resolves from the config, else the ``GEMINI_API_KEY``
        environment variable, else raises (see :func:`resolve_api_key`).
        ``base_url`` is passed through as configured; the endpoint is resolved
        later, in ``completion_kwargs()``, so it observes the environment the
        call actually goes out in (see :func:`resolve_base_url`).
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
