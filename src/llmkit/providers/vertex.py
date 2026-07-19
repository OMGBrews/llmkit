"""Google Vertex AI LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit._types import ReasoningEffort
from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    require_google_auth_sdk,
    resolve_gemini_structured_output,
)


class VertexProvider(BaseProvider):
    """Google Vertex AI LLM provider.

    Routes Gemini through Google Cloud's Vertex AI (``vertex_ai/<model>``) — the
    enterprise path to the same Gemini models the direct
    :class:`~llmkit.providers.google.GoogleProvider` reaches through Google AI
    Studio (``gemini/<model>``). The relationship mirrors Anthropic-direct vs
    Bedrock: same underlying model, but a different routing namespace, a
    different credential story, and a region knob that controls data residency.

    Like Bedrock, Vertex does **not** authenticate with a bearer ``api_key``:
    LiteLLM mints and refreshes a Google OAuth access token from the standard
    **Application Default Credentials (ADC)** chain (``gcloud auth
    application-default login``, ``GOOGLE_APPLICATION_CREDENTIALS``, or a
    workload-identity/metadata-server token). This provider therefore carries no
    secret — only the non-secret routing fields ``vertex_project`` and
    ``vertex_location``, and each only when set; the credentials stay in the
    ambient chain and never pass through :class:`~llmkit.LLMClientConfig`. When
    both are left unset they resolve from the environment (``VERTEXAI_PROJECT`` /
    ``VERTEXAI_LOCATION``), with the location otherwise falling back to Google's
    default region.

    ``vertex_location`` is the **data-processing residency** control: it selects
    the regional endpoint (``<location>-aiplatform.googleapis.com``) where the
    request is processed. Pass a regional value (e.g. ``"europe-west4"``,
    ``"asia-northeast1"``) to pin in-region processing; the ``"global"`` endpoint
    gives no residency guarantee. Note Gemini availability is **region-specific**,
    so a region chosen for residency may not host every model — including the
    ``gemini-2.5-flash-lite`` default. A model not deployed in the region fails
    with a Vertex ``400 FAILED_PRECONDITION`` ("Precondition check failed."), an
    *availability* error distinct from auth/permission failures; pass a ``model``
    the region actually serves.

    Structured output defaults to ``instructor.Mode.JSON_SCHEMA`` — Gemini's
    native JSON-schema mode, the same mode the direct ``GoogleProvider`` pins,
    for the same reason: the underlying model is Gemini. Note instructor's
    ``Mode.VERTEXAI_JSON`` / ``Mode.VERTEXAI_TOOLS`` are deliberately **not**
    used: they target instructor's native ``from_vertexai`` client (the
    ``google-cloud-aiplatform`` SDK), not this library's ``from_litellm`` call
    seam — the same kind of mismatch that rules out ``Mode.BEDROCK_JSON`` for
    Bedrock. Over LiteLLM the working Gemini modes are ``JSON_SCHEMA`` and
    ``JSON`` (instructor auto-mode would fall back to ``Mode.TOOLS``, which
    measurably regresses Gemini structured output to empty/invalid shapes).

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

    ``reasoning_effort`` is forwarded to LiteLLM for Vertex Gemini models that
    support thinking and is harmless on those that don't. As with Google AI
    Studio, Gemini 2.5 models think by default, spending reasoning tokens
    against ``max_tokens``; pass ``reasoning_effort="disable"`` to turn it off
    so a small ``max_tokens`` cap doesn't truncate structured output.

    Vertex needs the ``google-auth`` library to mint its OAuth token; it ships
    via the ``omg-llmkit[vertex]`` extra rather than the core install, so
    non-Vertex users take on no Google dependency. It is checked *eagerly* at
    construction: constructing this provider without ``google-auth`` raises a
    clear "install omg-llmkit[vertex]" error instead of a cryptic
    ``ModuleNotFound`` deep on the first completion.
    """

    _provider_name: ClassVar[str] = "Google Vertex AI"
    _model_prefix: ClassVar[str] = "vertex_ai/"
    # The default strategy, and the value ``__init_subclass__`` requires every
    # concrete provider to name. The *effective* mode is resolved per instance
    # from ``structured_output`` and returned by the ``instructor_mode``
    # property below — the ClassVar stays the declared contract value, never
    # shadowed through ``self`` (which would fail basedpyright and break the
    # ``_mode`` → ``instructor_mode`` pattern for the other six providers).
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"

    def __init__(
        self,
        model: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        structured_output: str = "schema",
    ):
        require_google_auth_sdk(self._provider_name)
        super().__init__(model, reasoning_effort)
        self._vertex_project: str | None = vertex_project
        self._vertex_location: str | None = vertex_location
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
        """Return Vertex routing kwargs forwarded to ``litellm``.

        Only ``vertex_project`` / ``vertex_location`` are sent, and only when
        set; the OAuth credentials resolve from the ambient ADC chain. An empty
        dict leaves project/location resolution to the environment
        (``VERTEXAI_PROJECT`` / ``VERTEXAI_LOCATION``) and Google's default
        region as well.
        """
        kwargs: dict[str, object] = {}
        if self._vertex_project:
            kwargs["vertex_project"] = self._vertex_project
        if self._vertex_location:
            kwargs["vertex_location"] = self._vertex_location
        return kwargs

    @classmethod
    def build(cls, config: LLMClientConfig) -> VertexProvider:
        """Construct from an :class:`LLMClientConfig` (project/location only; no secrets)."""
        return cls(
            model=config.model,
            vertex_project=config.vertex_project,
            vertex_location=config.vertex_location,
            reasoning_effort=config.reasoning_effort,
            structured_output=config.gemini_structured_output,
        )
