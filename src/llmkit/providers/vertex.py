"""Google Vertex AI LLM provider."""

from __future__ import annotations

from typing import ClassVar, override

import instructor

from llmkit.providers.base import (
    BaseProvider,
    LLMClientConfig,
    require_google_auth_sdk,
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

    Structured output is pinned to ``instructor.Mode.JSON_SCHEMA`` — Gemini's
    native JSON-schema mode, the same mode the direct ``GoogleProvider`` pins,
    for the same reason: the underlying model is Gemini. Note instructor's
    ``Mode.VERTEXAI_JSON`` / ``Mode.VERTEXAI_TOOLS`` are deliberately **not**
    used: they target instructor's native ``from_vertexai`` client (the
    ``google-cloud-aiplatform`` SDK), not this library's ``from_litellm`` call
    seam — the same kind of mismatch that rules out ``Mode.BEDROCK_JSON`` for
    Bedrock. Over LiteLLM the working Gemini mode is ``JSON_SCHEMA`` (instructor
    auto-mode would fall back to ``Mode.TOOLS``, which measurably regresses
    Gemini structured output to empty/invalid shapes).

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
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON_SCHEMA
    _default_model: ClassVar[str] = "gemini-2.5-flash-lite"

    def __init__(
        self,
        model: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        reasoning_effort: str | None = None,
    ):
        require_google_auth_sdk(self._provider_name)
        super().__init__(model, reasoning_effort)
        self._vertex_project: str | None = vertex_project
        self._vertex_location: str | None = vertex_location

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
        )
