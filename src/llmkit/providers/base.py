"""Provider-agnostic core for the provider package.

Defines the shared contract every LLM provider builds on: the
library-owned :class:`Provider` enum, the :class:`LLMClientConfig`
credentials carrier, the module-level config-source registration
(:func:`configure_llm_client`), the :class:`LLMProviderInterface`
protocol, the :class:`BaseProvider` ABC, and the :class:`LLMInfo`
display snapshot.

This module knows nothing about any concrete provider. Each provider
lives in its own sibling module (``openrouter.py``, ``ollama.py``, …)
and supplies what the LiteLLM call layer needs: a LiteLLM **model
string** (provider-prefixed, e.g. ``openrouter/<model>``,
``gemini/<model>``), the completion kwargs that carry credentials
(``api_key`` / ``api_base``), and the :class:`instructor.Mode` that pins
the provider's structured-output strategy. The mode map is explicit on
purpose — instructor's auto-mode falls back to ``Mode.TOOLS``, which
silently regresses Gemini structured output (measured 0% valid /
silent-empty shapes), so every provider names its mode. Since instructor
1.15.3, ``from_litellm`` (this library's call seam) validates the mode
against its LiteLLM registry at client construction, so only the core
modes (``JSON``, ``JSON_SCHEMA``, ``TOOLS``, …) are pinnable — the
per-provider modes it dropped raise ``RegistryError`` (see
``tests/providers/test_provider_contract.py``, which constructs every
shipped pin against the real registry).

Dispatch (config → provider) and the public facade live in this
package's ``__init__.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from enum import StrEnum
from typing import ClassVar, Literal, Protocol, cast, overload, override, runtime_checkable

import instructor

from llmkit._types import ReasoningEffort

logger = logging.getLogger(__name__)


def _require_sdk(
    import_name: str, *, provider_name: str, extra: str, label: str | None = None, note: str = ""
) -> None:
    """Raise a clear, actionable error if ``import_name`` is not importable.

    Generic backing for the per-dependency ``require_*_sdk`` wrappers below.
    Optional provider dependencies ship in opt-in pip extras (so hosts that
    never touch a given provider take on no extra dependency) and are checked
    *eagerly* at provider construction, so a missing dep fails with a fix
    instead of as a cryptic ``ModuleNotFound`` deep on the first completion.

    ``label`` is the human-readable dependency name used in the message (e.g.
    ``"the Anthropic SDK"``); it defaults to ``import_name``. ``note`` is an
    optional trailing sentence appended to the message (e.g. to cross-reference
    a related extra).
    """
    try:
        spec = importlib.util.find_spec(import_name)
    except ModuleNotFoundError:
        # A dotted ``import_name`` (e.g. ``google.auth``) makes ``find_spec``
        # import the parent package first; when the parent itself is absent it
        # raises ``ModuleNotFoundError`` rather than returning ``None``. Treat
        # that as "not installed" so the actionable message below fires either
        # way (a plain top-level name like ``boto3`` returns ``None`` and never
        # reaches here, so existing callers are unaffected).
        spec = None
    if spec is None:
        message = (
            f"The {provider_name} provider requires {label or import_name}, which is "
            "not installed. It ships in an opt-in extra so hosts that don't use "
            "this provider take on no extra dependency. Install it with:\n\n"
            f"    pip install 'omg-llmkit[{extra}]'"
        )
        if note:
            message += f"\n\n{note}"
        raise ModuleNotFoundError(message)


def require_boto3_sdk(provider_name: str) -> None:
    """Raise a clear, actionable error if ``boto3`` is not installed.

    LiteLLM's Bedrock path uses ``boto3`` to sign requests (AWS SigV4), so the
    Bedrock provider needs it at call time. It ships in the opt-in
    ``omg-llmkit[bedrock]`` extra rather than the core install. Call this at
    provider construction so a missing ``boto3`` fails *eagerly* with a fix,
    instead of as a cryptic ``ModuleNotFound`` deep on the first completion.
    """
    _require_sdk(
        "boto3",
        provider_name=provider_name,
        extra="bedrock",
        note="(boto3 signs Bedrock requests with AWS SigV4.)",
    )


def require_google_auth_sdk(provider_name: str) -> None:
    """Raise a clear, actionable error if ``google-auth`` is not installed.

    LiteLLM's ``vertex_ai/`` path signs Gemini-on-Vertex requests with a Google
    OAuth access token that it mints and refreshes through ``google.auth`` (the
    Application Default Credentials chain), so the Vertex provider needs it at
    call time. It is the minimal credential dependency the gemini-on-Vertex chat
    path actually imports — deliberately **not** the much heavier
    ``google-cloud-aiplatform`` (which targets instructor's ``from_vertexai``
    google-SDK client, a path this library does not use). It ships in the opt-in
    ``omg-llmkit[vertex]`` extra rather than the core install. Call this at
    provider construction so a missing ``google-auth`` fails *eagerly* with a
    fix, instead of as a cryptic ``ModuleNotFound`` deep on the first completion.
    """
    _require_sdk(
        "google.auth",
        provider_name=provider_name,
        extra="vertex",
        label="the google-auth library (google.auth)",
        note=(
            "(google-auth mints and refreshes the OAuth access token LiteLLM "
            "signs Vertex AI requests with, via Application Default Credentials.)"
        ),
    )


#: The Gemini structured-output strategies a host may select, mapped to the
#: instructor ``Mode`` each one pins. ``"schema"`` is Gemini's native
#: JSON-schema constrained decoding (``responseSchema`` / ``Mode.JSON_SCHEMA``),
#: the pre-existing default; ``"json"`` is JSON-mime-type-only output with the
#: schema moved into the system prompt and validated client-side
#: (``Mode.JSON``). See :func:`resolve_gemini_structured_output` and the Gemini
#: provider docstrings for the measurement that motivates the choice.
_GEMINI_STRUCTURED_OUTPUT_MODES: dict[str, instructor.Mode] = {
    "schema": instructor.Mode.JSON_SCHEMA,
    "json": instructor.Mode.JSON,
}


def resolve_gemini_structured_output(strategy: str) -> instructor.Mode:
    """Map a Gemini structured-output strategy name to its instructor ``Mode``.

    Single-sources the ``"schema"`` / ``"json"`` → ``Mode`` resolution the two
    Gemini providers (:class:`~llmkit.providers.vertex.VertexProvider`,
    :class:`~llmkit.providers.google.GoogleProvider`) both consume, so the valid
    values and the loud-failure behavior live in exactly one place.

    An unrecognized ``strategy`` raises :class:`ValueError` naming the valid
    values. The failure is *deliberate*: the config field is typed
    ``Literal["schema", "json"]`` but that annotation is not enforced at runtime,
    so a typo (or a value from a dynamic/untyped config source) must fail loudly
    at provider construction rather than silently falling back to a default and
    shipping the wrong structured-output guarantee.
    """
    try:
        return _GEMINI_STRUCTURED_OUTPUT_MODES[strategy]
    except KeyError:
        valid = ", ".join(repr(name) for name in _GEMINI_STRUCTURED_OUTPUT_MODES)
        raise ValueError(
            f"Unknown Gemini structured-output strategy {strategy!r}; "
            + f"valid values are {valid}."
        ) from None


def resolve_api_key(configured: str | None, *, env_var: str, provider_name: str) -> str:
    """Resolve a bearer provider's API key: explicit config, then env var, else raise.

    The five key-authenticated providers (OpenRouter, Anthropic, OpenAI,
    DeepSeek, Google AI Studio) previously coerced a missing ``api_key`` to
    ``""`` and handed that to LiteLLM, which then *silently* fell back to the
    provider's ambient environment key (``ANTHROPIC_API_KEY`` and friends). A
    caller who believed they were pinning credentials explicitly could bill or
    authenticate as a different identity than intended — an undocumented,
    invisible swap (and, because importing LiteLLM runs ``load_dotenv()``, even
    a stray ``.env`` could inject that key).

    This makes the same env-var fallback an *explicit, documented, tested*
    library step instead: the configured value wins when set; otherwise the
    provider's own environment variable is consulted (the first-party-SDK idiom
    each bearer provider already follows); and when neither is present the
    provider fails loud at construction, naming exactly what to set — never an
    empty key wandering into the transport to fail (or silently succeed) at call
    time. An empty string is treated as unset, matching the falsy-``model``
    convention on :class:`LLMClientConfig`.
    """
    if configured:
        return configured
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env
    raise ValueError(
        f"{provider_name} requires an API key and none is configured: set "
        + f"LLMClientConfig.api_key (or the {env_var} environment variable)."
    )


@overload
def resolve_base_url(configured: str | None, *, env_vars: tuple[str, ...], default: str) -> str: ...


@overload
def resolve_base_url(
    configured: str | None, *, env_vars: tuple[str, ...], default: None
) -> str | None: ...


def resolve_base_url(
    configured: str | None, *, env_vars: tuple[str, ...], default: str | None
) -> str | None:
    """Resolve a provider's endpoint: explicit config, then env var(s), else our default.

    The endpoint half of what :func:`resolve_api_key` did for the credential, and
    live for the same reason. Omitting ``api_base`` hands endpoint selection to
    LiteLLM's own chain (``api_base or litellm.api_base or
    get_secret("OPENAI_BASE_URL") or get_secret("OPENAI_API_BASE") or
    "https://api.openai.com/v1"``, and the per-provider equivalents), so
    something other than the caller picks the endpoint while their *explicitly
    pinned* ``api_key`` rides along to whatever it names. That "something" is an
    open-ended set: an in-process module global, a configured LiteLLM
    key-management backend (``get_secret`` consults one as well as
    ``os.environ``), the ambient environment — reachable from a stray ``.env``
    anywhere up the tree, since importing LiteLLM runs ``load_dotenv()`` — and
    whatever a future release adds. Enumerating those in order to *reject* them
    would be a blocklist against an unversioned dependency; returning a value the
    caller's provider then always sends is closed instead, because an explicit
    ``api_base`` outranks every other entry in the chains of the four routes this
    serves. (Ollama's chain is inverted and is *not* one of them — see
    :class:`~llmkit.providers.ollama.OllamaProvider`.)

    Unlike the key this never raises: an endpoint has a correct publishable
    default and a secret does not. It returns ``None`` only when ``default`` is
    ``None`` and nothing is configured — the Gemini case below.

    **Call this at the read point, not at construction.** LiteLLM consulted these
    variables when the request was built, and importing LiteLLM runs
    ``load_dotenv()``, so the environment a provider is *constructed* in is
    routinely not the environment the call goes out in: ``import llmkit`` does not
    import LiteLLM (it is deferred to first call), so a provider built at startup
    would freeze a default and then miss the very ``.env`` value LiteLLM would
    have honoured — silently sending the call to the public endpoint instead of
    the host's gateway. Each provider therefore calls this from
    ``completion_kwargs()``, which ``llmkit._litellm`` reaches only after
    importing LiteLLM. (Measured 2026-07-21: resolving at construction re-pointed
    a ``.env``-configured gateway to ``api.openai.com``.)

    **What that does and does not close.** It removes every endpoint source
    llmkit does not read. It deliberately does *not* remove ``env_vars``: the
    fallback is *kept*, with llmkit reading the same names LiteLLM read, in
    LiteLLM's own measured precedence, so a host that configures its endpoint
    that way is not silently redirected to the public one — which would be the
    very data-egress surprise this exists to prevent, merely aimed at different
    users. A set variable therefore still selects the endpoint, now by llmkit's
    documented decision and visible in ``completion_kwargs()``; the same bargain
    :func:`resolve_api_key` struck for the credential. An empty string is treated
    as unset, matching it too.

    **A ``None`` default means "let LiteLLM derive it".** Only the Google AI
    Studio route passes one, because its base carries an API *version* that
    LiteLLM picks from the model (``v1alpha`` for Gemini 3 and newer,
    ``v1beta`` otherwise — measured). Pinning a static version-bearing default
    there would silently downgrade every Gemini 3 call, so when nothing is
    configured that provider sends no ``api_base`` and lets LiteLLM choose the
    version. Deriving it here instead would be endpoint *computation*, which is
    the gateway-shaped work this library does not do.
    """
    if configured:
        return configured
    for env_var in env_vars:
        from_env = os.environ.get(env_var)
        if from_env:
            return from_env
    return default


class Provider(StrEnum):
    """LLM providers the library can route to.

    Library-owned enum (no dependency on the host's configuration).
    """

    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


@dataclass(frozen=True, repr=False)
class LLMClientConfig:
    """Provider selection + credentials for the active LLM provider.

    The library is config-source-agnostic: the host application builds
    this from whatever settings system it uses and registers a source
    via :func:`configure_llm_client`, so the library never imports the
    host's config module.

    Only the *active* provider's fields need be populated — and populating a
    provider-shaped field the selected provider does **not** read is rejected at
    :func:`~llmkit.providers.build_provider`, not silently ignored (each provider
    declares the knobs it honours; see ``_accepted_config_fields``). ``api_key``
    is a bearer credential for the five key-authenticated providers (OpenRouter,
    OpenAI, Anthropic, Google, DeepSeek); it resolves from this value, else the
    provider's environment variable (``ANTHROPIC_API_KEY`` and friends), else the
    provider raises at construction. It is not accepted by Ollama (local) or
    Bedrock/Vertex (ambient AWS/Google chains). The credential is masked in this
    config's ``repr`` — a set key shows as ``api_key=<redacted>`` — so a stray
    ``print`` or traceback never leaks it. ``base_url`` sets the endpoint for the
    five key-authenticated providers and Ollama: it resolves from this value,
    else the provider's own documented endpoint variable(s) (``OPENAI_BASE_URL``
    / ``OPENAI_API_BASE``, ``ANTHROPIC_API_BASE`` and friends — still honoured,
    now read explicitly by llmkit in a tested order), else that provider's own
    measured default endpoint. The endpoint is therefore llmkit's decision,
    readable from ``completion_kwargs()`` rather than inferred from LiteLLM's own
    chain, and it is resolved per call so it reflects the environment the call
    goes out in (see :func:`resolve_base_url`). Google AI Studio is the one
    provider with no default — its base carries an API version LiteLLM picks from
    the model — so with neither this field nor ``GEMINI_API_BASE`` set it sends no
    ``api_base`` and lets LiteLLM choose. It is not accepted by
    Bedrock or Vertex, whose endpoints derive from ``aws_region_name`` /
    ``vertex_location`` instead. Per-call ``model`` overrides (e.g. the
    strong/small roles the host resolves) are passed at call time and are not
    part of this config — this carries only the provider's *default* model.

    ``model`` is optional: leave it ``None`` (or empty) to inherit the
    selected provider's own default model rather than naming one here. A
    falsy ``model`` resolves to the concrete provider's built-in default
    (e.g. ``claude-sonnet-4-6`` for Anthropic), so the assembled LiteLLM
    id is always well-formed — never a dangling ``"anthropic/"``.

    ``reasoning_effort`` controls provider "thinking"/reasoning tokens,
    with portable aliases (``"disable" | "low" | "medium" | "high"``).
    ``None`` (the default) sends no reasoning control, leaving the provider's
    own default in place — byte-identical to prior behaviour. Set
    it once here (e.g. ``"disable"``) to have **every** call inherit the
    setting; Gemini's default-on thinking otherwise spends reasoning tokens
    against ``max_tokens`` and can truncate small-capped structured output.

    ``aws_region_name`` carries the AWS region for the Bedrock provider and
    is not accepted by any other provider (which authenticate with ``api_key`` /
    ``base_url``). It is the *only* AWS-shaped field on this config on
    purpose: Bedrock's secrets (access key / secret / session token, or
    instance-role credentials) resolve from the **ambient AWS credential
    chain** — environment, shared config, or instance/role — so they never
    pass through ``LLMClientConfig``. Leave it ``None`` to let the region
    resolve from the chain too (``AWS_REGION_NAME`` / ``AWS_REGION``).

    ``vertex_project`` and ``vertex_location`` are the Vertex AI analog of
    ``aws_region_name`` and are not accepted by any other provider. They are the
    only Vertex-shaped fields on purpose: like Bedrock, the Vertex provider
    carries **no secret** here — Google credentials resolve from **Application
    Default Credentials** (``gcloud auth application-default login``,
    ``GOOGLE_APPLICATION_CREDENTIALS``, or a workload-identity/metadata-server
    token), never through ``LLMClientConfig``. ``vertex_location`` is the
    **data-processing residency** control: it selects the endpoint where the
    request runs (LiteLLM derives three shapes from it —
    ``https://aiplatform.googleapis.com`` for ``"global"``,
    ``https://aiplatform.<geo>.rep.googleapis.com`` for a multi-region
    geography, else ``https://<location>-aiplatform.googleapis.com``), so a
    regional value (e.g. ``"europe-west4"``) pins in-region processing while
    the ``"global"`` endpoint gives no residency guarantee. That pin is
    overridable from outside the config — see
    :class:`~llmkit.providers.vertex.VertexProvider`. Leave either
    ``None`` to let it resolve from the environment (``VERTEXAI_PROJECT`` /
    ``VERTEXAI_LOCATION``), with the location otherwise falling back to
    Google's default region.

    ``gemini_structured_output`` selects the structured-output *strategy* for
    the two Gemini providers (Vertex, Google AI Studio); a non-default value is
    not accepted by any other provider (the default ``"schema"`` is always
    allowed). ``"schema"`` (the default) preserves the pre-existing wire
    behavior exactly — Gemini's native JSON-schema constrained decoding
    (``responseSchema`` / ``instructor.Mode.JSON_SCHEMA``), with server-side
    schema enforcement. ``"json"`` switches to JSON-mime-type-only output
    (``Mode.JSON``): the schema moves into the system prompt and is validated
    client-side, trading server-side schema enforcement for an escape from
    Gemini's constrained-decoding repetition-loop trap (measured 67-83%
    first-attempt runaway under ``"schema"`` vs 0% under ``"json"`` on real
    workloads — PIA Maker, 2026-07-15; see the Gemini provider docstrings). An
    unrecognized value fails loudly at provider construction — the ``Literal``
    type is not runtime-enforced, so a typo must not silently fall back.
    """

    provider: Provider
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    aws_region_name: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    gemini_structured_output: Literal["schema", "json"] = "schema"

    @override
    def __repr__(self) -> str:
        # The generated dataclass repr renders ``api_key='sk-...'`` verbatim,
        # so any host-side ``print(config)``, log line, or exception reporter's
        # locals capture would exfiltrate the credential. Mask it: a *set* key
        # shows ``api_key=<redacted>`` (presence stays debuggable, the value
        # never prints), while an empty ``""`` stays visible as itself since it
        # is not a secret and an accidental empty key is worth seeing.
        #
        # Only fields that differ from their default are shown — the same
        # signal-over-noise choice ``LLMCallOptions.__repr__`` makes; ``provider``
        # has no default, so it always prints. ``field(repr=False)`` on ``api_key``
        # alone was rejected for the reason ``options.py`` documents: it would drop
        # the field entirely, hiding whether a key is set at all.
        parts: list[str] = []
        for f in fields(self):
            value = cast("object", getattr(self, f.name))
            default = cast("object", f.default)
            if default is not MISSING and value == default:
                continue
            shown = repr(value)
            if f.name == "api_key" and value:
                shown = "<redacted>"
            parts.append(f"{f.name}={shown}")
        return f"{type(self).__name__}({', '.join(parts)})"


# Module-level config source, registered once by the host application at a
# universally-imported wiring edge (see ``app/__init__.py``). Mirrors the
# ``configure_rate_limit`` module-level pattern so call sites that only pass
# ``feature``/``label``/``model`` keep working with zero per-call wiring.
_config_source: Callable[[], LLMClientConfig] | None = None


def configure_llm_client(config_source: Callable[[], LLMClientConfig]) -> None:
    """Register the source that supplies the active :class:`LLMClientConfig`.

    The source is a zero-arg callable invoked on each :func:`build_provider`
    call, mirroring the previous behaviour of reading the host's
    (cached) settings on every provider construction — so a settings
    change is picked up without re-registration.
    """
    global _config_source
    _config_source = config_source


def active_config() -> LLMClientConfig:
    """Return the configured :class:`LLMClientConfig`, or raise if unset."""
    if _config_source is None:
        raise RuntimeError(
            "LLM client not configured: call configure_llm_client(...) at "
            + "application startup before constructing a provider."
        )
    return _config_source()


@runtime_checkable
class LLMProviderInterface(Protocol):
    """Protocol for LLM provider implementations."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def model(self) -> str:
        """Current default model name (unprefixed)."""
        ...

    @property
    def instructor_mode(self) -> instructor.Mode:
        """The instructor mode pinning structured output for this provider."""
        ...

    @property
    def strict_json_schema(self) -> bool:
        """Whether structured requests upgrade ``response_format`` to strict enforcement."""
        ...

    @property
    def compose_tools_schema(self) -> bool:
        """Whether this provider has a measured stable tools+schema capability."""
        ...

    def guard_compose_tools_schema(self, model: str | None) -> None:
        """Refuse a model that would silently fall back while composing."""
        ...

    @property
    def supports_tool_choice(self) -> bool:
        """Whether this route accepts LiteLLM's ``tool_choice`` control."""
        ...

    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        """Configured reasoning/thinking effort, or ``None`` for provider default."""
        ...

    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Translate a resolved reasoning effort into provider request kwargs."""
        ...

    def litellm_model(self, model: str | None = None) -> str:
        """The provider-prefixed LiteLLM model string for ``model`` (or the default)."""
        ...

    def completion_kwargs(self) -> dict[str, object]:
        """Credential + routing kwargs forwarded to ``litellm`` calls.

        Carries the credential kwargs (``api_key`` / ``api_base`` /
        ``aws_region_name``) and any provider-specific routing preference (e.g.
        OpenRouter's ``extra_body`` ``require_parameters``). The value type is
        ``object`` rather than ``str`` because a routing preference is a nested
        mapping, not a string.
        """
        ...


#: The provider-shaped :class:`LLMClientConfig` knobs — the ones a given provider
#: may or may not read, as opposed to ``provider`` / ``model`` /
#: ``reasoning_effort``, which every provider reads. Each provider declares the
#: subset it honours in ``_accepted_config_fields``; the rest are rejected by
#: :meth:`BaseProvider.reject_unread_config_fields`.
_PROVIDER_SHAPED_FIELDS: frozenset[str] = frozenset(
    {
        "api_key",
        "base_url",
        "aws_region_name",
        "vertex_project",
        "vertex_location",
        "gemini_structured_output",
    }
)


class BaseProvider(ABC):
    """Base class for LLM providers.

    Implements the shared public interface; subclasses supply the
    LiteLLM-specific routing via the ``_model_prefix`` / ``instructor_mode``
    hooks and ``completion_kwargs``.

    Each concrete subclass also provides a ``build(config)`` classmethod
    that maps an :class:`LLMClientConfig` to a constructed instance
    (including the provider's own credential/endpoint defaults). ``build``
    is a package-internal construction convention used by
    :func:`llmkit.providers.build_provider`; it is deliberately **not** part
    of :class:`LLMProviderInterface`, which is the surface consumers type
    against.
    """

    # Routing-contract hooks every concrete provider MUST set. They are
    # declared without class-level default *values* on purpose: a silent
    # default here is a footgun (a forgotten ``_mode`` would inherit a
    # JSON-schema mode that silently regresses Gemini/DeepSeek structured
    # output; a forgotten ``_default_model`` would assemble a dangling
    # ``"<prefix>/"`` id; a forgotten ``_provider_name`` would degrade logging,
    # rate-limit keying, and ``describe_llm``). ``__init_subclass__`` below
    # turns the "every provider names these" convention into an enforced
    # contract — an incomplete provider fails loudly at import, not at routing
    # time with a provider-shaped misbehavior.
    _provider_name: ClassVar[str]
    _mode: ClassVar[instructor.Mode]
    _default_model: ClassVar[str]
    # ``_model_prefix`` keeps a ``""`` default deliberately: an empty prefix is
    # legitimate (it still assembles a well-formed bare-model id), so unlike the
    # three hooks above a missing value is not a footgun and is left unguarded.
    _model_prefix: ClassVar[str] = ""
    #: Whether this provider runs against a local endpoint (no data leaves the
    #: host). A provider trait rather than a dispatch special-case, so locality
    #: stays defined alongside the provider it describes (see
    #: :func:`~llmkit.providers.describe_llm`). Cloud providers leave this False.
    is_local: ClassVar[bool] = False
    #: Whether structured requests upgrade instructor's ``json_schema``
    #: ``response_format`` to *strict* server-side enforcement (``"strict":
    #: true`` + top-level ``additionalProperties: false``) at the LiteLLM call
    #: seam (see ``llmkit._litellm``). instructor's ``JSON_SCHEMA`` handler
    #: sends neither, and some providers (OpenRouter) treat a non-strict
    #: json_schema as advisory — weak models then drift (measured: the schema
    #: echoed back with values nested inside ``properties``). A provider
    #: trait, not a global: providers whose enforcement was measured fine
    #: without it (OpenAI, Anthropic, Bedrock, Google, Ollama, Vertex) keep
    #: their measured wire shape untouched. Exposed via the
    #: ``strict_json_schema`` property (the ``_mode`` → ``instructor_mode``
    #: pattern).
    _strict_json_schema: ClassVar[bool] = False
    #: Whether tools plus a strict response schema are a measured, stable
    #: capability for this provider. Preview-only or unmeasured routes remain
    #: false so callers receive a loud steer to the portable two-step pattern.
    _compose_tools_schema: ClassVar[bool] = False
    _supports_tool_choice: ClassVar[bool] = True

    #: The routing-contract hooks validated for every concrete subclass.
    _required_hooks: ClassVar[tuple[str, ...]] = ("_provider_name", "_mode", "_default_model")

    #: The provider-shaped :class:`LLMClientConfig` knobs this provider actually
    #: reads in ``build`` (from :data:`_PROVIDER_SHAPED_FIELDS`). Declared per
    #: provider so :func:`reject_unread_config_fields` can turn "passed a knob the
    #: provider silently ignores" into a loud error — a new provider *cannot*
    #: forget to declare it (``__init_subclass__`` enforces its presence). It is
    #: annotation-only here (no value), so a concrete subclass that omits it trips
    #: that check; a provider reading none of the six may declare an empty
    #: frozenset. ``provider`` / ``model`` / ``reasoning_effort`` are universal
    #: and never listed.
    _accepted_config_fields: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject a concrete provider that omits a routing-contract hook.

        Fires at class definition (import time), so an incomplete provider
        fails the moment its module is imported — earlier and louder than a
        construction-time check, and before any call can route through it.

        Only *concrete* (instantiable) subclasses must satisfy the contract:
        an intermediate ABC that leaves an abstract method (e.g.
        ``completion_kwargs``) unimplemented is exempt until a concrete
        subclass fills it in. ``inspect.isabstract`` is the structural signal,
        so no provider needs to opt in or out by hand.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        missing = [hook for hook in cls._required_hooks if not getattr(cls, hook, None)]
        if missing:
            raise TypeError(
                f"{cls.__name__} is an incomplete provider: it must set "
                + f"{', '.join(missing)}. Every concrete BaseProvider subclass names its own "
                + "provider, instructor mode, and default model so structured-output routing, "
                + "rate-limit keying, and the assembled LiteLLM id are explicit — never "
                + "inherited from a silent class-level default."
            )
        # ``_accepted_config_fields`` is checked for *presence*, not truthiness:
        # an empty frozenset (a provider reading none of the provider-shaped
        # knobs) is a legitimate declaration, so ``not getattr(...)`` would wrongly
        # reject it. Annotation-only on the base means an omitting subclass reads
        # back the sentinel ``None`` here.
        if getattr(cls, "_accepted_config_fields", None) is None:
            raise TypeError(
                f"{cls.__name__} is an incomplete provider: it must set "
                + "_accepted_config_fields — the frozenset of LLMClientConfig knobs it "
                + "reads — so build_provider can reject a config field the provider would "
                + "silently ignore. Declare it (a frozenset of field names, possibly empty)."
            )

    def __init__(
        self, model: str | None = None, reasoning_effort: ReasoningEffort | None = None
    ) -> None:
        # A falsy model (None or "") falls back to the provider's own default
        # so the LiteLLM id assembled by ``litellm_model`` is always well-formed
        # (``<prefix><model>``) and never a dangling ``"<prefix>/"``. The
        # ``__init_subclass__`` contract guarantees ``_default_model`` is a
        # non-empty value every concrete provider set, so this fallback is
        # well-formed by construction, not merely by convention.
        self._model: str = model or self._default_model
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort

    @property
    def name(self) -> str:
        """Provider name."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Current default model name (unprefixed)."""
        return self._model

    @property
    def instructor_mode(self) -> instructor.Mode:
        """The instructor mode pinning structured output for this provider."""
        return self._mode

    @property
    def strict_json_schema(self) -> bool:
        """Whether structured requests upgrade ``response_format`` to strict enforcement.

        See the ``_strict_json_schema`` trait above for the rationale; the
        upgrade itself happens at the LiteLLM call seam in ``llmkit._litellm``.
        """
        return self._strict_json_schema

    @property
    def compose_tools_schema(self) -> bool:
        """Whether this provider supports a combined tools + schema request."""
        return self._compose_tools_schema

    def guard_compose_tools_schema(self, model: str | None) -> None:
        """Apply provider/model-specific compose safety checks.

        Most providers have no additional model split.  Routes with a
        synthetic-tool fallback override this before a transport call.
        """
        del model

    @property
    def supports_tool_choice(self) -> bool:
        """Whether this provider supports explicit tool selection."""
        return self._supports_tool_choice

    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        """Configured reasoning/thinking effort, or ``None`` for provider default.

        Translated by :meth:`reasoning_kwargs` when set; ``None`` sends
        nothing, so the provider's own thinking default stands.
        """
        return self._reasoning_effort

    def reasoning_kwargs(self, effort: ReasoningEffort, model: str) -> dict[str, object]:
        """Return LiteLLM's portable reasoning kwarg for ``effort``.

        Providers with a native reasoning control override this method.  It is
        concrete so existing provider subclasses only implementing the two
        abstract construction hooks remain valid.
        """
        del model
        return {"reasoning_effort": effort}

    def litellm_model(self, model: str | None = None) -> str:
        """Return the provider-prefixed LiteLLM model string.

        ``model`` overrides the provider's configured default when set
        (e.g. the host's strong/small role resolved to a concrete model).
        """
        return f"{self._model_prefix}{model or self._model}"

    @abstractmethod
    def completion_kwargs(self) -> dict[str, object]:
        """Return the credential + routing kwargs forwarded to every ``litellm`` call.

        Values are typed ``object`` (not ``str``) because a provider may add a
        nested routing preference (e.g. OpenRouter's ``extra_body``) alongside
        the string credential kwargs.
        """
        ...

    @classmethod
    @abstractmethod
    def build(cls, config: LLMClientConfig) -> BaseProvider:
        """Construct this provider from an :class:`LLMClientConfig`.

        The package-internal *construct-from-config* hook (see the class
        docstring): each provider maps the flat config onto its own constructor,
        applying its credential/endpoint defaults. Declared abstract so a new
        provider cannot forget it, and so :func:`~llmkit.providers.build_provider`
        can invoke it through the base type. It is deliberately **not** part of
        :class:`LLMProviderInterface`, the surface consumers type against.
        """
        ...

    @classmethod
    def reject_unread_config_fields(cls, config: LLMClientConfig) -> None:
        """Reject a populated config knob this provider will not read.

        Every provider reads ``provider`` / ``model`` / ``reasoning_effort``, but
        the six provider-shaped knobs (:data:`_PROVIDER_SHAPED_FIELDS`) are
        provider-specific. Passing one to a provider that never reads it — an
        ``api_key`` to Bedrock or Vertex (which authenticate via their ambient
        AWS/Google credential chains), a ``base_url`` to a fixed-endpoint
        provider, a non-default ``gemini_structured_output`` to a non-Gemini
        provider — used to be a silent no-op. That silence is a footgun: a host
        that populates a config generically believes it pinned a credential or
        endpoint that is in fact ignored.

        A knob counts as *populated* when it differs from its dataclass default
        (so an untouched field never trips this). Anything populated beyond this
        provider's ``_accepted_config_fields`` fails loud. Called from
        :func:`~llmkit.providers.build_provider` — the single seam ``make_provider``,
        the ``configure_llm_client`` source, and a direct ``build_provider(config)``
        all flow through — so a directly-built config is checked too.
        """
        accepted = cls._accepted_config_fields
        offenders: list[str] = []
        for f in fields(config):
            if f.name not in _PROVIDER_SHAPED_FIELDS or f.name in accepted:
                continue
            value = cast("object", getattr(config, f.name))
            default = cast("object", f.default)
            # A knob is "populated" when it differs from its default AND is not an
            # empty string. Empty is unset everywhere else in the library (the
            # bearer key resolver and each ``build`` treat ``""`` as no value), so
            # a generically-built config that fills an unused field with ``""``
            # (e.g. ``os.getenv("KEY", "")`` for a Bedrock deployment) must not
            # trip this — only a real value the provider would ignore does.
            if value != default and value != "":
                offenders.append(f.name)
        if not offenders:
            return
        reads = ", ".join(sorted(accepted)) if accepted else "none of the provider-shaped fields"
        raise ValueError(
            f"{cls._provider_name} does not read the config field(s) {sorted(offenders)}: it "
            + "would silently ignore them, which llmkit refuses. Populate only the fields the "
            + f"selected provider uses — {cls._provider_name} reads: {reads} (every provider "
            + "also reads 'model' and 'reasoning_effort')."
        )


@dataclass
class LLMInfo:
    """Descriptive metadata about the active LLM configuration.

    A read-only snapshot for display/telemetry (provider name, model,
    locality) — distinct from :class:`LLMClientConfig`, which carries the
    credentials used to *construct* a provider.
    """

    provider: Provider
    provider_name: str
    model: str
    is_local: bool
    compose_tools_schema: bool
