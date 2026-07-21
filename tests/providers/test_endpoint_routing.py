"""Wire-level: which endpoint LiteLLM actually resolves for a built provider.

Every other routing test in this suite captures the kwargs handed to
``litellm.acompletion`` (``test_openrouter_routing.py``, ``test_vertex_routing.py``)
— which by construction cannot answer the question that matters here, because the
endpoint is something LiteLLM *derives* from those kwargs plus the ambient
environment. A provider that sends no ``api_base`` at all looks identical, at the
kwargs seam, to one that sends the right one; the difference only becomes visible
one layer down, on the URL. So these tests go to the transport seam instead:
``httpx.AsyncClient.send`` is patched to raise a sentinel carrying
``str(request.url)``. That seam owns DNS resolution and connection, and it aborts
*before* either, so nothing leaves the process. Note the safety here is the
abort, not the hostnames: the steering cases use ``.invalid`` (RFC 6761
guarantees non-resolution) as a second layer, but the default-endpoint cases
necessarily name the providers' real hosts, since reproducing the real URL is the
whole assertion. Those requests are built and then dropped, never sent, and the
key is fake.

The bug being pinned: for the four providers that only forward ``api_base`` when
a ``base_url`` is configured (OpenAI, Anthropic, Google AI Studio, DeepSeek), the
endpoint was whatever LiteLLM's *own* resolution chain produced — an
open-ended list llmkit neither owned nor documented: ``litellm.api_base``
(a module global), a pluggable key-management backend behind ``get_secret``, the
per-provider environment aliases, and whatever LiteLLM adds next. The fix is for
those providers to resolve the endpoint themselves — config first, then the
provider's own documented environment aliases, then a library-owned default —
and send the result.

Two shipped subtleties these tests exist to hold in place. **Resolution happens
at the read point** (``completion_kwargs()``), not at construction: importing
LiteLLM is what runs ``load_dotenv()``, and ``import llmkit`` does not import
LiteLLM, so a provider built at startup would otherwise freeze a default and miss
the host's ``.env`` endpoint entirely. And **Google AI Studio owns no default**,
because its base carries a model-derived API version — see
``test_gemini_api_version_still_follows_the_model``.

**Be precise about what that closes, because these tests are the record of it.**
An explicit ``api_base`` outranks every other entry in the chains of the routes
this covers, so sending one makes the endpoint llmkit's decision. What that
*removes* is every source llmkit does not read — the module global, the
secret-manager backend, future aliases — for OpenAI, Anthropic and DeepSeek.
Google keeps that exposure in the one case where it sends nothing (no
``base_url``, no ``GEMINI_API_BASE``), and Ollama keeps it always, its chain
being inverted; both are documented carve-outs, not silent ones.
What it deliberately *keeps* is the six documented environment aliases: llmkit
now reads them itself, in LiteLLM's own measured precedence, so a host steering
its traffic with ``OPENAI_BASE_URL`` today is not silently redirected to the
public endpoint tomorrow. That is the same bargain ``resolve_api_key`` struck for
the credential (see ``opinions.md`` §3.9): the ambient fallback becomes explicit,
documented and tested rather than being deleted. So an ambient alias still
steers — by llmkit's own decision, observable in ``completion_kwargs()`` — and
the tests below assert exactly that, rather than pretending otherwise.

The literals below were measured on the wire against litellm 1.92.0 on
2026-07-21, not read out of LiteLLM's source: each provider's route derives its
URL differently. Anthropic appends its own ``/v1``, so a default pinned to
``https://api.anthropic.com/v1`` yields ``/v1/v1/messages``. Gemini's version
segment lives *in* the base and LiteLLM chooses it per model, which is why that
provider ends up with no constant at all. Both traps produce entirely
plausible-looking constants and are invisible to any test that exercises a single
model — which is exactly why these assertions are byte-for-byte rather than
reasoned about.

Every test clears all six endpoint environment variables before asserting:
importing ``litellm`` runs ``load_dotenv()``, so an ambient value can be injected
into the process, and a pass that depended on one would be worthless.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Final, cast
from unittest.mock import patch

import httpx
import litellm
import pytest

from llmkit import LLMClientConfig, Provider
from llmkit.providers import LLMProviderInterface, build_provider

#: Every endpoint environment variable any provider under test reads, cleared by
#: each test before it asserts. Order is irrelevant here — this is the *cleanup*
#: set; the precedence-carrying order lives in ``_ALIAS_PAIRS``.
_ALL_ENDPOINT_ENV_VARS: Final[tuple[str, ...]] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_BASE_URL",
    "GEMINI_API_BASE",
    "DEEPSEEK_API_BASE",
)

#: The measured base each provider's route resolves when nothing else names one.
#: For OpenAI, Anthropic and DeepSeek these are the values the providers'
#: ``_DEFAULT_BASE_URL`` constants must reproduce exactly. **Google's entry is
#: not a library constant** — that provider owns no default; it is recorded here
#: only as the base LiteLLM happens to derive for the *default* model, which is
#: what ``_MEASURED_DEFAULT_URL`` asserts against.
_MEASURED_DEFAULT_BASE: Final[dict[Provider, str]] = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.ANTHROPIC: "https://api.anthropic.com",
    Provider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
    Provider.DEEPSEEK: "https://api.deepseek.com/beta",
}

#: The full request URL each provider's route builds from that default — path
#: included, because the path is route-specific (``/chat/completions`` for the
#: OpenAI-shaped routes, ``/v1/messages`` for Anthropic, and Gemini's
#: ``:generateContent`` action on the default model).
_MEASURED_DEFAULT_URL: Final[dict[Provider, str]] = {
    Provider.OPENAI: "https://api.openai.com/v1/chat/completions",
    Provider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
    Provider.GOOGLE: (
        "https://generativelanguage.googleapis.com/v1beta"
        + "/models/gemini-2.5-flash-lite:generateContent"
    ),
    Provider.DEEPSEEK: "https://api.deepseek.com/beta/chat/completions",
}

#: (provider, endpoint env var) for every alias each provider's LiteLLM route
#: honours. Both OpenAI and Anthropic read two, in a measured precedence:
#: ``OPENAI_BASE_URL`` beats ``OPENAI_API_BASE``, and ``ANTHROPIC_API_BASE``
#: beats ``ANTHROPIC_BASE_URL``. Each alias gets its own case, because an
#: overlooked *secondary* alias re-points traffic just as completely as the
#: primary one — a fix that only covered ``OPENAI_BASE_URL`` would leave the
#: hole wide open.
_ALIAS_PAIRS: Final[tuple[tuple[Provider, str], ...]] = (
    (Provider.OPENAI, "OPENAI_BASE_URL"),
    (Provider.OPENAI, "OPENAI_API_BASE"),
    (Provider.ANTHROPIC, "ANTHROPIC_API_BASE"),
    (Provider.ANTHROPIC, "ANTHROPIC_BASE_URL"),
    (Provider.GOOGLE, "GEMINI_API_BASE"),
    (Provider.DEEPSEEK, "DEEPSEEK_API_BASE"),
)

#: A host that cannot resolve (RFC 6761 reserves ``.invalid``), standing in for
#: "some endpoint the caller never named".
_AMBIENT_BASE_URL: Final[str] = "https://ambient.invalid/v1"

#: The same, for the ``litellm.api_base`` module global — kept distinct from
#: ``_AMBIENT_BASE_URL`` so a failure message says which of the two sources won,
#: given that the library treats them oppositely (the global is inert; the
#: documented aliases are honoured).
_GLOBAL_BASE_URL: Final[str] = "https://global.invalid/v1"


class _CapturedRequest(Exception):
    """The transport seam's sentinel: carries the URL LiteLLM resolved.

    Raised *instead of* sending, so the request dies inside the process with the
    one fact these tests need. Deliberately an exception rather than a recorded
    side effect: aborting at ``send`` guarantees no socket is ever opened, where
    a "record and return a canned response" fake would still have to be trusted
    not to fall through on some untested route.
    """

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url: str = url


async def _abort_at_transport(
    _self: httpx.AsyncClient, request: httpx.Request, **_kwargs: object
) -> httpx.Response:
    """Stand in for ``httpx.AsyncClient.send``, capturing the URL and aborting.

    ``send`` is the chokepoint every LiteLLM route funnels through — the OpenAI
    SDK path (OpenAI/DeepSeek), LiteLLM's own async HTTP handler (Anthropic,
    Gemini) — and it sits *below* URL construction and *above* DNS and connect.
    That is what makes one patch enough to see every provider's resolved
    endpoint while guaranteeing the suite stays offline.
    """
    raise _CapturedRequest(str(request.url))


def _find_captured(exc: BaseException) -> _CapturedRequest | None:
    """Walk an exception's ``__cause__``/``__context__`` chain for the sentinel.

    LiteLLM catches transport failures and re-raises them as its own exception
    types, so the sentinel arrives buried rather than bare. The ``seen`` set
    guards against a cycle in the chain (an exception raised while handling
    itself, which would otherwise spin forever).
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _CapturedRequest):
            return current
        current = current.__cause__ or current.__context__
    return None


def capture_request_url(provider: LLMProviderInterface) -> str:
    """Return the URL LiteLLM resolves for one real call through ``provider``.

    Drives ``litellm.acompletion`` **directly** rather than
    ``llmkit._litellm.acompletion_text``: this file is about the endpoint LiteLLM
    derives, and going straight to it keeps llmkit's retry, rate-limit and
    logging layers out of the picture. That the call seam forwards
    ``completion_kwargs()`` verbatim is already pinned by ``test_vertex_routing.py``
    and ``test_openrouter_routing.py``, so nothing is lost by skipping it.

    ``num_retries=0`` matters: LiteLLM otherwise retries the aborted request,
    multiplying the work for an identical answer.

    Fails loudly when no request reached the seam. That is the whole safety
    property of this helper — a provider that raised during construction, or a
    route that resolved no URL at all, must never come back as a plausible
    string that a "not in" assertion would then happily pass.
    """

    # The same typed boundary ``llmkit._litellm`` puts over ``litellm.acompletion``:
    # LiteLLM's stubs leave several parameter slots ``Unknown``, so splatting the
    # provider's ``dict[str, object]`` credentials at the raw attribute floods the
    # file with ``reportArgumentType``. Narrowing it here keeps the credential
    # splat honest without a single ``Any``.
    acompletion = cast(
        "Callable[..., Coroutine[object, object, object]]",
        litellm.acompletion,
    )

    async def _drive() -> None:
        _ = await acompletion(
            model=provider.litellm_model(),
            messages=[{"role": "user", "content": "hi"}],
            num_retries=0,
            **provider.completion_kwargs(),
        )

    with patch.object(httpx.AsyncClient, "send", _abort_at_transport):
        try:
            asyncio.run(_drive())
        except Exception as exc:
            captured = _find_captured(exc)
            if captured is None:
                raise AssertionError(
                    "no request reached the transport seam, so no endpoint was "
                    + f"observed; the call failed first with {type(exc).__name__}: {exc}"
                ) from exc
            return captured.url
    raise AssertionError(
        "litellm.acompletion returned without sending a request, so no endpoint "
        + "was observed — the transport patch is not on the path this route takes."
    )


def _clear_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every endpoint variable, so no assertion rides on an ambient value."""
    for env_var in _ALL_ENDPOINT_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.mark.parametrize("provider", [Provider.OPENAI, Provider.ANTHROPIC, Provider.DEEPSEEK])
def test_litellm_global_api_base_no_longer_steers_the_request(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``litellm.api_base`` — a source llmkit never read — can no longer re-point us.

    The discriminating test of this file, and the sharpest statement of what
    always sending ``api_base`` actually buys. Pre-fix, with no ``base_url``
    configured, these providers sent no ``api_base`` at all and LiteLLM fell
    through to its own chain, where this in-process module global sits ahead of
    every environment name. Measured 2026-07-21: setting it re-pointed OpenAI,
    Anthropic and Google to ``global.invalid`` while the caller's pinned key rode
    along. Post-fix llmkit's explicit value is first in that chain, so the global
    is inert.

    This is the honest wire-level discriminator, *not* an ambient environment
    variable: the six documented aliases deliberately still steer (llmkit reads
    them itself now — see ``test_honoured_env_alias_steers_explicitly_through_the_library``).
    What the fix removes is every endpoint source llmkit does **not** read, of
    which this global is the one reachable offline. The others — a LiteLLM
    key-management backend serving these names through ``get_secret``, and any
    alias a future LiteLLM release adds — close by exactly the same mechanism and
    for exactly the same reason, which is why the fix is a closed one rather than
    a blocklist chasing an unversioned dependency.

    DeepSeek is a pin rather than a discriminator: its route resolves through its
    own transformation and was measured never to consult the global, so its case
    passes before and after. It is parametrized anyway so that a future LiteLLM
    that *does* route DeepSeek through the global finds this test already
    standing.

    **Google is deliberately absent**, and its absence is a documented
    limitation rather than an oversight. With nothing configured it sends no
    ``api_base`` at all — it owns no default endpoint, because the AI Studio base
    carries a model-derived API version (see
    ``test_gemini_api_version_still_follows_the_model``) — so for that one
    provider the global still wins. Naming a ``base_url`` or ``GEMINI_API_BASE``
    closes it; ``test_config_base_url_beats_ambient_env_on_the_wire`` covers the
    shape of that path.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setattr(litellm, "api_base", _GLOBAL_BASE_URL, raising=False)

    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))
    url = capture_request_url(built)

    assert "global.invalid" not in url, (
        f"litellm.api_base steered the request: {url} — an in-process global the "
        "library never reads won over the endpoint it resolved."
    )
    assert url.startswith(_MEASURED_DEFAULT_BASE[provider])


@pytest.mark.parametrize(("provider", "env_var"), _ALIAS_PAIRS)
def test_honoured_env_alias_steers_explicitly_through_the_library(
    provider: Provider, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A documented alias still reaches the wire — deliberately, and now by our hand.

    The migration promise, pinned where it can actually be checked. Taking
    ownership of ``api_base`` had one obvious failure mode: sending the library
    default unconditionally would silently redirect every host that steers its
    traffic with ``OPENAI_BASE_URL`` (or one of its five siblings) to the public
    endpoint — the sibling bug pointed at a different group of users. So the
    fallback is kept, and this test proves the request genuinely lands on the
    alias's host rather than merely appearing in a kwargs dict.

    The change here is *ownership*, not destination: the resolution happens in
    llmkit, in a measured precedence, tested and documented, and it is readable
    up front from ``completion_kwargs()`` instead of being inferred from a
    dependency's internals. Both facts are asserted together, because the whole
    point is that the value on the wire is the value the library says it picked.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv(env_var, _AMBIENT_BASE_URL)

    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))

    assert built.completion_kwargs()["api_base"] == _AMBIENT_BASE_URL
    assert capture_request_url(built).startswith(_AMBIENT_BASE_URL)


@pytest.mark.parametrize("provider", list(_MEASURED_DEFAULT_URL))
def test_default_endpoint_matches_the_measured_literal(
    provider: Provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unconfigured endpoint is byte-identical to the measured URL.

    This passes both before and after the fix, and that identity *is* the claim:
    making the four providers always send an ``api_base`` must not move a single
    byte of the wire shape they had when they sent none. Anyone rewriting a
    default has two ways to break it invisibly — pinning Anthropic to
    ``https://api.anthropic.com/v1`` (LiteLLM appends its own ``/v1``, giving
    ``/v1/v1/messages``) or dropping Gemini's ``/v1beta`` (the version segment
    lives in the base and is simply lost) — and both produce a perfectly
    plausible-looking constant. This is the only offline check that catches
    either, and the only guard against a default that goes stale when a provider
    moves its endpoint.

    So it is *not* redundant with the ambient test above, which asserts a prefix
    and would accept both mistakes. Please don't delete it as duplication.
    """
    _clear_endpoint_env(monkeypatch)

    built = build_provider(LLMClientConfig(provider=provider, api_key="sk-test"))

    assert capture_request_url(built) == _MEASURED_DEFAULT_URL[provider]


@pytest.mark.parametrize(
    ("model", "expected_version"),
    [
        ("gemini-2.5-flash-lite", "v1beta"),
        ("gemini-3-pro-preview", "v1alpha"),
    ],
)
def test_gemini_api_version_still_follows_the_model(
    model: str, expected_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Google AI Studio's API version is LiteLLM's to choose, and must stay that way.

    The trap that nearly shipped. Google AI Studio's base ends in an API
    *version*, and LiteLLM picks it from the model — ``v1alpha`` for Gemini 3 and
    newer, ``v1beta`` otherwise — applying it only when ``api_base`` is absent. A
    draft of this change gave the provider a static
    ``https://generativelanguage.googleapis.com/v1beta`` default like its three
    siblings, which measured correct against the default model and silently moved
    every Gemini 3 call from ``/v1alpha`` to ``/v1beta``: a wire-shape change
    doing exactly what owning the endpoint was meant to prevent, invisible to any
    test that only exercised one model.

    So the provider owns no default and omits ``api_base`` when nothing is
    configured. This test is the guard on that decision, and it is why the
    default is absent rather than merely undocumented — parametrize any new
    default here before adding one.
    """
    _clear_endpoint_env(monkeypatch)

    built = build_provider(
        LLMClientConfig(provider=Provider.GOOGLE, api_key="sk-test", model=model)
    )
    assert "api_base" not in built.completion_kwargs()

    url = capture_request_url(built)
    assert url == (
        f"https://generativelanguage.googleapis.com/{expected_version}"
        f"/models/{model}:generateContent"
    )


def test_config_base_url_beats_ambient_env_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly configured ``base_url`` wins over an ambient variable.

    Pins the property that makes the fix *complete* rather than partial: an
    explicit ``api_base`` is first in every LiteLLM resolution chain, so
    resolving config-first in the library and forwarding the result is enough —
    no extra step is needed to neutralize the environment. It already holds
    today (a configured ``base_url`` is the one case where these providers do
    forward ``api_base``), which is precisely why it is worth pinning: the fix
    reroutes the *unconfigured* path through the same kwarg, and must not
    disturb the configured one on its way past.

    OpenAI alone is enough — the resolution order being pinned is LiteLLM's,
    shared by every route, and the per-route URL shapes are covered above.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", _AMBIENT_BASE_URL)

    built = build_provider(
        LLMClientConfig(
            provider=Provider.OPENAI, api_key="sk-test", base_url="https://pinned.invalid/v1"
        )
    )
    url = capture_request_url(built)

    assert url.startswith("https://pinned.invalid/v1")
    assert "ambient.invalid" not in url
