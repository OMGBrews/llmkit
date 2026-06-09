"""The three-layer per-call option merge (config < options < explicit keyword).

A feature module builds one :class:`LLMCallOptions` and passes it as
``options=`` to every call function (:func:`~llmkit.structured_llm_call`,
:func:`~llmkit.text_llm_call`, :func:`~llmkit.stream_text_with_log`, and the
sync wrappers) instead of repeating the same per-call keyword block at each
site. This module owns only the *merge*: collapsing the three precedence
layers — the configured client, a shared ``LLMCallOptions``, and the explicit
per-call keywords — into the single :class:`_ResolvedCallArgs` shape every call
function forwards to the transport. It carries no logging or transport concern;
those live in :mod:`llmkit.capture` and :mod:`llmkit.structured_output`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llmkit.providers import LLMProviderInterface
from llmkit.retry import DEFAULT_RETRY_POLICY, RetryPolicy


class _Unset(Enum):
    """Sentinel type for an unset :class:`LLMCallOptions` field.

    A single-member ``Enum`` rather than a bare ``object()`` so the type
    checker can narrow ``field is _UNSET`` precisely: an option left unset
    is typed ``Literal[_Unset.UNSET]`` and a set one is its real type, so
    the merge helper resolves to the right branch with no ``cast``.
    """

    UNSET = "unset"


#: The "this option was not provided" sentinel. An unset option field defers
#: to the per-call kwarg (and, through it, to the configured client) — it
#: never clobbers a config value with a default.
_UNSET = _Unset.UNSET


@dataclass(frozen=True)
class LLMCallOptions:
    """A reusable bundle of the per-call keyword arguments.

    Opt-in ergonomics for the call functions
    (:func:`structured_llm_call`, :func:`structured_llm_call_sync`,
    :func:`text_llm_call`, :func:`text_llm_call_sync`, and
    :func:`stream_text_with_log`): a feature module builds one
    ``LLMCallOptions`` once and passes it as ``options=`` to every call,
    instead of repeating the same nine-keyword block at each site. The flat
    keyword path is untouched — pass no ``options`` and nothing changes.

    Every field is **optional and unset by default** (the private
    :data:`_UNSET` sentinel, not ``None``): an unset field defers to the
    per-call keyword, which in turn defers to the configured
    :class:`~llmkit.LLMClientConfig` for the dual-homed ``model`` /
    ``reasoning_effort``. This is what makes "unset option does not clobber
    config" work — only a field you *set* on the options participates in the
    merge. Precedence, lowest to highest: **config < options < explicit
    per-call keyword.**

    ``feature`` is deliberately **not** part of this bundle: it stays a
    required per-call keyword as a telemetry forcing function (it scopes the
    per-call log filenames and ``index.jsonl`` grouping operators grep), so
    it cannot be defaulted-away into a shared object and forgotten.

    Attributes:
        temperature: Sampling temperature. Unset defers to the per-call
            ``temperature`` keyword (which defaults to ``0.2``).
        model: Model override. Unset defers to the per-call ``model``
            keyword, then to the provider default.
        max_tokens: Completion-length cap. Unset defers to the per-call
            ``max_tokens`` keyword (uncapped when also unset).
        reasoning_effort: Provider reasoning/thinking effort. Unset defers
            to the per-call keyword, then to the configured client value.
        retry: Transient-error retry budget. Unset defers to the per-call
            ``retry`` keyword (:data:`~llmkit.DEFAULT_RETRY_POLICY`).
        provider: Per-call provider override. Unset defers to the per-call
            ``provider`` keyword (the globally-configured provider).
    """

    temperature: float | _Unset = _UNSET
    model: str | None | _Unset = _UNSET
    max_tokens: int | None | _Unset = _UNSET
    reasoning_effort: str | None | _Unset = _UNSET
    retry: RetryPolicy | _Unset = _UNSET
    provider: LLMProviderInterface | None | _Unset = _UNSET


@dataclass(frozen=True, slots=True)
class _ResolvedCallArgs:
    """The per-call arguments after merging ``options`` with the keywords.

    The single shape every call function forwards to the transport once the
    three-layer precedence (config < options < explicit keyword) has been
    collapsed. Internal to the call-function module.
    """

    temperature: float
    model: str | None
    max_tokens: int | None
    reasoning_effort: str | None
    retry: RetryPolicy
    provider: LLMProviderInterface | None


def resolve_call_args(
    options: LLMCallOptions | None,
    *,
    temperature: float,
    model: str | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
    retry: RetryPolicy,
    provider: LLMProviderInterface | None,
) -> _ResolvedCallArgs:
    """Merge ``options`` under the explicit per-call keywords.

    The shared seam all the call functions funnel through, so the
    precedence is defined in exactly one place. The keyword arguments are
    the values the call function received with its *own* defaults already
    applied; for each field the rule is:

    * an explicit keyword that differs from the call function's default
      wins outright (highest precedence);
    * otherwise a field that is *set* on ``options`` is used;
    * otherwise the call function's default stands.

    A keyword left at its default (``model=None``, ``temperature=0.2``,
    ``retry=DEFAULT_RETRY_POLICY``, …) is indistinguishable from "not
    passed", so it yields to ``options`` — which is exactly what lets a
    shared ``LLMCallOptions`` supply values while an explicit keyword still
    overrides it. When ``options is None`` the keywords pass straight
    through, so the flat-kwarg path is byte-for-byte unchanged.
    """
    if options is None:
        return _ResolvedCallArgs(
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            retry=retry,
            provider=provider,
        )
    return _ResolvedCallArgs(
        temperature=_pick(temperature, DEFAULT_TEMPERATURE, options.temperature),
        model=_pick(model, None, options.model),
        max_tokens=_pick(max_tokens, None, options.max_tokens),
        reasoning_effort=_pick(reasoning_effort, None, options.reasoning_effort),
        retry=_pick(retry, DEFAULT_RETRY_POLICY, options.retry),
        provider=_pick(provider, None, options.provider),
    )


def _pick[V](keyword: V, default: V, option: V | _Unset) -> V:
    """Resolve one field: explicit keyword > set option > keyword default.

    A ``keyword`` that differs from this call function's ``default`` was
    passed explicitly and wins. Otherwise a ``option`` that is set (not
    :data:`_UNSET`) is used; an unset option falls through to the
    (default) keyword, so it never clobbers a config-backed value.
    """
    if keyword != default:
        return keyword
    if option is _UNSET:
        return keyword
    return option


#: The shared default sampling temperature for the call functions — the
#: value a per-call ``temperature`` keyword carries when the caller does not
#: override it. Named so :func:`resolve_call_args` and every call signature
#: stay in agreement on what "unset temperature" means.
DEFAULT_TEMPERATURE = 0.2
