"""The three-layer per-call option merge (config < options < explicit keyword).

A feature module builds one :class:`LLMCallOptions` and passes it as
``options=`` to every call function (:func:`~llmkit.structured_llm_call`,
:func:`~llmkit.text_llm_call`, :func:`~llmkit.text_llm_call_stream`, and the
sync wrappers) instead of repeating the same per-call keyword block at each
site. This module owns only the *merge*: collapsing the three precedence
layers — the configured client, a shared ``LLMCallOptions``, and the explicit
per-call keywords — into the single :class:`_ResolvedCallArgs` shape every call
function forwards to the transport. It carries no logging or transport concern;
those live in :mod:`llmkit.capture` and :mod:`llmkit.structured_output`.

"Explicitly passed" is a structural fact here, not a guess: every mergeable
call-function keyword defaults to :data:`UNSET`, so a keyword that arrives as
anything else — including a value equal to the old default, and including
``None`` — was passed by the caller and wins outright. The merge never
compares values to infer intent.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import override

from llmkit._types import ReasoningEffort
from llmkit.providers import LLMProviderInterface
from llmkit.retry import DEFAULT_RETRY_POLICY, RetryPolicy


class Unset(Enum):
    """The type of the :data:`UNSET` sentinel — "this argument was not passed".

    A single-member ``Enum`` rather than a bare ``object()`` so the type
    checker can narrow ``x is not UNSET`` precisely: an unset value is typed
    ``Literal[Unset.UNSET]`` and a passed one is its real type, so the merge
    helper resolves to the right branch with no ``cast``. Public on purpose:
    it appears in the call functions' and :class:`LLMCallOptions`' signatures,
    and a caller's own typed wrapper can declare ``temperature: float | None
    | Unset = UNSET`` and forward unconditionally. Test with ``is``/``is not``
    only —
    ``UNSET`` is truthy, and truthiness on e.g. ``float | Unset`` is a bug
    either way (``0.0`` is falsy too).
    """

    UNSET = "unset"

    @override
    def __repr__(self) -> str:
        # ``help()``, ``inspect.signature``, and dataclass reprs all render
        # the default via ``repr``; "UNSET" reads as the sentinel it is,
        # where the generated "<Unset.UNSET: 'unset'>" reads as leaked
        # machinery.
        return "UNSET"


#: The "this argument was not passed" sentinel. An unset call-function keyword
#: defers to :class:`LLMCallOptions`, then to the true default (and, for the
#: config-backed fields, through it to the configured client). An unset
#: options field defers to the same chain minus itself — it never clobbers a
#: config value with a default.
UNSET = Unset.UNSET


@dataclass(frozen=True, repr=False)
class LLMCallOptions:
    """A reusable bundle of the per-call keyword arguments.

    Opt-in ergonomics for the call functions
    (:func:`structured_llm_call`, :func:`structured_llm_call_sync`,
    :func:`text_llm_call`, :func:`text_llm_call_sync`, and
    :func:`text_llm_call_stream`): a feature module builds one
    ``LLMCallOptions`` once and passes it as ``options=`` to every call,
    instead of repeating the same nine-keyword block at each site. The flat
    keyword path is untouched — pass no ``options`` and nothing changes.

    Every field is **optional and unset by default** (the :data:`UNSET`
    sentinel, not ``None``): an unset field defers to the per-call keyword,
    which in turn defers to the configured :class:`~llmkit.LLMClientConfig`
    for the dual-homed ``model`` / ``reasoning_effort``. This is what makes
    "unset option does not clobber config" work — only a field you *set* on
    the options participates in the merge. Precedence, lowest to highest:
    **config < options < explicit per-call keyword** — exactly, because the
    call functions' keywords also default to :data:`UNSET`, any keyword you
    pass (including ``None``, including a value equal to the documented
    default) overrides the matching options field. ``dataclasses.replace(
    opts, model=UNSET)`` legally *unsets* a field again; that is contract,
    not accident.

    ``feature`` is deliberately **not** part of this bundle: it stays a
    required per-call keyword as a telemetry forcing function (it scopes the
    per-call log filenames and ``index.jsonl`` grouping operators grep), so
    it cannot be defaulted-away into a shared object and forgotten.

    Attributes:
        temperature: Sampling temperature. Unset defers to the per-call
            ``temperature`` keyword (:data:`DEFAULT_TEMPERATURE`, ``0.2``,
            when that is also unset); an explicit ``None`` forwards no
            ``temperature`` kwarg at all, leaving the provider's default
            sampling in effect.
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

    temperature: float | None | Unset = UNSET
    model: str | None | Unset = UNSET
    max_tokens: int | None | Unset = UNSET
    reasoning_effort: ReasoningEffort | None | Unset = UNSET
    retry: RetryPolicy | Unset = UNSET
    provider: LLMProviderInterface | None | Unset = UNSET

    @override
    def __repr__(self) -> str:
        # Only the fields actually set: ``LLMCallOptions(temperature=0.9)``,
        # not six ``=UNSET`` entries drowning the one value a user debugging
        # a merge is looking for. (Generated repr suppressed via
        # ``repr=False``; ``field(repr=False)`` would wrongly hide set
        # fields too.)
        set_fields = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) is not UNSET
        )
        return f"{type(self).__name__}({set_fields})"


@dataclass(frozen=True, slots=True)
class _ResolvedCallArgs:
    """The per-call arguments after merging ``options`` with the keywords.

    The single shape every call function forwards to the transport once the
    three-layer precedence (config < options < explicit keyword) has been
    collapsed. Internal to the call-function module.
    """

    temperature: float | None
    model: str | None
    max_tokens: int | None
    reasoning_effort: ReasoningEffort | None
    retry: RetryPolicy
    provider: LLMProviderInterface | None


def resolve_call_args(
    options: LLMCallOptions | None,
    *,
    temperature: float | None | Unset,
    model: str | None | Unset,
    max_tokens: int | None | Unset,
    reasoning_effort: ReasoningEffort | None | Unset,
    retry: RetryPolicy | Unset,
    provider: LLMProviderInterface | None | Unset,
) -> _ResolvedCallArgs:
    """Merge ``options`` under the explicit per-call keywords.

    The shared seam all the call functions funnel through, so the
    precedence is defined in exactly one place. The keyword arguments arrive
    exactly as the call function received them — :data:`UNSET` when the
    caller didn't pass them — and for each field the rule is:

    * a keyword that was passed (is not :data:`UNSET`) wins outright,
      whatever its value (highest precedence);
    * otherwise a field that is *set* on ``options`` is used;
    * otherwise the true default stands (:data:`DEFAULT_TEMPERATURE`,
      :data:`~llmkit.DEFAULT_RETRY_POLICY`, ``None`` for the rest — the
      ``None``s the transport then resolves against the configured client).

    For ``temperature`` specifically, an explicit ``None`` is a real,
    resolved value: it forwards **no** ``temperature`` kwarg to the provider,
    so the provider's default sampling applies — the requested escape hatch
    from :data:`DEFAULT_TEMPERATURE` (``0.2``), which still stands whenever
    neither the keyword nor ``options`` supplies a value.

    ``options=None`` is simply an all-unset options, so the flat-kwarg path
    resolves to the same values as before options existed. The true defaults
    are applied here and only here — the call signatures carry :data:`UNSET`,
    not copies of the defaults, so the two can never drift.
    """
    opts = options if options is not None else _ALL_UNSET_OPTIONS
    return _ResolvedCallArgs(
        temperature=_pick(temperature, opts.temperature, DEFAULT_TEMPERATURE),
        model=_pick(model, opts.model, None),
        max_tokens=_pick(max_tokens, opts.max_tokens, None),
        reasoning_effort=_pick(reasoning_effort, opts.reasoning_effort, None),
        retry=_pick(retry, opts.retry, DEFAULT_RETRY_POLICY),
        provider=_pick(provider, opts.provider, None),
    )


def _pick[V](keyword: V | Unset, option: V | Unset, default: V) -> V:
    """Resolve one field: passed keyword > set option > true default.

    Structural identity checks only — no value equality, so a keyword whose
    value happens to equal the default is still recognized as passed, and a
    ``RetryPolicy`` equal to the default still wins.
    """
    if keyword is not UNSET:
        return keyword
    if option is not UNSET:
        return option
    return default


#: The shared default sampling temperature for the call functions — the value
#: a call resolves to when neither the ``temperature`` keyword nor ``options``
#: supplies one. Public so callers can reference the real default; applied in
#: :func:`resolve_call_args` (the signatures carry :data:`UNSET` instead), so
#: it has exactly one definition site.
DEFAULT_TEMPERATURE = 0.2

# Shared all-unset instance for the ``options=None`` path, so resolving the
# flat-keyword call doesn't allocate a throwaway options object per call.
_ALL_UNSET_OPTIONS = LLMCallOptions()
