# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **A config knob the selected provider does not read is now rejected, not
  silently ignored.** `LLMClientConfig` / `make_provider` accepted every
  provider-shaped field for every provider and each provider used only the ones
  it needed, dropping the rest — so a config populated generically (all fields
  filled from a settings object) *looked* like it pinned a credential or
  endpoint that was in fact ignored. `build_provider` — the single seam
  `make_provider`, `configure_llm_client`, and a direct `build_provider(config)`
  all flow through — now raises a `ValueError` naming the offending field(s) and
  what the provider does read. Each provider declares its accepted knobs
  (`_accepted_config_fields`), enforced at import. **Migration:** populate only
  the active provider's fields — an `api_key` on Bedrock/Vertex, a `base_url` on
  a fixed-endpoint provider, or a non-default `gemini_structured_output` on a
  non-Gemini provider now raises instead of being dropped. (The config docstring
  always said only the active provider's fields need be populated; this enforces
  it.)
- **Bearer providers resolve their API key explicitly: config, then the
  provider's environment variable, else raise.** The five key-authenticated
  providers (OpenRouter, Anthropic, OpenAI, DeepSeek, Google AI Studio) coerced
  a missing `api_key` to `""` and handed it to LiteLLM, which *silently* fell
  back to the ambient provider env key — an invisible identity/billing swap
  (reachable even from a stray `.env`, since importing LiteLLM runs
  `load_dotenv()`). Resolution is now explicit and documented: the configured
  key wins; else the provider's own variable (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`);
  else a `ValueError` at construction naming the variable. **Migration:**
  env-var-based setups keep working unchanged; a provider built with neither a
  configured key nor its env var now fails loudly at construction instead of at
  call time (or silently succeeding off an ambient key).
- **`$ref` structural siblings and empty `properties` now raise in
  `model_from_json_schema`.** A structural keyword beside a `$ref` (`type`,
  `enum`, `items`, `properties`, …) that differs from the referenced schema was
  silently dropped on the annotation path — most visibly a `$ref`-sibling
  `enum`, which widened the field to an unconstrained scalar. It now raises
  unless it restates the target's value (metadata and bounds still merge,
  outer-wins). An explicit empty `{"properties": {}}` now raises like an absent
  `properties` (both would build a zero-field model that rejects every
  response), unless `additionalProperties: true` is set.

- **`stream_text_with_log` is renamed to `text_llm_call_stream`.** The streaming
  call function now matches the `<shape>_llm_call[_sync|_stream]` grammar the
  other call functions follow (so `grep -r llm_call` finds the whole call
  surface); its old `with_log` suffix was noise (every call function logs). The
  old name stays as a deprecated alias through the pre-1.0 window (see
  **Deprecated**). **Migration:** rename `stream_text_with_log(...)` →
  `text_llm_call_stream(...)` — identical signature and behaviour.

- **The Python floor is lowered to 3.12, and the tested-platform surface is
  widened.** `requires-python` is now `>=3.12` (was `>=3.13`): every idiom the
  library uses — PEP 695 generics, `StrEnum`, `match`/`assert_never` — is fully
  available on 3.12, so the floor now sits at the lowest version that supports
  them, verified by CI. CI runs a matrix (3.12/3.13/3.14 on Linux plus one macOS
  and one Windows cell) so the `OS Independent` classifier is actually exercised,
  not just claimed. **Migration:** none — a strict superset of previously
  supported environments.

- **The public `dev` extra is removed; dev tooling is now a PEP 735 dependency
  group.** The maintainer toolchain (pytest, ruff, basedpyright) no longer ships
  as a public `omg-llmkit[dev]` extra — it moved to a `[dependency-groups]` entry
  that never lands in the wheel/sdist metadata, taking it off the published
  install surface. `uv sync` installs it by default. **Migration:** `pip install
  'omg-llmkit[dev]'` no longer resolves the tooling (it warns and installs core
  only); contributors use `uv sync`.

- **AIMD saturation is judged on the provider-wide in-flight aggregate, not one
  gate's local count.** The adaptive rate limiter decides whether a 429/503/529
  should halve a provider's concurrency limit by asking "were we at the shared
  limit?" — now measured across *every* population (each per-loop async gate plus
  the sync gate) rather than the single gate the throttled call sat on. A
  self-inflicted throttle in the multi-population regime (e.g. a host's own loop
  and llmkit's persistent sync loop together over the limit) now triggers a
  decrease it previously missed. Single-population workloads are byte-identical
  and admission caps are unchanged (still per-population); multi-population
  workloads may see legitimate `"throttle"` backpressure events they never got
  before, still bounded to one halving per cooldown.

### Deprecated

- **`stream_text_with_log` is a deprecated alias for `text_llm_call_stream`,
  removed in 1.0.** Calling it warns `DeprecationWarning` (eagerly, at call time)
  and otherwise behaves identically — same signature, same streamed chunks.
  Switch the call; the alias exists only to spare a hard break mid-cycle.

### Security

- **`LLMClientConfig` no longer leaks `api_key` in its `repr`.** The frozen
  dataclass's generated repr rendered `api_key='sk-...'` verbatim, so any
  host-side `print(config)`, log line, or exception reporter's locals capture
  could exfiltrate the credential. A set key now renders as `api_key=<redacted>`
  (presence stays debuggable; the value never prints), and the repr shows only
  fields that differ from their default. **Compatibility note:** code parsing
  the repr string will see the new format.

### Fixed

- **A `$def`'s `description` now survives a nullable-wrapped or chained `$ref`.**
  `model_from_json_schema` rescued a `$ref` target's model-facing `description`
  only for a bare top-level `$ref`, so wrapping it in `anyOf`+`null` (a nullable
  field) dropped the guidance instructor sends to the model. Unifying `$ref` /
  nullable resolution into one pass carries the description through for bare,
  nullable-wrapped, and multi-hop `$ref` fields alike (nearest hop wins).

- **Explicit per-call keywords now beat `options` even at default-equal
  values.** The documented precedence (**config < options < explicit
  keyword**) detected an "explicit" keyword by comparing its value against the
  signature default, so `structured_llm_call(..., temperature=0.2,
  options=LLMCallOptions(temperature=0.9))` silently ran at `0.9`, and an
  explicit `model=None` / `retry=DEFAULT_RETRY_POLICY` could never override an
  options field. The mergeable keywords (`temperature`, `model`, `max_tokens`,
  `reasoning_effort`, `retry`, `provider`) on all five call functions now
  default to the **`UNSET`** sentinel — "was it passed" is structural, and any
  passed value (including `None`) wins, exactly as the README promises. True
  defaults are applied in one place (`resolve_call_args`); behavior without
  `options`, or with disjoint keyword/options fields, is unchanged.
  **Compatibility note:** callers passing a keyword whose value equals the old
  default *and* setting the same field on `options` previously got the options
  value; they now get their explicit keyword. Code introspecting call-function
  signatures sees `float | Unset = UNSET`-style defaults.

- **A `run_sync` caller racing `shutdown()` no longer hangs or raises.** When a
  sync call obtained the persistent event loop just as `shutdown()` tore it down,
  the submit could land on a stopped loop (the caller blocked for the full
  timeout — 600 s by default — with the coroutine leaked) or a closed one (a
  spurious `RuntimeError`). Obtaining the loop and submitting to it are now atomic
  against `shutdown()`, so a racing caller transparently retries onto a lazily
  restarted loop; a call genuinely in flight at shutdown still gets a prompt
  `asyncio.CancelledError`, no longer maskable by a secondary `RuntimeError` from
  the cancel path.

### Added

- **`llmkit.Unset` / `llmkit.UNSET` / `llmkit.DEFAULT_TEMPERATURE` are
  public.** The not-passed sentinel that already appeared in
  `LLMCallOptions`' annotations is now a deliberate part of the typed surface
  (the openai-python `NotGiven`/`NOT_GIVEN` pattern) instead of a leaked
  private type: a caller's typed wrapper can declare `temperature: float |
  Unset = UNSET` and forward unconditionally. `repr(UNSET)` reads `UNSET`,
  and `LLMCallOptions`' repr now prints only the fields actually set —
  `LLMCallOptions(temperature=0.9)`, not six `=UNSET` entries.

- **`llmkit.Message` and `llmkit.ReasoningEffort` are public, and a prompt's
  message `content` may now be multimodal.** The call functions' `prompt`
  parameter was typed `str | list[dict[str, str]]` — the one place the
  transport's raw wire format leaked through an otherwise fully-typed surface,
  and wrong in both directions: a key typo (`{"roel": ...}`) type-checked, and
  the multimodal content-parts form LiteLLM accepts (a `content` that is a list
  of part dicts) was *rejected*. Prompts are now `str | list[Message]`, where
  `Message` is a `TypedDict` (`role: Literal["system", "user", "assistant"]`,
  `content: str | list[dict[str, object]]`): unknown keys and mistyped roles
  are type errors, and multimodal content type-checks. `reasoning_effort` is
  now the exported `ReasoningEffort` alias (`Literal["disable", "low",
  "medium", "high"] | str`) everywhere it was a bare `str`. Both types are
  exported so a caller can annotate their own prompt builders and effort
  constants. **`ReasoningEffort` is advisory, not enforcing:** under the type
  checker `Literal[...] | str` widens to `str`, so it documents the canonical
  set and drives editor autocomplete but does not statically reject a typo —
  the open `| str` arm is deliberate, since llmkit forwards the value verbatim
  to LiteLLM and providers accept values outside the set (e.g. OpenAI's
  `"minimal"`). Runtime is unchanged (a `TypedDict` is a plain dict; the alias
  is a `str`). **Compatibility note:** a caller passing a variable annotated
  `list[dict[str, str]]` as `prompt` now gets a type error — annotate it
  `list[Message]` (inline message-dict literals keep type-checking unchanged).

- **`except LLM_RECOVERABLE_ERRORS:` now actually catches a litellm-native
  503.** The litellm 503 entry in `LLM_TRANSPORT_ERRORS` matched only via a
  lazy `isinstance` hook (the metaclass stand-in that keeps `import llmkit`
  litellm-free), and Python's `except` matching bypasses such hooks — so the
  retry layer classified and retried genuine 503s correctly, but once the
  budget was exhausted the raw litellm class blew straight through the
  *documented* host degrade pattern with no warning. llmkit's transport
  boundary now re-raises every litellm-native 503 as a new llmkit-owned
  **`ServiceUnavailableError`** — a plain, statically-listed member of
  `LLM_TRANSPORT_ERRORS` (the `CircuitOpenError` / `OutputLimitError`
  precedent) — carrying `provider`, `model`, `status_code` (always 503), and
  the original `response`, with the litellm original on `__cause__`. Behaviour
  preserved by measurement: a server `Retry-After` stays readable through the
  copied `response`, and `status_code == 503` keeps the AIMD/breaker throttle
  classification intact. **Compatibility note:** code that caught
  `litellm.exceptions.ServiceUnavailableError` directly around llmkit calls
  must switch to `llmkit.ServiceUnavailableError` (or the documented
  catch-set); the raw litellm class still classifies as transport in
  `isinstance` checks, so a host's own litellm call wrapped in `with_retries`
  retries its 503s exactly as before.

### Added

- **`llmkit.__version__`.** The package now exposes its version, resolved from
  installed distribution metadata (`omg-llmkit`) so it can never drift from
  what pip installed. A source tree without dist metadata reports
  `"0.0.0+unknown"` rather than raising at import.
- **A host can pick Gemini's structured-output strategy.**
  `LLMClientConfig.gemini_structured_output` (`"schema"` | `"json"`, default
  `"schema"`) selects the instructor `Mode` for the two Gemini providers (Vertex,
  Google AI Studio) and is ignored by every other provider. `"schema"` preserves
  today's behavior exactly — Gemini's native JSON-schema constrained decoding
  (`Mode.JSON_SCHEMA`), with server-side schema enforcement. `"json"` switches to
  `Mode.JSON`: JSON-mime-type output with the schema moved into the system prompt
  and validated client-side. It exists because Gemini's constrained-decoding path
  is a measured **repetition-loop trap** — a token mask that, once the model
  starts looping, blocks exactly the tokens that would break the pattern, so the
  call spins until `max_tokens` kills it (PIA Maker measured 67-83% first-attempt
  runaway under `"schema"` vs 0% under `"json"` on real prompts, 2026-07-15).
  `"json"` trades server-side schema enforcement for an escape from that trap, at
  the cost of an occasional repair re-ask. Hosts that hit the loop no longer need
  to subclass a provider to override the private `_mode` ClassVar; the default
  changes nothing for anyone else. `make_provider(...)` accepts the same
  `gemini_structured_output` knob for the per-call `provider=` seam. An
  unrecognized value fails loudly at provider construction rather than silently
  falling back.

### Fixed

- **Structured-path transport errors are no longer double-sent inside the
  rate-limiter slot.** instructor's in-call re-ask (the loop llmkit feeds its
  `max_retries`) previously retried *every* failure except length truncation —
  so a 429/5xx/network or a permanent 401/400/403 was re-sent immediately, with
  no backoff and ignoring `Retry-After`, inside the single per-provider
  concurrency slot: one request became two, invisible to the outer retry layer.
  The in-call re-ask is now restricted to genuine parse failures (malformed
  JSON / `ValidationError`), matching instructor's own default. A transient
  error now makes **one** request per outer attempt (not two) and is retried by
  the outer layer with the full transport budget and `Retry-After` honored; a
  permanent error fails fast with no in-call duplicate. Every genuine parse
  failure still gets its one repair re-ask and is retried on the (lower)
  validation budget — this now correctly includes instructor's own
  `ResponseParsingError` (e.g. a blocked Gemini `Mode.JSON` response) and
  `AsyncValidationError` (a failing async field validator), which previously
  failed fast; length truncation still fails fast as `OutputLimitError`.
- **OpenRouter's default model works again.** `OpenRouterProvider._default_model`
  was `google/gemini-2.0-flash-001`, which OpenRouter has retired — the slug is
  gone from its catalog entirely, so *every* call that relied on the default
  (i.e. constructed the provider, or set `LLMClientConfig.model = None`, without
  naming a model) failed with `NotFoundError: "No endpoints found for
  google/gemini-2.0-flash-001"`. The default is now
  `google/gemini-2.5-flash-lite`: the same family as the retired id, the same
  model the `GOOGLE` and `VERTEX` providers already default to, and measured
  live at 10/10 valid strict-`json_schema` structured round-trips through the
  public surface (2026-07-14). Callers who pass an explicit `model` were never
  affected.

### Added

- **New logging surface, all additive:** `default_log_dir()` (top-level
  export) and the `LLMKIT_LOG_DIR` env override;
  `LocalYamlLogSink(retention_days=..., max_index_bytes=...)`; and three
  defaulted `LLMCallRecord` fields — `call_id`, `attempt`, `queue_wait_ms` —
  so every existing direct constructor and custom sink keeps working
  unchanged.

### Changed

- **⚠️ Log retention is on by default: the sink now deletes old logs.** The
  default `LocalYamlLogSink` prunes per-call YAML files older than **30 days**
  and rotates `index.jsonl` past **50 MiB** to a date-stamped generation that
  ages out under the same policy (housekeeping is hourly-throttled, on a
  worker thread, and rotation is an atomic rename that loses no lines). The
  first successful write announces the directory *and* the policy at INFO —
  before anything is ever deleted. Opt out with
  `LocalYamlLogSink(retention_days=None)` (and/or `max_index_bytes=None`) if
  your logs are an archive rather than a debugging aid.
- **⚠️ The default log directory moved for processes not launched from a
  project root.** It is now resolved lazily at first write — `LLMKIT_LOG_DIR`,
  else `data/llm-logs/` under the nearest ancestor with a
  `pyproject.toml`/`.git` (nearest wins; seeded with a `.gitignore` when the
  sink creates it), else a per-user state dir
  (`$XDG_STATE_HOME/llmkit/llm-logs`, `~/Library/Logs/llmkit`,
  `%LOCALAPPDATA%\llmkit\logs`) — and frozen, so a mid-run `chdir` can't split
  one process's logs. Launches from a repo root keep byte-identical paths;
  launches from a repo *subdirectory* consolidate to the repo root (previously
  they sprayed `subdir/data/llm-logs`); only processes outside any project
  move to the private state dir, announced by the first-write INFO. The
  `DEFAULT_LOG_DIR` constant is **removed** — use `default_log_dir()` or pass
  an explicit `log_dir`.
- **⚠️ Log files are private by default on POSIX.** A sink-created directory
  is `0o700` and log files `0o600`. Multi-reader deployments should pre-create
  the log directory with their desired mode — the sink never re-chmods a
  directory it didn't create.
- **The per-call YAML, `index.jsonl`, and header line 2 carry three new
  fields** (`call_id`, `attempt`, `queue_wait_ms`; header line 2 gains a
  `call=<id[:8]> attempt=<n>` suffix). Header line 1 — the `head -1` triage
  shape — is byte-identical. A strict index parser must accept the three new
  keys.
- **A persistently broken sink warns once, not once per call.** The first
  failure (and any *new* failure signature — different exception type or
  errno) logs a WARNING with traceback; identical repeats drop to DEBUG. The
  latch re-arms on success and on `configure_llm_logging`.

### Fixed

- **Sink I/O no longer blocks the event loop.** Every log write — `mkdir`,
  the full-payload `yaml.dump`, both file writes, retention housekeeping —
  runs via `asyncio.to_thread`, for the structured, buffered-text, and
  cleanly-finished stream paths (a stream abandoned mid-flight deliberately
  writes synchronously so its truncation-witness record can't be lost while
  the generator unwinds; a write that can't be offloaded — executor already
  shut down at teardown — degrades to blocking rather than lost). The
  documented capture contracts are unchanged: the record and written path are
  captured before the call returns, and both sync-bridge teardown paths drain
  the executor so a clean exit never abandons a final write.
- **Log records from retries of one call are now joinable.** Each logical
  call mints one `call_id` (uuid4 hex) and numbers attempts 1-based across
  all three call surfaces and the sync bridge — no more joining retry records
  by feature + timestamp proximity, which broke under concurrent same-feature
  fan-out.
- **`duration_ms` no longer silently conflates limiter queue time with
  provider latency.** New `queue_wait_ms` records time queued behind llmkit's
  own rate limiter (`0.0` when disabled, `None` when the attempt failed
  before acquiring); `duration_ms` keeps its meaning, so provider latency ≈
  `duration_ms - queue_wait_ms`.

- **The live smoke suite now exercises each OpenRouter path that can rot
  independently.** It previously always overrode the model, which is why a dead
  default shipped through a green release gate. There are now two OpenRouter
  live tests: one drives the provider's *own default* with no `model=` and no
  env override (so a retired default fails the gate instead of shipping), and
  the existing one keeps pinning the strict-`json_schema` wire shape against
  `mistralai/mistral-nemo`, the model measured to echo the schema back when the
  `response_format` is sent non-strict.

## [0.7.0] — 2026-07-14

Structured output works again for **Anthropic, Bedrock, and OpenRouter** on a
fresh install. instructor 1.15.3 removed `Mode.ANTHROPIC_JSON` and
`Mode.OPENROUTER_STRUCTURED_OUTPUTS` from the mode registry that
`from_litellm` validates at client construction, so under the unbounded
`instructor>=1.15.1` floor every structured call through those three
providers raised `RegistryError` before any request was sent. Every mode
repin below was verified by a live structured round-trip against all eight
providers (`--run-live`, 8/8 green, 2026-07-14).

### Fixed

- **Anthropic, Bedrock, and OpenRouter all repin to `Mode.JSON_SCHEMA`**
  (from `Mode.ANTHROPIC_JSON` / `Mode.ANTHROPIC_JSON` /
  `Mode.OPENROUTER_STRUCTURED_OUTPUTS`): the strict OpenAI-style
  `response_format` json-schema, which LiteLLM translates per provider and
  OpenRouter accepts natively (its `require_parameters` routing default
  continues to keep requests on endpoints that honor it). Chosen by live
  measurement (2026-07-14, Haiku 4.5 direct and on Bedrock): `JSON_SCHEMA`
  and `MD_JSON` validate every smoke-schema field; `JSON` fails because
  Claude wraps the JSON in a markdown fence that instructor's parse path
  does not strip; `TOOLS` cannot run on a lean install (LiteLLM's tools
  branch imports its proxy machinery, which needs `fastapi`).
- **OpenRouter structured requests are strict again.** The removed
  `OPENROUTER_STRUCTURED_OUTPUTS` handler sent `"strict": true` +
  `additionalProperties: false`; instructor's core `JSON_SCHEMA` handler
  sends neither, and OpenRouter treats a non-strict json_schema as
  *advisory* — measured: `mistralai/mistral-nemo` stochastically echoing
  the schema itself. The `OpenRouterProvider` now opts into a
  `strict_json_schema` provider trait and the LiteLLM call seam upgrades
  the `response_format` to the original measured wire shape. Other
  providers' requests are byte-identical to instructor's output.

### Changed

- **`instructor` is now declared `>=1.15.4,<2`** (was `>=1.15.1` unbounded).
  The mode registry `from_litellm` construction runs against is a coupling
  surface that has already broken once inside a patch series, so the range is
  pinned to the semantics the modes are verified on. An offline, keyless test
  now constructs every shipped provider's mode against the real registry, so
  the next registry drift fails CI instead of production.

### Removed

- **The `[anthropic]` extra and the eager Anthropic-SDK construction gate.**
  With the `Mode.JSON_SCHEMA` repin no llmkit path can reach the Anthropic SDK:
  LiteLLM speaks the Anthropic HTTP API directly, and instructor >=1.15.4
  touches the SDK only inside `try/except ImportError`. `AnthropicProvider`
  and `BedrockProvider` construct without it (previously they raised
  `ModuleNotFoundError` pointing at `omg-llmkit[anthropic]`), the `[bedrock]`
  extra no longer pulls the SDK in, and `require_anthropic_sdk` is gone from
  `llmkit.providers.base`. Hosts that installed `omg-llmkit[anthropic]` can
  simply drop the extra.

## [0.6.0] — 2026-07-12

Structured calls now **fail fast on output-token-limit truncation**
(`finish_reason='length'`) instead of blindly re-asking with the same budget.

### Changed

- **A length-truncated structured completion is no longer re-asked in-call.**
  Previously instructor's in-call repair loop (two total attempts) retried a
  truncation exactly like a malformed-JSON response — but a re-ask with an
  identical token budget can only truncate again. The motivating production
  failure was a stochastic degenerate repetition loop on Gemini: once looping,
  generation only stops at the provider's output ceiling (65,535 tokens,
  multiple minutes), and the blind re-ask looped again in **6 out of 6**
  observed failures — doubling every multi-minute burn while holding a
  concurrency slot. Truncation now surfaces immediately — one generation,
  seconds — as the new `OutputLimitError`. The schema-repair re-ask for
  genuinely repairable failures (malformed JSON / `ValidationError`) is
  untouched, as is the `InstructorRetryException` wrap on exhaustion; default
  requests remain byte-identical (still no `max_tokens` key unless set).
- **The retry layer never retries an output-limit truncation, under any
  configuration.** Both budgets of `RetryPolicy` / `with_retries` treat
  `OutputLimitError` as permanent — including the `with_retries` default
  `retry_on=None` ("retry on any Exception"), which would otherwise have
  charged the new bare exception to the transport budget. An explicit
  `retry_on` that lists the type still wins (opt-in resampling).

### Added

- **`OutputLimitError`** (exported from `llmkit`), following the
  `CircuitOpenError` fail-fast-but-catchable precedent. Carries `model`,
  `max_tokens` (`None` = no cap sent, provider ceiling), and
  `completion_tokens` (best-effort from the truncated completion's usage) so
  the failure is diagnosable from the error alone: `completion_tokens` at your
  cap → raise the cap; a huge count under no cap → the prompt induces runaway
  output.
- **`LLM_OUTPUT_LIMIT_ERRORS`** subset, and `LLM_RECOVERABLE_ERRORS` is now
  the union of **four** subsets (transport + schema + backpressure +
  output-limit) — hosts that catch `LLM_RECOVERABLE_ERRORS` keep degrading
  gracefully instead of crashing on a new uncaught type. Hosts composing
  their catch-nets from the *subsets* should add `LLM_OUTPUT_LIMIT_ERRORS`
  explicitly.
- `tenacity` is now a direct dependency (already present transitively via
  instructor; the floor matches instructor's own requirement).

### Migration / risk

- **Fail-fast trades away resample luck.** A caller running a *snug*
  `max_tokens` whose healthy responses occasionally graze the cap previously
  got a silent second sample that sometimes landed under the limit; it now
  hard-fails with `OutputLimitError` instead. That is the intended trade —
  the failure becomes legible ("raise your cap") rather than silent doubled
  latency — but a borderline caller should either set caps with real headroom
  (~8× median healthy output works well in the motivating host) or opt back
  into resampling with `retry_on=(OutputLimitError, ...)`.

## [0.5.0] — 2026-06-29

### Added

- **Google Vertex AI provider (`Provider.VERTEX` / `VertexProvider`).** A second
  path to the same Gemini models the `GOOGLE` (AI Studio) provider reaches, now
  through Google Cloud — the AI Studio ↔ Vertex split mirrors the existing
  Anthropic ↔ Bedrock one. Routes `vertex_ai/<model>` and pins Gemini's native
  `Mode.JSON_SCHEMA` for structured output. Like Bedrock it carries **no bearer
  key**: Google credentials resolve from **Application Default Credentials**
  (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or a
  workload-identity / metadata-server token), never through `LLMClientConfig`.
- **`vertex_location` data-residency selection.** `LLMClientConfig` gains
  `vertex_project` and `vertex_location` (and `make_provider` the matching
  keyword args); both are unused by every other provider. `vertex_location`
  selects the regional endpoint (`<location>-aiplatform.googleapis.com`) where the
  request is processed, so a regional value (e.g. `"europe-west4"`) pins in-region
  processing — the Vertex analog of Bedrock's `aws_region_name`. Left `None`, both
  resolve from the environment (`VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`), with the
  location otherwise falling back to Google's default region. Default model is
  Gemini 2.5 Flash-Lite (parity with the AI Studio provider). Note Gemini
  availability is region-specific: a region chosen for residency may not host
  every model (incl. the flash-lite default) — an unavailable model returns a
  Vertex `400 FAILED_PRECONDITION`, so pin a `model` the region serves.
- **`omg-llmkit[vertex]` opt-in extra.** Pulls only `google-auth` (the minimal
  credential dep LiteLLM's gemini-on-Vertex path imports to mint its OAuth token —
  not the heavier `google-cloud-aiplatform`), so a non-Vertex host takes on no
  Google dependency. Constructing `VertexProvider` without it raises a clear
  `install omg-llmkit[vertex]` error eagerly at construction. Also pulled into the
  `all` extra. New public symbol: `VertexProvider` (from `llmkit.providers`).

## [0.4.0] — 2026-06-28

Adaptive backpressure: the rate limiter now reacts to provider overload instead of
offering a constant load into it. Motivated by a consumer's sustained `503`
overload window that a fixed concurrency cap plus blind retries could not ride out.

### Added

- **Adaptive per-provider concurrency (AIMD), on by default.** The per-provider
  concurrency limit now *moves*: a provider overload signal (HTTP 429 / 503 / 529)
  received while saturated halves the limit (floored at 1, at most one decrease per
  refractory window so a correlated burst is a single step), and it recovers toward
  `max_concurrent` on a wall-clock interval once the provider stops pushing back. It
  only ever lowers the limit *below* `max_concurrent`, never above — so a provider
  that never throttles behaves identically to the previous fixed cap. Turn it off
  with `configure_rate_limit(adaptive=False)`. The throttle signal is read from the
  *unwrapped* provider error, so structured calls (where the error arrives wrapped
  in `InstructorRetryException`) drive it too. `configure_rate_limit` gains an
  `adaptive=` parameter and `RateLimitConfig` an `adaptive` field.
- **Per-provider circuit breaker, opt-in (off by default).** Arm it with
  `configure_rate_limit(breaker=True)`. Once a provider's throttle rate over a
  rolling window (the last 20 real outcomes, ≥ 50% throttled) trips it, the limiter
  **fails fast** for that provider — raising `CircuitOpenError` immediately, holding
  no concurrency slot and deducting no RPM token — for a 30s cooldown, after which a
  single probe tests recovery (a clean success closes it, any failure re-opens). It
  reads the *same* unwrapped per-provider outcome stream as adaptive concurrency, and
  is the "limit is effectively 0 while the provider is down" case that AIMD's
  floor-of-1 cannot express. **Off by default** because, unlike adaptive concurrency
  (which only ever *reduces* load), it flips "eventually succeeds" → "fails fast" — the
  host opts in. `CircuitOpenError` is a new fail-fast exception carrying `.provider`;
  it joins `LLM_RECOVERABLE_ERRORS` (via a new `LLM_BACKPRESSURE_ERRORS` subset) so a
  host's existing degrade-on-503 `except` keeps catching it, but it is **never
  retried** by the library. `BackpressureEvent.reason` widens with `"breaker_open"` /
  `"breaker_half_open"` / `"breaker_closed"`. New public symbols: `CircuitOpenError`,
  `LLM_BACKPRESSURE_ERRORS`; `configure_rate_limit` gains a `breaker=` parameter and
  `RateLimitConfig` a `breaker` field.
- **Backpressure observability.** `backpressure_callback(cb)` installs a callback
  (a context var, like `retry_progress_callback`, so it crosses the `run_sync`
  boundary) that receives a `BackpressureEvent(provider, old_limit, new_limit,
  reason)` on every adaptive limit change (and, when the breaker is armed, every
  breaker transition). New public symbols: `backpressure_callback`,
  `BackpressureCallback`, `BackpressureEvent`.
- **`Retry-After` is honoured.** When a retried provider error carries a
  `Retry-After` (a `retry-after` / `retry-after-ms` header, an HTTP-date, or the
  SDK's numeric attribute), the backoff waits that duration instead of the computed
  exponential — capped at the new `RetryPolicy.retry_after_cap` field (default 60s)
  so a hostile value can't wedge a call, read from the unwrapped error so structured
  calls honour it, and honoured even when `backoff_base_seconds` is 0. Absent a
  header, backoff is byte-identical to before. Applies to both `with_retries` and the
  streaming retry loop.

### Changed

- The per-provider concurrency primitive is now a **FIFO** gate (replacing the
  `asyncio.Semaphore`): waiters are admitted in strict arrival order, so a newcomer
  can no longer barge ahead of an older waiter under sustained saturation.
- The opt-in **RPM token-bucket** now admits its waiters in **FIFO** order too, on
  both the async and sync paths. Previously a waiter slept on a computed delay and
  then re-contended, so a newcomer could take the freshly refilled request token an
  older waiter was about to claim — leaving an individual waiter's latency unbounded
  under sustained self-saturation. The async path serializes waiters on a
  per-event-loop lock held across the refill wait; the sync path on a one-shot-ticket
  queue with no event loop. The aggregate rate is unchanged (only *who* is admitted
  next is ordered). The **TPM** gate is fair by construction — it deducts nothing at
  admission, so it cannot barge — and is left as the plain wait-for-budget loop.
- A call **cancelled mid-acquire** (after its RPM token is deducted but before it
  holds a concurrency slot — e.g. a `run_sync` timeout) now **refunds** the RPM
  token, so a systematically-cancelled workload no longer silently shrinks its own
  effective RPM.

No breaking changes: `adaptive` defaults on but can only reduce load below the
existing cap; the circuit breaker defaults **off** (a host arms it explicitly);
`RateLimitConfig` / `configure_rate_limit` gain trailing optional fields/parameters;
`RetryPolicy` gains a trailing field; `LLM_RECOVERABLE_ERRORS` keeps catching exactly
what it did plus the new fail-fast `CircuitOpenError` (which only ever arises once a
host opts into the breaker).

## [0.3.0] — 2026-06-15

### Changed

- **Synchronous calls now run on one persistent event loop.** `run_sync` — and
  the `structured_llm_call_sync` / `text_llm_call_sync` wrappers built on it —
  previously drove each call on a *fresh event loop per call*. Under a concurrent
  sync fan-out (many threads each making one sync call) those short-lived loops
  raced LiteLLM's process-global logging worker, flooding stderr and occasionally
  stalling an otherwise-successful call for up to LiteLLM's ~600 s request
  timeout. Every sync call now runs on a single, lazily-started, process-global
  loop on a dedicated daemon thread, so LiteLLM's logging worker binds once and is
  drained once — at interpreter exit, via `atexit`. A call made from *inside* an
  already-running event loop still falls back to a one-shot worker thread with its
  own loop. On the persistent-loop path a `timeout` now also **cancels** the
  coroutine on the loop (tearing down the in-flight provider request) instead of
  leaking it. **No public API changed.** Internally, the per-calling-thread
  concurrency `threading.Semaphore` the `*_sync` wrappers used is gone: the
  per-provider concurrency cap is now enforced inside the async path by the shared
  loop's semaphore, so a cross-thread sync fan-out is still bounded by it. The
  hand-rolled public sync acquire path (`rate_limit_acquire_sync`) is unchanged.

## [0.2.0] — 2026-06-09

Everything accumulated since `0.1.2` in one MINOR release: it carries
default-behavior changes and a small breaking surface. (The `0.1.3`/`0.1.4`
numbers were bumped internally but never published; their work ships here.)

**Migrating from 0.1.2.** Most code keeps working unchanged, but four changes
flip a default, change a contract, or move a symbol — review these first:

- **Transient-error retries are on by default.** Every call function now retries
  the recoverable set on its own. If you already wrap calls in your own retry
  loop, pass `retry=NO_RETRY` (or wrap with `with_retries`, which auto-collapses
  the inner pass) to avoid multiplied budgets. Permanent 4xx (401/400/403) fail
  fast and are never retried.
- **Per-provider concurrency limiting is on by default** (cap **8** concurrent
  calls per provider). Lower it with `configure_rate_limit(max_concurrent=...)`,
  or `enabled=False` to turn it off. RPM/TPM remain opt-in (off unless set), so
  an unset request is byte-identical to before — note a migrator's old
  per-minute tuning stays inert until you set `rpm=`/`tpm=`.
- **The `LogSink` protocol changed** (breaking for custom sinks): `write(record)`
  now returns `None`, not `Path | None`. A third-party sink must drop the
  return; path tracking moved to `capture_llm_log_paths()` and
  `LocalYamlLogSink.write_returning_path()` (see _Removed_).
- **A few symbols moved or were removed** (all breaking — see _Removed_):
  `get_provider`/`get_llm_config` → `build_provider`/`describe_llm`;
  `with_retries(max_retries=...)` → `max_attempts=...`; the `*Provider` classes,
  `with_retries`, and `GlobalRateLimiter` are no longer re-exported from the
  package root (import them from `llmkit.providers` / `llmkit.retry` /
  `llmkit.rate_limiting`); `capture_llm_log_paths` / `capture_llm_records` moved
  from `llmkit.structured_output` to the new `llmkit.capture` module (the
  top-level `from llmkit import ...` is unchanged). The Anthropic SDK is now the
  opt-in `omg-llmkit[anthropic]` extra — install it (or `[bedrock]` / `[all]`)
  only if you route Anthropic or Bedrock.

### Added

- `RetryPolicy.max_backoff_seconds` (default `30.0`) — a ceiling on any single
  backoff sleep. Previously the full-jitter ceiling `base * 2**(attempt-1)`
  was uncapped, so a large `max_attempts` could sleep hours on late attempts.
  The cap propagates through the call functions and `with_retries`.
- `text_llm_call_sync(...)` — a synchronous wrapper around `text_llm_call`,
  matching the existing structured sync wrappers and removing the call-surface
  asymmetry for non-streaming LLM calls.
- `model_from_json_schema(schema, *, name=None)` — converts a **JSON-schema
  dict** into a Pydantic model class at runtime, so consumers who declare their
  structured-output contracts as JSON-schema dicts (shared across Node /
  frontend / Python) no longer hand-write a converter before calling
  `structured_llm_call`. The intended pattern is **build-once-reuse**
  (`Invoice = model_from_json_schema(schema)` at import, then pass `Invoice` as
  `output_schema` on every call); `structured_llm_call`'s signature is
  unchanged (still `output_schema: type[T] -> T`, no `dict` overload). Built on
  `pydantic.create_model` (no new third-party dependency). Supported subset:
  `object` with `properties` and a `required` array; scalars (`string` /
  `integer` / `number` / `boolean`, plus `null` / nullable via
  `["string", "null"]` or an `anyOf` null branch); `array` with `items`
  (including arrays of objects); `enum` (string or integer members); nested
  objects inline or via local `$ref` (`#/$defs/...` / `#/definitions/...`); and
  `additionalProperties` as `true` / `false` / absent (a *typed*
  `additionalProperties` map is rejected). Anything outside the subset raises a
  clear `ValueError` naming the construct and its path, rather than silently
  producing a wrong model. Two footguns the CaCL dogfood hit are handled and
  tested: (1) a non-required field maps to an *optional* Pydantic field
  defaulting to `None`, and the generated model's `model_dump` /
  `model_dump_json` drop a `None` left in an **optional** field by default, so
  an omitted optional is **absent** rather than `"field": null` (which would
  fail downstream re-validation against the same schema) while an
  explicitly-null *required* nullable field is kept; pass `exclude_none=False`
  to keep every null, or `exclude_none=True` to drop them all. (2) A title-less
  or empty-titled schema still yields a validly-named class (default
  `JsonSchemaModel`), which `create_model` and `instructor` both require.
  Generated models default to `extra="forbid"`, so a response carrying a key
  not in the schema is rejected rather than silently kept (a hallucinated extra
  field fails loudly — stricter than JSON Schema's permissive
  `additionalProperties` default); `"additionalProperties": true` opts an
  object into `extra="allow"`, and an explicit `"type": "object"` with no
  `properties` raises (set `"additionalProperties": true` for an intentionally
  free-form object) instead of silently building a zero-field strict model that
  rejects every real response. Per-field bounds outside the supported set are
  dropped, and a constraint that doesn't match the field's type, or a mixed
  string/integer `enum`, is handled safely (see _Fixed_). Exported from the
  package root.
- A `py.typed` marker so consumers' type checkers honor llmkit's type hints
  (the package is basedpyright-clean and already declares the `Typing :: Typed`
  classifier, but without the PEP 561 marker downstream tools treated it as
  untyped and required an `ignore_missing_imports` override).
- **Opt-in per-provider requests-per-minute (RPM) and tokens-per-minute (TPM)
  rate limiting**, alongside the existing concurrency cap. `configure_rate_limit`
  gains `rpm=` / `tpm=` arguments (both `None` / off by default), and
  `RateLimitConfig` / `get_rate_limit_config()` now report them. RPM and TPM are
  **opt-in** — unlike concurrency, there is no universally sane per-minute
  default (it's your account's metered limit), so leaving them unset sends a
  request **byte-identical** to the prior behaviour. Each is a per-provider
  **token bucket** with a deliberately *small* burst depth — `min(max_concurrent,
  rpm)` requests for RPM, roughly one second of tokens for TPM, **not** a full
  minute's quota — so a cold or idle bucket admits a fan-out-sized burst and
  then smooths to the sustained rate without ever emitting ~2× the published
  per-minute limit inside one provider-side window; TPM is debited by each
  call's measured
  `usage.total_tokens` (a streamed call usually reports no usage and so does not
  debit TPM, consistent with cost being `None` for streams). This addresses the
  CaCL dogfood gap where a migrator's old `rate_limit_rpm` went inert under the
  concurrency-only model — the binding limit on a metered account is usually
  RPM/TPM, not concurrency, and the concurrency cap does not stand in for an RPM
  limit. The acquire context managers (`rate_limit_acquire_async` /
  `rate_limit_acquire_sync`, and `GlobalRateLimiter.acquire_async` /
  `acquire_sync`) now yield a `RateLimitSlot` whose `record_tokens(...)` debits
  the TPM budget; a host joining the limit by hand calls it once it knows a
  call's usage. `RateLimitSlot` is exported from the package root. Per-credential
  scoping for multi-tenant hosts remains a deliberate non-goal (llmkit is
  single-tenant by design; see `docs/planning/opinions.md` §6.4).
- `get_rate_limit_config()` (returning a frozen `RateLimitConfig`) — the
  symmetric read for `configure_rate_limit`, so a host can log or assert its
  effective `enabled` / `max_concurrent` / `rpm` / `tpm` at startup without
  reaching into limiter internals. Both are exported from the package root.
- **AWS Bedrock** provider (`Provider.BEDROCK` / `BedrockProvider`, `bedrock/`
  LiteLLM prefix), giving first-class access to Claude-on-Bedrock. Unlike every
  other provider, Bedrock authenticates through the standard **AWS credential
  chain** rather than a bearer `api_key`: the new optional
  `LLMClientConfig.aws_region_name` carries only the region (the single
  AWS-shaped field; unused by all other providers), while secrets resolve from
  the ambient chain (environment / shared config / instance role) and never
  pass through the config. Structured output is pinned to
  `instructor.Mode.ANTHROPIC_JSON` — the same Claude-native mode the direct
  Anthropic provider uses (the model is Claude). `Mode.BEDROCK_JSON` is
  deliberately avoided: it targets instructor's `from_bedrock` (boto3) client
  and drops `model` when driven through `from_litellm`, this library's call
  seam. `reasoning_effort` is forwarded where the underlying model supports it. The
  default model is Claude Haiku 4.5 via its **cross-region inference profile**
  id (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) — current Claude models on
  Bedrock are typically reached through inference profiles rather than plain
  on-demand ids; pass a different profile-, partition-, or on-demand-prefixed id
  as `model` to route elsewhere. `boto3` (for SigV4 signing) ships via the opt-in
  `omg-llmkit[bedrock]` extra, so non-Bedrock installs gain no AWS dependency.
  `BedrockProvider` is exported from the package root.
- Direct **DeepSeek** provider (`Provider.DEEPSEEK` / `DeepSeekProvider`,
  `deepseek/` LiteLLM prefix), giving first-class access to `deepseek-chat` (V3)
  and `deepseek-reasoner` (R1) on a first-party key rather than the indirect
  `openrouter/deepseek/...` hop (which adds a gateway markup). Structured output
  is pinned to DeepSeek's native JSON mode (`instructor.Mode.JSON`); the strict
  `Mode.JSON_SCHEMA` the other direct providers use is rejected by DeepSeek's
  API, while `Mode.JSON` is measured to validate on both models (live smoke
  test). `reasoning_effort` is forwarded for `deepseek-reasoner` and is harmless
  on `deepseek-chat`. `DeepSeekProvider` is exported from the package root.
- Direct **OpenAI** provider (`Provider.OPENAI` / `OpenAIProvider`, `openai/`
  LiteLLM prefix), giving first-class access to GPT / o-series / GPT-5 models
  without the indirect `openrouter/openai/...` hop (which adds a markup and a
  different structured-output mode). Structured output is pinned to OpenAI's
  native structured-outputs mode (`instructor.Mode.JSON_SCHEMA`), and
  `reasoning_effort`
  is forwarded for reasoning models on the same seam as the other providers. An
  optional `base_url` points the provider at OpenAI-compatible gateways; left
  unset, LiteLLM uses OpenAI's default endpoint. `OpenAIProvider` is exported
  from the package root.
- The retry helpers `retry_progress_callback` and `RetryProgressCallback` are
  now exported from the package root, so callers no longer reach into
  `llmkit.retry` directly. (`with_retries` is **not** part of the headline
  surface — it stays importable from `llmkit.retry`; see _Removed_ below.)
- `RetryPolicy` (plus `DEFAULT_RETRY_POLICY` and `NO_RETRY`) is exported from
  the package root — the per-call knob for the now default-on transient-error
  retry layer (see _Changed_ below). `RetryPolicy` is a frozen dataclass
  (`max_attempts`, `backoff_base_seconds`, `retry_on`); pass `retry=NO_RETRY`
  to opt a call out, or a custom instance to tune the budget.
- Reasoning/thinking control via `LLMClientConfig.reasoning_effort` (and a
  per-call `reasoning_effort` override on `structured_llm_call` /
  `structured_llm_call_sync` / `text_llm_call`), forwarded to LiteLLM. Lets
  callers disable Gemini thinking (`reasoning_effort="disable"`) so it
  doesn't consume the `max_tokens` budget and truncate structured output.
  Set once on the config and every call inherits it; the per-call value
  overrides it. Backward-compatible; defaults to provider behaviour — when
  unset (`None`) no `reasoning_effort` kwarg is sent, so the outbound
  request is byte-identical to before. `LLMCallRecord` also records the
  per-call setting for log completeness.
- `make_provider(provider, *, api_key=None, model=None, base_url=None,
  reasoning_effort=None, aws_region_name=None)` — build a provider from raw
  credentials in one line for the per-call `provider=` override, without an
  `LLMClientConfig`. Exported from the package root.
- `build_provider(config)` as the canonical construct-from-config function
  (replaces `get_provider`). Exported from the package root.
- `describe_llm(config)` as the read-snapshot accessor (replaces
  `get_llm_config`), importable from `llmkit.providers`.
- `model_from_json_schema` now carries per-field JSON-schema constraints into
  the generated Pydantic model: numeric `minimum`/`maximum`/`exclusiveMinimum`/
  `exclusiveMaximum` map to `ge`/`le`/`gt`/`lt`, and `minLength`/`maxLength`
  plus `minItems`/`maxItems` map to `min_length`/`max_length`, so generated
  models validate value bounds, not just shape. Bounds are resolved through
  `$ref` chains of any depth and through nullable wrappers; exclusive bounds
  are recognised in their numeric (Draft 2020-12) form only — the Draft-4 /
  OpenAPI-3.0 boolean form (`"exclusiveMinimum": true` beside a `"minimum"`) is
  dropped, leaving the sibling *inclusive* bound; constraints outside the
  supported set (e.g. `pattern`, `format`, `multipleOf`) remain silently
  dropped with no partial enforcement, and per-field `description` passthrough
  is unchanged.
- `capture_llm_records()` context manager — captures the per-call
  `LLMCallRecord` objects (approximate cost, resolved model/provider, duration,
  error) for calls in its scope, with no custom sink, across both the async
  call functions and the `run_sync` sync bridge. Exported from the package
  root.
- `LLMCallOptions`, an opt-in **frozen** bundle of the per-call keyword
  arguments (`temperature` / `model` / `max_tokens` / `reasoning_effort` /
  `retry` / `provider`) accepted as `options=` on `structured_llm_call`,
  `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and
  `stream_text_with_log`, to cut the repeated per-call keyword block; the
  flat-keyword path is unchanged. Precedence is config < `options` < explicit
  per-call keyword. `feature` is intentionally **not** part of `LLMCallOptions`
  — it stays a required per-call keyword as a deliberate telemetry forcing
  function. Exported from the package root.
- `rate_limit_acquire_async(provider_key)` and
  `rate_limit_acquire_sync(provider_key)` context managers in
  `llmkit.rate_limiting` — the public way to join the global per-provider rate
  limit by hand (e.g. from a LangChain chat-model wrapper) without referencing
  `GlobalRateLimiter`. Each yields a `RateLimitSlot` whose `record_tokens(...)`
  debits the TPM budget when configured.
- An `on_result` semantic-validation **re-roll hook** on the call functions
  (`structured_llm_call` / `_sync`, `text_llm_call` / `_sync`) plus a new
  exported `ResultValidationError`. The callback is invoked
  with each attempt's result; raising `ResultValidationError` from it rejects a
  result that *parsed* but is *semantically* wrong (an empty register, an
  unresolved citation, a total that doesn't reconcile) and re-rolls the call.
  The re-roll is charged against the **validation budget**
  (`RetryPolicy.validation_max_attempts`, default 2) — the same budget a schema
  failure uses, so a deterministically-bad result can't burn the full transport
  budget; on exhaustion the last `ResultValidationError` propagates, and each
  attempt (rejected ones included) is its own logged call. Folds an
  LLM-then-validate-then-re-roll loop consumers hand-rolled into the call
  itself. `text_llm_call`'s hook receives the response text. Like
  `feature`, `on_result` is a per-call keyword, not part of `LLMCallOptions`.
- An `omg_llmkit` **import shim**: the distribution installs as `omg-llmkit` but
  imports as `llmkit`, and a mistaken `import omg_llmkit` (e.g. a post-install
  smoke test using the install name) now raises a clear one-line redirect to
  `import llmkit` instead of a bare `ModuleNotFoundError`.

### Changed

- **The Anthropic SDK moved from a core dependency to the opt-in
  `omg-llmkit[anthropic]` extra.** It was core through 0.1.2 on the belief that
  `instructor` forced it at import; measured against `instructor>=1.15.1`, that
  is false — `instructor` reaches the SDK only at *call time*, on its
  `ANTHROPIC_*` usage-accounting path (`from anthropic.types import Usage` inside
  `instructor/core/retry.py`, guarded by the mode), so plain `import llmkit` and
  a Google-only flow never touch it. Non-Anthropic hosts now take on no Anthropic
  dependency. The `AnthropicProvider` and `BedrockProvider` (both pin
  `ANTHROPIC_JSON`, so both need the SDK at call time) raise a clear
  `install omg-llmkit[anthropic]` error *at construction* when the SDK is absent,
  rather than failing cryptically on the first completion. The `[bedrock]` extra
  pulls in `[anthropic]` (it routes Claude), and a new convenience `[all]` extra
  installs every provider's optional dependencies at once. **Migration:** hosts
  that route Anthropic or Bedrock should install `omg-llmkit[anthropic]` (or
  `omg-llmkit[bedrock]`, or `omg-llmkit[all]`); hosts on other providers need no
  change.
- **Transient-error retries are now on by default.** `structured_llm_call`,
  `structured_llm_call_sync`, `text_llm_call`, `text_llm_call_sync`, and
  `stream_text_with_log` retry
  the curated recoverable set (429 / 503 / 5xx, timeouts, transient
  provider errors) on their own — with full-jitter backoff — so
  reliability no longer depends on every caller wrapping each call. The transient
  set names the specific transient `openai` subclasses (`RateLimitError` /
  `InternalServerError` / `APIConnectionError`) rather than their broad
  `openai.APIError` base, so **permanent 4xx errors — authentication (401),
  bad-request (400), permission (403) — fail fast** instead of burning the retry
  budget. Programming errors (e.g. `TypeError`) still propagate immediately. The budget is the new
  per-call `retry: RetryPolicy` argument: pass `retry=NO_RETRY` to opt out or a
  custom `RetryPolicy` to tune it. This layer stays **separate** from
  instructor's in-call schema-repair budget (instructor's `max_retries`, which
  llmkit pins to `2` — two in-call attempts, i.e. one repair re-ask) —
  no double-counting — and each attempt remains its own logged call (so
  `capture_llm_log_paths` sees one path per attempt). Streaming retries only a
  failure that occurs **before the first chunk** is yielded; a partially
  consumed stream cannot be transparently restarted. `with_retries()` remains
  exported as the explicit, composable path for wrapping any awaitable, and now
  takes a `retry_on` filter.
- **Transport and schema-validation retries now have separate budgets.** The
  recoverable set is split into `LLM_TRANSPORT_ERRORS` (rate limits, transient
  5xx, network/timeout) and `LLM_SCHEMA_ERRORS` (pydantic `ValidationError`,
  instructor `InstructorRetryException`), and `RetryPolicy` counts each against
  its own budget: transport keeps `max_attempts` (default **3**), while the new
  `validation_max_attempts` (default **2** = one retry) governs schema failures.
  A deterministically-wrong schema or impossible constraint can no longer burn
  the full transport budget on doomed re-asks, while a *transiently*-malformed
  JSON response still earns one cross-call retry. `LLM_RECOVERABLE_ERRORS`
  remains the **union** of the two subsets, so its documented `except`-clause
  catch-set contract is unchanged. `NO_RETRY` disables **both** budgets (a single
  attempt). `RetryPolicy` gains `validation_max_attempts` and
  `validation_retry_on` (defaulting to `LLM_SCHEMA_ERRORS`); this cross-call
  layer remains separate from instructor's in-call repair (instructor's
  `max_retries`, pinned to `2` — one repair re-ask that fixes malformed JSON
  *within* one attempt before any `ValidationError` reaches the retry layer).
  Flagged by the PIA Maker and FiW dogfoods of the 0.2.0 RC.
- **Unified the public retry attempt-count on `max_attempts`.** `RetryPolicy`
  already used `max_attempts`; `with_retries(...)` now uses it too, with the same
  semantics everywhere — *total attempts including the first* (`N`, not `1 + N`).
  The old `with_retries(..., max_retries=...)` keyword and the progress-callback
  `max_retries` keyword were **removed outright** (hard cut, no shim — see
  _Removed_ below); both now use `max_attempts`. `LLM_TRANSPORT_ERRORS` and
  `LLM_SCHEMA_ERRORS` are exported from the package root alongside
  `LLM_RECOVERABLE_ERRORS`.
- **`with_retries` now guards against nested retry-budget multiplication.**
  Because the call functions retry internally by default, wrapping one in
  `with_retries` previously multiplied the budgets silently (the `3 × 3 = 9`
  trap the PIA Maker dogfood hit). `with_retries` now detects (via a context
  variable) when it runs *inside* an already-active llmkit retry loop and
  collapses that inner layer to a **single pass** — so the budgets no longer
  multiply, and per-attempt logging is preserved. An *accidental* double-wrap
  additionally emits a filterable `RuntimeWarning` (so consumers running under
  `-W error` can filter it). The guard does not affect `with_retries` wrapping a
  plain (non-llmkit) awaitable, which still retries normally. The clean way to
  drive retries from an outer wrapper is to opt the inner call out with
  `retry=NO_RETRY` — which collapses the inner pass **without** warning; this is
  documented in `with_retries`' docstring.
- **Concurrency rate limiting is now on by default, scoped per provider.**
  Previously the limiter was off until a host opted in, and a single process-wide
  budget fronted every provider. It is now active out of the box and bounds
  concurrent calls **per provider** (keyed by the effective provider name, the
  same value logging records), so fan-out to one provider can't overrun its rate
  limits or eat another provider's budget. The default cap is **8 concurrent
  calls per provider** — headroom for the fan-out workloads consumers actually
  run, while still bounding a self-inflicted burst; a tightly-metered account can
  lower it. `configure_rate_limit(max_concurrent=..., enabled=...)` changes the
  cap or turns it off, and `get_rate_limit_config()` reads back the effective
  values.
- The internal LiteLLM call layer now forwards a provider's **full**
  `completion_kwargs()` dict (splatting it into the call) instead of
  cherry-picking `api_key` / `api_base`. This lets a provider carry whatever
  credential kwargs LiteLLM needs — e.g. Bedrock's `aws_region_name` — without
  the shared call layer growing per-provider knowledge. No change for the
  existing providers (their kwarg dicts are unchanged); the request shape is
  byte-identical for `api_key` / `api_base` providers.
- Provider dispatch (`build_provider`, the renamed `get_provider` — see
  _Removed_) now **fails loud** on an unknown provider instead of silently
  constructing an `OllamaProvider`. The previous `else` catch-all meant a
  newly-added `Provider` enum member routed to a confusing local-Ollama failure;
  dispatch is now an exhaustive `match` whose fall-through calls
  `typing.assert_never`, so an unwired member is caught statically by
  basedpyright, raises `AssertionError` at runtime, and fails a dedicated
  exhaustiveness test. No behaviour change for the existing providers.
- The provider layer is reorganized from a single `providers.py` module into a
  `llmkit.providers` **package** with one module per provider over a
  provider-agnostic `base` module, so adding a provider is a self-contained new
  file plus one wiring line. The reorganization itself is purely internal —
  `Provider`, `LLMClientConfig`, and the `*Provider` classes keep importing from
  `llmkit.providers` — though this release *separately* renames
  `get_provider`/`get_llm_config` and trims the root re-exports (see _Removed_).
- `LLMClientConfig.model` is now optional (`str | None`, default `None`); a
  falsy model resolves to the selected provider's own default model instead of
  producing a broken `"<prefix>/"` LiteLLM id.
- Reserved `get_*` for reads and `build_*` / `make_*` for construction:
  `build_provider` replaces `get_provider`, and `describe_llm` replaces
  `get_llm_config`. (Both old names are removed from the public surface — see
  _Removed_ below.)
- The `LogSink` protocol method is now `write(record) -> None` (was
  `-> Path | None`); the file detail no longer leaks into the shared contract.
  `LocalYamlLogSink` exposes its written path via a new `write_returning_path()`
  method, and `capture_llm_log_paths()` continues to work unchanged.
- Documented the dual-homed `model` / `reasoning_effort` precedence explicitly
  — config < `options` < explicit per-call keyword — across the call-function
  docstrings and README, and documented `feature` as a deliberate required
  telemetry forcing function (intentionally excluded from `LLMCallOptions`).
- **OpenRouter now requests schema-honoring routing by default.**
  `OpenRouterProvider` sets OpenRouter's `provider.require_parameters` routing
  preference (via `extra_body`), so a structured-output request only lands on a
  serving endpoint that honors the strict `response_format` — closing the sharp
  edge where OpenRouter advertises `structured_outputs` at the *model* level but
  a routed *serving* endpoint silently ignores the schema and returns free-form
  JSON (a confusing downstream validation failure). It restricts routing to
  capable endpoints, which can in principle reduce availability or shift cost;
  construct `OpenRouterProvider(..., require_parameters=False)` to opt out. The
  `LLMProviderInterface.completion_kwargs` return type widened from
  `dict[str, str]` to `dict[str, object]` to carry the nested routing
  preference alongside the string credential kwargs.
- Documented that a **fully per-call host needs no global config source**: a
  caller that passes `provider=` on every call can run without
  `configure_llm_client(...)` — the call runs on the per-call provider alone and
  logging records it as the effective provider. Also flagged the
  `temperature=0.2` default and the per-call `provider=` override prominently in
  the README (onboarding papercuts from the greenfield integration).
- **`import llmkit` no longer imports LiteLLM eagerly**, roughly halving import
  time (measured ~5.3s → ~2.4s in a CI-like environment); LiteLLM loads on the
  first call. The one observable seam: the litellm-native 503 entry in
  `LLM_TRANSPORT_ERRORS` (`litellm.exceptions.ServiceUnavailableError`) is now
  a lazy stand-in resolved at `isinstance` time — classification behaves
  identically once litellm is loaded, but a bare `except LLM_TRANSPORT_ERRORS:`
  / `except LLM_RECOVERABLE_ERRORS:` clause cannot catch that one
  litellm-native class (Python's `except` matching bypasses the lazy check).
  Every other member still catches as usual, an openai-SDK 503 arrives as
  `openai.InternalServerError` and matches, and `isinstance` checks — what the
  retry layer uses — are unaffected.
- **The sync bridge's default `timeout` is now 600 seconds** (was 60), and
  `run_sync` accepts `timeout=None` for unbounded. The budget governs a
  `*_sync` call made from inside a running event loop (the worker-thread path);
  60s was shorter than a routine slow completion plus on-by-default retries,
  so the old default could abandon a healthy call. The same release also makes
  the timeout actually fire at the deadline (see _Fixed_).
- `structured_llm_call` / `structured_llm_call_sync` now bound their output
  type parameter to Pydantic (`[T: BaseModel]`). With the new `py.typed`
  marker, passing a dataclass or plain class as `output_schema` is a *type
  error* in your checker instead of a runtime failure inside instructor.
  Non-breaking for every valid caller.
- Dependency floors raised to the effective minimums the resolver already
  forced in practice: `openai>=2.20.0`, `pydantic>=2.8.0`, `httpx>=0.28.0` —
  lower-bound resolution (e.g. `uv --resolution lowest`) now succeeds instead
  of reporting no solution. Package metadata also moved to a PEP 639 SPDX
  license expression (`License-Expression: MIT`, with the LICENSE file still
  shipped).

### Removed

- **Breaking:** `with_retries` no longer accepts the deprecated `max_retries`
  keyword (hard-cut, no shim) — use `max_attempts` (same meaning: total attempts
  including the first). Passing `max_retries` now raises `TypeError`. The
  retry-progress callback keyword was likewise renamed `max_retries` ->
  `max_attempts` (the `RetryProgressCallback` protocol and the
  `handle_retry_failure` invocation), so callbacks now receive `max_attempts`.
- **Breaking:** `get_provider` and `get_llm_config` are removed from the public
  surface; use `build_provider` and `describe_llm` (importable from
  `llmkit.providers`) respectively.
- **Breaking:** `GlobalRateLimiter` removed from `llmkit.__all__` (still
  importable from `llmkit.rate_limiting` for internal use); replace
  `GlobalRateLimiter.acquire_async` / `acquire_sync` with `rate_limit_acquire_async`
  / `rate_limit_acquire_sync`, and `GlobalRateLimiter.is_enabled()` with
  `get_rate_limit_config().enabled`.
- **Breaking:** third-party `LogSink` implementations must drop the
  `Path | None` return from `write` and return `None`. The shipped
  `LocalYamlLogSink` is updated; consumers tracking file paths use
  `capture_llm_log_paths()` or `LocalYamlLogSink.write_returning_path()`.
- **Breaking:** `capture_llm_log_paths` (and the new `capture_llm_records`) no
  longer live in `llmkit.structured_output` — the capture seam moved to the new
  `llmkit.capture` module, so the 0.1.x deep import
  `from llmkit.structured_output import capture_llm_log_paths` now raises
  `ImportError`. Import from `llmkit.capture`, or use the unchanged top-level
  `from llmkit import capture_llm_log_paths, capture_llm_records`. (Relatedly,
  the new `LLMCallOptions` is defined in `llmkit.options`; the top-level
  `from llmkit import LLMCallOptions` is the supported path.)
- **Headline-surface trim (Breaking for `from llmkit import` of these names).**
  The seven concrete `*Provider` classes
  (`OpenRouterProvider` / `OllamaProvider` / `GoogleProvider` /
  `AnthropicProvider` / `OpenAIProvider` / `DeepSeekProvider` /
  `BedrockProvider`), `LLMInfo`, `describe_llm`, `with_retries`, and
  `GlobalRateLimiter` are demoted out of `llmkit.__all__`. They remain
  importable from their submodules (`llmkit.providers`, `llmkit.retry`,
  `llmkit.rate_limiting`); only the top-level re-export is gone.

### Fixed

- **Logging is now best-effort against *any* exception, not an enumerated
  set.** The YAML sink's handlers caught only `(OSError, yaml.YAMLError,
  UnicodeError)`, so an exotic failure (e.g. `RecursionError` from a cyclic
  structure) escaped a sink documented as never breaking the call, and the
  truncated exclusive-create file was left orphaned. The sink and the
  `index.jsonl` append now swallow any `Exception` (warning with context), the
  orphan is unlinked on any post-create failure, and an index-line failure no
  longer discards the already-written YAML path. Sanitized `feature`/`label`
  filename components are also clamped to 80 UTF-8 bytes, so a long label can
  no longer make every log write fail with `ENAMETOOLONG`.
- **`LocalYamlLogSink` is hardened against collisions, path traversal, and
  header forgery.** Per-call YAML filenames now carry a `uuid4` suffix and are
  opened with exclusive-create, so two concurrent calls with the same
  timestamp/feature/label can no longer share a path or silently truncate each
  other's log — note this changes the **filename format**, so tooling that
  globs or parses log filenames should expect the extra suffix. `feature` and
  `label` are sanitized before they reach the filesystem (separators and
  control characters stripped, `.` runs collapsed), so a value like
  `"../escape"` can no longer write outside `log_dir`. And the values
  interpolated into the `#` verdict header are flattened to one line, so an
  embedded newline cannot forge a second verdict line and corrupt `head -1`
  triage.
- **A failed `text_llm_call` attempt logs `response: null`, not `""`.** The
  empty-string initializer made a pre-content failure indistinguishable from a
  successful empty completion; the record now matches the documented
  "accumulated text, or None" contract (streaming keeps its intentional
  partial-transcript-on-error behavior).
- **A raising custom serializer can no longer mask a structured call's
  result.** The `model_dump()` on the logging path ran unprotected in the
  `finally`; a schema with a raising `@model_serializer` would replace the
  successful return value (or the real provider error) with the serializer
  exception. The dump now degrades the *logged* response to `None` with a
  warning.
- **Ctrl-C abandons an in-flight sync call instead of resuming it during
  teardown.** After `KeyboardInterrupt` stopped the loop mid-call, the
  logging-drain `finally` restarted the loop and resumed the interrupted call
  for up to the full drain timeout, then discarded its result. The bridge now
  cancels and settles the pending main task before draining, without masking
  the caller's `KeyboardInterrupt`.
- **`with_retries(retry_on=None)` routes wrapped transport causes to the
  transport budget.** With the documented "retry on any Exception" default, an
  `InstructorRetryException` wrapping a 429/503 was charged to the *lower*
  validation budget — the exact misclassification `underlying_provider_error`
  exists to prevent. The unwrapped cause is now classified against
  `LLM_TRANSPORT_ERRORS` when `retry_on` is unset.
- **Rate-limit provider keys are case-insensitive.** The limiter keyed budgets
  by exact string, but llmkit's own calls use display names (`"OpenAI"`) while
  the public helpers' examples said `"openai"` — a host following the docs
  silently acquired against a separate budget. Keys are now casefolded at the
  acquire boundary, so any casing joins the same rpm/tpm/concurrency budget;
  examples updated to the canonical names.

- **A real provider 503 is retried again.** LiteLLM maps HTTP 503 to its own
  `ServiceUnavailableError`, which subclasses `openai.APIStatusError` directly —
  not `openai.InternalServerError` — so the canonical transient "model
  overloaded" error matched neither retry budget and propagated unretried.
  `litellm.exceptions.ServiceUnavailableError` is now listed explicitly in
  `LLM_TRANSPORT_ERRORS`.
- **`stream_text_with_log` now participates in the nested-retry guard.** Its
  hand-rolled retry loop ignored the `_retry_active` flag that `with_retries`
  documents (and the other call functions honor), so wrapping stream consumption
  in `with_retries` could multiply pre-first-chunk attempts up to
  `max_attempts × max_attempts` with no warning. The stream loop now collapses
  to a single pass under an outer llmkit retry loop (with the same
  `RuntimeWarning`) and arms the guard around its own attempts.
- **A stream abandoned by its consumer is no longer logged as a clean `ok`
  call.** Breaking out of (or closing) `stream_text_with_log`, or cancelling
  the consuming task, produced a log record indistinguishable from a successful
  call carrying a partial transcript. The record now keeps the partial
  transcript but carries the error marker
  `llmkit.STREAM_ABANDONED_ERROR` ("Abandoned: stream closed
  by consumer before completion"), so the YAML verdict header reads `# ERROR`;
  `GeneratorExit` / `CancelledError` are always re-raised, never swallowed, and
  the abandoned-stream record is written deterministically at close time rather
  than whenever garbage collection finalizes the generator.
- **Structured calls get the documented single in-call schema repair.**
  instructor turns an integer `max_retries` into `stop_after_attempt(n)` — *n
  attempts total* — so the previous `max_retries=1` meant zero repair re-asks
  and every validation failure burned a full cross-call attempt (new request,
  log record, rate-limit slot). Now `max_retries=2`: two in-call attempts, one
  repair.
- **`LocalYamlLogSink` no longer drops `max_tokens` and `reasoning_effort`.**
  `LLMCallRecord` carried both fields but the YAML body never serialized them,
  so file-sink users couldn't tell from a log whether a truncated response was
  cap-limited or thinking was disabled. Both now appear in the per-call YAML
  (the compact `index.jsonl` deliberately omits them).
- **Per-call YAML logs are now always `yaml.safe_load`-able.** The sink dumped
  with the default (unsafe) Dumper, so a response containing an `Enum`,
  `Decimal`, or `set` was written as `!!python/object` tags — unparseable by
  `safe_load` and an arbitrary-code-execution hazard under full `yaml.load`.
  The sink now uses a `SafeDumper` subclass that renders enums as their values
  and any other non-plain object as a string.
- **`configure_rate_limit(max_concurrent=0)` now raises instead of hanging
  every call forever.** `Semaphore(0)` is a legal, permanently-locked
  semaphore, so a zero cap was accepted at configure time and every subsequent
  call blocked with nothing pointing at the misconfiguration; negative values
  raised only at the first call. `configure` now validates `max_concurrent < 1`
  with a `ValueError`, symmetric with the rpm/tpm checks.
- **`model_from_json_schema` rejects typeless non-object roots with a clear
  error.** A root carrying `enum`/`anyOf`/`oneOf` (or nothing recognizable as
  an object) without `"type": "object"` previously fell through to the object
  builder and silently produced a zero-field, `extra="forbid"` model that
  rejected every real response.
- **`model_from_json_schema` rejects a propertyless `{"type": "object"}` with
  a clear error.** An explicit object — root or nested — with no `properties`
  key likewise built a zero-field strict model that rejected every real
  response; it now raises a `ValueError` naming the path and suggesting
  `"additionalProperties": true` for an intentionally free-form object (which
  builds an open `extra="allow"` model).
- **`model_from_json_schema` validates the `required` array instead of
  silently mis-building.** A non-list `required` (or a non-string entry) was
  silently ignored — making *every* field optional — and a `required` name
  with no matching property built a model that was wrong in both directions.
  Both now raise a clear `ValueError` naming the offending value.
- **`model_from_json_schema` enforces constraints on array items.** Bounds on
  an array's `items` schema (`minLength`, `minimum`, …) — keywords in the
  documented supported set — were silently dropped, so schema-violating
  elements validated. Item constraints now wrap the element annotation,
  composing with nullable, enum, and `$ref` items.
- **`model_from_json_schema` honors constraint keywords that are siblings of a
  `$ref`.** Resolution replaced the schema dict outright, so
  `{"$ref": ..., "minimum": 5}` lost the bound (while a sibling `description`
  was honored). Sibling keywords now merge over the resolved target with
  outer-wins precedence, matching Draft 2020-12 and the module's existing
  merge behavior.
- **`model_from_json_schema` resolves `$ref` chains of arbitrary depth.** Field
  type resolution unwrapped at most two `$ref` hops, so a three-or-more-hop
  chain (`A → B → C → scalar`) fell through to the self-contradictory
  `no 'type', 'enum', or '$ref' — got keys ['$ref']` error. Resolution now
  loops; a cycle made purely of `$ref`s fails with the same clear
  recursive-schema error that object-level recursion already raised.
- **`model_from_json_schema` rejects the property names pydantic genuinely
  cannot carry — and only those — with a clear error.** A property with a
  leading underscore (like `_id`, silently dropped as a private attribute) or
  in pydantic's `model_*` protected namespace (`model_config` → `TypeError`,
  `model_dump` → a leaked protected-namespace error) crashed or corrupted deep
  inside `create_model`; such names now fail with the module's standard
  `ValueError` naming the property and its path. Names that merely shadow
  pydantic's *deprecated v1 shims* (`schema`, `json`, `dict`, `copy`,
  `validate`, `construct`, `parse_obj`, …) work fine as fields, so they build
  working fields — without pydantic's shadows-an-attribute warning spam.
- **`model_from_json_schema` rejects a non-dict `anyOf`/`oneOf` branch with a
  clear error.** A branch list like `["string", {"type": "null"}]` raised an
  opaque `AttributeError` (calling `.get` on the `str`); it now fails with a
  `ValueError` naming the keyword and field path.
- **Transient transport failures in a *structured* call now get the full
  transport retry budget.** instructor wraps every exhausted attempt —
  including a 429/503/network failure — in `InstructorRetryException`, which is
  in `LLM_SCHEMA_ERRORS`, so a wrapped transport error was charged the lower
  `validation_max_attempts` budget (2) instead of `max_attempts` (3) — strictly
  fewer retries than the identical error gets on the plain-text path. The retry
  layer now unwraps `InstructorRetryException` to its underlying provider error
  (`underlying_provider_error`) and routes a transport cause to the transport
  budget; a genuine schema failure still uses the validation budget.
- **A *permanent* 4xx inside a structured call now fails fast, as documented.**
  The same instructor wrapping caught a 401/400/403 too, and because the
  wrapper is in `LLM_SCHEMA_ERRORS`, a structured call with (say) a bad API key
  was *retried* on the validation budget — four provider requests plus a
  backoff sleep before the caller saw the error, while the plain-text path
  failed after one. A wrapped cause that is neither transport- nor
  schema-shaped now re-raises immediately — exactly one attempt — matching the
  fail-fast contract; an explicit `retry_on` that lists the wrapper type still
  retries, preserving caller intent.
- **The per-provider concurrency cap now binds the sync wrappers across
  threads.** Each `*_sync` call runs on a fresh event loop whose per-`(provider,
  loop)` asyncio semaphore was always uncontended, so a thread-pool fan-out of
  `structured_llm_call_sync` / `text_llm_call_sync` was effectively unlimited.
  The sync wrappers now hold a loop-agnostic per-provider `threading.Semaphore`
  in the calling thread around the whole bridged call, so cross-thread sync
  fan-out shares one cap. One honest caveat (documented in
  `llmkit.rate_limiting`): async callers on a shared loop and sync callers in
  other threads are capped on independent semaphores, so a workload mixing both
  can momentarily hold up to 2 × `max_concurrent` in-flight calls per provider;
  RPM/TPM budgets are shared across both populations.
- **The on-by-default concurrency limiter no longer raises "bound to a different
  event loop" across sync calls.** The per-provider async semaphore was cached
  in a process-global registry keyed by provider name alone, but an
  `asyncio.Semaphore` binds to the event loop it first *blocks* on. Because the
  sync bridge (`*_llm_call_sync`) runs a fresh loop per call, a saturated
  provider on one call's loop would hand its now-bound semaphore to the next
  call's loop and raise `RuntimeError` the moment it had to block. The registry
  is now keyed per `(provider, loop)` and prunes closed loops, so a contended
  cap survives the loop change. (Only surfaced under genuine contention, which
  is why it escaped the earlier per-group review.)
- **`capture_llm_records()` / `capture_llm_log_paths()` now capture across a
  `*_sync` call made from inside a running event loop.** On that path `run_sync`
  offloads to a worker thread, and a bare `executor.submit` does not propagate
  `contextvars`, so the capture buffer (held in a `ContextVar`) was invisible to
  the worker and the call's record was silently dropped. The worker now runs
  inside a copy of the caller's context, so capture (and the retry progress
  callback) cross the boundary as documented.
- **`model_from_json_schema` no longer crashes at validation time on a
  constraint keyword that doesn't match the field's type.** A length bound on a
  numeric field (or a numeric bound on a string field) was passed straight to
  pydantic's `Field`, which accepts it at build time but raises `TypeError` on
  the first validation — turning a stray keyword in an otherwise-valid schema
  into an opaque crash on the first response. Such a mismatched constraint is
  now dropped (gated by the field's resolved JSON type), honoring the
  drop-the-unsupported contract instead of crashing.
- **`model_from_json_schema` now rejects a mixed string/integer `enum` with a
  clear error** naming the construct and its path, instead of silently building
  a model that coerces members to one base and then rejects its own
  schema-valid values (e.g. integer `1` stored — and required — as `"1"`). The
  supported subset is a homogeneous string *or* integer enum.
- **`model_from_json_schema` no longer crashes on signed / colliding integer
  enums, and enum fields now dump as raw scalars.** A non-contiguous integer
  enum such as `[-1, 1, 2, 3, 4, 5]` (FiW's eval-judge schema, where `-1` is an
  "Unknown" sentinel) raised `ValueError: _sunder_ names ... are reserved`: the
  member-name builder stripped the sign, so `-1` and `1` collided on key `"1"`
  and the collision suffix bumped one to the reserved `_1_`. Member names are
  now letter-led and sign-preserving (`NEG_1`, `N_1`), so they can never form a
  reserved `_sunder_`/`__dunder__` name. Generated models also set
  `use_enum_values=True`, so a validated instance stores the raw scalar —
  `model_dump()` yields `2`, not `<Enum._2: 2>` (so a dict consumer that
  `model_dump()`s the result gets raw values). Surfaced by the FiW
  0.2.0 RC dogfood.
- **Synchronous calls no longer leak a `coroutine 'Logging.async_success_handler'
  was never awaited` `RuntimeWarning` to stderr.** LiteLLM logs successes
  asynchronously without awaiting inline — it queues the
  `async_success_handler` coroutine on a background worker — so the sync bridge
  (`structured_llm_call_sync` and the sync text path, via `run_sync`) could
  close its event loop with that coroutine still un-awaited, surfacing the
  warning on an otherwise clean call (flagged by the FiW dogfood of the 0.2.0 RC
  in both a CLI and a Lambda). `run_sync` now **drains** LiteLLM's pending async
  logging before tearing the loop down: it flushes the logging worker's queue
  (awaiting the queued handlers) and then cancels any remaining background tasks
  — bounded by the same `timeout` budget so a hung callback can't wedge the
  bridge, and applied to **both** `run_sync` branches (the `asyncio.run` path
  and the worker-thread path used when a loop is already running). The drain is
  best-effort and never converts a successful call into a failure; genuine call
  errors still propagate.
- `text_llm_call` now coerces provider **list-content** responses to a single
  string. Some providers return `message.content` as a list of content blocks
  rather than a string; the call previously returned that list verbatim,
  violating its `str` return annotation (and the README's documented coercion).
  Text blocks are now joined and non-text blocks skipped, so the return value
  is always a string.
- The Anthropic provider raised `ModuleNotFoundError: No module named
  'anthropic'` on the first call, because the SDK `instructor` requires for
  `ANTHROPIC_JSON` usage accounting was never declared as a dependency. A clean
  `pip install omg-llmkit` can now call the Anthropic provider out of the box.
- **A `*_sync` call made from inside a running event loop now honors its
  `timeout`.** On that path `run_sync` drove the worker through a
  `with ThreadPoolExecutor(...)` block whose `__exit__` calls
  `shutdown(wait=True)`, so a timed-out `future.result(timeout=...)` could not
  return until the in-flight provider request finished — the `timeout` argument
  was effectively inert in the documented async-framework case, contradicting
  the docstring. The executor is now torn down with `wait=False`, so a
  `TimeoutError` reaches the caller at the deadline while the abandoned worker
  drains in the background (one leaked non-daemon thread + request until they
  finish on their own, as the docstring already warned).
- **A streaming text call no longer dies with `IndexError` on an empty-`choices`
  chunk.** `_chunk_delta_text` indexed `chunk.choices[0]` unconditionally, but
  LiteLLM preserves an explicitly empty `choices=[]` — which the first-class
  Gemini provider emits for metadata-only / keepalive frames. The error is not
  in the retry set and, once any content has been yielded, was re-raised
  unretried, killing the stream mid-flight and discarding partial output. Such a
  frame is now skipped (treated as an empty delta).
- **A non-streaming text completion no longer crashes with `IndexError` on an
  empty-`choices` response.** `acompletion_text` read `response.choices[0]`
  before the `None` → `""` coercion, so a degenerate `{"choices": []}` body
  (e.g. from a malformed OpenAI-compatible proxy) raised instead of degrading to
  an empty string. The index is now guarded.
- **Per-call YAML logs are written as UTF-8.** The per-call log file was opened
  with `open(candidate, "x")` (platform-default encoding) while the dump emits
  raw non-ASCII (`allow_unicode=True`), so on a non-UTF-8 host any call
  containing emoji/accents/CJK raised `UnicodeEncodeError` — which, subclassing
  `ValueError`, escaped both the `OSError`/`YAMLError` handlers, bypassing
  orphan cleanup and the index write. The file is now opened with
  `encoding="utf-8"` (matching the index writer) and the handlers also catch
  `UnicodeError` so any residual encode failure still cleans up and degrades.
- **`model_from_json_schema`'s default dump no longer drops an explicitly-null
  *required* field.** The generated model's `model_dump`/`model_dump_json`
  excluded *every* `None` to keep an unset optional from round-tripping as
  `"field": null` — but that also dropped a field in the schema's `required`
  array that is legitimately nullable (`["string", "null"]` or an `anyOf` null
  branch) and set to `None`, so `model(a=None).model_dump()` returned `{}` and
  then failed re-validation with `'a' is a required property` — the exact
  footgun the exclusion exists to prevent, inverted. The drop is now scoped to
  optional fields; a required null is kept. `exclude_none=False` (keep every
  null) and `exclude_none=True` (drop every null) remain available explicitly.
- **`model_from_json_schema` accepts the canonical nullable-enum spelling.** A
  nullable enum's standard JSON Schema form carries `null` as an enum member
  (`{"type": ["string", "null"], "enum": ["a", null]}`) — JSON Schema requires
  it there for an actual `null` to validate — but the enum builder rejected any
  non-`str`/`int` member, so it failed with `Unsupported enum value None` while
  accepting the looser null-free spelling. The `null` member is now dropped once
  the field has resolved as nullable (its nullability rides the `X | None`
  union), for both the `type`-list and `anyOf`/`oneOf` shapes. A `null` member
  on a *non-nullable* field still fails loud (a type that forbids `null` but an
  enum that permits it is contradictory), as does an enum whose only member is
  `null` — with an error naming the actual defect (a nullable enum needs at
  least one non-null member), not a misleading "'enum' must be a non-empty
  list".

## [0.1.2] — 2026-06-05

### Added

- `max_tokens` parameter on `structured_llm_call` /
  `structured_llm_call_sync`, forwarded to the provider (parity with
  `text_llm_call`). Backward-compatible; defaults to no cap. When unset
  (`None`) the outbound request is byte-identical to before — no
  `max_tokens` key is sent. `LLMCallRecord` also records the cap for log
  completeness.

## [0.1.1] — 2026-06-05

### Added

- Per-call `provider` override on `structured_llm_call`,
  `structured_llm_call_sync`, `text_llm_call`, and `stream_text_with_log`.
  Passing an explicit `LLMProviderInterface` (e.g. an `OpenRouterProvider`
  built from credentials) routes that single call through a different
  provider family without changing the app-wide `configure_llm_client`
  registration; `None` (the default) preserves the existing behaviour, so
  every current caller is unchanged. The invocation log records the
  provider that actually ran.

## [0.1.0] — 2026-06-05

Initial public release.

### Added

- Provider-agnostic call surface over LiteLLM (with `instructor` for structured
  output) across OpenRouter, Google, Anthropic, and local Ollama.
- `structured_llm_call` / `structured_llm_call_sync` — validated Pydantic output,
  with each provider pinned to its native JSON-schema mode (never auto-`Mode.TOOLS`).
- `text_llm_call` and `stream_text_with_log` for plain-text and streamed calls.
- Process-global async rate limiter (`GlobalRateLimiter`, `configure_rate_limit`).
- Transient-error retries (`with_retries`, `LLM_RECOVERABLE_ERRORS`), kept
  separate from instructor's schema-repair retries.
- Agent-readable logging: `LocalYamlLogSink` writes verdict-first per-call YAML
  plus an append-only `index.jsonl`; pluggable `LogSink` protocol for custom sinks.
- Approximate per-call cost (`approximate_cost`) sourced from LiteLLM's response
  estimate, for budget visibility.

[0.7.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.7.0
[0.6.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.6.0
[0.5.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.5.0
[0.4.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.4.0
[0.3.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.3.0
[0.2.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.2.0
[0.1.2]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.2
[0.1.1]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.1
[0.1.0]: https://github.com/OMGBrews/llmkit/releases/tag/v0.1.0
