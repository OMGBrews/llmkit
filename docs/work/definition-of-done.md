# Definition of done

The evidence a change to `llmkit` needs before it counts as done, for contributors and
maintainers alike. The shape follows the shared
[definition-of-done schema](https://github.com/OMGBrewmaster/workshop/blob/main/docs/definition-of-done-schema.md)
in the public OMG Brews Workshop repository, which also holds the
[standing rule](https://github.com/OMGBrewmaster/workshop/blob/main/docs/definition-of-done.md)
it serves.

## Repository profile

- Declared unit: `OMGBrews/llmkit`, published to PyPI as `omg-llmkit` (import name
  `llmkit`). The maintainers' work environment also pins it as a submodule
  ([`consumed-by.md`](consumed-by.md)); that environment declares its own evidence and
  nothing here is inherited from it.
- Profile: [shared or released software](https://github.com/OMGBrewmaster/workshop/blob/main/docs/verification-profiles.md#profiles).
- Production evidence: Not applicable — shared or released software.
- Default landing route: pull request, squash-merged.
- Landing enforcement: procedural.
- Enforcement evidence: observed 2026-10-08 with
  `gh api repos/OMGBrews/llmkit/rules/branches/main` — the active `protect-main` ruleset
  carries only `deletion` and `non_fast_forward`, and classic branch protection is not
  configured (`branches/main/protection` returns 404). Nothing requires a pull request or a
  green status; the requirements below hold because maintainers follow this page.
- Strengthened for: immutable PyPI releases — a published version can never be replaced,
  so a release re-runs CI at the exact tag and needs the live provider suite (see
  [Release requirements](#release-requirements)).

## Pre-commit feedback

None — no hooks are configured; the landing commands below are fast enough to run before
each commit.

## Landing requirements

Run `uv sync` once first; it installs the `dev` group that provides `ruff`, `basedpyright`
and `pytest`. Every command runs from the repository root. In CI the `check` job runs the
first four requirements in five matrix cells (`check (ubuntu-latest, 3.12)` and its
siblings), and `ci-ok` succeeds only when every CI job does.

- **Lint**
  - Command: `uv run ruff check .`
  - Pass condition: exit 0.
  - Applies to: every change outside the docs-only surface.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `check (…)` matrix cells, `ci-ok`
- **Format**
  - Command: `uv run ruff format --check .`
  - Pass condition: exit 0, using the exactly pinned `ruff` from `uv sync` — format output
    is version-sensitive, so a system `ruff` is not this evidence.
  - Applies to: every change outside the docs-only surface.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `check (…)` matrix cells, `ci-ok`
- **Types**
  - Command: `uv run basedpyright`
  - Pass condition: 0 errors, 0 warnings. The `recommended` tier runs with no baseline, so
    any new finding fails.
  - Applies to: every change outside the docs-only surface.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `check (…)` matrix cells, `ci-ok`
- **Offline tests**
  - Command: `uv run pytest`
  - Pass condition: exit 0 with no network and no credentials. New behavior needs a test.
  - Applies to: every change outside the docs-only surface.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `check (…)` matrix cells, `ci-ok`
- **Dependency floors**
  - Command: `uv sync --resolution lowest-direct && uv run pytest`
  - Pass condition: exit 0, testing against the lowest declared versions of the direct
    dependencies.
  - Applies to: any change to `pyproject.toml` or to code that may rely on a newer
    `litellm`, `instructor` or `pydantic` API than the declared floor.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `floors`, `ci-ok`
  - Run `uv sync` afterwards to restore the normal environment.
- **Wheel smoke test**
  - Command: `uv build`
  - Pass condition: the build succeeds and the built wheel, installed into a clean
    environment, imports with its `py.typed` marker, a version matching the wheel, and the
    `omg_llmkit` redirect shim. CI performs the install-and-import half.
  - Applies to: every change outside the docs-only surface.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `package`, `ci-ok`
- **Documentation links**
  - Command: `python3 scripts/check_doc_links.py`
  - Pass condition: exit 0; the final line reports that all relative links resolve.
  - Applies to: every change, including docs-only changes.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
  - CI status: `docs-links`, `ci-ok`
- **Live provider suite**
  - Command: `uv run pytest tests/integration --run-live`
  - Pass condition: every live test passes with every provider's credentials loaded. A
    missing credential is a failure, not a skip; the only allowed skips are the structural
    `importorskip`s for the optional Bedrock and Vertex extras.
  - Applies to: a change to a structured-output mode pin, a default model id, or a
    provider adapter.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available to maintainers; contributors without credentials ask a
    maintainer to run it against the pull request.
  - CI status: none

### Docs-only surface

The paths below are read by no landing check except **Documentation links**:
`ruff` and `basedpyright` read Python only (`pyrightconfig.json` includes `src`, `tests`
and `scripts`), `pytest` collects only `tests/` with no doctest globs, and the package
build reads only `README.md`, which `pyproject.toml` names as the long description — so
`README.md`, `LICENSE`, and everything under `src/`, `tests/`, `scripts/` and `.github/`
are deliberately absent. A diff confined to these paths takes the docs-only lane: run the
link check and nothing else.

<!-- DOCS-ONLY:BEGIN — paths no check in this repo reads except the docs-links check; read by Workshop's Tools/docs-only-diff.sh. DO NOT REMOVE. -->
docs/
CHANGELOG.md
CONTRIBUTING.md
PRINCIPLES.md
SECURITY.md
AGENTS.md
CLAUDE.md
<!-- DOCS-ONLY:END -->

Classify a diff with the predicate rather than by eye, from a checkout of the public
[Workshop](https://github.com/OMGBrewmaster/workshop) repository:

```bash
bash <workshop>/Tools/docs-only-diff.sh <base-sha>   # 0 = docs-only; 1 = full requirements; 2 = cannot decide
```

Exit 2 means "cannot decide" and selects the full requirements, never a pass. The
predicate exits 2 when this block is missing or malformed, and `/ship` stops on a
malformed `CONSUMED-BY` list in [`consumed-by.md`](consumed-by.md), so a damaged
machine-read block in this directory fails closed rather than silently.

## Release requirements

Maintainers cut releases; authorization and the release runbook live in the maintainers'
work environment, not in this repository.

- **Live provider suite at the release candidate**
  - Command: `uv run pytest tests/integration --run-live`
  - Pass condition: every live test passes at the commit to be tagged, with every
    provider's credentials loaded.
  - Applies to: every release.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available to maintainers
- **CI at the exact released ref**
  - Command: none — `.github/workflows/publish.yml` calls `ci.yml` on the release ref.
  - Pass condition: every CI job passes at the tagged ref before anything is built for
    upload; `publish.yml`'s `build` job needs it.
  - Applies to: every release.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available
- **Immutable version identity**
  - Command: none — performed by `publish.yml`.
  - Pass condition: the version built from git tags equals the `vX.Y.Z` release tag (or the
    `confirm_version` input of a manual dispatch), and the wheel is published to PyPI as
    `omg-llmkit` through Trusted Publishing.
  - Applies to: every release.
  - Evidence type: automated
  - Enforcement: procedural
  - Evidence state: available

## Deployment requirements

Not applicable — `llmkit` is a library; the applications that install it own their
deployments.

## Known gaps

- No landing requirement is platform-enforced. The `ci-ok` comments in
  `.github/workflows/ci.yml` describe it as the context branch protection requires, but as
  of 2026-10-08 no rule requires it (see the enforcement evidence above).
- `scripts/check_doc_links.py` checks relative file targets only: `#heading` anchors and
  external URLs are not validated.
- The live provider suite does not run in this repository's CI, because it needs provider
  credentials; its evidence comes from a maintainer's run.
- Whether a PyPI upload route exists outside `publish.yml` is unconfirmed, so the release
  requirements are recorded as procedural.

## Wrinkles

- **Plain `uv run pytest` is only half the suite.** Live tests are selected by `--run-live`
  and by nothing else — never by whether a key happens to be in the environment — so the
  same command behaves identically on every machine. Both that rule and the
  missing-key-fails rule are deliberate design, not bugs to fix.
- **`uv.lock` is gitignored**, so every CI run re-resolves from `pyproject.toml`; the weekly
  scheduled CI run re-tests the newest resolution even when nobody pushes.

## See also

- [`docs/README.md`](../README.md) — what each document under `docs/` is for
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — setup and the contribution workflow
- [`AGENTS.md`](../../AGENTS.md) — guidance for coding agents
- [`consumed-by.md`](consumed-by.md) — the maintainers' parent repository
