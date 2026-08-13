# Definition of done

The gates a change to `llmkit` must pass. This is the deliverable library — published to
PyPI as `omg-llmkit`, import name `llmkit` — and its offline gates are exactly what CI runs
on every push and pull request. The shared convention behind this file, including the
exit-code-safe way to run any command below, lives in the OMG Brews `devtools` repo at
`docs/definition-of-done.md`; this repo carries no `devtools/` checkout to link into.

Run `uv sync` once first — it installs the `dev` group, which provides all three tools.
Every command runs from the repo root.

| Gate | Command | Pass condition | Applies to |
|------|---------|----------------|------------|
| Lint | `uv run ruff check .` | Exit 0. | Any Python change. |
| Format | `uv run ruff format --check .` | Exit 0. | Any Python change. |
| Types | `uv run basedpyright` | 0 errors, 0 warnings. | Any Python change. The `recommended` tier with no baseline, so any new finding fails. |
| Offline tests | `uv run pytest` | Exit 0. | Any Python change; new behavior needs a test. Green with no network and no credentials, everywhere. A README or docstring-only diff does not need it. |
| Live provider suite | `uv run pytest tests/integration --run-live` | Every live test passes. | A change to a mode pin, a default model id, or a provider adapter. Needs real provider credentials, so in practice a maintainer runs it against the PR. |

## The prose surface no gate reads

Not one gate above opens a Markdown file. `ruff check`/`ruff format` process Python only;
`pyrightconfig.json` scopes the type-check to `include: ["src", "tests"]`; pytest runs
`testpaths = ["tests"]` with no `--doctest-glob`, no `--doctest-modules`, and no test that
reads a repo-relative document (every filesystem read in the suite is under `tmp_path`); and
nothing generates code from prose. The one document a gate *does* read is **`README.md`** —
`pyproject.toml` names it as `readme`, so hatchling parses it into the wheel's long
description and CI's wheel build-and-import job would fail on a diff that moved or deleted
it. README is therefore deliberately absent from the list, as are `LICENSE` (`license-files`)
and every path under `src/`, `tests/`, and `.github/`.

<!-- DOCS-ONLY:BEGIN — paths no gate in this repo reads; read by the OMG Brews devtools repo's Tools/docs-only-diff.sh. DO NOT REMOVE. -->
docs/
CHANGELOG.md
CONTRIBUTING.md
PRINCIPLES.md
SECURITY.md
AGENTS.md
CLAUDE.md
<!-- DOCS-ONLY:END -->

Ask the predicate rather than judging a diff by eye:

```bash
bash <devtools>/Tools/docs-only-diff.sh <base-sha>   # exit 0 = docs-only; 1 = run the gates; 2 = undecidable
```

It is the single definition of the docs-only test, and it lives in the OMG Brews `devtools`
repo, which **this repo does not vendor** — so the fast lane is available only to a caller
that already has a devtools checkout on disk (a maintainer working from the `llmkit-dev`
wrapper, say). Anyone else runs the gates. Exit 2 means "cannot decide" and must be treated
as "not docs-only", never as a pass.

## Wrinkles

- **Plain `uv run pytest` is only half the suite.** Live tests are selected by `--run-live`
  and by nothing else — never by whether a key happens to be in the environment — so the
  same command behaves identically on every machine. Under the flag a missing key is a hard
  failure, not a skip; the only allowed skips are structural `importorskip`s for the
  optional Bedrock and Vertex extras. Both rules are deliberate design, not bugs to fix.
- **`ruff` is pinned exactly** (`ruff==0.15.0` in `pyproject.toml`) because
  `ruff format --check` output is version-sensitive. Run it through `uv`, not a system
  ruff, or the format gate is not reproducible.
- **Green gates are not all of CI.** `uv.lock` is gitignored, so every run re-resolves from
  `pyproject.toml`, and CI adds a `--resolution lowest-direct` floors job plus a wheel
  build-and-import smoke test. A change that starts relying on a newer-than-floor API of
  litellm, instructor, or pydantic passes every gate above and still fails CI.
- **This repo keeps no task buckets.** Planning, task documents, and maintainer runbooks
  live in `OMGBrews/llmkit-dev`, which mounts this repo as a submodule
  ([`consumed-by.md`](consumed-by.md)); `docs/work/` here holds this file and the
  consumed-by list, and nothing else.
