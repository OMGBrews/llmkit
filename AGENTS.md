# Agent guide

Guidance for AI coding agents working in this repository. Everything here is
harness-agnostic; the human-facing context lives in [README.md](README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md), which this file points into rather than
repeats.

## What this is

`llmkit` — a thin, opinionated, local-first structured-output and logging
layer over LiteLLM. Published to PyPI as **`omg-llmkit`**; the import name is
`llmkit`. The library owns call ergonomics, structured-output mode pinning,
rate limiting, and logging; it is deliberately **not** a gateway and does not
reimplement transport. Design promises: [PRINCIPLES.md](PRINCIPLES.md).

Layout: `src/llmkit/` is the package; `tests/` is the offline suite;
`tests/integration/` is the live half (see below before touching it).

## Setup

```bash
uv sync            # installs the dev group; the [bedrock]/[vertex] extras come with it
```

## Gates — all four must pass before a PR

CI runs these four gates on every push and pull request — the ones to run
before opening a PR (it also runs a lowest-versions resolution job, a wheel
smoke test, and a weekly unlocked-resolution run that rarely concern a PR
author; [CONTRIBUTING.md](CONTRIBUTING.md) has the full context):

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest
```

Notes that bite agents:

- `basedpyright` runs in its `recommended` tier with **no baseline** — any
  new finding fails CI. Prefer precise types; when a suppression is forced,
  it must name the rule and say why (the `raw-*` tag convention is in
  CONTRIBUTING).
- Plain `uv run pytest` runs only the offline half of the suite: green with
  no network and no credentials, everywhere.
- New behavior needs a test.

## Live tests — read before touching `tests/integration/`

The live tests make real provider calls and run **only** under
`pytest --run-live`, which needs provider credentials you almost certainly
don't have. Two rules protect the suite's design — do not "fix" either as if
it were a bug:

- Live tests never run merely because a key is present in the environment;
  only the explicit `--run-live` flag selects them.
- Under `--run-live`, a missing key is a **hard failure, not a skip**. The
  only allowed skips are structural (`importorskip` on an absent optional
  extra: Bedrock, Vertex). This fail-loud behavior is deliberate.

Provider behavior (mode pins, default model ids) is measured against live
APIs, not assumed — a change to a pin needs a live measurement behind it,
which in practice means a maintainer runs the live suite against your PR.

## Conventions

- `main` is linear: pull requests are **squash-merged**, one commit per
  change. Force pushes to `main` are blocked.
- Keep the public surface small — `llmkit` owns call ergonomics, not
  transport.
- No `dict[str, Any]` / bare `Any`; use precise types.
- Hard cuts over deprecation shims for internal changes.
