# Consumed by

The superproject that records this repository as a submodule, and why a merged change
here is not quite the end of the job. `/ship` reads the machine-readable list between the
sentinels every time; the prose is for whoever has to decide what to do about a lagging
pointer. Nothing here affects users of the published `omg-llmkit` package — it is about
the OMG Brews work environment this library is maintained from.

<!-- CONSUMED-BY:BEGIN — DO NOT REMOVE: /ship parses the parent list between these sentinels. -->
OMGBrews/llmkit-dev
<!-- CONSUMED-BY:END -->

`OMGBrews/llmkit-dev` is the maintainers' wrapper — planning, task buckets, and the
release runbook, none of which ever land here — and it records this repository at
`library` with `branch = main`. The gitlink may only ever name a commit reachable from
this repo's `main`, never a PR-branch SHA (squash merges orphan those) and never moving
backwards; the wrapper's `.github/workflows/pointer-check.yml` enforces both on every PR
and push.

Nothing moves that pointer on a schedule. The wrapper ran a daily Dependabot roll as a
backstop until 2026-08-07 and retired it, so this edge now follows the same rule as every
other one in the maintainers' fleet: **pointers move only when someone asks.** A wrapper
pointer that lags this repo's `main` is therefore the expected state between deliberate
bumps, not a fault — it means nobody has recorded the change upward yet, and it is
reported rather than healed. The deliberate bump is what closes it:
`/ship` includes the parent pointer-bump in its plan **by default** when a
session ships this repo without `OMGBrews/llmkit-dev` attached — the maintainer strikes
it to decline, and the decline is recorded in the ship report rather than the parent
being reported as current. In practice that path runs from the wrapper, which mounts
this repo and carries the shared skills; this repository has no `.claude/` of its own.
Locally, hq's `./scripts/sync/push.sh` sweep records it.

A lagging pointer is expected and safe; an unrecorded one is invisible, which is the
failure this file exists to prevent (found-in-words cms, 2026-07-21: 14 commits and 11
task documents invisible behind a stale pointer, in a repo with no such backstop).

The list is edited by hand. Adding a consumer is a human act, and this file is where it is
written down.
