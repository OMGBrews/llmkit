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

This edge is the fleet's one exception to "pointers move only when someone asks." Since
the scheduled bump bot was retired 2026-07-28 that is true everywhere else, but the
wrapper keeps a repo-local backstop: `.github/dependabot.yml` rolls the `library` pointer
to this repo's `main` daily, and `dependabot-automerge.yml` lands a green roll without a
human click. The backstop bounds the lag at about a day; it does not make the bump
semantic or immediate, and a red roll simply stays open. So the deliberate bump still
happens: `/ship` includes the parent pointer-bump in its plan **by default** when a
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
