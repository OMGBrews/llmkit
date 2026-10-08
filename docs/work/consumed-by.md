# Consumed by

The repository that records this one as a submodule. It matters only to OMG Brews
maintainers; outside contributors and users of the published `omg-llmkit` package can
ignore it. The list between the sentinels is read by the shared `/ship` skill and is
edited by hand.

<!-- CONSUMED-BY:BEGIN — DO NOT REMOVE: /ship parses the parent list between these sentinels. -->
OMGBrews/llmkit-dev
<!-- CONSUMED-BY:END -->

`OMGBrews/llmkit-dev` is the maintainers' private work environment. It records this
repository at `library` and moves that pointer separately, after a change here merges.
The pointer may name only a commit reachable from this repository's `main`, and it never
moves backwards. The wrapper's `pointer-check.yml` workflow checks and reports both on
pull requests and pushes; it does not block them. The reasoning, history, and tooling for
pointer bumps live in that repository, not here.

## See also

- [`definition-of-done.md`](definition-of-done.md) — the evidence a change here needs
  before it lands
- [`docs/README.md`](../README.md) — what each document under `docs/` is for
