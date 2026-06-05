# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/OMGBrews/llmkit/security/advisories/new)
(the **Security** tab → **Report a vulnerability**). This opens a private
advisory visible only to the maintainers.

This is a small, best-effort project. There is no formal SLA, but reports are
taken seriously and acknowledged as soon as practical.

## Scope

`llmkit` is a thin client layer; it holds no credentials of its own and runs no
network services. The most security-relevant surfaces are:

- **Provider API keys** passed through `LLMClientConfig` — these live in your
  process, never in `llmkit`.
- **The default log sink** (`LocalYamlLogSink`) writes prompts and responses to
  local files under `data/llm-logs/`. Treat that directory as sensitive if your
  prompts carry secrets, and add it to `.gitignore` (the default for this repo).

## Supported versions

Only the latest released version receives fixes.
