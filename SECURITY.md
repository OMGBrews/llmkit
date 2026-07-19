# Security policy

How to report a security vulnerability in `llmkit` and which versions receive
fixes. Read this before filing a security issue.

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
  local files (see the README's "Where the logs go": `LLMKIT_LOG_DIR`, else
  `data/llm-logs/` under the enclosing project root, else a per-user state
  directory). Treat that directory as sensitive if your prompts carry secrets.
  On POSIX the sink **promises** owner-only access for everything it creates:
  the directory is `0o700` and log files are `0o600`. A directory you create
  yourself is never re-chmodded (that is the escape hatch for multi-reader
  setups), and when the sink creates the project-root default location it
  seeds a `.gitignore` so prompt logs cannot be committed. The default
  **30-day retention** also acts as data minimization — logged prompts don't
  outlive their debugging value unless you opt into keeping them
  (`retention_days=None`).

## Supported versions

Only the latest released version receives fixes.
