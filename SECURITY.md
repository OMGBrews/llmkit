# Security policy

How to report a security vulnerability in `llmkit` and which versions receive
fixes. Read this before filing a security issue.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/OMGBrews/llmkit/security/advisories/new)
(the **Security** tab → **Report a vulnerability**). This opens a private
advisory shared with the reporter, maintainers, and invited collaborators.

Include affected versions, reproduction steps, and the potential impact. Use
sanitized examples rather than live credentials or sensitive prompts.

This is a small, best-effort project. There is no formal SLA, but reports are
taken seriously and acknowledged as soon as practical.

## Scope

`llmkit` is a thin client layer that runs in your application's process, not a
network service. The most security-relevant surfaces are:

- **Provider API keys** passed through `LLMClientConfig` are held in configuration
  and provider objects in your process and passed to the provider transport.
- **The default log sink** (`LocalYamlLogSink`) writes prompts and responses to
  local files by default. Treat them as sensitive. Call
  `configure_llm_logging(None)` to disable the sink, or use a
  [custom sink](README.md#write-your-own-logsink) to control what is stored.
  See [where the logs go](README.md#where-the-logs-go) for directory selection.
  On POSIX, the sink creates the leaf log directory with mode `0o700` and new
  log files with mode `0o600`; parent directories and pre-existing files are
  not covered by this creation-mode guarantee. Existing directory permissions
  are preserved, but a shared directory alone does not grant other users read
  access to new log files. When creating the project-root default directory,
  the sink attempts to seed a `.gitignore` to reduce accidental commits; this
  is best effort and does not prevent forced staging.
- **Log retention** defaults to 30 days, with best-effort cleanup after
  successful writes, at most once per hour. Inactive applications can leave
  old logs indefinitely, and deletion failures can also leave files behind.
  The active `index.jsonl` is not age-pruned. Applications requiring a fixed
  deletion deadline need their own cleanup policy. See
  [retention details](README.md#retention-bounded-by-default), including the
  `retention_days=None` opt-out.
- **Response caches** are opt-in and can contain sensitive responses. The host
  application and its [cache store](README.md#write-your-own-store) own access
  controls, tenant isolation, and expiry. Log permissions and retention do not
  apply to cached copies.

## Supported versions

Only the latest released version receives fixes.
