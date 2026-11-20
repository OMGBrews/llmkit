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
- **The default log sink** (`LocalYamlLogSink`) is on by default and writes
  what each call attempt asked and answered to local files, in full and
  without redaction: the prompt with its system turns, tool results and
  multimodal parts (inline base64 images and files verbatim), the tool
  definitions offered and the tool calls returned with their arguments, a
  per-dispatch `requests` copy of the messages sent (including rejected
  answers that repairs send back), the answer, `partial_text`, and error text,
  which can quote what the provider sent. An inline image is stored at least
  twice per attempt, so images multiply file size. Treat the files as
  sensitive. Call `configure_llm_logging(None)`, or set `LLMKIT_LOG=off` in the
  environment, to disable the default sink, give it a
  [`transform`](README.md#redacting-records-transform) that redacts or skips
  records before they are written, or use a
  [custom sink](README.md#write-your-own-logsink) to control what is stored.
  An unrecognised `LLMKIT_LOG` value also disables it. See
  [where the logs go](README.md#where-the-logs-go) for directory selection.
  On POSIX, the sink creates the leaf log directory with mode `0o700` and new
  log files with mode `0o600`; parent directories and pre-existing files are
  not covered by this creation-mode guarantee. Existing directory permissions
  are preserved, but a shared directory alone does not grant other users read
  access to new log files. At the project-root default directory, the sink
  seeds a `.gitignore` when the directory has none, whoever created the
  directory, to reduce accidental commits; this is best effort, does not
  untrack files already committed, and does not prevent forced staging.
- **Log retention** defaults to 30 days, with best-effort cleanup after
  successful writes, at most once per hour. Inactive applications can leave
  old logs indefinitely, and deletion failures can also leave files behind.
  The active `index.jsonl` is not age-pruned. Applications requiring a fixed
  deletion deadline need their own cleanup policy. `LLMKIT_LOG_RETENTION_DAYS`
  changes the default without a code change; an invalid value turns age
  pruning off rather than deleting anything. See
  [retention details](README.md#retention-bounded-by-default), including the
  `retention_days=None` opt-out.
- **Response caches** are opt-in and can contain sensitive responses. The host
  application and its [cache store](README.md#write-your-own-store) own access
  controls, tenant isolation, and expiry. Log permissions and retention do not
  apply to cached copies.

## Supported versions

Only the latest released version receives fixes.
