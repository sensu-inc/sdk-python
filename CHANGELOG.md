# `sensu-sdk` (Python) changelog

## 0.7.0 — 2026-05-11

### Added — opt-in message-body capture for Replay v1

- **`MessageSnapshotItem.body: NotRequired[str]`** — new optional field
  on the message snapshot TypedDict. Existing callers that don't set
  it see no behavior change.
- **`capture_message_bodies` client option** — default `False`. When
  `True`, raw message bodies on `messages_snapshot` are forwarded to
  the Sensu API on each LLM call. The API masks PII via its shared
  pipeline at ingest, stores the masked form for display, and keeps
  raw bodies tenant-side for the Replay scrubber's audited unmask flow.
- **`SensuClient.sanitize_messages_snapshot()`** — the wire sanitizer
  used by `track_llm()` before the snapshot is flushed. Strips `body`
  when `capture_message_bodies` is `False`; otherwise caps body length
  at the server schema limit of 65,536 chars.

### Semver notes

Pre-1.0 minor bump. **Fully backward compatible** — the default for
`capture_message_bodies` is `False`, so existing SDK callers that
never sent `body` on `messages_snapshot` continue to see exactly the
same wire payload they did under 0.6.x. Opting in is a deliberate
per-client config change.

## 0.6.0 and earlier

Pre-changelog. See git history.
