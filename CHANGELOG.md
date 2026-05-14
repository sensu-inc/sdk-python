# `sensu-sdk` (Python) changelog

## 0.8.0 — 2026-05-13

### Added — per-call tool I/O body capture

- **`TrackToolOptions.args: NotRequired[Any]`** — new optional field
  on the step-level `step.track_tool({ … })` call. JSON-serialized
  into `input_body` on `tool.call.completed` when `capture_bodies`
  is true.
- **`TrackToolOptions.capture_bodies: NotRequired[bool]`** — default
  `False`. When `True`, the call's `args` and the awaited result of
  `fn` are JSON-stringified and shipped on `tool.call.completed` as
  `input_body` + `output_body`. The Sensu API runs its shared PII
  pipeline at ingest and surfaces the raw bodies only via the
  audited Replay unmask flow. Per-call opt-in (not per-client) so
  storage and PII exposure are explicit decisions. See
  `planning/TOOL_IO_CAPTURE_PLAN.md §5.2` in the platform repo.
- **Top-level `SensuClient.track_tool(tool_name, fn, *, args=…, capture_bodies=…)`**
  — the convenience helper gains the two new keyword-only options.
  An internal sentinel distinguishes "args not provided" from
  "args=None" so callers retain the right to capture explicit `None`
  arguments without auto-skipping.
- **256 KB per-field cap** with the cross-SDK `' …[truncated]'`
  marker on overflow. Cross-SDK invariant: when serialization fails
  for either side (circular reference, anything `default=str` can't
  handle) BOTH body fields are skipped — never half-captured.
- **`default=str, ensure_ascii=False` JSON encoding** — `datetime`,
  `Decimal`, `UUID`, and custom objects fall back to `str(obj)`
  rather than raising. The narrower serialization-failure surface
  (vs `sdk-ts`, where these types throw and skip capture) is
  intentional — Python idiom is to lean on `__str__`.

### Changed

- No breaking changes. Default `capture_bodies` is `False`, so
  existing `track_tool` calls continue to emit the v1 metadata-only
  `tool.call.completed` event.

### Semver notes

Pre-1.0 minor bump. **Fully backward compatible.** Opting in
requires passing `capture_bodies=True` per call.

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
