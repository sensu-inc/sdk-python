# `sensu-sdk` (Python) changelog

## 0.12.3 — 2026-05-20

### Added — pricing cache TTL

New `pricing_cache_ttl_ms` option on `SensuClientOptions`. Default
**1 hour** (3,600,000 ms). After expiry the next `resolve_pricing()`
call refetches from the live API and replaces the cached entry. Set
`0` to disable caching entirely.

Parity with sdk-ts `pricingCacheTtlMs` and sdk-go
`PricingCacheTTLMs`. Closes v1.1 follow-on #1 from
[`[DONE]_SDK_CONSOLIDATION_PLAN.md` §9](https://github.com/sensu-inc/sensu/blob/main/planning/%5BDONE%5D_SDK_CONSOLIDATION_PLAN.md).

`resolve_pricing()` also gains an optional `now_monotonic: float`
parameter for deterministic TTL testing without monkey-patching
`time.monotonic()`. Production callers leave it `None` (default
`time.monotonic()` is used).

5 new pytest cases covering TTL expiry, rate updates across the
expiry boundary, `cache_ttl_ms=0` disables caching, per-(provider,
model) independence, and client → resolver round-trip with default
+ custom + zero values.

Patch bump: 0.12.2 → 0.12.3.

## 0.12.2 — 2026-05-20

### Fixed — capture_message_bodies missing from SensuClientOptions TypedDict

The runtime client has always honored `capture_message_bodies` (see
[REPLAY_V1_PLAN.md §7](https://github.com/sensu-inc/sensu/blob/main/planning/REPLAY_V1_PLAN.md)
in the platform repo) but the field was missing from the public
`SensuClientOptions` TypedDict — so mypy/pyright didn't surface it
in autocomplete, and customers had to discover it from internals.

Caught during the cross-SDK parity audit
([SDK_CONSOLIDATION_PLAN.md](https://github.com/sensu-inc/sensu/blob/main/planning/SDK_CONSOLIDATION_PLAN.md)
Phase 2 PR 2) — sdk-ts and sdk-go both exposed the option in their
public type, Python was the outlier.

**Behavior:** unchanged. Existing code that passed the option keeps
working; code that didn't was already getting the `False` default.
This release just makes the option discoverable.

```python
client = sensu.SensuClient({
    "api_key": "...",
    "capture_message_bodies": True,  # now type-checked
})
```

5 new pytest cases in `tests/test_client_options.py`:
- TypedDict declares `capture_message_bodies` (load-bearing
  type-checker discoverability assertion + sentinel against
  accidental removal)
- TypedDict declares the full expected key set (drift sentinel)
- runtime round-trip True / False
- runtime default False

25/25 pass across capture_message_bodies + pricing + client_options +
register_agent_version. No regressions.

Patch bump: 0.12.1 → 0.12.2.

## 0.12.1 — 2026-05-20

### Changed — surface pricing failures via UserWarning

The pricing resolver already returned `(0.0, 0.0)` on API failure
(unlike sdk-ts and sdk-go, Python never shipped a bundled fallback).
The platform [SDK_CONSOLIDATION_PLAN.md §3c](https://github.com/sensu-inc/sensu/blob/main/planning/SDK_CONSOLIDATION_PLAN.md)
formalizes that behavior as the new design across all three SDKs.
This release surfaces the failure so customers can tell when costs
are zeros:

- `resolve_pricing()` now emits a `UserWarning` on each failure path
  (4xx/5xx, network error, 200 with null rates, `disable_live_pricing`,
  client `disabled`, missing API key) — **at most once per
  `(provider, model)` per client lifetime** so logs don't spam.
- The warning message points customers to
  `POST /api/v1/pricing/org-models` for registering custom-model
  pricing (the new self-serve path shipped on the platform).
- New optional `warned: Set[str]` parameter on `resolve_pricing()`
  (service-level callers that don't pass it get silent sentinels —
  backward-compatible).

**No behavior change** for callers who already handled `(0.0, 0.0)`
as "no estimate" — those keep working unchanged. The server's
ingest pipeline reconciles cost from `llm_calls` + the catalog at
query time regardless, so dashboards stay correct even when the
SDK sends 0.

9 new pytest cases covering success cache, 4xx/5xx, network error,
null-rates, three short-circuit paths, warn-at-most-once semantics,
and per-(provider, model) warning isolation.

## 0.12.0 — 2026-05-19

### Added — agent version registry for eval-gated CI/CD (§5.2)

- **`client.register_agent_version({...})`** — new run-less async
  helper that wraps `POST /api/v1/agents/:id/versions`. Lets
  customers register the candidate config (system prompt + optional
  model) used at a given commit, then reference the returned
  versionId from the Sensu eval-gate Action instead of inlining the
  full config in every PR check.
- New exported TypedDicts: `CandidateConfig`,
  `RegisterAgentVersionOptions`, `AgentVersion`.
- Owner/admin role required server-side (the registration represents
  a deploy fact); an API key with `full` scope works as expected. See
  the platform repo's `planning/EVAL_GATED_CI_PLAN.md` PR 5 for the
  matching backend.

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
