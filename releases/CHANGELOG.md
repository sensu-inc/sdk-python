# Changelog

All notable changes to `senzu-sdk` are documented here.

Format: each release has a **Breaking Changes**, **Added**, **Changed**, and **Fixed** section. Sections are omitted when empty.

---

## [0.5.3] — 2026-05-07

### Added
- Initial public release of the Python SDK.
- `SenzuClient` — core telemetry client with batched event flushing, async context propagation via `contextvars.ContextVar`, thread-safe buffer, and `atexit` process-exit flush.
- `RunHandle` — represents an agent run; exposes `start_step()`, `record_feedback()`, `record_eval_score()`, `handoff()`, `end()`, and `async with` context manager support.
- `StepHandle` — represents a step within a run; exposes `track_llm()`, `track_streaming_llm()`, `record_llm()`, `track_tool()`, `track_retrieval()`, `record_retrieval()`, `track_embedding()`, `record_embedding()`, `track_guardrail()`, `record_guardrail()`, `record_prompt_render()`, `end()`, and `async with` support.
- `senzu.run()` high-level API — wraps an async function with automatic run start/end and ContextVar propagation so nested `track_*` calls resolve the active run implicitly.
- Client-level shortcut methods — `track_tool()`, `track_retrieval()`, `track_embedding()`, `track_guardrail()` auto-find the active run from context.
- Multi-agent support — `spawn_run()` (shared trace and session) and `handoff()`.
- Session management — `start_session()` and `resume_session()`.
- Prompt management — `deploy_prompt_version()` and `StepHandle.record_prompt_render()`.
- Loop detection — configurable `loop_threshold` and `on_loop_detected` callback.
- Dynamic pricing — `resolve_pricing()` fetches live rates from `GET /api/v1/pricing/models/{provider}/{model}` with per-client session cache; returns a `(0.0, 0.0)` sentinel when unavailable rather than fabricating a number.
- `senzu.integrations.anthropic` — `wrap_anthropic()` patches `client.messages.create()` to auto-track LLM calls including cache token counts.
- `senzu.integrations.openai` — `wrap_openai()` patches `client.chat.completions.create()` to auto-track LLM calls.
- `senzu.integrations.langchain` — `SenzuCallbackHandler` for LangChain chains and agents; tracks LLM calls, tool calls, streaming TTFT, and retry detection.
- Full `TypedDict` type annotations throughout; compatible with Python 3.9+.
- GitHub Actions workflow for Trusted Publishing to PyPI on `sdk-python/v*` tag push.

---

## How to release

1. Update `version` in [pyproject.toml](../pyproject.toml) and `__version__` in [senzu/\_\_init\_\_.py](../senzu/__init__.py).
2. Add a release entry at the top of this file.
3. Commit: `git commit -m "chore(sdk-python): release vX.Y.Z"`
4. Tag and push: `git tag sdk-python/vX.Y.Z && git push origin sdk-python/vX.Y.Z`
5. GitHub Actions builds and publishes to PyPI automatically via Trusted Publishing.

## Version policy

This SDK follows [Semantic Versioning](https://semver.org):

- **Patch** (0.5.x) — bug fixes, non-breaking additions (new optional fields, new event types).
- **Minor** (0.x.0) — new features, new methods, new integrations. Backwards compatible.
- **Major** (x.0.0) — breaking changes to public method signatures or wire format.
