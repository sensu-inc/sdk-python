from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
import time
import warnings
from contextvars import ContextVar
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional, Set, Tuple, TypeVar

import httpx

from sensu._pricing import estimate_cost, resolve_pricing
from sensu._types import (
    AgentVersion,
    CandidateConfig,
    ContextBreakdown,
    DeployPromptVersionOptions,
    GuardrailResult,
    GuardrailType,
    HandoffOptions,
    RawEmbeddingOptions,
    RawGuardrailOptions,
    RawLlmCallOptions,
    RawRetrievalOptions,
    FeedbackOptions,
    RecordEvalScoreOptions,
    RecordFeedbackOptions,
    RecordPromptRenderOptions,
    RegisterAgentVersionOptions,
    ScoreOptions,
    ResumeSessionOptions,
    SpawnRunOptions,
    StartRunOptions,
    StartSessionOptions,
    StartStepOptions,
    TrackEmbeddingOptions,
    TrackGuardrailOptions,
    TrackLlmOptions,
    TrackRetrievalOptions,
    TrackStreamingLlmOptions,
    TrackToolOptions,
)
from sensu._utils import estimate_bytes, extract_stream_chunk_text, extract_usage, format_debug_event, new_id, utcnow_iso

T = TypeVar("T")

# Module-level ContextVar — one per process; task-isolated under asyncio
_active_run_var: ContextVar["RunHandle"] = ContextVar("sensu_active_run")


# ---------------------------------------------------------------------------
# Tool I/O body capture (TOOL_IO_CAPTURE_PLAN.md §5.2 + §11.3 + §5.4)
# ---------------------------------------------------------------------------

# 256 KB per field. Wider than the LLM message-body cap (64 KB) because
# real tool outputs — JSON manifests, HTML excerpts, search results —
# routinely run past 64 KB. The Sensu API enforces the same cap
# defensively via ``z.string().max(262144)`` on the tool.call.completed
# schema.
_MAX_TOOL_BODY_CHARS = 262_144

# Cross-SDK truncation marker (§5.4). Leading space is intentional so
# the marker lands cleanly on a word boundary in the inspector. Same
# byte sequence as ``sdk-ts`` and ``sdk-go`` — keeps the on-the-wire
# shape uniform across languages.
_TRUNCATION_MARKER = " …[truncated]"

# Sentinel for "no args passed". Using a unique object (rather than
# ``None``) preserves the user's right to explicitly pass ``None`` as
# the tool's input — that serializes to ``"null"`` and is captured.
# Cross-SDK parity with sdk-ts, where omitted args (undefined) skip
# capture but an explicit ``null`` argument does not.
_ARGS_NOT_PROVIDED = object()


def serialize_tool_bodies_for_capture(
    args: Any,
    result: Any,
    *,
    capture_bodies: bool,
    args_provided: bool,
) -> Dict[str, str]:
    """Serialize tool I/O bodies for transport on tool.call.completed.

    Implements TOOL_IO_CAPTURE_PLAN.md §5.2 + §11.4. Returns a dict
    suitable for splatting into the event payload:

    - opt-out (capture_bodies=False) → empty dict; neither body field
      emitted. Matches v1 metadata-only behavior.
    - opt-in but caller omitted ``args`` → empty dict. Cross-SDK
      parity with sdk-ts (undefined args → skip capture).
    - opt-in + both sides JSON-serialize cleanly → dict with
      ``input_body`` and ``output_body``, each ≤ 256 KB.
    - opt-in + serialization fails for either side (circular reference,
      anything ``default=str`` can't handle) → empty dict. Skip BOTH
      bodies rather than half-capturing — keeps the server's
      "snapshotMissing" affordance coherent (§11.4 lean A).

    ``json.dumps(default=str)`` (per §5.2) means datetime / Decimal /
    UUID / custom objects fall back to ``str(obj)`` instead of raising.
    The narrower failure surface (vs sdk-ts, where TypeError on these
    types skips capture) is intentional — Python's idiom is to lean on
    ``__str__`` rather than refuse the call.

    Exported for unit tests so the serialization rules can be pinned
    without standing up a full client + run + step + mock httpx.
    """
    if not capture_bodies or not args_provided:
        return {}
    try:
        input_body  = json.dumps(args,   default=str, ensure_ascii=False)
        output_body = json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return {}
    return {
        "input_body":  _truncate_tool_body_for_transport(input_body),
        "output_body": _truncate_tool_body_for_transport(output_body),
    }


def _truncate_tool_body_for_transport(s: str) -> str:
    if len(s) <= _MAX_TOOL_BODY_CHARS:
        return s
    return s[: _MAX_TOOL_BODY_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


# ---------------------------------------------------------------------------
# StepHandle
# ---------------------------------------------------------------------------


class StepHandle:
    def __init__(
        self,
        client: "SensuClient",
        *,
        step_id: str,
        run_id: str,
        session_id: str,
        agent_id: str,
        org_id: str,
        trace_id: str,
        span_id: str,
        sequence: int,
        name: Optional[str] = None,
        step_type: Optional[str] = None,
    ) -> None:
        self._client = client
        self.step_id = step_id
        self.run_id = run_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.org_id = org_id
        self.trace_id = trace_id
        self.span_id = span_id
        self._sequence = sequence
        self._name = name
        self._step_type = step_type

    def _base(self, **extra: Any) -> Dict[str, Any]:
        return {
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "org_id": self.org_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            **extra,
        }

    # -- LLM tracking --------------------------------------------------------

    async def track_llm(self, opts: TrackLlmOptions) -> Any:
        provider = opts["provider"]
        model = opts["model"]
        fn = opts["fn"]
        llm_call_id = opts.get("llm_call_id") or new_id()
        max_ctx = opts.get("max_context_tokens")
        msgs = opts.get("messages_snapshot")
        chunk_ids = opts.get("referenced_chunk_ids")
        extract_ctx = opts.get("extract_context_breakdown")

        self._client.enqueue(self._base(
            event_type="llm.request.started",
            provider=provider,
            model=model,
            llm_call_id=llm_call_id,
        ))

        start_ms = time.monotonic() * 1000
        result: Any = None
        status = "success"
        exc: Optional[BaseException] = None

        try:
            result = await fn()
        except Exception as e:
            status = "error"
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms
        usage = extract_usage(result, model) if result is not None else {}

        # Resolve live pricing from the API
        if usage.get("input_tokens") is not None:
            input_price, output_price = await self._client.resolve_pricing(provider, model)
            cost = estimate_cost(input_price, output_price, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            if cost is not None:
                usage["cost_usd_estimate"] = cost

        ctx_breakdown: Optional[ContextBreakdown] = None
        if extract_ctx and result is not None:
            ctx_breakdown = extract_ctx(result)

        event: Dict[str, Any] = self._base(
            event_type="llm.request.completed",
            provider=provider,
            model=model,
            llm_call_id=llm_call_id,
            latency_ms=latency_ms,
            status=status,
            **usage,
        )
        if max_ctx is not None:
            event["max_context_tokens"] = max_ctx
        if ctx_breakdown is not None:
            event["context_breakdown"] = ctx_breakdown
        if msgs is not None:
            event["messages_snapshot"] = self._client.sanitize_messages_snapshot(msgs)
        if chunk_ids is not None:
            event["referenced_chunk_ids"] = chunk_ids

        self._client.enqueue(event)
        self._client.notify_tool_call(self.run_id, "__llm__")

        if exc is not None:
            raise exc
        return result

    async def track_streaming_llm(self, opts: TrackStreamingLlmOptions) -> str:
        provider = opts["provider"]
        model = opts["model"]
        stream = opts["stream"]
        llm_call_id = opts.get("llm_call_id") or new_id()
        emit_every = opts.get("emit_every_n_tokens", 10)
        on_complete = opts.get("on_complete")
        max_context_tokens = opts.get("max_context_tokens")

        start_ms = time.monotonic() * 1000
        first_token_ms: Optional[float] = None
        token_count = 0
        text_parts: List[str] = []

        async for chunk in stream:
            text = extract_stream_chunk_text(chunk)
            if text:
                if first_token_ms is None:
                    first_token_ms = time.monotonic() * 1000
                text_parts.append(text)
                token_count += 1
                if token_count % emit_every == 0:
                    ttft = (first_token_ms - start_ms) if first_token_ms else None
                    self._client.enqueue(self._base(
                        event_type="stream.token.received",
                        llm_call_id=llm_call_id,
                        tokens_so_far=token_count,
                        ttft_ms=ttft,
                    ))

        full_text = "".join(text_parts)
        latency_ms = time.monotonic() * 1000 - start_ms
        ttft_ms = (first_token_ms - start_ms) if first_token_ms else None

        completion: Dict[str, Any] = self._base(
            event_type="llm.request.completed",
            provider=provider,
            model=model,
            llm_call_id=llm_call_id,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status="success",
        )
        if max_context_tokens is not None:
            completion["max_context_tokens"] = max_context_tokens
        self._client.enqueue(completion)

        if on_complete:
            on_complete(full_text, ttft_ms)

        return full_text

    def record_llm(self, opts: RawLlmCallOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="llm.request.completed",
            provider=opts["provider"],
            model=opts["model"],
        )
        for key in (
            "input_tokens", "output_tokens", "cached_input_tokens", "total_tokens",
            "max_context_tokens", "context_used_tokens", "latency_ms", "ttft_ms",
            "cost_usd_estimate", "status", "context_breakdown", "referenced_chunk_ids",
        ):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    # -- Tool tracking -------------------------------------------------------

    async def track_tool(self, opts: TrackToolOptions) -> Any:
        tool_name = opts["tool_name"]
        fn = opts["fn"]
        retry_of = opts.get("retry_of")
        tool_call_id = new_id()

        event: Dict[str, Any] = self._base(
            event_type="tool.call.started",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        if retry_of:
            event["retry_of"] = retry_of
        self._client.enqueue(event)
        self._client.notify_tool_call(self.run_id, tool_name)

        start_ms = time.monotonic() * 1000
        result: Any = None
        status = "success"
        exc: Optional[BaseException] = None

        try:
            result = await fn()
        except Exception as e:
            status = "error"
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms
        output_bytes = estimate_bytes(result) if result is not None else 0

        body_fields = serialize_tool_bodies_for_capture(
            opts.get("args"),
            result,
            capture_bodies=bool(opts.get("capture_bodies", False)),
            args_provided="args" in opts,
        )

        self._client.enqueue(self._base(
            event_type="tool.call.completed",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            latency_ms=latency_ms,
            status=status,
            output_size_bytes=output_bytes,
            **body_fields,
        ))

        if exc is not None:
            raise exc
        return result

    # -- Retrieval tracking --------------------------------------------------

    async def track_retrieval(self, opts: TrackRetrievalOptions) -> Any:
        fn = opts["fn"]
        vector_store_id = opts.get("vector_store_id")
        top_k = opts.get("top_k")

        start_ms = time.monotonic() * 1000
        result: Any = None
        status = "success"
        exc: Optional[BaseException] = None

        try:
            result = await fn()
        except Exception as e:
            status = "error"
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms

        event: Dict[str, Any] = self._base(
            event_type="retrieval.completed",
            latency_ms=latency_ms,
            status=status,
        )
        if vector_store_id:
            event["vector_store_id"] = vector_store_id
        if top_k is not None:
            event["top_k"] = top_k

        # Support { result, chunks } return shape
        actual_result = result
        if isinstance(result, dict) and "result" in result and "chunks" in result:
            actual_result = result["result"]
            chunks = result["chunks"]
            event["chunks_returned"] = len(chunks)
            event["chunks"] = chunks

        self._client.enqueue(event)

        if exc is not None:
            raise exc
        return actual_result

    def record_retrieval(self, opts: RawRetrievalOptions) -> None:
        event: Dict[str, Any] = self._base(event_type="retrieval.completed")
        for key in (
            "vector_store_id", "top_k", "latency_ms", "chunks_returned",
            "tokens_injected", "similarity_score_avg", "status", "chunks",
        ):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    # -- Embedding tracking --------------------------------------------------

    async def track_embedding(self, opts: TrackEmbeddingOptions) -> Any:
        fn = opts["fn"]
        model = opts["model"]
        input_text_length = opts.get("input_text_length")
        batch_size = opts.get("batch_size")

        start_ms = time.monotonic() * 1000
        result: Any = None
        exc: Optional[BaseException] = None

        try:
            result = await fn()
        except Exception as e:
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms

        event: Dict[str, Any] = self._base(
            event_type="embedding.created",
            model=model,
            latency_ms=latency_ms,
        )
        if input_text_length is not None:
            event["input_text_length"] = input_text_length
        if batch_size is not None:
            event["batch_size"] = batch_size
        self._client.enqueue(event)

        if exc is not None:
            raise exc
        return result

    def record_embedding(self, opts: RawEmbeddingOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="embedding.created",
            model=opts["model"],
        )
        for key in ("input_text_length", "token_count", "latency_ms", "cost_usd_estimate", "batch_size"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    # -- Guardrail tracking --------------------------------------------------

    async def track_guardrail(self, opts: TrackGuardrailOptions) -> GuardrailResult:
        guardrail_id = opts["guardrail_id"]
        fn = opts["fn"]
        guardrail_type = opts.get("guardrail_type")
        input_hash = opts.get("input_hash")

        start_evt: Dict[str, Any] = self._base(
            event_type="guardrail.check.started",
            guardrail_id=guardrail_id,
        )
        if guardrail_type:
            start_evt["guardrail_type"] = guardrail_type
        if input_hash:
            start_evt["input_hash"] = input_hash
        self._client.enqueue(start_evt)

        start_ms = time.monotonic() * 1000
        guard_result: GuardrailResult = "pass"
        exc: Optional[BaseException] = None

        try:
            guard_result = await fn()
        except Exception as e:
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms

        done_evt: Dict[str, Any] = self._base(
            event_type="guardrail.check.completed",
            guardrail_id=guardrail_id,
            result=guard_result,
            latency_ms=latency_ms,
            blocked=guard_result == "fail",
        )
        if guardrail_type:
            done_evt["guardrail_type"] = guardrail_type
        self._client.enqueue(done_evt)

        if guard_result == "fail":
            self._client.enqueue(self._base(
                event_type="guardrail.blocked",
                guardrail_id=guardrail_id,
            ))

        if exc is not None:
            raise exc
        return guard_result

    def record_guardrail(self, opts: RawGuardrailOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="guardrail.check.completed",
            guardrail_id=opts["guardrail_id"],
        )
        for key in ("guardrail_type", "input_hash", "result", "block_reason", "severity", "latency_ms", "blocked"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)
        if opts.get("blocked") or opts.get("result") == "fail":
            self._client.enqueue(self._base(
                event_type="guardrail.blocked",
                guardrail_id=opts["guardrail_id"],
            ))

    # -- Prompt render -------------------------------------------------------

    def record_prompt_render(self, opts: RecordPromptRenderOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="prompt.rendered",
            template_id=opts["template_id"],
        )
        for key in ("template_version", "rendered_token_count", "variable_count", "latency_ms"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    # -- Lifecycle -----------------------------------------------------------

    async def end(self) -> None:
        self._client.enqueue(self._base(event_type="agent.step.completed"))

    async def __aenter__(self) -> "StepHandle":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.end()


# ---------------------------------------------------------------------------
# RunHandle
# ---------------------------------------------------------------------------


class RunHandle:
    def __init__(
        self,
        client: "SensuClient",
        *,
        run_id: str,
        session_id: str,
        agent_id: str,
        org_id: str,
        trace_id: str,
        span_id: str,
    ) -> None:
        self._client = client
        self.run_id = run_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.org_id = org_id
        self.trace_id = trace_id
        self.span_id = span_id
        self._step_sequence = 0

    def _base(self, **extra: Any) -> Dict[str, Any]:
        return {
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "org_id": self.org_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            **extra,
        }

    def start_step(self, opts: Optional[StartStepOptions] = None) -> StepHandle:
        opts = opts or {}
        self._step_sequence += 1
        step_id = opts.get("step_id") or new_id()
        sequence = opts.get("sequence", self._step_sequence)
        name = opts.get("name")
        step_type = opts.get("step_type", "generic")

        step = StepHandle(
            self._client,
            step_id=step_id,
            run_id=self.run_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            org_id=self.org_id,
            trace_id=self.trace_id,
            span_id=new_id(),
            sequence=sequence,
            name=name,
            step_type=step_type,
        )

        evt: Dict[str, Any] = step._base(
            event_type="agent.step.started",
            step_type=step_type,
            sequence=sequence,
        )
        if name:
            evt["step_name"] = name
        self._client.enqueue(evt)
        return step

    def record_feedback(self, opts: RecordFeedbackOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="feedback.received",
            type=opts["type"],
        )
        for key in ("score", "comment", "end_user_id"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    def record_eval_score(self, opts: RecordEvalScoreOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="eval.score.recorded",
            metric=opts["metric"],
            score=opts["score"],
        )
        for key in ("evaluator_id", "model_used_for_eval", "step_id", "llm_call_id"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    def handoff(self, opts: HandoffOptions) -> None:
        event: Dict[str, Any] = self._base(
            event_type="agent.handoff",
            to_agent_id=opts["to_agent_id"],
        )
        for key in ("reason", "context_tokens_transferred"):
            if key in opts:
                event[key] = opts[key]  # type: ignore[literal-required]
        self._client.enqueue(event)

    async def end(self, status: Literal["completed", "failed"] = "completed") -> None:
        self._client.clear_run_loop_state(self.run_id)
        event_type = "agent.run.completed" if status == "completed" else "agent.run.failed"
        self._client.enqueue(self._base(event_type=event_type, status=status))
        await self._client.flush()

    async def __aenter__(self) -> "RunHandle":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status: Literal["completed", "failed"] = "failed" if exc_type is not None else "completed"
        await self.end(status)


# ---------------------------------------------------------------------------
# SensuClient
# ---------------------------------------------------------------------------


_LEGACY_ENV_WARNED: set = set()


def _env_with_legacy(new_name: str, old_name: str, default: str) -> str:
    val = os.environ.get(new_name)
    if val is not None:
        return val
    val = os.environ.get(old_name)
    if val is not None:
        if old_name not in _LEGACY_ENV_WARNED:
            _LEGACY_ENV_WARNED.add(old_name)
            warnings.warn(
                f"[sensu] {old_name} is deprecated; use {new_name} instead. "
                "Support will be removed in a future release."
            )
        return val
    return default


class SensuClient:
    def __init__(self, opts: Optional[Dict[str, Any]] = None) -> None:
        opts = opts or {}

        if opts.get("from_env"):
            api_key = _env_with_legacy("SENSU_API_KEY", "SENZU_API_KEY", "")
            base_url = _env_with_legacy("SENSU_BASE_URL", "SENZU_BASE_URL", "http://localhost:3001")
            agent_id = _env_with_legacy("SENSU_AGENT_ID", "SENZU_AGENT_ID", "")
            org_id = _env_with_legacy("SENSU_ORG_ID", "SENZU_ORG_ID", "")
        else:
            api_key = opts.get("api_key", "")
            base_url = opts.get("base_url", "http://localhost:3001")
            agent_id = opts.get("agent_id", "")
            org_id = opts.get("org_id", "")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._org_id = org_id
        self._batch_size: int = opts.get("batch_size", 10)
        self._flush_interval_s: float = opts.get("flush_interval_ms", 2000) / 1000.0
        self.disabled: bool = bool(opts.get("disabled", False))
        self._on_loop_detected: Optional[Callable[[str, int], None]] = opts.get("on_loop_detected")
        self._loop_threshold: int = opts.get("loop_threshold", 5)
        self._disable_live_pricing: bool = bool(opts.get("disable_live_pricing", False))
        # Default 1 hour. Set 0 to disable caching (every call hits the API).
        self._pricing_cache_ttl_ms: float = float(opts.get("pricing_cache_ttl_ms", 3_600_000))
        self._debug_mode: bool = bool(opts.get("debug_mode", False))
        # When False (default), the SDK strips `body` from every message
        # snapshot before flushing. Setting True opts the org into the
        # Replay v1 unmask flow — the API masks PII server-side; raw
        # bodies stay tenant-side and require an audited unmask to read.
        # See planning/REPLAY_V1_PLAN.md §7.
        self.capture_message_bodies: bool = bool(opts.get("capture_message_bodies", False))

        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        # Cache value shape: (rates_tuple, time.monotonic() timestamp).
        # Entries older than _pricing_cache_ttl_ms are treated as misses
        # on read (see _pricing.resolve_pricing).
        self._pricing_cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
        # Set of "provider:model" keys we've already warned about for live
        # pricing failures. Keeps logs quiet under repeated failures.
        self._warned_pricing_misses: Set[str] = set()
        self._run_tool_counts: Dict[str, Dict[str, int]] = {}
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._stopped = False
        self._async_http: Optional[httpx.AsyncClient] = (
            None if self.disabled else httpx.AsyncClient(timeout=10.0)
        )

        if not self.disabled:
            atexit.register(self._atexit_flush)

    # -- Message-snapshot sanitization --------------------------------------

    _MAX_BODY_CHARS = 65_536  # matches server schema cap

    def sanitize_messages_snapshot(self, msgs: List[Any]) -> List[Dict[str, Any]]:
        """Strip ``body`` from every snapshot unless ``capture_message_bodies``
        is True; cap body length at 65,536 chars to match the server schema.

        Called from track_llm() before the event hits the wire. Centralizes
        the capture decision so future producers can reuse it.
        """
        out: List[Dict[str, Any]] = []
        for m in msgs:
            d = dict(m)  # TypedDict → plain dict, also a defensive copy
            if not self.capture_message_bodies:
                d.pop("body", None)
            elif isinstance(d.get("body"), str) and len(d["body"]) > self._MAX_BODY_CHARS:
                d["body"] = d["body"][: self._MAX_BODY_CHARS]
            out.append(d)
        return out

    # -- Buffer & flushing ---------------------------------------------------

    def enqueue(self, event: Dict[str, Any]) -> None:
        if self.disabled:
            return
        if self._debug_mode:
            print(f"[sensu] {format_debug_event(event)}")
        with self._buffer_lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._batch_size
        if should_flush:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush())
            except RuntimeError:
                pass  # no running loop — periodic timer or atexit handles it

    async def flush(self) -> None:
        if self.disabled:
            return
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer = []

        try:
            resp = await self._async_http.post(
                f"{self._base_url}/api/v1/events",
                json={"events": batch},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self._api_key,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                for err in data.get("errors") or []:
                    warnings.warn(f"[sensu] Event error at index {err.get('index')}: {err.get('error')}")
            else:
                # re-queue on non-2xx
                with self._buffer_lock:
                    self._buffer = batch + self._buffer
        except Exception:
            # re-queue on network error
            with self._buffer_lock:
                self._buffer = batch + self._buffer

    # -- Run-less feedback / eval helpers ------------------------------------

    async def feedback(self, opts: FeedbackOptions) -> Optional[Dict[str, Any]]:
        """Post end-user feedback for a run. Run-less helper — no active sensu.run() context required.

        Hits POST /api/v1/feedback directly (not the event buffer).
        Returns the parsed JSON response (``{"id": "..."}``) or None on failure.
        """
        if self.disabled or not self._api_key or self._async_http is None:
            return None
        body: Dict[str, Any] = {"runId": opts["run_id"], "type": opts["type"]}
        for src, dest in (("score", "score"), ("comment", "comment"), ("end_user_id", "endUserId")):
            if src in opts:
                body[dest] = opts[src]  # type: ignore[literal-required]
        try:
            resp = await self._async_http.post(
                f"{self._base_url}/api/v1/feedback",
                json=body,
                headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            )
            if resp.status_code >= 400:
                warnings.warn(f"[sensu] feedback failed {resp.status_code}: {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            warnings.warn(f"[sensu] feedback network error: {e}")
            return None

    async def score(self, opts: ScoreOptions) -> Optional[Dict[str, Any]]:
        """Post an automated eval score for a run. Run-less helper.

        Hits POST /api/v1/eval-scores directly (not the event buffer).
        Returns the parsed JSON response (``{"id": "..."}``) or None on failure.
        """
        if self.disabled or not self._api_key or self._async_http is None:
            return None
        body: Dict[str, Any] = {
            "runId":  opts["run_id"],
            "metric": opts["metric"],
            "score":  opts["score"],
        }
        for src, dest in (
            ("evaluator_id", "evaluatorId"),
            ("model_used_for_eval", "modelUsedForEval"),
            ("step_id", "stepId"),
            ("llm_call_id", "llmCallId"),
        ):
            if src in opts:
                body[dest] = opts[src]  # type: ignore[literal-required]
        try:
            resp = await self._async_http.post(
                f"{self._base_url}/api/v1/eval-scores",
                json=body,
                headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            )
            if resp.status_code >= 400:
                warnings.warn(f"[sensu] score failed {resp.status_code}: {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            warnings.warn(f"[sensu] score network error: {e}")
            return None

    async def register_agent_version(
        self, opts: "RegisterAgentVersionOptions",
    ) -> Optional[Dict[str, Any]]:
        """Register a candidate config used at a given commit so eval-gate
        checks (§5.2) can reference it as ``versionId`` instead of inlining
        the full config in every request. Run-less helper.

        Hits ``POST /api/v1/agents/:id/versions`` directly. Returns the
        parsed JSON response (AgentVersion dict) or None on failure.

        Customers typically call this from their deploy step:

            await sensu.register_agent_version({
                "agent_id": "cust-support-v3",
                "sha":      os.environ["GITHUB_SHA"],
                "config":   {"system_prompt": PROMPT, "model": "claude-sonnet-4-6"},
            })
        """
        if self.disabled or not self._api_key or self._async_http is None:
            return None
        from urllib.parse import quote
        agent_id = opts.get("agent_id")
        if not agent_id:
            warnings.warn("[sensu] register_agent_version: agent_id is required")
            return None
        try:
            resp = await self._async_http.post(
                f"{self._base_url}/api/v1/agents/{quote(agent_id, safe='')}/versions",
                json={"sha": opts["sha"], "config": opts["config"]},
                headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            )
            if resp.status_code >= 400:
                warnings.warn(
                    f"[sensu] register_agent_version failed {resp.status_code}: {resp.text}",
                )
                return None
            return resp.json()
        except Exception as e:
            warnings.warn(f"[sensu] register_agent_version network error: {e}")
            return None

    def _atexit_flush(self) -> None:
        """Called by atexit — may run outside any event loop."""
        try:
            loop = asyncio.get_running_loop()
            # Still inside an event loop (e.g. uvloop shutdown) — schedule task
            loop.create_task(self.flush())
        except RuntimeError:
            # No running loop — safe to use asyncio.run()
            with self._buffer_lock:
                if not self._buffer:
                    return
            try:
                asyncio.run(self.flush())
            except Exception:
                pass

    async def _flush_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self._flush_interval_s)
            try:
                await self.flush()
            except Exception:
                pass

    def _ensure_flush_task(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = loop.create_task(self._flush_loop(), name="sensu-flush")

    # -- Context propagation -------------------------------------------------

    def get_active_run(self) -> Optional[RunHandle]:
        return _active_run_var.get(None)

    async def run(
        self,
        opts: StartRunOptions,
        fn: Callable[["RunHandle"], Coroutine[Any, Any, T]],
    ) -> T:
        self._ensure_flush_task()
        run_handle = self.start_run(opts)
        token = _active_run_var.set(run_handle)
        succeeded = False
        result: Any = None
        try:
            result = await fn(run_handle)
            succeeded = True
        finally:
            _active_run_var.reset(token)
            try:
                await run_handle.end("completed" if succeeded else "failed")
            except Exception:
                pass
        return result  # type: ignore[return-value]

    # -- Run management ------------------------------------------------------

    def start_run(self, opts: Optional[StartRunOptions] = None) -> RunHandle:
        opts = opts or {}
        run_id = opts.get("run_id") or new_id()
        session_id = opts.get("session_id") or new_id()
        trace_id = new_id()
        span_id = new_id()

        run = RunHandle(
            self,
            run_id=run_id,
            session_id=session_id,
            agent_id=self._agent_id,
            org_id=self._org_id,
            trace_id=trace_id,
            span_id=span_id,
        )

        evt: Dict[str, Any] = run._base(event_type="agent.run.started")
        if opts.get("run_type"):
            evt["run_type"] = opts["run_type"]
        if opts.get("end_user_id"):
            evt["end_user_id"] = opts["end_user_id"]
        self.enqueue(evt)
        return run

    def spawn_run(self, parent_run: RunHandle, opts: SpawnRunOptions) -> RunHandle:
        child_agent_id = opts["child_agent_id"]
        child_run_id = opts.get("child_run_id") or new_id()
        session_id = opts.get("session_id") or parent_run.session_id

        self.enqueue({
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "event_type": "agent.spawned",
            "org_id": parent_run.org_id,
            "agent_id": parent_run.agent_id,
            "session_id": parent_run.session_id,
            "run_id": parent_run.run_id,
            "trace_id": parent_run.trace_id,
            "span_id": new_id(),
            "child_agent_id": child_agent_id,
            "child_run_id": child_run_id,
            **({"spawn_reason": opts["spawn_reason"]} if opts.get("spawn_reason") else {}),
        })

        child = RunHandle(
            self,
            run_id=child_run_id,
            session_id=session_id,
            agent_id=child_agent_id,
            org_id=parent_run.org_id,
            trace_id=parent_run.trace_id,  # shared trace
            span_id=new_id(),
        )

        child_evt: Dict[str, Any] = child._base(event_type="agent.run.started")
        if opts.get("run_type"):
            child_evt["run_type"] = opts["run_type"]
        self.enqueue(child_evt)
        return child

    # -- Client-level shortcut methods (auto-find active run) ----------------

    async def track_tool(
        self,
        tool_name: str,
        fn: Callable[[], Coroutine[Any, Any, T]],
        *,
        retry_of: Optional[str] = None,
        args: Any = _ARGS_NOT_PROVIDED,
        capture_bodies: bool = False,
    ) -> T:
        """Track a tool call inside the active ``sensu.run()`` context.

        Pass ``args`` + ``capture_bodies=True`` to ship the tool's input
        and result on ``tool.call.completed``. The Sensu API runs its
        shared PII pipeline at ingest — raw bodies never leave the
        tenant boundary unmasked. Per-call opt-in
        (TOOL_IO_CAPTURE_PLAN.md §11.2).
        """
        run = self.get_active_run()
        if run is None:
            return await fn()
        step = run.start_step({"name": tool_name, "step_type": "tool"})
        try:
            opts: TrackToolOptions = {"tool_name": tool_name, "fn": fn}
            if retry_of:
                opts["retry_of"] = retry_of
            if capture_bodies:
                opts["capture_bodies"] = True
            if args is not _ARGS_NOT_PROVIDED:
                opts["args"] = args
            result = await step.track_tool(opts)
        finally:
            await step.end()
        return result  # type: ignore[return-value]

    async def track_retrieval(
        self,
        retrieval_store_id: str,
        fn: Callable[[], Coroutine[Any, Any, T]],
        *,
        top_k: Optional[int] = None,
    ) -> T:
        run = self.get_active_run()
        if run is None:
            return await fn()
        step = run.start_step({"name": retrieval_store_id, "step_type": "retrieval"})
        try:
            ropts: TrackRetrievalOptions = {"fn": fn, "vector_store_id": retrieval_store_id}
            if top_k is not None:
                ropts["top_k"] = top_k
            result = await step.track_retrieval(ropts)
        finally:
            await step.end()
        return result  # type: ignore[return-value]

    async def track_embedding(
        self,
        model: str,
        fn: Callable[[], Coroutine[Any, Any, T]],
        *,
        input_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> T:
        run = self.get_active_run()
        if run is None:
            return await fn()
        step = run.start_step({"name": "embedding", "step_type": "embedding"})
        try:
            eopts: TrackEmbeddingOptions = {"model": model, "fn": fn}
            if input_length is not None:
                eopts["input_text_length"] = input_length
            if batch_size is not None:
                eopts["batch_size"] = batch_size
            result = await step.track_embedding(eopts)
        finally:
            await step.end()
        return result  # type: ignore[return-value]

    async def track_guardrail(
        self,
        guardrail_id: str,
        guardrail_type: GuardrailType,
        fn: Callable[[], Coroutine[Any, Any, GuardrailResult]],
    ) -> GuardrailResult:
        run = self.get_active_run()
        if run is None:
            return await fn()
        step = run.start_step({"name": guardrail_id, "step_type": "guardrail"})
        try:
            gopts: TrackGuardrailOptions = {
                "guardrail_id": guardrail_id,
                "guardrail_type": guardrail_type,
                "fn": fn,
            }
            result = await step.track_guardrail(gopts)
        finally:
            await step.end()
        return result

    # -- Loop detection ------------------------------------------------------

    def notify_tool_call(self, run_id: str, tool_name: str) -> None:
        if run_id not in self._run_tool_counts:
            self._run_tool_counts[run_id] = {}
        counts = self._run_tool_counts[run_id]
        counts[tool_name] = counts.get(tool_name, 0) + 1
        if counts[tool_name] >= self._loop_threshold and self._on_loop_detected:
            try:
                self._on_loop_detected(tool_name, counts[tool_name])
            except Exception:
                pass

    def clear_run_loop_state(self, run_id: str) -> None:
        self._run_tool_counts.pop(run_id, None)

    # -- Pricing -------------------------------------------------------------

    async def resolve_pricing(self, provider: str, model: str) -> Tuple[float, float]:
        return await resolve_pricing(
            provider,
            model,
            base_url=self._base_url,
            api_key=self._api_key,
            cache=self._pricing_cache,
            disable_live_pricing=self._disable_live_pricing,
            disabled=self.disabled,
            warned=self._warned_pricing_misses,
            cache_ttl_ms=self._pricing_cache_ttl_ms,
        )

    # -- Session management --------------------------------------------------

    def start_session(self, opts: Optional[StartSessionOptions] = None) -> str:
        opts = opts or {}
        session_id = opts.get("session_id") or new_id()
        evt: Dict[str, Any] = {
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "event_type": "session.started",
            "org_id": self._org_id,
            "agent_id": self._agent_id,
            "session_id": session_id,
            "run_id": new_id(),
            "trace_id": new_id(),
            "span_id": new_id(),
        }
        if opts.get("channel"):
            evt["channel"] = opts["channel"]
        if opts.get("end_user_id"):
            evt["end_user_id"] = opts["end_user_id"]
        self.enqueue(evt)
        return session_id

    def resume_session(self, opts: ResumeSessionOptions) -> str:
        session_id = opts.get("session_id") or new_id()
        evt: Dict[str, Any] = {
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "event_type": "session.resumed",
            "org_id": self._org_id,
            "agent_id": self._agent_id,
            "session_id": session_id,
            "run_id": new_id(),
            "trace_id": new_id(),
            "span_id": new_id(),
            "resumed_from_session_id": opts["resumed_from_session_id"],
        }
        if opts.get("channel"):
            evt["channel"] = opts["channel"]
        if opts.get("end_user_id"):
            evt["end_user_id"] = opts["end_user_id"]
        self.enqueue(evt)
        return session_id

    # -- Prompt management ---------------------------------------------------

    def deploy_prompt_version(self, opts: DeployPromptVersionOptions) -> None:
        evt: Dict[str, Any] = {
            "event_id": new_id(),
            "timestamp": utcnow_iso(),
            "event_type": "prompt.version.deployed",
            "org_id": self._org_id,
            "agent_id": self._agent_id,
            "session_id": new_id(),
            "run_id": new_id(),
            "trace_id": new_id(),
            "span_id": new_id(),
            "template_id": opts["template_id"],
            "new_version": opts["new_version"],
        }
        if opts.get("old_version"):
            evt["old_version"] = opts["old_version"]
        if opts.get("deployed_by"):
            evt["deployed_by"] = opts["deployed_by"]
        self.enqueue(evt)

    # -- Lifecycle -----------------------------------------------------------

    def destroy(self) -> None:
        self._stopped = True
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        try:
            atexit.unregister(self._atexit_flush)
        except Exception:
            pass
        if self._async_http is not None:
            # Schedule close if loop is running; otherwise skip (GC handles it)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_http.aclose())
            except RuntimeError:
                pass

    def __enter__(self) -> "SensuClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()

    async def __aenter__(self) -> "SensuClient":
        self._ensure_flush_task()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.flush()
        self.destroy()
