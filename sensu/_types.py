from __future__ import annotations

import sys
from typing import Any, Callable, Coroutine, List, Literal, Optional

if sys.version_info >= (3, 11):
    from typing import TypedDict, NotRequired
else:
    from typing_extensions import TypedDict, NotRequired

# ---------------------------------------------------------------------------
# Callable alias for async functions used as `fn` in track_* methods
# ---------------------------------------------------------------------------

AnyCoroutine = Coroutine[Any, Any, Any]
AsyncFn = Callable[[], AnyCoroutine]

# ---------------------------------------------------------------------------
# Literal aliases
# ---------------------------------------------------------------------------

GuardrailResult = Literal["pass", "fail", "modified"]
GuardrailType = Literal["content", "pii", "jailbreak", "custom"]

# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class SensuClientOptions(TypedDict, total=False):
    api_key: str
    base_url: str
    agent_id: str
    org_id: str
    from_env: bool
    batch_size: int
    flush_interval_ms: int
    disabled: bool
    on_loop_detected: Callable[[str, int], None]
    loop_threshold: int
    disable_live_pricing: bool
    debug_mode: bool
    # When True, llm.message_snapshot events carry raw message bodies in
    # addition to token counts. The server masks PII at ingest; raw
    # bodies stay tenant-side and require an audited unmask to read.
    # Default: False. See planning/REPLAY_V1_PLAN.md §7 in the platform
    # repo. Parity with sdk-ts (captureMessageBodies) and sdk-go
    # (CaptureMessageBodies) — the field was already honored by the
    # client internally, this declaration just exposes it in the public
    # TypedDict for type-checker discoverability.
    capture_message_bodies: bool


# ---------------------------------------------------------------------------
# Run / Step options
# ---------------------------------------------------------------------------


class StartRunOptions(TypedDict, total=False):
    session_id: str
    run_type: str
    end_user_id: str
    run_id: str


RunOptions = StartRunOptions


class StartStepOptions(TypedDict, total=False):
    name: str
    step_type: str
    sequence: int
    step_id: str


# ---------------------------------------------------------------------------
# LLM tracking
# ---------------------------------------------------------------------------


class ContextBreakdown(TypedDict, total=False):
    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    tool_tokens: int
    retrieval_tokens: int


class MessageSnapshotItem(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    token_count: int
    content_hash: str
    tool_name: NotRequired[str]
    # Optional raw message body. Only forwarded to the API when the client
    # was constructed with capture_message_bodies=True. The API masks PII
    # via its shared pipeline at ingest. Max 65,536 chars (longer bodies
    # are truncated client-side to match the server schema cap).
    body: NotRequired[str]


class TrackLlmOptions(TypedDict, total=False):
    # required
    provider: str
    model: str
    fn: AsyncFn
    # optional
    max_context_tokens: int
    extract_context_breakdown: Callable[[Any], Optional[ContextBreakdown]]
    llm_call_id: str
    messages_snapshot: List[MessageSnapshotItem]
    referenced_chunk_ids: List[str]


class TrackStreamingLlmOptions(TypedDict, total=False):
    # required
    provider: str
    model: str
    stream: Any  # AsyncIterable[Any]
    # optional
    max_context_tokens: int
    llm_call_id: str
    emit_every_n_tokens: int
    on_complete: Callable[[str, Optional[float]], None]


class RawLlmCallOptions(TypedDict, total=False):
    # required
    provider: str
    model: str
    # optional
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    total_tokens: int
    max_context_tokens: int
    context_used_tokens: int
    latency_ms: float
    ttft_ms: float
    cost_usd_estimate: float
    status: Literal["success", "error", "timeout"]
    context_breakdown: ContextBreakdown
    referenced_chunk_ids: List[str]


# ---------------------------------------------------------------------------
# Tool tracking
# ---------------------------------------------------------------------------


class TrackToolOptions(TypedDict, total=False):
    # required
    tool_name: str
    fn: AsyncFn
    # optional
    retry_of: str
    # Tool I/O body capture (TOOL_IO_CAPTURE_PLAN.md §5.2). When
    # ``capture_bodies`` is True, ``args`` and the awaited result of
    # ``fn`` are JSON-serialized and shipped on tool.call.completed as
    # ``input_body`` + ``output_body``. The Sensu API masks PII via
    # its shared pipeline at ingest and surfaces the raw bodies only
    # through the audited Replay unmask flow. Per-call opt-in (not
    # per-client) so storage and PII exposure are explicit decisions.
    args: Any
    capture_bodies: bool


# ---------------------------------------------------------------------------
# Retrieval tracking
# ---------------------------------------------------------------------------


class TrackRetrievalOptions(TypedDict, total=False):
    # required
    fn: AsyncFn
    # optional
    vector_store_id: str
    top_k: int


class RetrievalChunkInput(TypedDict, total=False):
    # required
    chunk_id: str
    token_count: int
    # optional
    source: str
    similarity_score: float
    content_preview: str


class RawRetrievalOptions(TypedDict, total=False):
    vector_store_id: str
    top_k: int
    latency_ms: float
    chunks_returned: int
    tokens_injected: int
    similarity_score_avg: float
    status: Literal["success", "error"]
    chunks: List[RetrievalChunkInput]


# ---------------------------------------------------------------------------
# Embedding tracking
# ---------------------------------------------------------------------------


class TrackEmbeddingOptions(TypedDict, total=False):
    # required
    model: str
    fn: AsyncFn
    # optional
    input_text_length: int
    batch_size: int


class RawEmbeddingOptions(TypedDict, total=False):
    # required
    model: str
    # optional
    input_text_length: int
    token_count: int
    latency_ms: float
    cost_usd_estimate: float
    batch_size: int


# ---------------------------------------------------------------------------
# Feedback & eval
# ---------------------------------------------------------------------------


class RecordFeedbackOptions(TypedDict, total=False):
    # required
    type: Literal["thumbs_up", "thumbs_down", "score", "correction"]
    # optional
    score: float
    comment: str
    end_user_id: str


class RecordEvalScoreOptions(TypedDict, total=False):
    # required
    metric: str
    score: float
    # optional
    evaluator_id: str
    model_used_for_eval: str
    step_id: str
    llm_call_id: str


class FeedbackOptions(TypedDict, total=False):
    """Options for the run-less, top-level ``client.feedback()`` helper."""
    # required
    run_id: str
    type: Literal["thumbs_up", "thumbs_down", "score", "correction"]
    # optional
    score: float
    comment: str
    end_user_id: str


class ScoreOptions(TypedDict, total=False):
    """Options for the run-less, top-level ``client.score()`` helper."""
    # required
    run_id: str
    metric: str
    score: float
    # optional
    evaluator_id: str
    model_used_for_eval: str
    step_id: str
    llm_call_id: str


# ---------------------------------------------------------------------------
# Eval-gated CI/CD (§5.2) — agent versions registry
# ---------------------------------------------------------------------------


class CandidateConfig(TypedDict, total=False):
    """Candidate config registered under an agent version. Mirrors the API's
    CandidateConfig shape — system_prompt is required, model is optional
    (defaults to the sampled run's source model at gate time)."""
    # required
    system_prompt: str
    # optional
    model: str


class RegisterAgentVersionOptions(TypedDict, total=False):
    """Options for ``client.register_agent_version()``."""
    # required — customer-facing agent name (server prepends org id)
    agent_id: str
    # required — opaque identifier (usually a git commit SHA)
    sha: str
    # required
    config: CandidateConfig


class AgentVersion(TypedDict):
    """Server response shape for a registered agent version."""
    id: str
    agentId: str
    sha: str
    config: CandidateConfig
    createdAt: str


# ---------------------------------------------------------------------------
# Multi-agent
# ---------------------------------------------------------------------------


class SpawnRunOptions(TypedDict, total=False):
    # required
    child_agent_id: str
    # optional
    child_run_id: str
    spawn_reason: str
    run_type: str
    session_id: str


class HandoffOptions(TypedDict, total=False):
    # required
    to_agent_id: str
    # optional
    reason: str
    context_tokens_transferred: int


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class TrackGuardrailOptions(TypedDict, total=False):
    # required
    guardrail_id: str
    fn: Callable[[], Coroutine[Any, Any, GuardrailResult]]
    # optional
    guardrail_type: GuardrailType
    input_hash: str


class RawGuardrailOptions(TypedDict, total=False):
    # required
    guardrail_id: str
    # optional
    guardrail_type: GuardrailType
    input_hash: str
    result: GuardrailResult
    block_reason: str
    severity: Literal["low", "medium", "high"]
    latency_ms: float
    blocked: bool


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------


class RecordPromptRenderOptions(TypedDict, total=False):
    # required
    template_id: str
    # optional
    template_version: str
    rendered_token_count: int
    variable_count: int
    latency_ms: float


class DeployPromptVersionOptions(TypedDict, total=False):
    # required
    template_id: str
    new_version: str
    # optional
    old_version: str
    deployed_by: str


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class StartSessionOptions(TypedDict, total=False):
    session_id: str
    channel: Literal["web", "api", "mobile"]
    end_user_id: str


class ResumeSessionOptions(TypedDict, total=False):
    # required
    resumed_from_session_id: str
    # optional
    session_id: str
    channel: Literal["web", "api", "mobile"]
    end_user_id: str
