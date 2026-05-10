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
