"""
Sensu Python SDK — AI observability for agent systems.

Quick start (async context propagation):

    import sensu

    client = sensu.SensuClient({"from_env": True})
    anthropic = sensu.wrap_anthropic(Anthropic(), sensu.WrapAnthropicOptions(client=client))

    async def handle_request(user_input: str) -> str:
        async with client.run({}) as run:
            step = run.start_step({"name": "main", "step_type": "llm"})
            result = await anthropic.messages.create(model="claude-sonnet-4-6", ...)
            await step.end()
            return result.content[0].text

Quick start (explicit run/step):

    client = sensu.SensuClient({"api_key": "...", "agent_id": "...", "org_id": "..."})
    run = client.start_run({"session_id": "abc"})
    step = run.start_step({"name": "fetch", "step_type": "tool"})
    result = await step.track_tool({"tool_name": "web_search", "fn": search})
    await step.end()
    await run.end()
"""

from sensu._client import RunHandle, SensuClient, StepHandle
from sensu._types import (
    AgentVersion,
    CandidateConfig,
    ContextBreakdown,
    DeployPromptVersionOptions,
    FeedbackOptions,
    GuardrailResult,
    GuardrailType,
    HandoffOptions,
    MessageSnapshotItem,
    RawEmbeddingOptions,
    RawGuardrailOptions,
    RawLlmCallOptions,
    RawRetrievalOptions,
    RecordEvalScoreOptions,
    RecordFeedbackOptions,
    RecordPromptRenderOptions,
    RegisterAgentVersionOptions,
    ResumeSessionOptions,
    RetrievalChunkInput,
    RunOptions,
    ScoreOptions,
    SensuClientOptions,
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

# Integration option classes — re-exported for ergonomic imports
from sensu.integrations.anthropic import WrapAnthropicOptions
from sensu.integrations.openai import WrapOpenAIOptions


def wrap_anthropic(anthropic_client: object, opts: WrapAnthropicOptions) -> object:
    """Wrap an Anthropic client to auto-track messages.create() calls."""
    from sensu.integrations.anthropic import wrap_anthropic as _wrap
    return _wrap(anthropic_client, opts)


def wrap_openai(openai_client: object, opts: WrapOpenAIOptions) -> object:
    """Wrap an OpenAI client to auto-track chat.completions.create() calls."""
    from sensu.integrations.openai import wrap_openai as _wrap
    return _wrap(openai_client, opts)


def __getattr__(name: str) -> object:
    """Lazy re-exports so importing ``langchain``/``langgraph``/``crewai`` is
    only required when the relevant handler is actually referenced. Keeps the
    core import path zero-dep."""
    if name == "SensuCallbackHandler":
        from sensu.integrations.langchain import SensuCallbackHandler as _Handler
        return _Handler
    if name == "SensuLangGraphHandler":
        from sensu.integrations.langgraph import SensuLangGraphHandler as _Handler
        return _Handler
    if name == "SensuCrewListener":
        from sensu.integrations.crewai import SensuCrewListener as _Handler
        return _Handler
    raise AttributeError(f"module 'sensu' has no attribute {name!r}")


__version__ = "0.11.0"

__all__ = [
    # Core classes
    "SensuClient",
    "RunHandle",
    "StepHandle",
    # Integration helpers
    "wrap_anthropic",
    "wrap_openai",
    "WrapAnthropicOptions",
    "WrapOpenAIOptions",
    "SensuCallbackHandler",      # lazy — requires the [langchain] extra at use time
    "SensuLangGraphHandler",     # lazy — requires the [langgraph] extra at use time
    "SensuCrewListener",         # lazy — requires the [crewai] extra at use time
    # TypedDicts — option types
    "SensuClientOptions",
    "StartRunOptions",
    "RunOptions",
    "StartStepOptions",
    "TrackLlmOptions",
    "TrackStreamingLlmOptions",
    "TrackToolOptions",
    "TrackRetrievalOptions",
    "TrackEmbeddingOptions",
    "TrackGuardrailOptions",
    "RawLlmCallOptions",
    "RawRetrievalOptions",
    "RawEmbeddingOptions",
    "RawGuardrailOptions",
    "RecordFeedbackOptions",
    "RecordEvalScoreOptions",
    "FeedbackOptions",
    "ScoreOptions",
    "SpawnRunOptions",
    "HandoffOptions",
    "RecordPromptRenderOptions",
    "DeployPromptVersionOptions",
    "StartSessionOptions",
    "ResumeSessionOptions",
    # Eval-gated CI/CD (§5.2)
    "CandidateConfig",
    "RegisterAgentVersionOptions",
    "AgentVersion",
    # Supporting types
    "ContextBreakdown",
    "MessageSnapshotItem",
    "RetrievalChunkInput",
    "GuardrailResult",
    "GuardrailType",
]
