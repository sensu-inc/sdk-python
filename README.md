# sensu-sdk

Python SDK for the [Sensu](https://sensu-ai.com) AI observability platform.

> Migrating from `senzu-sdk`? The package was renamed; install `sensu-sdk` and replace `from senzu import SenzuClient` with `from sensu import SensuClient`. The legacy `senzu-sdk` v0.5.x package will continue to work but receives no further updates.

## Installation

```bash
pip install sensu-sdk                    # core
pip install "sensu-sdk[anthropic]"       # + Anthropic auto-tracking
pip install "sensu-sdk[openai]"          # + OpenAI auto-tracking
pip install "sensu-sdk[langchain]"       # + LangChain callback handler
pip install "sensu-sdk[langgraph]"       # + LangGraph node-aware handler
pip install "sensu-sdk[crewai]"          # + CrewAI multi-agent listener
pip install "sensu-sdk[all]"             # everything
```

## Quick start

```python
import sensu

client = sensu.SensuClient({
    "api_key": "your-api-key",
    "agent_id": "your-agent-id",
    "org_id": "your-org-id",
})

# High-level API with automatic context propagation
async def handle_request():
    async with client.run({"session_id": "abc"}) as run:
        step = run.start_step({"name": "fetch", "step_type": "tool"})
        result = await step.track_tool({"tool_name": "search", "fn": search})
        await step.end()

# Anthropic auto-tracking
from anthropic import AsyncAnthropic
from sensu.integrations.anthropic import WrapAnthropicOptions, wrap_anthropic

anthropic = wrap_anthropic(
    AsyncAnthropic(),
    WrapAnthropicOptions(client=client),
)

async def chat():
    async with client.run({}) as run:
        # messages.create() is tracked automatically
        return await anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
```

## LangChain

Drop the Sensu callback handler into any LangChain chain, agent, or LLM.
Chain boundaries, LLM calls (with streaming TTFT and retry/fallback
detection), and tool calls are captured automatically. Compatible with
LangChain 0.x and 1.x.

```python
import sensu
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

client = sensu.SensuClient({"from_env": True})
handler = sensu.SensuCallbackHandler(client=client)

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
llm = ChatAnthropic(model="claude-sonnet-4-6")
chain = prompt | llm

result = await chain.ainvoke(
    {"question": "What is observability?"},
    config={"callbacks": [handler]},
)
```

**Tying events to a specific run.** Pass `session_id` and `run_id`
explicitly to correlate LangChain telemetry with a run started elsewhere:

```python
run = client.start_run({"session_id": "user-session-1"})
handler = sensu.SensuCallbackHandler(
    client=client,
    session_id="user-session-1",
    run_id=run.run_id,
)
```

**What's captured.** Chain start/end → `agent.step.*`; LLM start/end/error
→ `llm.request.*` (provider, model, tokens, latency, TTFT); streaming
tokens → `stream.token.received` every 10th token; tool start/end/error
→ `tool.call.*` with `retry_of` when the same tool re-invokes after error
and `is_fallback` on the next LLM after an error.

**Limitations.** LangChain's callback interface exposes aggregate token
counts only — per-role context breakdown is not surfaced through this
path. For context-window analysis, use the low-level `track_llm()` /
`record_llm()` APIs directly.

Requires the `langchain` extra (`pip install 'sensu-sdk[langchain]'`).

## LangGraph

Building agentic workflows with [LangGraph](https://langchain-ai.github.io/langgraph/)?
Use `SensuLangGraphHandler` to get per-node steps in the Sensu trace tree.
Each node execution emits an `agent.step.started` event with
`step_type='langgraph_node'` and the node name; the internal `ChannelWrite`
plumbing wrappers (tagged `langsmith:hidden` by LangGraph) are filtered out
so the event stream stays clean.

```python
import sensu
from langgraph.graph import StateGraph, START, END

client = sensu.SensuClient({"from_env": True})
handler = sensu.SensuLangGraphHandler(client=client)

graph = (
    StateGraph(MyState)
    .add_node("plan_step", planner_node)
    .add_node("research_step", researcher_node)
    .add_node("write_step", writer_node)
    .add_edge(START, "plan_step")
    .add_edge("plan_step", "research_step")
    .add_edge("research_step", "write_step")
    .add_edge("write_step", END)
    .compile()
)

result = await graph.ainvoke(
    {"topic": "observability"},
    config={"callbacks": [handler]},
)
```

**What's added beyond the LangChain handler.** A node execution emits
`step_type='langgraph_node'` (vs `'chain'`), with extra fields:

- `node_name` — the `add_node()` name (e.g. `"plan_step"`)
- `langgraph_step` — LangGraph's monotonic step counter (when emitted)

Everything else — LLM calls, tool calls, streaming TTFT, retry/fallback —
works identically to the LangChain integration, because `SensuLangGraphHandler`
subclasses `SensuCallbackHandler`. Mixed LangChain + LangGraph projects work
with either handler; the parent class auto-detects LangGraph nodes from the
callback metadata too.

Requires the `langgraph` extra (`pip install 'sensu-sdk[langgraph]'`).

## CrewAI

Running multi-agent workflows with [CrewAI](https://crewai.com)? Use
`SensuCrewListener` to attach Sensu to CrewAI's native event bus. Each
agent gets a stable child id (`{orchestrator}::{role}`), task lifecycle
events become `agent.step.*`, role switches emit `agent.handoff`, and
LLM + tool calls land as the same `llm.request.*` / `tool.call.*` events
the LangChain integration produces.

```python
import sensu
from sensu.integrations.crewai import SensuCrewListener
from crewai import Agent, Task, Crew, Process

# The Sensu agent_id is the "crew orchestrator" label; each CrewAI agent
# inside the crew becomes a child agent like "research-crew::researcher".
client = sensu.SensuClient({"from_env": True, "agent_id": "research-crew"})
SensuCrewListener(client=client)  # one-line attach to the global event bus

researcher = Agent(role="researcher", goal="...", backstory="...", llm=llm)
writer     = Agent(role="writer",     goal="...", backstory="...", llm=llm)

task1 = Task(description="Research", expected_output="...", agent=researcher)
task2 = Task(description="Write",    expected_output="...", agent=writer)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
)
result = crew.kickoff(inputs={"topic": "observability"})
```

**Event mapping.**

| CrewAI event | Sensu event |
|---|---|
| `CrewKickoffStartedEvent`              | (no Sensu event — run lifecycle owned by `SensuClient`) |
| `TaskStartedEvent`                     | `agent.step.started` (`step_type='crewai_task'`) |
| `TaskCompletedEvent` / `TaskFailedEvent` | `agent.step.completed` (with `status: success` / `error`) |
| `AgentExecutionStartedEvent`           | `agent.spawned` (once per role) + `agent.handoff` on role switch |
| `LLMCallStartedEvent` / Completed / Failed | `llm.request.started` / `llm.request.completed` (with `is_fallback` after error) |
| `ToolUsageStartedEvent` / Finished / Error | `tool.call.started` / `tool.call.completed` (with `retry_of` on retry) |

The multi-agent identity mapping is what makes this useful: every spawn
populates `agent_spawns.child_agent_id` so the multi-agent system map
gets a real topology to render once that feature ships.

Requires the `crewai` extra (`pip install 'sensu-sdk[crewai]'`).

## Environment variables

| Variable | Description |
|---|---|
| `SENSU_API_KEY` | API key |
| `SENSU_BASE_URL` | API base URL (default: `http://localhost:3001`) |
| `SENSU_AGENT_ID` | Agent ID |
| `SENSU_ORG_ID` | Organisation ID |

> The legacy `SENZU_*` names are still read as a fallback and emit a deprecation warning. They will be removed in a future release.

Use `from_env=True` to load from environment:

```python
client = sensu.SensuClient({"from_env": True})
```

## License

MIT
