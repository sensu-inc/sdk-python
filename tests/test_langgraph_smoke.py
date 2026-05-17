"""
End-to-end smoke test for the Python LangGraph SensuLangGraphHandler.

Builds a real 4-node LangGraph StateGraph (plan_step → research_step →
write_step → review_step), runs it with the handler attached, and asserts:
 - one agent.step.started fires per node with step_type='langgraph_node'
 - node_name matches the StateGraph node name
 - all events share the same session_id / run_id / trace_id

No external services — nodes are local string manipulation, so this runs
in CI without provider keys. Skipped if langgraph isn't installed.

Run:  python -m pytest tests/test_langgraph_smoke.py -v
"""
from __future__ import annotations

import pytest

from sensu import SensuClient


def make_client() -> SensuClient:
    return SensuClient({
        "api_key": "test-key",
        "base_url": "http://localhost:9999",
        "agent_id": "agent-smoke",
        "org_id": "org-smoke",
        "batch_size": 100,
        "flush_interval_ms": 60_000,
        "disable_live_pricing": True,
    })


@pytest.mark.asyncio
async def test_langgraph_e2e_smoke_via_state_graph() -> None:
    """Real StateGraph through real LangGraph runtime → 4 langgraph_node steps."""
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain")

    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from sensu.integrations.langgraph import SensuLangGraphHandler

    class PipelineState(TypedDict, total=False):
        topic: str
        outline: str
        research: str
        draft: str
        final: str

    async def planner(s: PipelineState) -> PipelineState:
        return {"outline": f"outline for: {s['topic']}"}

    async def researcher(s: PipelineState) -> PipelineState:
        return {"research": f"research notes on: {s['topic']}"}

    async def writer(s: PipelineState) -> PipelineState:
        return {"draft": f"draft of [{s.get('outline')}] using {s.get('research')}"}

    async def reviewer(s: PipelineState) -> PipelineState:
        return {"final": f"reviewed: {s.get('draft')}"}

    graph = (
        StateGraph(PipelineState)
        .add_node("plan_step", planner)
        .add_node("research_step", researcher)
        .add_node("write_step", writer)
        .add_node("review_step", reviewer)
        .add_edge(START, "plan_step")
        .add_edge("plan_step", "research_step")
        .add_edge("research_step", "write_step")
        .add_edge("write_step", "review_step")
        .add_edge("review_step", END)
        .compile()
    )

    client = make_client()
    handler = SensuLangGraphHandler(client=client)

    result = await graph.ainvoke(
        {"topic": "observability for AI agents"},
        config={"callbacks": [handler]},
    )
    assert result["final"].startswith("reviewed:")

    events = client._buffer
    node_starts = [
        e for e in events
        if e["event_type"] == "agent.step.started" and e.get("step_type") == "langgraph_node"
    ]
    node_names = [e["node_name"] for e in node_starts]

    # Each named node fires exactly one langgraph_node start
    for expected in ("plan_step", "research_step", "write_step", "review_step"):
        matches = [n for n in node_names if n == expected]
        assert len(matches) == 1, f"expected 1 start for {expected!r}, got {len(matches)}: {node_names}"

    # Step ID pairing
    chain_ends = [e for e in events if e["event_type"] == "agent.step.completed"]
    for start in node_starts:
        matched = [e for e in chain_ends if e.get("step_id") == start["step_id"]]
        assert matched, f"no chain end for node {start['node_name']!r}"

    # All events share identity
    assert len({e["session_id"] for e in events}) == 1
    assert len({e["run_id"] for e in events}) == 1
    assert len({e["trace_id"] for e in events}) == 1
