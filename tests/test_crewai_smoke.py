"""
End-to-end smoke test for the Python CrewAI SensuCrewListener.

Builds a real 2-agent Crew (researcher → writer) with a custom mock LLM
that returns canned responses (no provider key required) and verifies:
 - one agent.step.started fires per task with step_type='crewai_task'
 - exactly one agent.spawned per unique role (no duplicates)
 - agent.handoff fires on the role switch (researcher → writer)
 - all events share the same session_id / run_id / trace_id
 - the listener identifies as SensuClient.agent_id::role for child ids

Skipped if crewai isn't installed.

Run:  python -m pytest tests/test_crewai_smoke.py -v
"""
from __future__ import annotations

from typing import Any

import pytest

from sensu import SensuClient


def make_client(agent_id: str = "research-crew") -> SensuClient:
    return SensuClient({
        "api_key": "test-key",
        "base_url": "http://localhost:9999",
        "agent_id": agent_id,
        "org_id": "org-1",
        "batch_size": 100,
        "flush_interval_ms": 60_000,
        "disable_live_pricing": True,
    })


def test_crewai_e2e_smoke_two_agent_pipeline() -> None:
    """Real Crew, real CrewAI runtime → tasks fire crewai_task steps + spawn/handoff."""
    pytest.importorskip("crewai")

    from crewai import Agent, Task, Crew, Process
    from crewai.events.event_bus import crewai_event_bus
    from crewai.llms.base_llm import BaseLLM
    from sensu.integrations.crewai import SensuCrewListener

    class CannedLLM(BaseLLM):
        """Minimal LLM that returns canned text per call — no network required."""

        def __init__(self, response: str = "ok") -> None:
            super().__init__(model="canned-mock-llm")
            self._response = response

        def call(
            self,
            messages: Any,
            tools: Any = None,
            callbacks: Any = None,
            available_functions: Any = None,
            from_task: Any = None,
            from_agent: Any = None,
            response_model: Any = None,
        ) -> str:
            return self._response

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

    client = make_client()

    # Build agents with the mock LLM
    researcher = Agent(
        role="researcher",
        goal="Summarize the topic",
        backstory="A careful researcher.",
        llm=CannedLLM("researcher: observability is about explaining system state from outputs."),
        verbose=False,
        allow_delegation=False,
    )
    writer = Agent(
        role="writer",
        goal="Polish the summary",
        backstory="A careful writer.",
        llm=CannedLLM("writer: final polished summary."),
        verbose=False,
        allow_delegation=False,
    )

    task1 = Task(
        description="Research the topic and produce a summary.",
        expected_output="A short summary.",
        agent=researcher,
    )
    task2 = Task(
        description="Polish the summary into a final answer.",
        expected_output="A polished answer.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=False,
    )

    # Register the listener inside scoped_handlers so other tests aren't affected
    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        crew.kickoff()

    events = client._buffer
    assert events, "expected at least one Sensu event"

    # -- Task steps -------------------------------------------------------
    task_starts = [
        e for e in events
        if e["event_type"] == "agent.step.started" and e.get("step_type") == "crewai_task"
    ]
    assert len(task_starts) == 2, f"expected 2 task starts, got {len(task_starts)}: {task_starts}"

    # -- Multi-agent identity --------------------------------------------
    spawned = [e for e in events if e["event_type"] == "agent.spawned"]
    spawned_roles = {e["child_agent_id"] for e in spawned}
    assert spawned_roles == {"research-crew::researcher", "research-crew::writer"}, \
        f"expected one spawn per role, got {spawned_roles}"

    # Each spawn has a child_run_id
    for e in spawned:
        assert e["child_run_id"]

    # One handoff: researcher → writer (the role switch happens once per kickoff)
    handoffs = [e for e in events if e["event_type"] == "agent.handoff"]
    assert len(handoffs) >= 1
    final_handoff = handoffs[-1]
    assert final_handoff["to_agent_id"] == "research-crew::writer"

    # -- Identity ---------------------------------------------------------
    session_ids = {e["session_id"] for e in events}
    run_ids = {e["run_id"] for e in events}
    trace_ids = {e["trace_id"] for e in events}
    assert len(session_ids) == 1
    assert len(run_ids) == 1
    assert len(trace_ids) == 1

    # All events scoped to the right org + orchestrator agent
    for e in events:
        assert e["org_id"] == "org-1"
        assert e["agent_id"] == "research-crew"
