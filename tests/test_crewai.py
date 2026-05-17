"""
Unit tests for the CrewAI SensuCrewListener.

Tests exercise the listener by emitting events through the real
``crewai_event_bus`` and verifying ``client.enqueue`` gets the expected
Sensu-shaped events. No real LLM or tool execution — events are
constructed with ``model_construct`` to bypass strict pydantic validation
on agent / task fields.

Each test runs inside ``crewai_event_bus.scoped_handlers()`` so handlers
are cleared between tests (the bus is a module-level singleton).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from sensu import SensuClient


def _emit_sync(bus: Any, event: Any) -> None:
    """Emit an event and block until all handlers complete.

    crewai_event_bus.emit() dispatches handlers in a thread pool and returns
    a Future. For deterministic test assertions we must wait for that future.
    """
    fut = bus.emit(None, event)
    if fut is not None:
        fut.result(timeout=5.0)


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


def _mock_task(name: str = "task-x", description: str = "do the thing", agent_role: str | None = None) -> Any:
    t = MagicMock()
    t.id = f"task-{name}"
    t.name = name
    t.description = description
    if agent_role:
        t.agent = MagicMock()
        t.agent.role = agent_role
    else:
        t.agent = None
    t.fingerprint = None
    return t


def _mock_agent(role: str) -> Any:
    a = MagicMock()
    a.role = role
    a.fingerprint = None
    return a


def _make_task_started_event(task: Any) -> Any:
    from crewai.events.types.task_events import TaskStartedEvent
    return TaskStartedEvent.model_construct(type="task_started", context=None, task=task)


def _make_task_completed_event(task: Any, output: Any = None) -> Any:
    from crewai.events.types.task_events import TaskCompletedEvent
    return TaskCompletedEvent.model_construct(
        type="task_completed", task=task, output=output or MagicMock(),
    )


def _make_task_failed_event(task: Any, error: str = "boom") -> Any:
    from crewai.events.types.task_events import TaskFailedEvent
    return TaskFailedEvent.model_construct(
        type="task_failed", task=task, error=error,
    )


def _make_agent_exec_start_event(agent: Any) -> Any:
    from crewai.events.types.agent_events import AgentExecutionStartedEvent
    return AgentExecutionStartedEvent.model_construct(
        type="agent_execution_started",
        agent=agent, task=None, tools=None, task_prompt="prompt",
    )


def _make_llm_start_event(model: str) -> Any:
    from crewai.events.types.llm_events import LLMCallStartedEvent
    return LLMCallStartedEvent.model_construct(
        type="llm_call_started", model=model, messages="hi",
    )


def _make_llm_completed_event(model: str) -> Any:
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
    return LLMCallCompletedEvent.model_construct(
        type="llm_call_completed", model=model, messages="hi",
        response="hello", call_type=LLMCallType.LLM_CALL,
    )


def _make_llm_failed_event(model: str, error: str = "boom") -> Any:
    from crewai.events.types.llm_events import LLMCallFailedEvent
    return LLMCallFailedEvent.model_construct(
        type="llm_call_failed", model=model, error=error,
    )


def _make_tool_start_event(tool_name: str, agent_role: str = "r") -> Any:
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent
    return ToolUsageStartedEvent.model_construct(
        type="tool_usage_started", tool_name=tool_name, tool_args={},
        agent_role=agent_role,
    )


def _make_tool_finished_event(tool_name: str, output: Any = "result", agent_role: str = "r") -> Any:
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
    now = datetime.utcnow()
    return ToolUsageFinishedEvent.model_construct(
        type="tool_usage_finished", tool_name=tool_name, tool_args={},
        agent_role=agent_role, started_at=now, finished_at=now,
        output=output,
    )


def _make_tool_error_event(tool_name: str, error: str = "timeout", agent_role: str = "r") -> Any:
    from crewai.events.types.tool_usage_events import ToolUsageErrorEvent
    return ToolUsageErrorEvent.model_construct(
        type="tool_usage_error", tool_name=tool_name, tool_args={},
        agent_role=agent_role, error=error,
    )


# ---------------------------------------------------------------------------
# Import-error gate
# ---------------------------------------------------------------------------


def test_crew_listener_import_error_without_crewai() -> None:
    """SensuCrewListener raises ImportError if the [crewai] extra isn't installed."""
    import sys
    from unittest.mock import patch

    blocked = {
        "crewai": None,
        "crewai.events": None,
        "crewai.events.base_event_listener": None,
        "crewai.events.event_bus": None,
        "crewai.events.types": None,
        "crewai.events.types.crew_events": None,
        "crewai.events.types.task_events": None,
        "crewai.events.types.agent_events": None,
        "crewai.events.types.llm_events": None,
        "crewai.events.types.tool_usage_events": None,
    }
    sys.modules.pop("sensu.integrations.crewai", None)
    try:
        with patch.dict(sys.modules, blocked):
            from sensu.integrations.crewai import SensuCrewListener
            client = make_client()
            with pytest.raises(ImportError, match="crewai"):
                SensuCrewListener(client=client)
    finally:
        sys.modules.pop("sensu.integrations.crewai", None)


# ---------------------------------------------------------------------------
# Lifecycle event mappings
# ---------------------------------------------------------------------------


def test_task_started_emits_step_started_with_crewai_step_type() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        task = _mock_task(name="research", agent_role="researcher")
        _emit_sync(crewai_event_bus, _make_task_started_event(task))

    started = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert started["step_type"] == "crewai_task"
    assert started["task_id"] == "task-research"
    assert started["task_name"] == "research"
    assert started["agent_role"] == "researcher"
    assert started["child_agent_id"] == "research-crew::researcher"
    assert started["step_id"]


def test_task_completed_pairs_with_step_id_from_task_start() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()
    task = _mock_task(name="t1")

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_task_started_event(task))
        _emit_sync(crewai_event_bus, _make_task_completed_event(task))

    started = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    completed = next(e for e in client._buffer if e["event_type"] == "agent.step.completed")
    assert completed["step_id"] == started["step_id"]
    assert completed["status"] == "success"


def test_task_failed_status_error() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()
    task = _mock_task(name="t2")

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_task_started_event(task))
        _emit_sync(crewai_event_bus, _make_task_failed_event(task))

    completed = next(e for e in client._buffer if e["event_type"] == "agent.step.completed")
    assert completed["status"] == "error"


# ---------------------------------------------------------------------------
# Multi-agent identity
# ---------------------------------------------------------------------------


def test_first_agent_execution_per_role_emits_agent_spawned() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client(agent_id="research-crew")

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(_mock_agent("researcher")))

    spawned = next(e for e in client._buffer if e["event_type"] == "agent.spawned")
    assert spawned["child_agent_id"] == "research-crew::researcher"
    assert spawned["child_run_id"]
    assert "researcher" in spawned["spawn_reason"]


def test_agent_role_switch_emits_handoff() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(_mock_agent("researcher")))
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(_mock_agent("writer")))

    handoff_events = [e for e in client._buffer if e["event_type"] == "agent.handoff"]
    assert len(handoff_events) == 1
    assert handoff_events[0]["to_agent_id"] == "research-crew::writer"
    assert "researcher" in handoff_events[0]["reason"]
    assert "writer" in handoff_events[0]["reason"]


def test_agent_re_execution_does_not_emit_spawned_twice() -> None:
    """Same agent role re-executed within a crew = no duplicate spawn event."""
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()
    agent = _mock_agent("researcher")

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(agent))
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(agent))

    spawned_events = [e for e in client._buffer if e["event_type"] == "agent.spawned"]
    assert len(spawned_events) == 1


# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------


def test_llm_call_start_end_emits_request_pair() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_llm_start_event("claude-sonnet-4-6"))
        _emit_sync(crewai_event_bus, _make_llm_completed_event("claude-sonnet-4-6"))

    started = next(e for e in client._buffer if e["event_type"] == "llm.request.started")
    completed = next(e for e in client._buffer if e["event_type"] == "llm.request.completed")
    assert started["provider"] == "anthropic"
    assert started["model"] == "claude-sonnet-4-6"
    assert completed["status"] == "success"
    assert completed["llm_call_id"] == started["llm_call_id"]


def test_llm_call_failed_marks_next_start_as_fallback() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_llm_start_event("gpt-4o"))
        _emit_sync(crewai_event_bus, _make_llm_failed_event("gpt-4o"))
        _emit_sync(crewai_event_bus, _make_llm_start_event("gpt-4o-mini"))

    fallback = next(
        e for e in client._buffer
        if e["event_type"] == "llm.request.started" and e.get("is_fallback")
    )
    assert fallback["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tool lifecycle
# ---------------------------------------------------------------------------


def test_tool_usage_start_end_emits_call_pair() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_tool_start_event("web_search", agent_role="researcher"))
        _emit_sync(crewai_event_bus, _make_tool_finished_event("web_search", agent_role="researcher"))

    started = next(e for e in client._buffer if e["event_type"] == "tool.call.started")
    completed = next(e for e in client._buffer if e["event_type"] == "tool.call.completed")
    assert started["tool_name"] == "web_search"
    assert started["tool_call_id"]
    assert completed["tool_call_id"] == started["tool_call_id"]
    assert completed["status"] == "success"


def test_tool_retry_after_error_sets_retry_of() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client)
        _emit_sync(crewai_event_bus, _make_tool_start_event("flaky"))
        first_id = client._buffer[-1]["tool_call_id"]
        _emit_sync(crewai_event_bus, _make_tool_error_event("flaky"))
        _emit_sync(crewai_event_bus, _make_tool_start_event("flaky"))

    retry_starts = [
        e for e in client._buffer
        if e["event_type"] == "tool.call.started" and e.get("retry_of")
    ]
    assert len(retry_starts) == 1
    assert retry_starts[0]["retry_of"] == first_id


# ---------------------------------------------------------------------------
# Base fields
# ---------------------------------------------------------------------------


def test_all_events_share_session_run_trace_identity() -> None:
    pytest.importorskip("crewai")
    from sensu.integrations.crewai import SensuCrewListener
    from crewai.events.event_bus import crewai_event_bus

    client = make_client()

    with crewai_event_bus.scoped_handlers():
        SensuCrewListener(client=client, session_id="s-1", run_id="r-1")
        _emit_sync(crewai_event_bus, _make_task_started_event(_mock_task("t")))
        _emit_sync(crewai_event_bus, _make_agent_exec_start_event(_mock_agent("r")))

    assert client._buffer
    for evt in client._buffer:
        for k in ("event_id", "timestamp", "org_id", "agent_id", "session_id", "run_id", "trace_id", "span_id"):
            assert k in evt, f"missing {k!r} on {evt['event_type']}"
        assert evt["session_id"] == "s-1"
        assert evt["run_id"] == "r-1"
