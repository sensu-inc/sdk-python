"""
Unit tests for SensuClient, RunHandle, and StepHandle.

All tests run without a real API server — the HTTP flush is patched.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sensu
from sensu import SensuClient, RunHandle, StepHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(**overrides: Any) -> SensuClient:
    opts: Dict[str, Any] = {
        "api_key": "test-key",
        "base_url": "http://localhost:9999",
        "agent_id": "agent-1",
        "org_id": "org-1",
        "batch_size": 100,       # large so auto-flush never fires in tests
        "flush_interval_ms": 60_000,
        "disable_live_pricing": True,
        **overrides,
    }
    return SensuClient(opts)


def collected_events(client: SensuClient) -> List[Dict[str, Any]]:
    """Return buffered events without flushing."""
    return list(client._buffer)


# ---------------------------------------------------------------------------
# Enqueue / disabled
# ---------------------------------------------------------------------------


def test_enqueue_adds_event() -> None:
    client = make_client()
    client.enqueue({"event_type": "test.event"})
    assert len(client._buffer) == 1
    assert client._buffer[0]["event_type"] == "test.event"


def test_disabled_client_drops_events() -> None:
    client = make_client(disabled=True)
    client.enqueue({"event_type": "test.event"})
    assert len(client._buffer) == 0


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


def test_start_run_emits_event() -> None:
    client = make_client()
    run = client.start_run({"session_id": "sess-1", "run_type": "test"})
    events = collected_events(client)
    assert any(e["event_type"] == "agent.run.started" for e in events)
    assert run.session_id == "sess-1"
    assert run.agent_id == "agent-1"
    assert run.org_id == "org-1"


def test_start_run_generates_ids_when_omitted() -> None:
    client = make_client()
    run = client.start_run()
    assert run.run_id
    assert run.session_id
    assert run.trace_id


# ---------------------------------------------------------------------------
# RunHandle.start_step
# ---------------------------------------------------------------------------


def test_start_step_emits_event() -> None:
    client = make_client()
    run = client.start_run()
    client._buffer.clear()
    step = run.start_step({"name": "my-step", "step_type": "tool"})
    events = collected_events(client)
    assert any(e["event_type"] == "agent.step.started" for e in events)
    assert step.run_id == run.run_id
    assert step.step_id


def test_start_step_auto_increments_sequence() -> None:
    client = make_client()
    run = client.start_run()
    step1 = run.start_step()
    step2 = run.start_step()
    assert step2._sequence == step1._sequence + 1


# ---------------------------------------------------------------------------
# RunHandle.end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_end_completed_emits_event() -> None:
    client = make_client()
    with patch.object(client, "flush", new_callable=AsyncMock):
        run = client.start_run()
        client._buffer.clear()
        await run.end("completed")
    events = collected_events(client)
    assert any(e["event_type"] == "agent.run.completed" for e in events)


@pytest.mark.asyncio
async def test_run_end_failed_emits_event() -> None:
    client = make_client()
    with patch.object(client, "flush", new_callable=AsyncMock):
        run = client.start_run()
        client._buffer.clear()
        await run.end("failed")
    events = collected_events(client)
    assert any(e["event_type"] == "agent.run.failed" for e in events)


# ---------------------------------------------------------------------------
# RunHandle context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_context_manager_success() -> None:
    client = make_client()
    with patch.object(client, "flush", new_callable=AsyncMock):
        async with client.start_run() as run:
            assert isinstance(run, RunHandle)
    events = collected_events(client)
    assert any(e["event_type"] == "agent.run.completed" for e in events)


@pytest.mark.asyncio
async def test_run_context_manager_failure() -> None:
    client = make_client()
    with patch.object(client, "flush", new_callable=AsyncMock):
        with pytest.raises(ValueError):
            async with client.start_run():
                raise ValueError("boom")
    events = collected_events(client)
    assert any(e["event_type"] == "agent.run.failed" for e in events)


# ---------------------------------------------------------------------------
# sensu.run() high-level API + ContextVar propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sensu_run_sets_active_run() -> None:
    client = make_client()
    captured: list[RunHandle | None] = []

    async def fn(run: RunHandle) -> None:
        captured.append(client.get_active_run())

    with patch.object(client, "flush", new_callable=AsyncMock):
        await client.run({}, fn)

    assert len(captured) == 1
    assert isinstance(captured[0], RunHandle)


@pytest.mark.asyncio
async def test_sensu_run_resets_context_after_completion() -> None:
    client = make_client()

    async def fn(run: RunHandle) -> None:
        pass

    with patch.object(client, "flush", new_callable=AsyncMock):
        await client.run({}, fn)

    assert client.get_active_run() is None


@pytest.mark.asyncio
async def test_sensu_run_nested_contexts_isolated() -> None:
    client = make_client()
    inner_run_ids: list[str] = []

    async def inner(run: RunHandle) -> None:
        active = client.get_active_run()
        if active:
            inner_run_ids.append(active.run_id)

    async def outer(run: RunHandle) -> None:
        with patch.object(client, "flush", new_callable=AsyncMock):
            await client.run({"run_id": "inner-id"}, inner)

    with patch.object(client, "flush", new_callable=AsyncMock):
        await client.run({"run_id": "outer-id"}, outer)

    # Inner context saw the inner run, not the outer
    assert inner_run_ids == ["inner-id"]


# ---------------------------------------------------------------------------
# StepHandle.track_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_track_tool_calls_fn_and_emits_events() -> None:
    client = make_client()
    run = client.start_run()
    step = run.start_step()
    client._buffer.clear()

    async def search() -> str:
        return "result"

    result = await step.track_tool({"tool_name": "search", "fn": search})

    assert result == "result"
    types = [e["event_type"] for e in collected_events(client)]
    assert "tool.call.started" in types
    assert "tool.call.completed" in types


@pytest.mark.asyncio
async def test_track_tool_records_error_status() -> None:
    client = make_client()
    run = client.start_run()
    step = run.start_step()
    client._buffer.clear()

    async def fail() -> str:
        raise RuntimeError("oops")

    with pytest.raises(RuntimeError):
        await step.track_tool({"tool_name": "fail_tool", "fn": fail})

    completed = [e for e in collected_events(client) if e["event_type"] == "tool.call.completed"]
    assert completed[0]["status"] == "error"


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def test_loop_detection_fires_callback() -> None:
    fired: list[tuple[str, int]] = []
    client = make_client(loop_threshold=3, on_loop_detected=lambda t, c: fired.append((t, c)))
    run = client.start_run()
    for _ in range(4):
        client.notify_tool_call(run.run_id, "my_tool")
    assert len(fired) >= 1
    assert fired[0][0] == "my_tool"
    assert fired[0][1] >= 3


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def test_start_session_returns_id_and_emits_event() -> None:
    client = make_client()
    sid = client.start_session({"channel": "web"})
    assert sid
    events = collected_events(client)
    assert any(e["event_type"] == "session.started" for e in events)


def test_resume_session_emits_event() -> None:
    client = make_client()
    sid = client.resume_session({"resumed_from_session_id": "old-session"})
    assert sid
    events = collected_events(client)
    resumed = [e for e in events if e["event_type"] == "session.resumed"]
    assert resumed
    assert resumed[0]["resumed_from_session_id"] == "old-session"


# ---------------------------------------------------------------------------
# Feedback & eval
# ---------------------------------------------------------------------------


def test_record_feedback_emits_event() -> None:
    client = make_client()
    run = client.start_run()
    client._buffer.clear()
    run.record_feedback({"type": "thumbs_up", "comment": "great"})
    events = collected_events(client)
    fb = [e for e in events if e["event_type"] == "feedback.received"]
    assert fb
    assert fb[0]["feedback_type"] == "thumbs_up"
    assert fb[0]["comment"] == "great"


def test_record_eval_score_emits_event() -> None:
    client = make_client()
    run = client.start_run()
    client._buffer.clear()
    run.record_eval_score({"metric": "faithfulness", "score": 0.95})
    events = collected_events(client)
    evals = [e for e in events if e["event_type"] == "eval.score.recorded"]
    assert evals
    assert evals[0]["metric"] == "faithfulness"
    assert evals[0]["score"] == 0.95


# ---------------------------------------------------------------------------
# Multi-agent: spawn_run / handoff
# ---------------------------------------------------------------------------


def test_spawn_run_shares_trace_and_session() -> None:
    client = make_client()
    parent = client.start_run({"session_id": "shared-sess"})
    client._buffer.clear()
    child = client.spawn_run(parent, {"child_agent_id": "child-agent"})
    assert child.trace_id == parent.trace_id
    assert child.session_id == parent.session_id
    events = collected_events(client)
    assert any(e["event_type"] == "agent.spawned" for e in events)
    assert any(e["event_type"] == "agent.run.started" and e["agent_id"] == "child-agent" for e in events)


def test_handoff_emits_event() -> None:
    client = make_client()
    run = client.start_run()
    client._buffer.clear()
    run.handoff({"to_agent_id": "next-agent", "reason": "delegating"})
    events = collected_events(client)
    handoffs = [e for e in events if e["event_type"] == "agent.handoff"]
    assert handoffs
    assert handoffs[0]["to_agent_id"] == "next-agent"


# ---------------------------------------------------------------------------
# Prompt management
# ---------------------------------------------------------------------------


def test_deploy_prompt_version_emits_event() -> None:
    client = make_client()
    client.deploy_prompt_version({
        "template_id": "greeting",
        "new_version": "v2",
        "old_version": "v1",
    })
    events = collected_events(client)
    deploys = [e for e in events if e["event_type"] == "prompt.version.deployed"]
    assert deploys
    assert deploys[0]["new_version"] == "v2"


# ---------------------------------------------------------------------------
# flush — HTTP path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_posts_events_and_clears_buffer() -> None:
    client = make_client()
    client.enqueue({"event_type": "test"})
    assert len(client._buffer) == 1

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"processed": 1, "errors": []}

    with patch("httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_http

        # Patch the lazy client creation
        client._async_http = mock_http
        await client.flush()

    assert len(client._buffer) == 0


@pytest.mark.asyncio
async def test_flush_requeues_on_network_error() -> None:
    client = make_client()
    client.enqueue({"event_type": "test"})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=ConnectionError("down"))
    client._async_http = mock_http

    await client.flush()

    # Event should be back in the buffer
    assert len(client._buffer) == 1
