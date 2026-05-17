"""
Unit tests for the Anthropic, OpenAI, and LangChain integrations.

All LLM calls are mocked — no real API keys required.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sensu import SensuClient
from sensu.integrations.anthropic import WrapAnthropicOptions, wrap_anthropic
from sensu.integrations.openai import WrapOpenAIOptions, wrap_openai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> SensuClient:
    return SensuClient({
        "api_key": "test-key",
        "base_url": "http://localhost:9999",
        "agent_id": "agent-1",
        "org_id": "org-1",
        "batch_size": 100,
        "flush_interval_ms": 60_000,
        "disable_live_pricing": True,
    })


def _fake_anthropic_response(model: str = "claude-sonnet-4-6") -> Any:
    resp = MagicMock()
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    resp.usage.cache_read_input_tokens = 20
    resp.usage.cache_creation_input_tokens = 0
    return resp


def _fake_openai_response(model: str = "gpt-4o") -> Any:
    resp = MagicMock()
    resp.model = model
    resp.choices = [MagicMock()]
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 80
    resp.usage.completion_tokens = 40
    resp.usage.total_tokens = 120
    return resp


# ---------------------------------------------------------------------------
# Anthropic integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_anthropic_tracks_llm_call_in_run() -> None:
    client = make_client()

    fake_resp = _fake_anthropic_response()
    original_create = AsyncMock(return_value=fake_resp)

    anthropic_mock = MagicMock()
    anthropic_mock.messages.create = original_create

    wrapped = wrap_anthropic(anthropic_mock, WrapAnthropicOptions(client=client))

    from unittest.mock import patch
    with patch.object(client, "flush", new_callable=AsyncMock):
        async with client.start_run({"session_id": "s1"}) as run:
            client._buffer.clear()
            await wrapped.messages.create(model="claude-sonnet-4-6", messages=[])

    events = client._buffer
    completed = [e for e in events if e["event_type"] == "llm.request.completed"]
    assert completed, "Expected llm.request.completed event"
    assert completed[0]["provider"] == "anthropic"
    assert completed[0]["model"] == "claude-sonnet-4-6"
    assert completed[0]["input_tokens"] == 100   # 100 + 0 cache_creation
    assert completed[0]["output_tokens"] == 50
    assert completed[0]["cached_input_tokens"] == 20


@pytest.mark.asyncio
async def test_wrap_anthropic_standalone_event_without_run() -> None:
    client = make_client()

    fake_resp = _fake_anthropic_response()
    anthropic_mock = MagicMock()
    anthropic_mock.messages.create = AsyncMock(return_value=fake_resp)

    wrapped = wrap_anthropic(anthropic_mock, WrapAnthropicOptions(client=client))
    await wrapped.messages.create(model="claude-sonnet-4-6", messages=[])

    events = client._buffer
    completed = [e for e in events if e["event_type"] == "llm.request.completed"]
    assert completed, "Standalone event should be emitted even without an active run"
    assert completed[0]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_wrap_anthropic_reraises_exception() -> None:
    client = make_client()

    anthropic_mock = MagicMock()
    anthropic_mock.messages.create = AsyncMock(side_effect=RuntimeError("api error"))
    wrapped = wrap_anthropic(anthropic_mock, WrapAnthropicOptions(client=client))

    with pytest.raises(RuntimeError, match="api error"):
        await wrapped.messages.create(model="claude-sonnet-4-6", messages=[])

    # Error event should still be recorded
    events = client._buffer
    completed = [e for e in events if e["event_type"] == "llm.request.completed"]
    assert completed
    assert completed[0]["status"] == "error"


@pytest.mark.asyncio
async def test_wrap_anthropic_explicit_run_handle() -> None:
    client = make_client()

    fake_resp = _fake_anthropic_response()
    anthropic_mock = MagicMock()
    anthropic_mock.messages.create = AsyncMock(return_value=fake_resp)

    from unittest.mock import patch
    with patch.object(client, "flush", new_callable=AsyncMock):
        run = client.start_run()
        wrapped = wrap_anthropic(
            anthropic_mock,
            WrapAnthropicOptions(client=client, run_handle=run),
        )
        client._buffer.clear()
        await wrapped.messages.create(model="claude-sonnet-4-6", messages=[])
        await run.end()

    completed = [e for e in client._buffer if e["event_type"] == "llm.request.completed"]
    assert completed
    # Should be attributed to the explicit run
    assert completed[0]["run_id"] == run.run_id


# ---------------------------------------------------------------------------
# OpenAI integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_openai_tracks_llm_call_in_run() -> None:
    client = make_client()

    fake_resp = _fake_openai_response()
    openai_mock = MagicMock()
    openai_mock.chat = MagicMock()
    openai_mock.chat.completions = MagicMock()
    openai_mock.chat.completions.create = AsyncMock(return_value=fake_resp)

    wrapped = wrap_openai(openai_mock, WrapOpenAIOptions(client=client))

    from unittest.mock import patch
    with patch.object(client, "flush", new_callable=AsyncMock):
        async with client.start_run({"session_id": "s2"}) as run:
            client._buffer.clear()
            await wrapped.chat.completions.create(model="gpt-4o", messages=[])

    completed = [e for e in client._buffer if e["event_type"] == "llm.request.completed"]
    assert completed
    assert completed[0]["provider"] == "openai"
    assert completed[0]["model"] == "gpt-4o"
    assert completed[0]["input_tokens"] == 80
    assert completed[0]["output_tokens"] == 40
    assert completed[0]["total_tokens"] == 120


@pytest.mark.asyncio
async def test_wrap_openai_standalone_event_without_run() -> None:
    client = make_client()

    fake_resp = _fake_openai_response()
    openai_mock = MagicMock()
    openai_mock.chat.completions.create = AsyncMock(return_value=fake_resp)
    wrapped = wrap_openai(openai_mock, WrapOpenAIOptions(client=client))

    await wrapped.chat.completions.create(model="gpt-4o", messages=[])

    completed = [e for e in client._buffer if e["event_type"] == "llm.request.completed"]
    assert completed


@pytest.mark.asyncio
async def test_wrap_openai_reraises_exception() -> None:
    client = make_client()

    openai_mock = MagicMock()
    openai_mock.chat.completions.create = AsyncMock(side_effect=ValueError("rate limited"))
    wrapped = wrap_openai(openai_mock, WrapOpenAIOptions(client=client))

    with pytest.raises(ValueError, match="rate limited"):
        await wrapped.chat.completions.create(model="gpt-4o", messages=[])

    completed = [e for e in client._buffer if e["event_type"] == "llm.request.completed"]
    assert completed
    assert completed[0]["status"] == "error"


# ---------------------------------------------------------------------------
# LangChain integration
# ---------------------------------------------------------------------------


def test_langchain_handler_import_error_without_langchain() -> None:
    """SensuCallbackHandler should raise ImportError when langchain is absent.

    The handler probes both `langchain_core.callbacks.base` (LangChain 1.x) and
    `langchain.callbacks.base` (0.x), so both must be unavailable to trigger.
    """
    import sys
    from unittest.mock import patch

    blocked = {
        "langchain": None,
        "langchain.callbacks": None,
        "langchain.callbacks.base": None,
        "langchain_core": None,
        "langchain_core.callbacks": None,
        "langchain_core.callbacks.base": None,
    }
    with patch.dict(sys.modules, blocked):
        from sensu.integrations.langchain import SensuCallbackHandler
        client = make_client()
        with pytest.raises(ImportError, match="langchain"):
            SensuCallbackHandler(client=client)


@pytest.mark.asyncio
async def test_langchain_handler_llm_start_end() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client, session_id="lc-sess")

    run_id = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatAnthropic"}, [], run_id=run_id)
    await handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

    types = [e["event_type"] for e in client._buffer]
    assert "llm.request.started" in types
    assert "llm.request.completed" in types


@pytest.mark.asyncio
async def test_langchain_handler_tool_start_end() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)

    run_id = uuid.uuid4()
    await handler.on_tool_start({"name": "calculator"}, "1+1", run_id=run_id)
    await handler.on_tool_end("2", run_id=run_id)

    started = [e for e in client._buffer if e["event_type"] == "tool.call.started"]
    completed = [e for e in client._buffer if e["event_type"] == "tool.call.completed"]
    assert started
    assert completed
    assert started[0]["tool_name"] == "calculator"
    assert completed[0]["status"] == "success"


@pytest.mark.asyncio
async def test_langchain_handler_retry_detection() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)

    # First call — fails
    run_id_1 = uuid.uuid4()
    await handler.on_tool_start({"name": "search"}, "query", run_id=run_id_1)
    await handler.on_tool_error(RuntimeError("timeout"), run_id=run_id_1)

    # Second call of same tool — should be detected as retry
    run_id_2 = uuid.uuid4()
    await handler.on_tool_start({"name": "search"}, "query", run_id=run_id_2)

    retry_events = [
        e for e in client._buffer
        if e["event_type"] == "tool.call.started" and e.get("retry_of")
    ]
    assert retry_events, "Second call should be flagged as a retry"


@pytest.mark.asyncio
async def test_langchain_handler_llm_model_provider_carry_forward() -> None:
    """on_llm_end must preserve the model + provider captured at start (not 'unknown')."""
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)
    run_id = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatAnthropic"}, [], run_id=run_id)
    await handler.on_llm_end(MagicMock(llm_output={}), run_id=run_id)

    end = next(e for e in client._buffer if e["event_type"] == "llm.request.completed")
    assert end["model"] == "ChatAnthropic"
    assert end["provider"] == "anthropic"
    assert end["status"] == "success"


@pytest.mark.asyncio
async def test_langchain_handler_llm_error_status() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)
    run_id = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatOpenAI"}, [], run_id=run_id)
    await handler.on_llm_error(RuntimeError("boom"), run_id=run_id)

    end = next(e for e in client._buffer if e["event_type"] == "llm.request.completed")
    assert end["status"] == "error"
    assert end["model"] == "ChatOpenAI"
    assert end["provider"] == "openai"


@pytest.mark.asyncio
async def test_langchain_handler_is_fallback_after_error() -> None:
    """The next LLM start after an error must be tagged is_fallback=True."""
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)

    rid_a = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatOpenAI"}, [], run_id=rid_a)
    await handler.on_llm_error(RuntimeError("boom"), run_id=rid_a)

    rid_b = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatAnthropic"}, [], run_id=rid_b)

    fallback = next(
        e for e in client._buffer
        if e["event_type"] == "llm.request.started" and e.get("is_fallback")
    )
    assert fallback["model"] == "ChatAnthropic"


@pytest.mark.asyncio
async def test_langchain_handler_streaming_emits_every_tenth_token() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)
    run_id = uuid.uuid4()
    await handler.on_llm_start({"name": "ChatAnthropic"}, [], run_id=run_id)
    for _ in range(25):
        await handler.on_llm_new_token("x", run_id=run_id)

    stream_events = [e for e in client._buffer if e["event_type"] == "stream.token.received"]
    # STREAM_EMIT_EVERY = 10 → events at 10 and 20
    assert len(stream_events) == 2
    assert stream_events[0]["tokens_so_far"] == 10
    assert stream_events[1]["tokens_so_far"] == 20


@pytest.mark.asyncio
async def test_langchain_handler_base_fields_on_every_event() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client, session_id="s-1", run_id="r-1")
    await handler.on_chain_start({}, {}, run_id=uuid.uuid4())
    await handler.on_llm_start({"name": "ChatAnthropic"}, [], run_id=uuid.uuid4())

    for ev in client._buffer:
        for k in ("event_id", "timestamp", "org_id", "agent_id", "session_id", "run_id", "trace_id", "span_id"):
            assert k in ev, f"missing {k!r} on {ev['event_type']}"
        assert ev["session_id"] == "s-1"
        assert ev["run_id"] == "r-1"


@pytest.mark.asyncio
async def test_langchain_handler_chain_step_lifecycle() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)
    rid = uuid.uuid4()
    await handler.on_chain_start({}, {}, run_id=rid)
    started = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert started["step_type"] == "chain"
    assert started["step_id"]

    await handler.on_chain_end({}, run_id=rid)
    completed = next(e for e in client._buffer if e["event_type"] == "agent.step.completed")
    assert completed.get("step_id") == started["step_id"]


# ---------------------------------------------------------------------------
# LangGraph (Phase B)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_langgraph_handler_basic_identity() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    assert handler.name == "sensu_langgraph_handler"


@pytest.mark.asyncio
async def test_langgraph_node_detection_in_chain_start() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler
    import uuid

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    await handler.on_chain_start(
        {},
        {},
        run_id=uuid.uuid4(),
        tags=["graph:step:1"],
        metadata={"langgraph_node": "researcher", "langgraph_step": 2},
    )
    evt = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert evt["step_type"] == "langgraph_node"
    assert evt["node_name"] == "researcher"
    assert evt["langgraph_step"] == 2


@pytest.mark.asyncio
async def test_langgraph_handler_fallback_to_chain_when_no_metadata() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler
    import uuid

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    await handler.on_chain_start({}, {}, run_id=uuid.uuid4())
    evt = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert evt["step_type"] == "chain"
    assert "node_name" not in evt


@pytest.mark.asyncio
async def test_langgraph_skips_langsmith_hidden_wrappers() -> None:
    """Channel-write wrappers tagged `langsmith:hidden` must not emit."""
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler
    import uuid

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    rid = uuid.uuid4()
    await handler.on_chain_start(
        {},
        {},
        run_id=rid,
        tags=["langsmith:hidden"],
        metadata={"langgraph_node": "plan_step", "langgraph_step": 0},
    )
    # And the matching end is also a no-op
    await handler.on_chain_end({}, run_id=rid)
    assert client._buffer == []


@pytest.mark.asyncio
async def test_langgraph_plain_chain_with_hidden_tag_but_no_node_still_emits() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler
    import uuid

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    await handler.on_chain_start(
        {}, {}, run_id=uuid.uuid4(), tags=["langsmith:hidden"], metadata=None,
    )
    # Without langgraph_node in metadata, we don't have a clear plumbing
    # signal, so we fall through to a normal chain step.
    evt = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert evt["step_type"] == "chain"


@pytest.mark.asyncio
async def test_langgraph_parent_handler_auto_detects_nodes() -> None:
    """Customers using just SensuCallbackHandler in a mixed project should
    still get langgraph_node steps — detection lives in the parent."""
    pytest.importorskip("langchain")
    from sensu.integrations.langchain import SensuCallbackHandler
    import uuid

    client = make_client()
    handler = SensuCallbackHandler(client=client)
    await handler.on_chain_start(
        {}, {}, run_id=uuid.uuid4(), metadata={"langgraph_node": "analyst"},
    )
    evt = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    assert evt["step_type"] == "langgraph_node"
    assert evt["node_name"] == "analyst"


@pytest.mark.asyncio
async def test_langgraph_node_end_uses_same_step_id() -> None:
    pytest.importorskip("langchain")
    from sensu.integrations.langgraph import SensuLangGraphHandler
    import uuid

    client = make_client()
    handler = SensuLangGraphHandler(client=client)
    rid = uuid.uuid4()
    await handler.on_chain_start(
        {}, {}, run_id=rid, metadata={"langgraph_node": "writer"},
    )
    await handler.on_chain_end({}, run_id=rid)
    started = next(e for e in client._buffer if e["event_type"] == "agent.step.started")
    completed = next(e for e in client._buffer if e["event_type"] == "agent.step.completed")
    assert completed["step_id"] == started["step_id"]
