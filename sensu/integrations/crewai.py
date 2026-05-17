"""
CrewAI event listener for Sensu telemetry.

CrewAI doesn't use LangChain callbacks natively — it has its own event bus
(``crewai_event_bus``) that emits richer multi-agent lifecycle events than
the LangChain callback surface. This listener subscribes to that bus and
maps each event into Sensu's wire format.

Mapping:
- ``CrewKickoffStartedEvent``        → (run boundary; no Sensu event emitted —
                                        run lifecycle is owned by SensuClient)
- ``TaskStartedEvent``                → ``agent.step.started`` (step_type='crewai_task')
- ``TaskCompletedEvent`` / ``TaskFailedEvent`` → ``agent.step.completed``
- ``AgentExecutionStartedEvent``      → ``agent.spawned`` (once per role) +
                                        ``agent.handoff`` on role switch
- ``LLMCallStartedEvent``             → ``llm.request.started``
- ``LLMCallCompletedEvent`` / Failed  → ``llm.request.completed``
- ``ToolUsageStartedEvent``           → ``tool.call.started``
- ``ToolUsageFinishedEvent`` / Error  → ``tool.call.completed``

Multi-agent identity mapping (see NATIVE_FRAMEWORK_INTEGRATIONS_V1.md §2.3):

  SensuClient.agent_id == the **Crew orchestrator** label (e.g. ``"research-crew"``).
  Each CrewAI agent inside the crew gets a child id ``"{orchestrator}::{role}"``,
  which populates ``agent_spawns.child_agent_id`` and feeds the multi-agent map.

Usage::

    import sensu
    from sensu.integrations.crewai import SensuCrewListener
    from crewai import Agent, Task, Crew

    client = sensu.SensuClient({"from_env": True, "agent_id": "research-crew"})
    listener = SensuCrewListener(client=client)

    crew = Crew(agents=[...], tasks=[...])
    result = crew.kickoff(inputs={"topic": "..."})  # listener fires automatically

Requires the ``crewai`` extra::

    pip install 'sensu-sdk[crewai]'
"""
from __future__ import annotations

import datetime
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from sensu._client import SensuClient

# Module-load import. crewai is a peer/extra; importing this module without
# crewai installed gives a clear ImportError at use time via the constructor.
try:
    from crewai.events.base_event_listener import BaseEventListener as _BaseEventListener
    from crewai.events.event_bus import CrewAIEventsBus as _CrewAIEventsBus
    from crewai.events.types.crew_events import (
        CrewKickoffStartedEvent as _CrewKickoffStartedEvent,
        CrewKickoffCompletedEvent as _CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent as _CrewKickoffFailedEvent,
    )
    from crewai.events.types.task_events import (
        TaskStartedEvent as _TaskStartedEvent,
        TaskCompletedEvent as _TaskCompletedEvent,
        TaskFailedEvent as _TaskFailedEvent,
    )
    from crewai.events.types.agent_events import (
        AgentExecutionStartedEvent as _AgentExecutionStartedEvent,
        AgentExecutionCompletedEvent as _AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent as _AgentExecutionErrorEvent,
    )
    from crewai.events.types.llm_events import (
        LLMCallStartedEvent as _LLMCallStartedEvent,
        LLMCallCompletedEvent as _LLMCallCompletedEvent,
        LLMCallFailedEvent as _LLMCallFailedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageStartedEvent as _ToolUsageStartedEvent,
        ToolUsageFinishedEvent as _ToolUsageFinishedEvent,
        ToolUsageErrorEvent as _ToolUsageErrorEvent,
    )
    _CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover — covered by the import-error test
    _CREWAI_AVAILABLE = False

    # Fallback stub so the module can still be imported in environments
    # without crewai. The constructor raises a clear ImportError.
    class _BaseEventListener:  # type: ignore[no-redef]
        def __init__(self) -> None:  # pragma: no cover
            pass

        def setup_listeners(self, bus: Any) -> None:  # pragma: no cover
            pass


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _new_id() -> str:
    return str(uuid.uuid4())


class SensuCrewListener(_BaseEventListener):  # type: ignore[misc,valid-type]
    """CrewAI BaseEventListener subclass that emits Sensu telemetry."""

    def __init__(
        self,
        client: "SensuClient",
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not _CREWAI_AVAILABLE:
            raise ImportError(
                "crewai is required for SensuCrewListener. "
                "Install with: pip install 'sensu-sdk[crewai]'"
            )

        self.client = client
        self._session_id = session_id or _new_id()
        self._run_id = run_id or _new_id()
        self._trace_id = _new_id()

        # Per-task step bookkeeping (CrewAI task.id → Sensu step_id)
        self._task_step_ids: Dict[str, str] = {}
        self._task_start_times: Dict[str, float] = {}

        # Per-role agent identity bookkeeping. Roles map to stable child_agent_ids
        # and child_run_ids — first time we see a role we emit agent.spawned;
        # subsequent role switches emit agent.handoff.
        self._spawned_roles: Set[str] = set()
        self._role_child_run_ids: Dict[str, str] = {}
        self._last_role: Optional[str] = None
        self._current_role: Optional[str] = None

        # Tool + LLM call bookkeeping
        self._tool_start_times: Dict[str, float] = {}
        self._tool_call_ids: Dict[str, str] = {}  # tool_name → tool_call_id
        self._failed_tool_calls: Set[str] = set()
        self._last_tool_call_id_by_name: Dict[str, str] = {}

        # LLM lifecycle. CrewAI doesn't tag LLM events with a stable
        # correlation id like LangChain does, so we use a simple stack:
        # the most recent llm_call_id is paired with the next completion.
        self._llm_call_id_stack: List[str] = []
        self._llm_start_times: List[float] = []
        self._last_llm_errored: bool = False

        super().__init__()  # registers handlers via setup_listeners

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _base(self, span_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "event_id": _new_id(),
            "timestamp": _utcnow(),
            "org_id": self.client._org_id,
            "agent_id": self.client._agent_id,
            "session_id": self._session_id,
            "run_id": self._run_id,
            "trace_id": self._trace_id,
            "span_id": span_id or _new_id(),
        }

    def _child_agent_id(self, role: str) -> str:
        """Compose the child agent id from the SensuClient orchestrator + role."""
        orchestrator = self.client._agent_id or "crew"
        return f"{orchestrator}::{role}"

    def _on_role_seen(self, role: str) -> None:
        """Emit agent.spawned on first sight of a role; emit agent.handoff on
        role switches between executions. Idempotent."""
        if role not in self._spawned_roles:
            child_run_id = self._role_child_run_ids.setdefault(role, _new_id())
            self.client.enqueue({
                **self._base(),
                "event_type": "agent.spawned",
                "child_run_id": child_run_id,
                "child_agent_id": self._child_agent_id(role),
                "spawn_reason": f"crewai agent execution: {role}",
            })
            self._spawned_roles.add(role)

        if self._last_role is not None and self._last_role != role:
            self.client.enqueue({
                **self._base(),
                "event_type": "agent.handoff",
                "to_agent_id": self._child_agent_id(role),
                "reason": f"crewai handoff: {self._last_role} → {role}",
            })

        self._last_role = role
        self._current_role = role

    # -----------------------------------------------------------------------
    # Event registration (called by BaseEventListener.__init__)
    # -----------------------------------------------------------------------

    def setup_listeners(self, bus: "_CrewAIEventsBus") -> None:
        listener = self  # closure capture for the decorator handlers

        @bus.on(_CrewKickoffStartedEvent)
        def _on_crew_start(source: Any, event: Any) -> None:
            # No Sensu event emitted — run lifecycle is owned by SensuClient.
            # The listener simply ensures session/run/trace ids stay stable.
            _ = source, event

        @bus.on(_CrewKickoffCompletedEvent)
        def _on_crew_end(source: Any, event: Any) -> None:
            _ = source, event

        @bus.on(_CrewKickoffFailedEvent)
        def _on_crew_failed(source: Any, event: Any) -> None:
            _ = source, event

        # -- Tasks ----------------------------------------------------------

        @bus.on(_TaskStartedEvent)
        def _on_task_start(source: Any, event: Any) -> None:
            task = getattr(event, "task", None)
            task_id = str(task.id) if (task and hasattr(task, "id")) else _new_id()
            step_id = _new_id()
            listener._task_step_ids[task_id] = step_id
            listener._task_start_times[task_id] = time.monotonic() * 1000

            task_name = (
                getattr(task, "name", None)
                or (getattr(task, "description", "") or "")[:80]
                or "task"
            ) if task else "task"

            evt: Dict[str, Any] = {
                **listener._base(),
                "step_id": step_id,
                "event_type": "agent.step.started",
                "step_type": "crewai_task",
                "sequence": 0,
                "task_id": task_id,
                "task_name": task_name,
            }
            if task and hasattr(task, "agent") and task.agent is not None:
                role = getattr(task.agent, "role", None)
                if role:
                    evt["agent_role"] = role
                    evt["child_agent_id"] = listener._child_agent_id(role)
            listener.client.enqueue(evt)

        @bus.on(_TaskCompletedEvent)
        def _on_task_end(source: Any, event: Any) -> None:
            task = getattr(event, "task", None)
            task_id = str(task.id) if (task and hasattr(task, "id")) else None
            step_id = listener._task_step_ids.pop(task_id, None) if task_id else None
            start_ms = listener._task_start_times.pop(task_id, None) if task_id else None
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None

            listener.client.enqueue({
                **listener._base(),
                **({"step_id": step_id} if step_id else {}),
                "event_type": "agent.step.completed",
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                "status": "success",
            })

        @bus.on(_TaskFailedEvent)
        def _on_task_failed(source: Any, event: Any) -> None:
            task = getattr(event, "task", None)
            task_id = str(task.id) if (task and hasattr(task, "id")) else None
            step_id = listener._task_step_ids.pop(task_id, None) if task_id else None
            start_ms = listener._task_start_times.pop(task_id, None) if task_id else None
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None

            listener.client.enqueue({
                **listener._base(),
                **({"step_id": step_id} if step_id else {}),
                "event_type": "agent.step.completed",
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                "status": "error",
            })

        # -- Agent execution ------------------------------------------------

        @bus.on(_AgentExecutionStartedEvent)
        def _on_agent_start(source: Any, event: Any) -> None:
            agent = getattr(event, "agent", None)
            role = getattr(agent, "role", None) if agent else None
            if role:
                listener._on_role_seen(role)

        @bus.on(_AgentExecutionCompletedEvent)
        def _on_agent_end(source: Any, event: Any) -> None:
            _ = source, event  # no Sensu event — task lifecycle covers it

        @bus.on(_AgentExecutionErrorEvent)
        def _on_agent_error(source: Any, event: Any) -> None:
            _ = source, event  # captured at task level

        # -- LLM ------------------------------------------------------------

        @bus.on(_LLMCallStartedEvent)
        def _on_llm_start(source: Any, event: Any) -> None:
            llm_call_id = _new_id()
            listener._llm_call_id_stack.append(llm_call_id)
            listener._llm_start_times.append(time.monotonic() * 1000)

            is_fallback = listener._last_llm_errored
            listener._last_llm_errored = False

            model = getattr(event, "model", None) or "unknown"
            provider = _infer_provider(model)

            evt: Dict[str, Any] = {
                **listener._base(),
                "event_type": "llm.request.started",
                "llm_call_id": llm_call_id,
                "provider": provider,
                "model": model,
            }
            if is_fallback:
                evt["is_fallback"] = True
            if listener._current_role:
                evt["agent_role"] = listener._current_role
                evt["child_agent_id"] = listener._child_agent_id(listener._current_role)
            listener.client.enqueue(evt)

        @bus.on(_LLMCallCompletedEvent)
        def _on_llm_end(source: Any, event: Any) -> None:
            llm_call_id = listener._llm_call_id_stack.pop() if listener._llm_call_id_stack else None
            start_ms = listener._llm_start_times.pop() if listener._llm_start_times else None
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None

            model = getattr(event, "model", None) or "unknown"
            provider = _infer_provider(model)

            listener.client.enqueue({
                **listener._base(),
                "event_type": "llm.request.completed",
                **({"llm_call_id": llm_call_id} if llm_call_id else {}),
                "provider": provider,
                "model": model,
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                "status": "success",
            })

        @bus.on(_LLMCallFailedEvent)
        def _on_llm_failed(source: Any, event: Any) -> None:
            llm_call_id = listener._llm_call_id_stack.pop() if listener._llm_call_id_stack else None
            start_ms = listener._llm_start_times.pop() if listener._llm_start_times else None
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None
            listener._last_llm_errored = True

            model = getattr(event, "model", None) or "unknown"

            listener.client.enqueue({
                **listener._base(),
                "event_type": "llm.request.completed",
                **({"llm_call_id": llm_call_id} if llm_call_id else {}),
                "provider": _infer_provider(model),
                "model": model,
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                "status": "error",
            })

        # -- Tools ----------------------------------------------------------

        @bus.on(_ToolUsageStartedEvent)
        def _on_tool_start(source: Any, event: Any) -> None:
            tool_name = getattr(event, "tool_name", "unknown")
            tool_call_id = _new_id()
            listener._tool_start_times[tool_call_id] = time.monotonic() * 1000
            listener._tool_call_ids[tool_name] = tool_call_id

            prev_id = listener._last_tool_call_id_by_name.get(tool_name)
            retry_of = prev_id if (prev_id and prev_id in listener._failed_tool_calls) else None
            listener._last_tool_call_id_by_name[tool_name] = tool_call_id

            evt: Dict[str, Any] = {
                **listener._base(),
                "event_type": "tool.call.started",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
            }
            if retry_of:
                evt["retry_of"] = retry_of
            agent_role = getattr(event, "agent_role", None) or listener._current_role
            if agent_role:
                evt["agent_role"] = agent_role
            listener.client.enqueue(evt)

        @bus.on(_ToolUsageFinishedEvent)
        def _on_tool_end(source: Any, event: Any) -> None:
            tool_name = getattr(event, "tool_name", "unknown")
            tool_call_id = listener._tool_call_ids.pop(tool_name, None)
            start_ms = (
                listener._tool_start_times.pop(tool_call_id, None)
                if tool_call_id else None
            )
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None

            output = getattr(event, "output", None)
            output_size_bytes = len(str(output).encode("utf-8")) if output is not None else None

            listener.client.enqueue({
                **listener._base(),
                "event_type": "tool.call.completed",
                "tool_name": tool_name,
                **({"tool_call_id": tool_call_id} if tool_call_id else {}),
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                **({"output_size_bytes": output_size_bytes} if output_size_bytes is not None else {}),
                "status": "success",
            })

        @bus.on(_ToolUsageErrorEvent)
        def _on_tool_error(source: Any, event: Any) -> None:
            tool_name = getattr(event, "tool_name", "unknown")
            tool_call_id = listener._tool_call_ids.pop(tool_name, None)
            if tool_call_id:
                listener._failed_tool_calls.add(tool_call_id)
            start_ms = (
                listener._tool_start_times.pop(tool_call_id, None)
                if tool_call_id else None
            )
            latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None

            listener.client.enqueue({
                **listener._base(),
                "event_type": "tool.call.completed",
                "tool_name": tool_name,
                **({"tool_call_id": tool_call_id} if tool_call_id else {}),
                **({"latency_ms": latency_ms} if latency_ms is not None else {}),
                "status": "error",
            })


def _infer_provider(model: str) -> str:
    """Map a model string (e.g. "gpt-4o-mini", "claude-sonnet-4-6") to a provider."""
    if not model:
        return "crewai"
    n = model.lower()
    if "anthropic" in n or "claude" in n:
        return "anthropic"
    if "openai" in n or "gpt" in n or "o1" in n:
        return "openai"
    if "google" in n or "gemini" in n or "vertex" in n:
        return "google"
    if "ollama" in n or "local" in n:
        return "local"
    if "bedrock" in n:
        return "aws"
    if "cohere" in n:
        return "cohere"
    if "mistral" in n:
        return "mistral"
    return "crewai"


__all__ = ["SensuCrewListener"]
