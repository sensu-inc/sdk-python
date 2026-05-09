"""
LangChain callback handler for Sensu telemetry.

STATUS: Work in progress — mirrors the TypeScript LangChain integration.

Usage:
    from sensu.integrations.langchain import SensuCallbackHandler
    handler = SensuCallbackHandler(client=sensu_client)
    chain = LLMChain(..., callbacks=[handler])
"""
from __future__ import annotations

import datetime
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

if TYPE_CHECKING:
    from sensu._client import SensuClient


def _infer_provider(name: str) -> str:
    n = name.lower()
    if "anthropic" in n or "claude" in n:
        return "anthropic"
    if "openai" in n or "gpt" in n:
        return "openai"
    if "google" in n or "gemini" in n:
        return "google"
    if "ollama" in n or "local" in n:
        return "local"
    if "bedrock" in n:
        return "aws"
    if "cohere" in n:
        return "cohere"
    if "mistral" in n:
        return "mistral"
    return "langchain"


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _new_id() -> str:
    return str(uuid.uuid4())


class SensuCallbackHandler:
    """
    LangChain BaseCallbackHandler subclass that emits Sensu telemetry events.

    Requires: pip install 'sensu-sdk[langchain]'
    """

    STREAM_EMIT_EVERY = 10

    def __init__(
        self,
        client: "SensuClient",
        *,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        try:
            from langchain.callbacks.base import BaseCallbackHandler as _Base  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "langchain is required for SensuCallbackHandler. "
                "Install with: pip install 'sensu-sdk[langchain]'"
            ) from exc

        self.client = client
        self._session_id = session_id or _new_id()
        self._run_id = run_id or _new_id()
        self._trace_id = _new_id()

        # Keyed by LangChain run_id (UUID string)
        self._llm_start_times: Dict[str, float] = {}
        self._tool_start_times: Dict[str, float] = {}
        self._step_ids: Dict[str, str] = {}
        self._first_token_times: Dict[str, float] = {}
        self._stream_token_counts: Dict[str, int] = {}
        self._llm_call_ids: Dict[str, str] = {}
        self._tool_call_ids: Dict[str, str] = {}
        self._tool_names: Dict[str, str] = {}
        self._last_tool_call_id_by_name: Dict[str, str] = {}
        self._failed_tool_call_ids: Set[str] = set()

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

    # -- Chain ---------------------------------------------------------------

    async def on_chain_start(
        self, serialized: Any, inputs: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        step_id = _new_id()
        rid = str(run_id)
        self._step_ids[rid] = step_id
        self.client.enqueue({
            **self._base(),
            "step_id": step_id,
            "event_type": "agent.step.started",
            "step_type": "chain",
            "sequence": 0,
        })

    async def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        step_id = self._step_ids.pop(rid, None)
        self.client.enqueue({
            **self._base(),
            **({"step_id": step_id} if step_id else {}),
            "event_type": "agent.step.completed",
        })

    # -- LLM -----------------------------------------------------------------

    async def on_llm_start(
        self, serialized: Any, prompts: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        rid = str(run_id)
        llm_call_id = _new_id()
        self._llm_start_times[rid] = time.monotonic() * 1000
        self._llm_call_ids[rid] = llm_call_id
        self._stream_token_counts.pop(rid, None)
        self._first_token_times.pop(rid, None)
        name: str = (serialized or {}).get("name", "")
        self.client.enqueue({
            **self._base(),
            "event_type": "llm.request.started",
            "llm_call_id": llm_call_id,
            "provider": _infer_provider(name),
            "model": name or "unknown",
        })

    async def on_llm_new_token(self, token: str, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        now_ms = time.monotonic() * 1000
        self._first_token_times.setdefault(rid, now_ms)
        count = self._stream_token_counts.get(rid, 0) + 1
        self._stream_token_counts[rid] = count
        if count % self.STREAM_EMIT_EVERY == 0:
            start_ms = self._llm_start_times.get(rid)
            first_ms = self._first_token_times.get(rid)
            ttft = (first_ms - start_ms) if (start_ms and first_ms) else None
            self.client.enqueue({
                **self._base(),
                "event_type": "stream.token.received",
                "llm_call_id": self._llm_call_ids.get(rid),
                "tokens_so_far": count,
                "ttft_ms": ttft,
            })

    async def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        start_ms = self._llm_start_times.pop(rid, None)
        latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None
        first_ms = self._first_token_times.pop(rid, None)
        ttft = (first_ms - start_ms) if (start_ms and first_ms) else None
        self._stream_token_counts.pop(rid, None)
        llm_call_id = self._llm_call_ids.pop(rid, None)

        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = (llm_output.get("tokenUsage") or {}) if isinstance(llm_output, dict) else {}

        self.client.enqueue({
            **self._base(),
            "event_type": "llm.request.completed",
            "llm_call_id": llm_call_id,
            "provider": "langchain",
            "model": "unknown",
            "latency_ms": latency_ms,
            "ttft_ms": ttft,
            "status": "success",
            "input_tokens": token_usage.get("promptTokens"),
            "output_tokens": token_usage.get("completionTokens"),
            "total_tokens": token_usage.get("totalTokens"),
        })

    async def on_llm_error(self, error: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        start_ms = self._llm_start_times.pop(rid, None)
        latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None
        self._first_token_times.pop(rid, None)
        self._stream_token_counts.pop(rid, None)
        llm_call_id = self._llm_call_ids.pop(rid, None)
        self.client.enqueue({
            **self._base(),
            "event_type": "llm.request.completed",
            "llm_call_id": llm_call_id,
            "provider": "langchain",
            "model": "unknown",
            "latency_ms": latency_ms,
            "status": "error",
        })

    # -- Tool ----------------------------------------------------------------

    async def on_tool_start(
        self, serialized: Any, input_str: str, *, run_id: Any, **kwargs: Any
    ) -> None:
        rid = str(run_id)
        tool_name: str = (serialized or {}).get("name", "unknown")
        tool_call_id = _new_id()
        self._tool_start_times[rid] = time.monotonic() * 1000
        self._tool_call_ids[rid] = tool_call_id
        self._tool_names[rid] = tool_name

        prev_id = self._last_tool_call_id_by_name.get(tool_name)
        retry_of = prev_id if (prev_id and prev_id in self._failed_tool_call_ids) else None
        self._last_tool_call_id_by_name[tool_name] = tool_call_id

        evt = {
            **self._base(),
            "event_type": "tool.call.started",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
        if retry_of:
            evt["retry_of"] = retry_of
        self.client.enqueue(evt)

    async def on_tool_end(self, output: str, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        start_ms = self._tool_start_times.pop(rid, None)
        latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None
        tool_call_id = self._tool_call_ids.pop(rid, None)
        tool_name = self._tool_names.pop(rid, "unknown")
        self.client.enqueue({
            **self._base(),
            "event_type": "tool.call.completed",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "latency_ms": latency_ms,
            "status": "success",
            "output_size_bytes": len((output or "").encode("utf-8")),
        })

    async def on_tool_error(self, error: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        start_ms = self._tool_start_times.pop(rid, None)
        latency_ms = (time.monotonic() * 1000 - start_ms) if start_ms else None
        tool_call_id = self._tool_call_ids.pop(rid, None)
        tool_name = self._tool_names.pop(rid, "unknown")
        if tool_call_id:
            self._failed_tool_call_ids.add(tool_call_id)
        self.client.enqueue({
            **self._base(),
            "event_type": "tool.call.completed",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "latency_ms": latency_ms,
            "status": "error",
        })
