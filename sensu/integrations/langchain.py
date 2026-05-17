"""
LangChain callback handler for Sensu telemetry.

Drop into any LangChain chain, agent, or LLM via the ``callbacks`` list to
capture LLM calls, tool calls, streaming TTFT, retry/fallback chains, and
chain step boundaries automatically.

Usage::

    from sensu import SensuClient, SensuCallbackHandler

    client = SensuClient({"from_env": True})
    handler = SensuCallbackHandler(client=client)

    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
    await chain.ainvoke({"input": user_message})

Requires the ``langchain`` extra::

    pip install 'sensu-sdk[langchain]'
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
        # LangChain 1.x moved BaseCallbackHandler to `langchain_core.callbacks.base`.
        # Older 0.x releases still have it at `langchain.callbacks.base`. Accept either
        # so customers on either major version can use this handler.
        _err: Optional[ImportError] = None
        for _module in ("langchain_core.callbacks.base", "langchain.callbacks.base"):
            try:
                __import__(_module, fromlist=["BaseCallbackHandler"])
                _err = None
                break
            except ImportError as exc:
                _err = exc
        if _err is not None:
            raise ImportError(
                "langchain is required for SensuCallbackHandler. "
                "Install with: pip install 'sensu-sdk[langchain]'"
            ) from _err

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
        # Carry model + provider from start to end so completion events aren't 'unknown'.
        self._llm_models: Dict[str, str] = {}
        self._llm_providers: Dict[str, str] = {}
        self._tool_call_ids: Dict[str, str] = {}
        self._tool_names: Dict[str, str] = {}
        self._last_tool_call_id_by_name: Dict[str, str] = {}
        self._failed_tool_call_ids: Set[str] = set()
        # When the previous LLM call errored, the next start is tagged is_fallback.
        self._last_llm_errored: bool = False

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

        is_fallback = self._last_llm_errored
        self._last_llm_errored = False

        name: str = (serialized or {}).get("name", "")
        model = name or "unknown"
        provider = _infer_provider(name)
        self._llm_models[rid] = model
        self._llm_providers[rid] = provider

        evt: Dict[str, Any] = {
            **self._base(),
            "event_type": "llm.request.started",
            "llm_call_id": llm_call_id,
            "provider": provider,
            "model": model,
        }
        if is_fallback:
            evt["is_fallback"] = True
        self.client.enqueue(evt)

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
        is_streamed = rid in self._stream_token_counts
        self._stream_token_counts.pop(rid, None)
        llm_call_id = self._llm_call_ids.pop(rid, None)
        model = self._llm_models.pop(rid, "unknown")
        provider = self._llm_providers.pop(rid, "langchain")

        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = (llm_output.get("tokenUsage") or {}) if isinstance(llm_output, dict) else {}

        self.client.enqueue({
            **self._base(),
            "event_type": "llm.request.completed",
            "llm_call_id": llm_call_id,
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "ttft_ms": ttft,
            "streamed": is_streamed,
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
        model = self._llm_models.pop(rid, "unknown")
        provider = self._llm_providers.pop(rid, "langchain")
        self._last_llm_errored = True
        self.client.enqueue({
            **self._base(),
            "event_type": "llm.request.completed",
            "llm_call_id": llm_call_id,
            "provider": provider,
            "model": model,
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
