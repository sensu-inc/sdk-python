from __future__ import annotations

import json
from typing import Any, Dict, Optional


def extract_usage(result: Any, model: str) -> Dict[str, Any]:
    """
    Infer token usage from Anthropic and OpenAI response shapes.
    Returns a dict with keys matching the telemetry event schema.
    """
    
    # Anthropic Python SDK: result.usage.input_tokens / output_tokens
    usage_obj = getattr(result, "usage", None)
    if usage_obj is not None and hasattr(usage_obj, "input_tokens"):
        input_tokens = (getattr(usage_obj, "input_tokens", 0) or 0)
        input_tokens += (getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
        output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
        cached = getattr(usage_obj, "cache_read_input_tokens", None)
        out: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if cached is not None:
            out["cached_input_tokens"] = cached
        return out

    # OpenAI Python SDK: result.usage.prompt_tokens / completion_tokens
    if hasattr(result, "choices") and hasattr(result, "usage"):
        u = result.usage
        if u is not None:
            input_tokens = getattr(u, "prompt_tokens", 0) or 0
            output_tokens = getattr(u, "completion_tokens", 0) or 0
            total = getattr(u, "total_tokens", None)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total if total is not None else input_tokens + output_tokens,
            }

    return {}


def estimate_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except Exception:
        return 0


def extract_stream_chunk_text(chunk: Any) -> str:
    """Extract text delta from Anthropic or OpenAI streaming chunk shapes."""
    if isinstance(chunk, str):
        return chunk

    # Anthropic: ContentBlockDeltaEvent with text_delta
    if getattr(chunk, "type", None) == "content_block_delta":
        delta = getattr(chunk, "delta", None)
        if delta is not None and getattr(delta, "type", None) == "text_delta":
            return getattr(delta, "text", "") or ""

    # OpenAI: chunk.choices[0].delta.content
    choices = getattr(chunk, "choices", None)
    if choices:
        delta = getattr(choices[0], "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content

    return ""


def format_debug_event(event: Dict[str, Any]) -> str:
    etype = event.get("event_type", "?")

    if etype == "llm.request.completed":
        tokens = (event.get("input_tokens") or 0) + (event.get("output_tokens") or 0)
        cached = f" cached={event['cached_input_tokens']}" if event.get("cached_input_tokens") else ""
        cost = f" cost=${event['cost_usd_estimate']:.4f}" if event.get("cost_usd_estimate") else ""
        return (
            f"llm.request.completed   provider={event.get('provider','?')}  "
            f"model={event.get('model','?')}  tokens={tokens}{cached}  "
            f"latency={event.get('latency_ms','?')}ms{cost}"
        )
    if etype == "tool.call.completed":
        return (
            f"tool.call.completed     tool={event.get('tool_name','?')}  "
            f"latency={event.get('latency_ms','?')}ms  status={event.get('status','?')}"
        )
    if etype == "agent.run.started":
        return f"agent.run.started       run={str(event.get('run_id','?'))[:8]}"
    if etype == "agent.run.completed":
        return f"agent.run.completed     run={str(event.get('run_id','?'))[:8]}"
    if etype == "agent.run.failed":
        return f"agent.run.failed        run={str(event.get('run_id','?'))[:8]}"
    if etype == "agent.step.started":
        return f"agent.step.started      step={event.get('step_name') or event.get('step_type','?')}"
    if etype == "agent.step.completed":
        return f"agent.step.completed    step={str(event.get('step_id','?'))[:8]}"

    return etype


def utcnow_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def new_id() -> str:
    import uuid
    return str(uuid.uuid4())
