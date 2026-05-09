from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sensu._client import RunHandle, SensuClient


class WrapOpenAIOptions:
    def __init__(
        self,
        client: "SensuClient",
        *,
        run_handle: Optional["RunHandle"] = None,
        default_model: Optional[str] = None,
        default_provider: str = "openai",
    ) -> None:
        self.client = client
        self.run_handle = run_handle
        self.default_model = default_model
        self.default_provider = default_provider


def wrap_openai(openai_client: Any, opts: WrapOpenAIOptions) -> Any:
    """
    Wrap an OpenAI client so all chat.completions.create() calls are
    automatically tracked as LLM telemetry events.

    Context resolution order:
      1. opts.run_handle  — explicit handle
      2. client.get_active_run()  — ContextVar set by sensu.run()
      3. Standalone event  — emitted without a run/step

    Usage:
        sensu_client = SensuClient({"from_env": True})
        openai = wrap_openai(OpenAI(), WrapOpenAIOptions(client=sensu_client))

        async def handler():
            async with sensu_client.run({}) as run:
                await openai.chat.completions.create(model=..., messages=[...])
    """
    from sensu._pricing import estimate_cost
    from sensu._utils import new_id, utcnow_iso

    client = opts.client
    provider = opts.default_provider
    original_create = openai_client.chat.completions.create

    async def _patched_create(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}

        model: str = kwargs.get("model", opts.default_model or "unknown")

        run = opts.run_handle or client.get_active_run()
        step = run.start_step({"name": "openai-completion", "step_type": "llm"}) if run else None

        start_ms = time.monotonic() * 1000
        result: Any = None
        status = "success"
        exc: Optional[BaseException] = None

        try:
            result = await original_create(**kwargs)
        except Exception as e:
            status = "error"
            exc = e

        latency_ms = time.monotonic() * 1000 - start_ms

        usage = getattr(result, "usage", None)
        input_tokens: Optional[int] = getattr(usage, "prompt_tokens", None)
        output_tokens: Optional[int] = getattr(usage, "completion_tokens", None)
        total_tokens: Optional[int] = getattr(usage, "total_tokens", None)
        actual_model: str = getattr(result, "model", model)

        # Resolve pricing dynamically
        cost: Optional[float] = None
        if input_tokens is not None and output_tokens is not None:
            input_price, output_price = await client.resolve_pricing(provider, actual_model)
            cost = estimate_cost(input_price, output_price, input_tokens, output_tokens)

        raw_opts: dict[str, Any] = {
            "provider": provider,
            "model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "status": status,
        }
        if cost is not None:
            raw_opts["cost_usd_estimate"] = cost

        if step:
            step.record_llm(raw_opts)  # type: ignore[arg-type]
            await step.end()
        else:
            client.enqueue({
                "event_id": new_id(),
                "event_type": "llm.request.completed",
                "timestamp": utcnow_iso(),
                "org_id": client._org_id,
                "agent_id": client._agent_id,
                "session_id": new_id(),
                "run_id": new_id(),
                "trace_id": new_id(),
                "span_id": new_id(),
                **{k: v for k, v in raw_opts.items() if v is not None},
            })

        if exc is not None:
            raise exc
        return result

    openai_client.chat.completions.create = _patched_create
    return openai_client
