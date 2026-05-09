from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sensu._client import RunHandle, SensuClient


class WrapAnthropicOptions:
    def __init__(
        self,
        client: "SensuClient",
        *,
        run_handle: Optional["RunHandle"] = None,
        default_provider: str = "anthropic",
    ) -> None:
        self.client = client
        self.run_handle = run_handle
        self.default_provider = default_provider


def wrap_anthropic(anthropic_client: Any, opts: WrapAnthropicOptions) -> Any:
    """
    Wrap an Anthropic client so all messages.create() calls are automatically
    tracked as LLM telemetry events.

    Context resolution order:
      1. opts.run_handle  — explicit handle (useful in sync or non-async contexts)
      2. client.get_active_run()  — ContextVar set by sensu.run()
      3. Standalone event  — emitted without a run/step so data is never lost

    Usage with async context propagation:
        sensu_client = SensuClient({"from_env": True})
        anthropic = wrap_anthropic(Anthropic(), WrapAnthropicOptions(client=sensu_client))

        async def handler():
            async with sensu_client.run({}) as run:
                await anthropic.messages.create(model=..., ...)  # auto-tracked

    Usage with an explicit run handle:
        run = sensu_client.start_run()
        anthropic = wrap_anthropic(Anthropic(), WrapAnthropicOptions(client=sensu_client, run_handle=run))
        await anthropic.messages.create(...)
        await run.end()
    """
    from sensu._pricing import estimate_cost
    from sensu._utils import new_id, utcnow_iso

    client = opts.client
    provider = opts.default_provider
    original_create = anthropic_client.messages.create

    async def _patched_create(*args: Any, **kwargs: Any) -> Any:
        # Accept both positional-dict and keyword-only call styles
        if args and isinstance(args[0], dict):
            kwargs = {**args[0], **kwargs}

        model: str = kwargs.get("model", "unknown")

        run = opts.run_handle or client.get_active_run()
        step = run.start_step({"name": "anthropic-completion", "step_type": "llm"}) if run else None

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
        input_tokens = (getattr(usage, "input_tokens", 0) or 0)
        input_tokens += (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cached = getattr(usage, "cache_read_input_tokens", None)
        actual_model: str = getattr(result, "model", model)

        # Resolve pricing dynamically
        cost: Optional[float] = None
        if input_tokens or output_tokens:
            input_price, output_price = await client.resolve_pricing(provider, actual_model)
            cost = estimate_cost(input_price, output_price, input_tokens, output_tokens)

        raw_opts: dict[str, Any] = {
            "provider": provider,
            "model": actual_model,
            "input_tokens": input_tokens or None,
            "output_tokens": output_tokens or None,
            "total_tokens": (input_tokens + output_tokens) or None,
            "latency_ms": latency_ms,
            "status": status,
        }
        if cached is not None:
            raw_opts["cached_input_tokens"] = cached
        if cost is not None:
            raw_opts["cost_usd_estimate"] = cost

        if step:
            step.record_llm(raw_opts)  # type: ignore[arg-type]
            await step.end()
        else:
            # No active run — emit a standalone event so data is never silently dropped
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

    anthropic_client.messages.create = _patched_create
    return anthropic_client
