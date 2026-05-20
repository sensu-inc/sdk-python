from __future__ import annotations

import warnings
from typing import Dict, Optional, Set, Tuple
from urllib.parse import quote

# Sentinel returned when pricing cannot be resolved from the API.
# Results in a cost estimate of zero rather than a misleading number.
_UNKNOWN: Tuple[float, float] = (0.0, 0.0)


def _warn_once(warned: Set[str], key: str, reason: str) -> None:
    """Emit a single warning per (provider, model) per client lifetime so
    repeated calls with a misconfigured model don't spam the logs.
    """
    if key in warned:
        return
    warned.add(key)
    warnings.warn(
        f"[sensu] live pricing unavailable for {key} ({reason}); "
        f"cost estimates for this model will be 0 until the API call succeeds. "
        f"If this is a custom model, register it via "
        f"POST /api/v1/pricing/org-models.",
        stacklevel=3,
    )


async def resolve_pricing(
    provider: str,
    model: str,
    *,
    base_url: str,
    api_key: str,
    cache: Dict[str, Tuple[float, float]],
    disable_live_pricing: bool,
    disabled: bool,
    warned: Optional[Set[str]] = None,
) -> Tuple[float, float]:
    """
    Fetch model pricing from the Sensu API.

    Resolution order:
      1. Per-client session cache (avoids redundant network calls)
      2. Live API: GET /api/v1/pricing/models/{provider}/{model}
      3. Returns (0.0, 0.0) sentinel if unavailable — cost estimate is omitted
         rather than fabricated.

    On failure (API unreachable, 4xx/5xx, null rates, ``disable_live_pricing``,
    ``disabled``, or missing ``api_key``) emits a ``UserWarning`` at most once
    per ``(provider, model)`` per client lifetime so customers can tell when
    costs are zeros. The server's ingest pipeline reconciles cost from
    ``llm_calls`` + the catalog at query time, so dashboards stay correct.

    Set ``disable_live_pricing=True`` to skip the network call entirely and
    always return the sentinel (useful in tests or as a debug toggle).

    Custom-model pricing should be registered via
    ``POST /api/v1/pricing/org-models`` (see platform docs).
    """
    cache_key = f"{provider}:{model}"
    if cache_key in cache:
        return cache[cache_key]

    if disable_live_pricing or disabled or not api_key:
        if warned is not None:
            reason = (
                "disable_live_pricing=True" if disable_live_pricing
                else "client disabled" if disabled
                else "no API key"
            )
            _warn_once(warned, cache_key, reason)
        return _UNKNOWN

    try:
        import httpx
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                f"{base_url}/api/v1/pricing/models/{quote(provider)}/{quote(model)}",
                headers={"X-API-Key": api_key},
                timeout=5.0,
            )
        if resp.status_code == 200:
            data = resp.json()
            inp = data.get("inputPricePer1mTokens")
            out = data.get("outputPricePer1mTokens")
            if inp is not None and out is not None:
                pair: Tuple[float, float] = (float(inp), float(out))
                cache[cache_key] = pair
                return pair
            if warned is not None:
                _warn_once(warned, cache_key, "API returned 200 with null rates")
        elif warned is not None:
            _warn_once(warned, cache_key, f"API returned {resp.status_code}")
    except Exception as e:
        if warned is not None:
            _warn_once(warned, cache_key, f"network error: {e!r}")

    return _UNKNOWN


def estimate_cost(
    input_price_per_1m: float,
    output_price_per_1m: float,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """
    Compute cost in USD from resolved pricing rates.
    Returns None when rates are the sentinel (0, 0) so callers can omit the
    field rather than emitting a misleading $0.00 estimate.
    """
    if input_price_per_1m == 0.0 and output_price_per_1m == 0.0:
        return None
    return (
        (input_tokens / 1_000_000) * input_price_per_1m
        + (output_tokens / 1_000_000) * output_price_per_1m
    )
