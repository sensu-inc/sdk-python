"""
Tests for the live-pricing resolution path (post-pivot, v0.12.1).

The Python SDK has never bundled a pricing fallback table; the
`(0.0, 0.0)` sentinel-on-failure behavior is now the correct design
per SDK_CONSOLIDATION_PLAN.md §3c. This test suite locks in:

  - Cache: successful API call is cached + reused without a second fetch
  - Failure paths return (0.0, 0.0) and emit a UserWarning
  - The warning fires at most once per (provider, model) per client lifetime
  - Short-circuit paths (disable_live_pricing, disabled, missing api_key)
    skip the network call entirely
"""
from __future__ import annotations

from typing import Any, Dict, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sensu._pricing import resolve_pricing


def _fresh_state() -> Tuple[Dict[str, Tuple[float, float]], Set[str]]:
    return {}, set()


def _make_resp(status: int, body: Dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=body or {})
    return resp


def _patch_httpx_get(resp: MagicMock | None = None, side_effect: Exception | None = None):
    """Patch httpx.AsyncClient so resolve_pricing's GET is controllable."""
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client_cm)
    client_cm.__aexit__  = AsyncMock(return_value=None)
    if side_effect is not None:
        client_cm.get = AsyncMock(side_effect=side_effect)
    else:
        client_cm.get = AsyncMock(return_value=resp)
    return patch("httpx.AsyncClient", return_value=client_cm)


@pytest.mark.asyncio
async def test_success_caches_and_reuses() -> None:
    cache, warned = _fresh_state()
    resp = _make_resp(200, {"inputPricePer1mTokens": 15, "outputPricePer1mTokens": 75})
    with _patch_httpx_get(resp) as client_cls:
        first = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
        second = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert first == (15.0, 75.0)
    assert second == (15.0, 75.0)
    assert client_cls.call_count == 1  # second call hit the cache
    assert warned == set()


@pytest.mark.asyncio
async def test_4xx_returns_sentinel_and_warns_once() -> None:
    import warnings as warnings_module
    cache, warned = _fresh_state()
    resp = _make_resp(404)

    # First call: warning fires.
    with _patch_httpx_get(resp), pytest.warns(UserWarning, match="API returned 404"):
        first = await resolve_pricing(
            "cohere", "command-r-future",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert first == (0.0, 0.0)
    assert "cohere:command-r-future" in warned

    # Second call: no new sensu warning, still returns sentinel.
    with _patch_httpx_get(resp), warnings_module.catch_warnings(record=True) as record:
        warnings_module.simplefilter("always")
        second = await resolve_pricing(
            "cohere", "command-r-future",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert second == (0.0, 0.0)
    sensu_warnings = [w for w in record if "live pricing unavailable" in str(w.message)]
    assert sensu_warnings == []


@pytest.mark.asyncio
async def test_network_error_returns_sentinel_and_warns() -> None:
    cache, warned = _fresh_state()
    with _patch_httpx_get(side_effect=RuntimeError("ECONNREFUSED")), \
         pytest.warns(UserWarning, match="network error"):
        result = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert result == (0.0, 0.0)


@pytest.mark.asyncio
async def test_200_with_null_rates_treated_as_miss() -> None:
    cache, warned = _fresh_state()
    resp = _make_resp(200, {"inputPricePer1mTokens": None, "outputPricePer1mTokens": None})
    with _patch_httpx_get(resp), pytest.warns(UserWarning, match="null rates"):
        result = await resolve_pricing(
            "anthropic", "mystery",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert result == (0.0, 0.0)


@pytest.mark.asyncio
async def test_disable_live_pricing_skips_fetch_and_warns() -> None:
    cache, warned = _fresh_state()
    with _patch_httpx_get(_make_resp(200)) as client_cls, \
         pytest.warns(UserWarning, match="disable_live_pricing=True"):
        result = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=True, disabled=False, warned=warned,
        )
    assert result == (0.0, 0.0)
    assert client_cls.call_count == 0


@pytest.mark.asyncio
async def test_disabled_client_skips_fetch_and_warns() -> None:
    cache, warned = _fresh_state()
    with _patch_httpx_get(_make_resp(200)) as client_cls, \
         pytest.warns(UserWarning, match="client disabled"):
        result = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=True, warned=warned,
        )
    assert result == (0.0, 0.0)
    assert client_cls.call_count == 0


@pytest.mark.asyncio
async def test_missing_api_key_skips_fetch_and_warns() -> None:
    cache, warned = _fresh_state()
    with _patch_httpx_get(_make_resp(200)) as client_cls, \
         pytest.warns(UserWarning, match="no API key"):
        result = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    assert result == (0.0, 0.0)
    assert client_cls.call_count == 0


@pytest.mark.asyncio
async def test_different_models_warn_independently() -> None:
    cache, warned = _fresh_state()
    resp = _make_resp(500)
    with _patch_httpx_get(resp), pytest.warns(UserWarning) as record:
        await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    with _patch_httpx_get(resp), pytest.warns(UserWarning) as record2:
        await resolve_pricing(
            "openai", "gpt-4o",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False, warned=warned,
        )
    sensu_first  = [w for w in record  if "live pricing unavailable" in str(w.message)]
    sensu_second = [w for w in record2 if "live pricing unavailable" in str(w.message)]
    assert len(sensu_first)  == 1
    assert len(sensu_second) == 1
    assert "anthropic:claude-opus-4-7" in warned
    assert "openai:gpt-4o"             in warned


@pytest.mark.asyncio
async def test_warned_set_is_optional_for_backward_compat() -> None:
    """Service-level callers that don't supply ``warned`` still get the
    sentinel and don't raise — warnings are silently skipped."""
    cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
    with _patch_httpx_get(_make_resp(500)):
        result = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
        )
    assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Cache TTL — uses now_monotonic injection for deterministic time travel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_refetches_after_ttl_expires() -> None:
    cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
    resp = _make_resp(200, {"inputPricePer1mTokens": 15, "outputPricePer1mTokens": 75})

    # First call at t=0, second at t=59s (within 60s TTL — cache hit).
    with _patch_httpx_get(resp) as client_cls:
        await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
            cache_ttl_ms=60_000.0, now_monotonic=0.0,
        )
    with _patch_httpx_get(resp) as client_cls_2:
        await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
            cache_ttl_ms=60_000.0, now_monotonic=59.0,
        )
    # Second call hits cache (no second httpx.AsyncClient instantiation).
    assert client_cls.call_count == 1
    assert client_cls_2.call_count == 0

    # Third call at t=61s — past TTL → refetch.
    with _patch_httpx_get(resp) as client_cls_3:
        await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
            cache_ttl_ms=60_000.0, now_monotonic=61.0,
        )
    assert client_cls_3.call_count == 1


@pytest.mark.asyncio
async def test_cache_picks_up_updated_rates_on_refetch() -> None:
    cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
    with _patch_httpx_get(_make_resp(200, {
        "inputPricePer1mTokens": 15, "outputPricePer1mTokens": 75,
    })):
        before = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
            cache_ttl_ms=1_000.0, now_monotonic=0.0,
        )
    assert before == (15.0, 75.0)

    # Past TTL, server now returns updated rate.
    with _patch_httpx_get(_make_resp(200, {
        "inputPricePer1mTokens": 12, "outputPricePer1mTokens": 60,
    })):
        after = await resolve_pricing(
            "anthropic", "claude-opus-4-7",
            base_url="http://localhost", api_key="k",
            cache=cache, disable_live_pricing=False, disabled=False,
            cache_ttl_ms=1_000.0, now_monotonic=2.0,
        )
    assert after == (12.0, 60.0)


@pytest.mark.asyncio
async def test_ttl_zero_disables_caching() -> None:
    cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
    resp = _make_resp(200, {"inputPricePer1mTokens": 15, "outputPricePer1mTokens": 75})

    for i in range(5):
        with _patch_httpx_get(resp) as client_cls:
            await resolve_pricing(
                "anthropic", "claude-opus-4-7",
                base_url="http://localhost", api_key="k",
                cache=cache, disable_live_pricing=False, disabled=False,
                cache_ttl_ms=0.0, now_monotonic=float(i),
            )
        # Every iteration must have hit the wire (cache is bypassed).
        assert client_cls.call_count == 1, f"iteration {i} did not refetch"


@pytest.mark.asyncio
async def test_ttl_applies_per_provider_model_independently() -> None:
    cache: Dict[str, Tuple[Tuple[float, float], float]] = {}
    resp = _make_resp(200, {"inputPricePer1mTokens": 1, "outputPricePer1mTokens": 2})

    # Prime both cache entries at t=0.
    for provider, model in [("anthropic", "claude-opus-4-7"), ("openai", "gpt-4o")]:
        with _patch_httpx_get(resp):
            await resolve_pricing(
                provider, model,
                base_url="http://localhost", api_key="k",
                cache=cache, disable_live_pricing=False, disabled=False,
                cache_ttl_ms=60_000.0, now_monotonic=0.0,
            )
    assert len(cache) == 2

    # Both within TTL at t=30s — both cache-hit.
    for provider, model in [("anthropic", "claude-opus-4-7"), ("openai", "gpt-4o")]:
        with _patch_httpx_get(resp) as client_cls:
            await resolve_pricing(
                provider, model,
                base_url="http://localhost", api_key="k",
                cache=cache, disable_live_pricing=False, disabled=False,
                cache_ttl_ms=60_000.0, now_monotonic=30.0,
            )
        assert client_cls.call_count == 0

    # Both past TTL at t=90s — both refetch.
    for provider, model in [("anthropic", "claude-opus-4-7"), ("openai", "gpt-4o")]:
        with _patch_httpx_get(resp) as client_cls:
            await resolve_pricing(
                provider, model,
                base_url="http://localhost", api_key="k",
                cache=cache, disable_live_pricing=False, disabled=False,
                cache_ttl_ms=60_000.0, now_monotonic=90.0,
            )
        assert client_cls.call_count == 1


@pytest.mark.asyncio
async def test_client_propagates_pricing_cache_ttl_ms() -> None:
    """SensuClient round-trips pricing_cache_ttl_ms into the resolver and
    defaults to 3_600_000 (1 hour) when omitted."""
    from sensu import SensuClient

    default_client = SensuClient({
        "api_key": "k", "base_url": "http://localhost",
        "agent_id": "a", "org_id": "o",
        "batch_size": 100, "flush_interval_ms": 999_999,
        "disable_live_pricing": True,
    })
    assert default_client._pricing_cache_ttl_ms == 3_600_000

    custom_client = SensuClient({
        "api_key": "k", "base_url": "http://localhost",
        "agent_id": "a", "org_id": "o",
        "batch_size": 100, "flush_interval_ms": 999_999,
        "disable_live_pricing": True,
        "pricing_cache_ttl_ms": 60_000,
    })
    assert custom_client._pricing_cache_ttl_ms == 60_000

    zero_client = SensuClient({
        "api_key": "k", "base_url": "http://localhost",
        "agent_id": "a", "org_id": "o",
        "batch_size": 100, "flush_interval_ms": 999_999,
        "disable_live_pricing": True,
        "pricing_cache_ttl_ms": 0,
    })
    assert zero_client._pricing_cache_ttl_ms == 0
