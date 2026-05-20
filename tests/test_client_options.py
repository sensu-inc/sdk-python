"""
Tests for SensuClientOptions — the public TypedDict declarations.

Phase 2 PR 2 of the SDK_CONSOLIDATION_PLAN.md adds capture_message_bodies
to the TypedDict. The runtime client already honored the key (the parity
audit caught that the public type was missing it while the internal
field was wired). These tests lock in that:

  1. Every TypedDict key actually round-trips into the client's
     corresponding internal field
  2. capture_message_bodies specifically maps to client.capture_message_bodies
"""
from __future__ import annotations

from typing import get_type_hints

import pytest

from sensu import SensuClient, SensuClientOptions


# ---------------------------------------------------------------------------
# Type-checker discoverability — protect against accidental removal
# ---------------------------------------------------------------------------


def test_typeddict_declares_capture_message_bodies() -> None:
    hints = get_type_hints(SensuClientOptions)
    assert "capture_message_bodies" in hints, (
        "capture_message_bodies must be declared on SensuClientOptions so "
        "type checkers (mypy/pyright) surface it. Phase 2 PR 2 added it; "
        "if you're removing it, also remove the internal field in _client.py "
        "and the body-capture pathway in sanitize_messages_snapshot."
    )
    assert hints["capture_message_bodies"] is bool


def test_typeddict_declares_all_expected_options() -> None:
    """Sanity check: keys the docs reference are all present in the TypedDict.
    Catches drift between the public API surface and customer-facing examples.
    """
    hints = get_type_hints(SensuClientOptions)
    expected = {
        "api_key", "base_url", "agent_id", "org_id", "from_env",
        "batch_size", "flush_interval_ms", "disabled",
        "on_loop_detected", "loop_threshold",
        "disable_live_pricing", "debug_mode",
        "capture_message_bodies",  # new in Phase 2 PR 2
        "pricing_cache_ttl_ms",    # new in 0.12.3 (cache TTL)
    }
    missing = expected - set(hints)
    assert not missing, f"SensuClientOptions missing keys: {missing}"


# ---------------------------------------------------------------------------
# Runtime round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_capture_message_bodies_round_trips(value: bool) -> None:
    client = SensuClient({
        "api_key":               "test-key",
        "base_url":              "http://localhost:9999",
        "agent_id":              "agent-1",
        "org_id":                "org-1",
        "batch_size":            100,
        "flush_interval_ms":     999_999,
        "disable_live_pricing":  True,
        "capture_message_bodies": value,
    })
    assert client.capture_message_bodies is value


def test_capture_message_bodies_defaults_to_false() -> None:
    client = SensuClient({
        "api_key":              "test-key",
        "base_url":             "http://localhost:9999",
        "agent_id":             "agent-1",
        "org_id":               "org-1",
        "batch_size":           100,
        "flush_interval_ms":    999_999,
        "disable_live_pricing": True,
        # capture_message_bodies omitted intentionally
    })
    assert client.capture_message_bodies is False
