"""Tests for per-call ``capture_bodies`` on ``track_tool``.

Implements the SDK side of TOOL_IO_CAPTURE_PLAN.md §5.2 + §5.4 + §11.
Two layers:

1. Direct unit tests on ``serialize_tool_bodies_for_capture`` —
   pinned semantics for the cross-SDK invariants (default off,
   opt-in, JSON-serialize both, 256 KB truncation marker, skip-both
   on serialization failure, ``default=str`` fallback for datetime /
   Decimal / etc).
2. End-to-end wire-shape tests that drive ``client.track_tool`` and
   assert what ``tool.call.completed`` looks like on the buffer —
   whether ``input_body`` / ``output_body`` are present and what
   shape they take.
"""
from __future__ import annotations

import decimal
import datetime
import json
from typing import Any, Dict, List

import pytest

from sensu._client import SensuClient, serialize_tool_bodies_for_capture


# ---------------------------------------------------------------------------
# Layer 1 — pure helper
# ---------------------------------------------------------------------------


def test_opt_out_returns_empty_dict() -> None:
    out = serialize_tool_bodies_for_capture(
        {"q": "a"}, {"ok": 1}, capture_bodies=False, args_provided=True,
    )
    assert out == {}


def test_opt_in_but_args_not_provided_returns_empty_dict() -> None:
    # Cross-SDK parity with sdk-ts: a caller that opts in without
    # passing args gets metadata-only (no half-capture).
    out = serialize_tool_bodies_for_capture(
        None, {"ok": 1}, capture_bodies=True, args_provided=False,
    )
    assert out == {}


def test_opt_in_with_json_serializable_args_and_result() -> None:
    out = serialize_tool_bodies_for_capture(
        {"query": "find user@example.com"},
        {"matches": 1, "top": {"email": "user@example.com"}},
        capture_bodies=True,
        args_provided=True,
    )
    assert out["input_body"]  == json.dumps({"query": "find user@example.com"}, ensure_ascii=False)
    assert out["output_body"] == json.dumps(
        {"matches": 1, "top": {"email": "user@example.com"}}, ensure_ascii=False,
    )


def test_opt_in_with_primitives() -> None:
    out = serialize_tool_bodies_for_capture("hello", 42, capture_bodies=True, args_provided=True)
    assert out["input_body"]  == '"hello"'
    assert out["output_body"] == "42"


def test_opt_in_with_explicit_none_args_captures_null() -> None:
    # Cross-SDK parity: explicit None != omitted. Captured as JSON "null".
    out = serialize_tool_bodies_for_capture(
        None, {"ok": 1}, capture_bodies=True, args_provided=True,
    )
    assert out["input_body"]  == "null"
    assert out["output_body"] == '{"ok": 1}'


def test_opt_in_with_none_result_captures_null() -> None:
    out = serialize_tool_bodies_for_capture(
        {"q": "a"}, None, capture_bodies=True, args_provided=True,
    )
    assert out["input_body"]  == '{"q": "a"}'
    assert out["output_body"] == "null"


def test_opt_in_with_circular_args_skips_both_bodies() -> None:
    cyclic: Dict[str, Any] = {}
    cyclic["self"] = cyclic
    out = serialize_tool_bodies_for_capture(
        cyclic, {"ok": 1}, capture_bodies=True, args_provided=True,
    )
    # json.dumps raises ValueError on circular reference → skip both.
    assert out == {}


def test_opt_in_with_circular_result_skips_both_bodies() -> None:
    cyclic: Dict[str, Any] = {}
    cyclic["self"] = cyclic
    out = serialize_tool_bodies_for_capture(
        {"q": "a"}, cyclic, capture_bodies=True, args_provided=True,
    )
    assert out == {}


def test_default_str_fallback_handles_datetime() -> None:
    # Per §5.2, json.dumps(default=str) means datetime / Decimal / UUID /
    # custom objects fall back to str(obj) — they don't trigger the
    # serialization-failure path. Narrower failure surface than sdk-ts
    # by design (Python idiom: lean on __str__).
    ts = datetime.datetime(2026, 5, 13, 10, 0, 0, tzinfo=datetime.timezone.utc)
    out = serialize_tool_bodies_for_capture(
        {"called_at": ts}, {"ok": True}, capture_bodies=True, args_provided=True,
    )
    assert "2026-05-13" in out["input_body"]


def test_default_str_fallback_handles_decimal() -> None:
    out = serialize_tool_bodies_for_capture(
        {"amount": decimal.Decimal("12.34")},
        {"ok": True},
        capture_bodies=True,
        args_provided=True,
    )
    assert "12.34" in out["input_body"]


def test_body_exactly_at_cap_is_preserved_verbatim() -> None:
    # JSON.dumps wraps a string in quotes — for a final wire length
    # of exactly 262144, the inner string is 262142 chars.
    inner = "x" * 262_142
    out = serialize_tool_bodies_for_capture(
        inner, "ok", capture_bodies=True, args_provided=True,
    )
    assert len(out["input_body"]) == 262_144
    assert not out["input_body"].endswith("[truncated]")


def test_oversize_body_is_truncated_at_cap_with_marker() -> None:
    giant = "x" * 300_000
    out = serialize_tool_bodies_for_capture(
        giant, "ok", capture_bodies=True, args_provided=True,
    )
    assert len(out["input_body"]) == 262_144
    # Cross-SDK marker — same byte sequence as sdk-ts / sdk-go.
    assert out["input_body"].endswith(" …[truncated]")
    # The other side wasn't oversize → preserved unchanged.
    assert out["output_body"] == '"ok"'


def test_both_oversize_are_truncated_independently() -> None:
    big_in  = "a" * 300_000
    big_out = "b" * 300_000
    out = serialize_tool_bodies_for_capture(
        big_in, big_out, capture_bodies=True, args_provided=True,
    )
    assert len(out["input_body"])  == 262_144
    assert len(out["output_body"]) == 262_144
    assert out["input_body"].endswith(" …[truncated]")
    assert out["output_body"].endswith(" …[truncated]")


# ---------------------------------------------------------------------------
# Layer 2 — track_tool wire shape
# ---------------------------------------------------------------------------


def _make_client(**overrides: Any) -> SensuClient:
    return SensuClient({
        "api_key":               "test",
        "base_url":              "http://localhost:9999",
        "agent_id":              "agent-1",
        "org_id":                "org-1",
        "batch_size":            100,
        "flush_interval_ms":     999_999,
        "disable_live_pricing":  True,
        **overrides,
    })


def _collected_events(client: SensuClient) -> List[Dict[str, Any]]:
    return list(client._buffer)


def _find_tool_completed(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    for e in events:
        if e.get("event_type") == "tool.call.completed":
            return e
    raise AssertionError("tool.call.completed not found")


@pytest.mark.asyncio
async def test_track_tool_default_no_body_fields() -> None:
    client = _make_client()

    async def body(_run: Any) -> None:
        await client.track_tool("crm_lookup", lambda: _ok({"matches": 1}))

    await client.run({}, body)
    evt = _find_tool_completed(_collected_events(client))
    assert "input_body" not in evt
    assert "output_body" not in evt
    assert evt["status"] == "success"


@pytest.mark.asyncio
async def test_track_tool_opt_in_carries_both_bodies() -> None:
    client = _make_client()

    async def body(_run: Any) -> None:
        await client.track_tool(
            "crm_lookup",
            lambda: _ok({"matches": 1, "top": {"email": "user@example.com"}}),
            args={"query": "find user@example.com"},
            capture_bodies=True,
        )

    await client.run({}, body)
    evt = _find_tool_completed(_collected_events(client))
    assert evt["input_body"]  == json.dumps({"query": "find user@example.com"}, ensure_ascii=False)
    assert evt["output_body"] == json.dumps(
        {"matches": 1, "top": {"email": "user@example.com"}}, ensure_ascii=False,
    )
    assert evt["status"] == "success"
    assert isinstance(evt["latency_ms"], float)
    assert isinstance(evt["tool_call_id"], str)


@pytest.mark.asyncio
async def test_track_tool_opt_in_without_args_keyword_skips_capture() -> None:
    # Per the cross-SDK rule: capture_bodies=True but caller didn't
    # pass `args` keyword → metadata-only (no half-capture).
    client = _make_client()

    async def body(_run: Any) -> None:
        await client.track_tool(
            "crm_lookup", lambda: _ok({"ok": True}), capture_bodies=True,
        )

    await client.run({}, body)
    evt = _find_tool_completed(_collected_events(client))
    assert "input_body" not in evt
    assert "output_body" not in evt


@pytest.mark.asyncio
async def test_track_tool_opt_in_on_error_path_captures_input_and_null_output() -> None:
    # When fn raises, the SDK records status="error" and ``result``
    # stays ``None``. Python's ``json.dumps(None) == "null"`` is valid,
    # so both bodies still land — input_body has the user-supplied
    # args, output_body is the JSON null sentinel. This diverges from
    # sdk-ts (where undefined result skips capture) but mirrors
    # Python's idiom: ``None`` is a real, serializable value.
    client = _make_client()

    async def boom() -> Any:
        raise RuntimeError("nope")

    async def body(_run: Any) -> None:
        await client.track_tool(
            "crm_lookup", boom, args={"query": "a"}, capture_bodies=True,
        )

    with pytest.raises(RuntimeError):
        await client.run({}, body)
    evt = _find_tool_completed(_collected_events(client))
    assert evt["status"] == "error"
    assert evt["input_body"]  == '{"query": "a"}'
    assert evt["output_body"] == "null"


@pytest.mark.asyncio
async def test_track_tool_opt_in_circular_result_skips_both_bodies() -> None:
    client = _make_client()
    cyclic: Dict[str, Any] = {}
    cyclic["self"] = cyclic

    async def returns_cyclic() -> Any:
        return cyclic

    async def body(_run: Any) -> None:
        await client.track_tool(
            "crm_lookup", returns_cyclic, args={"q": "a"}, capture_bodies=True,
        )

    await client.run({}, body)
    evt = _find_tool_completed(_collected_events(client))
    assert "input_body" not in evt
    assert "output_body" not in evt
    assert evt["status"] == "success"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ok(value: Any) -> Any:
    return value
