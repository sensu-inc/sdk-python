"""Tests for capture_message_bodies — the Replay v1 wire contract.

When the flag is False (default), the SDK MUST strip ``body`` from every
message before flushing; when True, the body survives and oversize
bodies are capped at the server schema limit. See planning/REPLAY_V1_PLAN.md §7.
"""
from sensu._client import SensuClient


def _client(capture: bool) -> SensuClient:
    return SensuClient({
        "api_key":             "test",
        "base_url":            "http://localhost:9999",
        "agent_id":            "agent-1",
        "org_id":              "org-1",
        "batch_size":          100,
        "flush_interval_ms":   999_999,
        "disable_live_pricing": True,
        "capture_message_bodies": capture,
    })


def test_strips_body_when_capture_disabled() -> None:
    c = _client(False)
    out = c.sanitize_messages_snapshot([
        {"role": "user",      "token_count": 5, "content_hash": "h1", "body": "hello"},
        {"role": "assistant", "token_count": 7, "content_hash": "h2", "body": "world"},
        {"role": "system",    "token_count": 3, "content_hash": "h3"},
    ])
    assert len(out) == 3
    for m in out:
        assert "body" not in m


def test_preserves_body_when_capture_enabled() -> None:
    c = _client(True)
    out = c.sanitize_messages_snapshot([
        {"role": "user", "token_count": 5, "content_hash": "h1", "body": "hello"},
        {"role": "user", "token_count": 0, "content_hash": "h2", "body": ""},
        {"role": "user", "token_count": 1, "content_hash": "h3"},  # no body
    ])
    assert out[0]["body"] == "hello"
    assert out[1]["body"] == ""
    assert "body" not in out[2]


def test_caps_body_at_server_schema_limit() -> None:
    c = _client(True)
    giant = "x" * 80_000
    out = c.sanitize_messages_snapshot([
        {"role": "user", "token_count": 1, "content_hash": "h", "body": giant},
    ])
    assert len(out[0]["body"]) == 65_536


def test_preserves_non_body_fields_regardless_of_capture() -> None:
    msg = {
        "role":         "assistant",
        "tool_name":    "search",
        "token_count":  42,
        "content_hash": "abc123",
        "body":         "sensitive",
    }
    off = _client(False).sanitize_messages_snapshot([msg])[0]
    assert off["role"] == "assistant"
    assert off["tool_name"] == "search"
    assert off["token_count"] == 42
    assert off["content_hash"] == "abc123"
    assert "body" not in off

    on = _client(True).sanitize_messages_snapshot([msg])[0]
    assert on["body"] == "sensitive"
