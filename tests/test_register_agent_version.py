"""
Unit tests for SensuClient.register_agent_version() — the eval-gated
CI/CD (§5.2) convenience helper that wraps POST /api/v1/agents/:id/versions.

Mocks the async http client; no live server required.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from sensu import SensuClient


def make_client(**overrides: Any) -> SensuClient:
    opts: Dict[str, Any] = {
        "api_key":              "test-key",
        "base_url":             "http://localhost:9999",
        "agent_id":             "agent-1",
        "org_id":               "org-1",
        "batch_size":           100,
        "flush_interval_ms":    60_000,
        "disable_live_pricing": True,
        **overrides,
    }
    return SensuClient(opts)


@pytest.mark.asyncio
async def test_register_agent_version_posts_correct_body() -> None:
    client = make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "id":        "ver_xyz123",
        "agentId":   "org-1:cust-support-v3",
        "sha":       "a1b2c3d4",
        "config":    {"systemPrompt": "tighter rules", "model": "claude-sonnet-4-6"},
        "createdAt": "2026-05-19T12:00:00.000Z",
    }
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    client._async_http = mock_http

    result = await client.register_agent_version({
        "agent_id": "cust-support-v3",
        "sha":      "a1b2c3d4",
        "config":   {"system_prompt": "tighter rules", "model": "claude-sonnet-4-6"},
    })

    assert result is not None
    assert result["id"] == "ver_xyz123"
    assert result["agentId"] == "org-1:cust-support-v3"

    mock_http.post.assert_called_once()
    args, kwargs = mock_http.post.call_args
    assert args[0] == "http://localhost:9999/api/v1/agents/cust-support-v3/versions"
    assert kwargs["json"] == {
        "sha":    "a1b2c3d4",
        "config": {"system_prompt": "tighter rules", "model": "claude-sonnet-4-6"},
    }
    assert kwargs["headers"]["X-API-Key"] == "test-key"


@pytest.mark.asyncio
async def test_register_agent_version_url_encodes_agent_id() -> None:
    client = make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "id": "ver_1", "agentId": "org-1:agent/with/slashes", "sha": "s",
        "config": {"systemPrompt": "p"}, "createdAt": "2026-05-19T00:00:00.000Z",
    }
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    client._async_http = mock_http

    await client.register_agent_version({
        "agent_id": "agent/with/slashes",
        "sha":      "s",
        "config":   {"system_prompt": "p"},
    })

    args, _ = mock_http.post.call_args
    assert args[0] == "http://localhost:9999/api/v1/agents/agent%2Fwith%2Fslashes/versions"


@pytest.mark.asyncio
async def test_register_agent_version_returns_none_on_4xx() -> None:
    client = make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Agent not found"
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    client._async_http = mock_http

    with pytest.warns(UserWarning, match="register_agent_version failed 404"):
        result = await client.register_agent_version({
            "agent_id": "missing",
            "sha":      "x",
            "config":   {"system_prompt": "p"},
        })
    assert result is None


@pytest.mark.asyncio
async def test_register_agent_version_returns_none_on_network_error() -> None:
    client = make_client()
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=Exception("network down"))
    client._async_http = mock_http

    with pytest.warns(UserWarning, match="register_agent_version network error"):
        result = await client.register_agent_version({
            "agent_id": "a", "sha": "s", "config": {"system_prompt": "p"},
        })
    assert result is None


@pytest.mark.asyncio
async def test_register_agent_version_returns_none_when_disabled() -> None:
    client = make_client(disabled=True)
    result = await client.register_agent_version({
        "agent_id": "a", "sha": "s", "config": {"system_prompt": "p"},
    })
    assert result is None


@pytest.mark.asyncio
async def test_register_agent_version_returns_none_without_api_key() -> None:
    client = make_client(api_key="")
    result = await client.register_agent_version({
        "agent_id": "a", "sha": "s", "config": {"system_prompt": "p"},
    })
    assert result is None


@pytest.mark.asyncio
async def test_register_agent_version_warns_on_missing_agent_id() -> None:
    client = make_client()
    mock_http = AsyncMock()
    client._async_http = mock_http
    with pytest.warns(UserWarning, match="agent_id is required"):
        result = await client.register_agent_version({  # type: ignore[typeddict-item]
            "sha":    "x",
            "config": {"system_prompt": "p"},
        })
    assert result is None
    mock_http.post.assert_not_called()
