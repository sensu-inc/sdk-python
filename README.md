# sensu-sdk

Python SDK for the [Sensu](https://sensu-ai.com) AI observability platform.

> Migrating from `senzu-sdk`? The package was renamed; install `sensu-sdk` and replace `from senzu import SenzuClient` with `from sensu import SensuClient`. The legacy `senzu-sdk` v0.5.x package will continue to work but receives no further updates.

## Installation

```bash
pip install sensu-sdk                    # core
pip install "sensu-sdk[anthropic]"       # + Anthropic auto-tracking
pip install "sensu-sdk[openai]"          # + OpenAI auto-tracking
pip install "sensu-sdk[langchain]"       # + LangChain callback handler
pip install "sensu-sdk[all]"             # everything
```

## Quick start

```python
import sensu

client = sensu.SensuClient({
    "api_key": "your-api-key",
    "agent_id": "your-agent-id",
    "org_id": "your-org-id",
})

# High-level API with automatic context propagation
async def handle_request():
    async with client.run({"session_id": "abc"}) as run:
        step = run.start_step({"name": "fetch", "step_type": "tool"})
        result = await step.track_tool({"tool_name": "search", "fn": search})
        await step.end()

# Anthropic auto-tracking
from anthropic import AsyncAnthropic
from sensu.integrations.anthropic import WrapAnthropicOptions, wrap_anthropic

anthropic = wrap_anthropic(
    AsyncAnthropic(),
    WrapAnthropicOptions(client=client),
)

async def chat():
    async with client.run({}) as run:
        # messages.create() is tracked automatically
        return await anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
```

## Environment variables

| Variable | Description |
|---|---|
| `SENSU_API_KEY` | API key |
| `SENSU_BASE_URL` | API base URL (default: `http://localhost:3001`) |
| `SENSU_AGENT_ID` | Agent ID |
| `SENSU_ORG_ID` | Organisation ID |

> The legacy `SENZU_*` names are still read as a fallback and emit a deprecation warning. They will be removed in a future release.

Use `from_env=True` to load from environment:

```python
client = sensu.SensuClient({"from_env": True})
```

## License

MIT
