# mnemos-bridge-crewai

`mnemos-bridge-crewai` wraps each MNEMOS MCP tool as a CrewAI `BaseTool`
subclass. It connects to a MNEMOS MCP HTTP/SSE endpoint through
`mnemos-bridge-core`, lists the available MCP tools, and lazily converts their
JSON Schema inputs into Pydantic argument models for CrewAI.

## Install

```bash
pip install mnemos-bridge-crewai
```

Set the MNEMOS MCP endpoint and token before running your CrewAI process:

```bash
export MNEMOS_MCP_URL="http://192.168.207.67:5003/sse"
export MNEMOS_MCP_TOKEN="your-mnemos-token"
```

## Tool exposure is read-only by default

The adapter exposes only the tool names in its reviewed read-only allowlist
unless you opt in to broader access. The allowlist mirrors the canonical MNEMOS
server contract in `mnemos/mcp/tools/__init__.py`; unknown names are withheld,
even when they do not contain an obviously mutating verb. This fail-closed
behavior prevents a newly advertised or attacker-invented tool from acquiring
CrewAI access by default.

The default includes read operations such as `search_memories`, `log_memory`,
and `checkout_memory`. It excludes writes such as `create_memory`,
`update_memory`, `delete_memory`, and `branch_memory`.

```python
# Default: only explicitly known read-only tools.
adapter = await MnemosCrewAIAdapter.connect(url, token)

# Exact opt-in: an explicit allowlist supersedes read-only mode.
adapter = await MnemosCrewAIAdapter.connect(
    url,
    token,
    allowed_tools=["search_memories", "create_memory"],
)

# An empty allowlist permits nothing.
adapter = await MnemosCrewAIAdapter.connect(url, token, allowed_tools=[])

# A denylist is always subtractive and is applied last.
adapter = await MnemosCrewAIAdapter.connect(
    url,
    token,
    denied_tools=["checkout_memory"],
)

# Restore unrestricted pre-policy exposure explicitly.
adapter = await MnemosCrewAIAdapter.connect(url, token, readonly=False)
```

## Working Example

```python
import asyncio
import os

from crewai import Agent, Crew, Task
from mnemos_bridge_crewai import MnemosCrewAIAdapter


async def main() -> None:
    adapter = await MnemosCrewAIAdapter.connect(
        os.environ["MNEMOS_MCP_URL"],
        os.environ["MNEMOS_MCP_TOKEN"],
    )

    async with adapter:
        # Only tools on the explicit read-only allowlist are returned by default.
        mnemos_tools = await adapter.crewai_tools()

        researcher = Agent(
            role="Memory researcher",
            goal="Use MNEMOS memories to answer questions with relevant context.",
            backstory="You retrieve concise context from MNEMOS before answering.",
            tools=mnemos_tools,
            verbose=True,
        )

        task = Task(
            description=(
                "Search MNEMOS memories for notes about onboarding preferences. "
                "Summarize the most relevant result and cite the memory title when available."
            ),
            expected_output="A short summary of the most relevant MNEMOS memory.",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[task])
        result = await crew.kickoff_async()
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

The canonical proof of the integration is split into two tiers:

- Tier 1 offline adapter and tool-wrapper tests:

  ```bash
  pip install -e .
  pytest tests/test_adapter_offline.py tests/test_tool_factory_offline.py -v
  ```

- Tier 2 live CrewAI kickoff test against a real MNEMOS MCP HTTP/SSE endpoint:

  ```bash
  export MNEMOS_TEST_BASE="http://192.168.207.67:5003/sse"
  export MNEMOS_MCP_TOKEN="your-mnemos-token"
  export OPENAI_API_KEY="your-openai-key"
  pytest tests/integration/test_crewai_crew_kickoff.py -v
  ```

The tier-2 test builds a one-agent CrewAI Crew, calls `await crew.kickoff_async()`, records
the live MCP dispatcher boundary, and verifies that at least one MNEMOS MCP tool
was invoked end to end. It is skipped unless `MNEMOS_TEST_BASE`,
`MNEMOS_MCP_TOKEN`, and either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` are set.

For a full local verification pass, run:

```bash
pytest tests/ -v
pytest tests/integration -v --collect-only
```

## JSON Schema Coverage

The adapter supports the JSON Schema shape typically used by MCP tool inputs:

- Basic scalar types: `string`, `integer`, `number`, and `boolean`.
- `array` fields, exposed to Pydantic as `list`.
- `object` fields, exposed to Pydantic as `dict`.
- Required fields, represented as required Pydantic model fields.
- Optional fields, with JSON Schema defaults preserved when provided.
- One-level nested object schemas, represented as `dict`.

For schemas with exotic or ambiguous property internals, the adapter preserves
the MCP tool's top-level argument names and uses broad `dict`, `list`, or `Any`
annotations where a precise Python type cannot be inferred.

## Adapter API

### `MnemosCrewAIAdapter.connect(mcp_url, mcp_token, *, timeout=30, readonly=True, allowed_tools=None, denied_tools=None)`

Connects to a MNEMOS MCP HTTP/SSE endpoint, opens the underlying MCP client, and
stores the raw tool schemas returned by `list_tools()`. The tool-exposure
arguments establish the default policy for `crewai_tools()`.

### `adapter.crewai_tools(*, readonly=..., allowed_tools=..., denied_tools=...)`

Returns one CrewAI `BaseTool` instance for each permitted MNEMOS MCP tool. Any
argument left unset inherits the policy configured by `connect()`. Conversion is
lazy and generated tool instances are cached by name.

### `adapter.tool_schemas`

Returns every tool schema the server advertised before policy filtering, so a
caller can inspect which tools were withheld.

### `adapter.aclose()`

Closes the underlying MCP client session. The adapter also supports async context
manager cleanup:

```python
adapter = await MnemosCrewAIAdapter.connect(url, token)
async with adapter:
    tools = await adapter.crewai_tools()
```


## Build infrastructure & partners

Continuous integration and package distribution for this project are generously
supported by our open-source infrastructure partners:

- **[GitLab](https://gitlab.com/)** — canonical source hosting and CI pipelines
  (format / lint / test gates), via the
  [GitLab for Open Source](https://about.gitlab.com/solutions/open-source/) program.
- **[Buildkite](https://buildkite.com/)** — CI/CD orchestration with hosted macOS
  and Linux agents, and our APT package registry host
  (`packages.buildkite.com/ncz-os/ncz`), via the
  [Buildkite Open Source](https://buildkite.com/pricing) program.

Thank you to both for backing open-source software.
