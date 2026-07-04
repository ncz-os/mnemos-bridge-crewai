> # 📍 Moved to GitLab
> **The canonical, authoritative home of this project is GitLab — always:**
> ## 👉 https://gitlab.com/ncz-os/mnemos-bridge-crewai
>
> This GitHub repository is a **frozen, read-only mirror**. All development, issues, and releases happen on GitLab. Please open issues and merge requests there. The full history of this stub is preserved on GitLab.

---

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
export MNEMOS_MCP_URL="http://192.168.207.67:5003"
export MNEMOS_MCP_TOKEN="your-mnemos-token"
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
        result = crew.kickoff()
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
  export MNEMOS_TEST_BASE="http://192.168.207.67:5003"
  export MNEMOS_MCP_TOKEN="your-mnemos-token"
  export OPENAI_API_KEY="your-openai-key"
  pytest tests/integration/test_crewai_crew_kickoff.py -v
  ```

The tier-2 test builds a one-agent CrewAI Crew, calls `crew.kickoff()`, records
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
- Optional fields, represented as `Optional[...]` with a default of `None`.
- One-level nested object schemas, represented as `dict`.

Schemas with exotic or ambiguous features fall back to a generic Pydantic model
with a single `kwargs: dict` field. This includes `oneOf`, `allOf`, `anyOf`,
`$ref`, conditional schemas, negation schemas, and object nesting deeper than one
level. The generated fallback model includes a docstring note explaining why the
generic model was used.

## Adapter API

### `MnemosCrewAIAdapter.connect(mcp_url, mcp_token, *, timeout=30)`

Connects to a MNEMOS MCP HTTP/SSE endpoint, opens the underlying MCP client, and
stores the raw tool schemas returned by `list_tools()`.

### `adapter.crewai_tools()`

Returns one CrewAI `BaseTool` instance for each MNEMOS MCP tool. Conversion is
lazy: tool classes and Pydantic argument schemas are generated the first time this
method is called.

### `adapter.aclose()`

Closes the underlying MCP client session. The adapter also supports async context
manager cleanup:

```python
adapter = await MnemosCrewAIAdapter.connect(url, token)
async with adapter:
    tools = await adapter.crewai_tools()
```
