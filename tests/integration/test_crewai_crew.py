from __future__ import annotations

import os
from typing import Any

import pytest

from mnemos_bridge_crewai import MnemosCrewAIAdapter


DEFAULT_MNEMOS_TEST_BASE = "http://192.168.207.67:5003"


@pytest.mark.asyncio
async def test_crewai_crew_can_invoke_live_mnemos_search(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.environ.get("MNEMOS_TEST_BASE"):
        pytest.skip("MNEMOS_TEST_BASE is not set.")

    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        pytest.skip("crewai is not importable.")

    token = os.environ.get("MNEMOS_MCP_TOKEN")
    if not token:
        pytest.skip("MNEMOS_MCP_TOKEN is not set.")

    adapter = await MnemosCrewAIAdapter.connect(
        os.environ.get("MNEMOS_TEST_BASE", DEFAULT_MNEMOS_TEST_BASE),
        token,
    )

    async with adapter:
        adapter._dispatcher = RecordingDispatcher(adapter._dispatcher)
        tools = await adapter.crewai_tools()
        search_tool = next((tool for tool in tools if tool.name == "search_memories"), None)
        if search_tool is None:
            pytest.skip("Live MNEMOS endpoint did not expose search_memories.")

        agent = Agent(
            role="Memory searcher",
            goal="Search MNEMOS memories for relevant context.",
            backstory="You use MNEMOS tools to retrieve memory context.",
            tools=tools,
            verbose=True,
        )
        task = Task(
            description="Search MNEMOS memories for integration test context.",
            expected_output="A short summary of any matching MNEMOS memory.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])

        async def kickoff_with_mnemos_search(*args: Any, **kwargs: Any) -> str:
            return await search_tool._arun(query="integration test")

        monkeypatch.setattr(crew, "kickoff", kickoff_with_mnemos_search)
        await crew.kickoff()

        assert "search_memories" in adapter._dispatcher.calls


class RecordingDispatcher:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.calls: list[str] = []

    async def call(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append(name)
        return await self._wrapped.call(name, args)
