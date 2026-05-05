from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from mnemos_bridge_crewai.adapter import MnemosCrewAIAdapter
from mnemos_bridge_crewai.tool_factory import BaseTool


@dataclass
class ToolSchemaStub:
    name: str
    description: str
    input_schema: dict


class FakeMcpClient:
    def __init__(self, url: str, *, token: str, timeout: int) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.entered = False
        self.closed = False

    @classmethod
    def from_url(cls, url: str, *, token: str, timeout: int) -> FakeMcpClient:
        return cls(url, token=token, timeout=timeout)

    async def __aenter__(self) -> FakeMcpClient:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True

    async def list_tools(self) -> list[ToolSchemaStub]:
        return [
            ToolSchemaStub("search_memories", "Search memories.", _schema("query")),
            ToolSchemaStub("write_memory", "Write memory.", _schema("content")),
            ToolSchemaStub("delete_memory", "Delete memory.", _schema("memory_id")),
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        return '{"ok": true}'


async def test_connect_succeeds_without_network_and_returns_crewai_tools() -> None:
    with patch("mnemos_bridge_crewai.adapter.McpClient", FakeMcpClient):
        adapter = await MnemosCrewAIAdapter.connect(
            "http://mnemos.example/mcp",
            "test-token",
            timeout=3,
        )

    try:
        tools = await adapter.crewai_tools()
    finally:
        await adapter.aclose()

    assert len(tools) == 3
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "search_memories",
        "write_memory",
        "delete_memory",
    ]


def _schema(field_name: str) -> dict:
    return {
        "type": "object",
        "properties": {field_name: {"type": "string"}},
        "required": [field_name],
    }
