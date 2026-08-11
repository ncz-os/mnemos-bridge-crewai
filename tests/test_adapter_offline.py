from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from unittest.mock import AsyncMock, patch

import pytest

from mnemos_bridge_crewai.adapter import MnemosCrewAIAdapter, _McpDispatcher, _result_to_text
from mnemos_bridge_crewai.tool_factory import BaseTool, create_crewai_tool


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


async def test_sync_tool_run_from_worker_dispatches_on_connection_loop() -> None:
    owner_loop = asyncio.get_running_loop()

    class LoopCheckingClient:
        async def call_tool(self, name: str, args: dict) -> str:
            assert asyncio.get_running_loop() is owner_loop
            return f"{name}:{args['query']}"

    dispatcher = _McpDispatcher(LoopCheckingClient(), timeout=3)
    schema = ToolSchemaStub("search_memories", "Search.", _schema("query"))
    tool = create_crewai_tool(schema, dispatcher)()

    result = await asyncio.to_thread(tool.run, query="mnemos")

    assert result == "search_memories:mnemos"


def test_mcp_error_result_raises_with_server_text() -> None:
    @dataclass
    class TextBlock:
        text: str

    @dataclass
    class ErrorResult:
        content: list[TextBlock]
        is_error: bool = True

    with pytest.raises(RuntimeError, match="invalid limit"):
        _result_to_text(ErrorResult([TextBlock("invalid limit")]))


def test_mcp_camel_case_error_result_raises_with_server_text() -> None:
    @dataclass
    class ErrorResult:
        content: list[dict[str, str]]
        isError: bool = True

    with pytest.raises(RuntimeError, match="unauthorized"):
        _result_to_text(ErrorResult([{"text": "unauthorized"}]))


def test_mnemos_structured_error_returned_as_text_raises() -> None:
    payload = json.dumps({"success": False, "error": "limit must be an integer"})

    with pytest.raises(RuntimeError, match="limit must be an integer"):
        _result_to_text(payload)


async def test_dispatch_explicitly_disables_retries() -> None:
    result = type("Result", (), {"content": [], "is_error": False})()
    dispatch = AsyncMock(return_value=result)
    dispatcher = _McpDispatcher(FakeMcpClient("url", token="token", timeout=1), timeout=1)

    with patch("mnemos_bridge_crewai.adapter.core_dispatch", dispatch):
        await dispatcher.call("create_memory", {"content": "once"})

    dispatch.assert_awaited_once_with(
        dispatcher._client,
        "create_memory",
        {"content": "once"},
        retries=0,
        timeout=1,
    )


def test_sync_dispatch_rejects_stopped_owner_loop() -> None:
    loop = asyncio.new_event_loop()
    try:
        async def build_dispatcher() -> _McpDispatcher:
            return _McpDispatcher(FakeMcpClient("url", token="token", timeout=1), timeout=1)

        dispatcher = loop.run_until_complete(build_dispatcher())
        with pytest.raises(RuntimeError, match="not running"):
            dispatcher.call_sync("search_memories", {"query": "x"})
    finally:
        loop.close()


async def test_sync_dispatch_wait_is_bounded_and_cancels_call() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowClient:
        async def call_tool(self, name: str, args: dict) -> str:
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

    dispatcher = _McpDispatcher(SlowClient(), timeout=0.01)
    with patch("mnemos_bridge_crewai.adapter.core_dispatch", None):
        with pytest.raises(TimeoutError, match="did not complete"):
            await asyncio.to_thread(dispatcher.call_sync, "search_memories", {"query": "x"})

    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.wait_for(cancelled.wait(), timeout=1)


def _schema(field_name: str) -> dict:
    return {
        "type": "object",
        "properties": {field_name: {"type": "string"}},
        "required": [field_name],
    }
