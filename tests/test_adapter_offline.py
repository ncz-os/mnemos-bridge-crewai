from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from mnemos_bridge_crewai.adapter import MnemosCrewAIAdapter, _McpDispatcher, _result_to_text
from mnemos_bridge_crewai.tool_factory import BaseTool, create_crewai_tool
from mnemos_bridge_crewai.tool_policy import (
    READ_ONLY_TOOL_NAMES,
    filter_tool_schemas,
    is_destructive_tool_name,
    is_read_only_tool_name,
)


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
            ToolSchemaStub("log_memory", "Read commit history.", _schema("memory_id")),
            ToolSchemaStub("checkout_memory", "Read a commit.", _schema("memory_id")),
            ToolSchemaStub("write_memory", "Write memory.", _schema("content")),
            ToolSchemaStub("delete_memory", "Delete memory.", _schema("memory_id")),
            ToolSchemaStub("grant_memory_access", "Invented mutation.", _schema("user_id")),
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        return '{"ok": true}'


async def _connect(**kwargs: object) -> MnemosCrewAIAdapter:
    with patch("mnemos_bridge_crewai.adapter.McpClient", FakeMcpClient):
        return await MnemosCrewAIAdapter.connect(
            "http://mnemos.example/mcp",
            "test-token",
            timeout=3,
            **kwargs,
        )


async def test_connect_defaults_to_the_explicit_read_only_allowlist() -> None:
    adapter = await _connect()

    try:
        with pytest.warns(UserWarning, match="explicit safe allowlist"):
            tools = await adapter.crewai_tools()
    finally:
        await adapter.aclose()

    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "search_memories",
        "log_memory",
        "checkout_memory",
    ]
    assert [schema.name for schema in adapter.tool_schemas] == [
        "search_memories",
        "log_memory",
        "checkout_memory",
        "write_memory",
        "delete_memory",
        "grant_memory_access",
    ]


async def test_default_policy_rejects_invented_mutating_tool_name() -> None:
    adapter = await _connect()

    try:
        with pytest.warns(UserWarning, match="grant_memory_access"):
            tools = await adapter.crewai_tools()
    finally:
        await adapter.aclose()

    assert "grant_memory_access" not in {tool.name for tool in tools}


async def test_readonly_false_restores_unrestricted_exposure() -> None:
    adapter = await _connect(readonly=False)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tools = await adapter.crewai_tools()
    finally:
        await adapter.aclose()

    assert [tool.name for tool in tools] == [
        "search_memories",
        "log_memory",
        "checkout_memory",
        "write_memory",
        "delete_memory",
        "grant_memory_access",
    ]


async def test_explicit_allowlist_can_opt_in_a_write_and_denied_tools_wins() -> None:
    adapter = await _connect(
        allowed_tools=["search_memories", "write_memory"],
        denied_tools=["search_memories"],
    )

    try:
        tools = await adapter.crewai_tools()
    finally:
        await adapter.aclose()

    assert [tool.name for tool in tools] == ["write_memory"]


async def test_empty_explicit_allowlist_permits_nothing() -> None:
    adapter = await _connect(allowed_tools=[])

    try:
        assert await adapter.crewai_tools() == []
    finally:
        await adapter.aclose()


def test_read_only_allowlist_exactly_matches_the_server_contract() -> None:
    assert READ_ONLY_TOOL_NAMES == {
        "search_memories",
        "get_memory",
        "list_memories",
        "get_stats",
        "kg_search",
        "kg_timeline",
        "log_memory",
        "diff_memory_commits",
        "checkout_memory",
        "recommend_model",
        "pantheon_list_models",
        "pantheon_route_explain",
        "graeae_consult",
        "graeae_get_consultation",
        "list_deletions",
        "kronos_anomalies",
        "kronos_forecast",
    }


@pytest.mark.parametrize("name", ["log_memory", "checkout_memory"])
def test_dag_read_tools_are_not_misclassified_as_destructive(name: str) -> None:
    assert is_read_only_tool_name(name)
    assert not is_destructive_tool_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "create_memory",
        "update_memory",
        "delete_memory",
        "branch_memory",
        "grant_memory_access",
        "rotate_api_key",
    ],
)
def test_mutating_and_unknown_names_are_withheld_by_default(name: str) -> None:
    assert not is_read_only_tool_name(name)
    assert is_destructive_tool_name(name)


def test_filter_tool_schemas_fails_closed_for_unknown_names() -> None:
    schemas = [
        {"name": "search_memories"},
        {"name": "checkout_memory"},
        {"name": "grant_memory_access"},
    ]

    assert filter_tool_schemas(schemas) == schemas[:2]
    assert filter_tool_schemas(schemas, readonly=False) == schemas


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
