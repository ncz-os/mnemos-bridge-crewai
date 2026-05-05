from __future__ import annotations

import json
import sys
from types import TracebackType
from typing import Any

from mnemos_bridge_core import McpClient

try:
    from mnemos_bridge_core.dispatch import dispatch as core_dispatch
except ImportError:  # pragma: no cover - compatibility fallback for older core builds.
    core_dispatch = None

from .tool_factory import create_crewai_tool


class MnemosCrewAIAdapter:
    def __init__(
        self,
        *,
        client: Any,
        client_context: Any,
        tool_schemas: list[Any],
        dispatcher: Any,
        owns_context: bool,
    ) -> None:
        self._client = client
        self._client_context = client_context
        self._tool_schemas = tool_schemas
        self._dispatcher = dispatcher
        self._owns_context = owns_context
        self._tools: list[Any] | None = None
        self._closed = False

    @classmethod
    async def connect(
        cls,
        mcp_url: str,
        mcp_token: str,
        *,
        timeout: int = 30,
    ) -> MnemosCrewAIAdapter:
        """Connect to MNEMOS MCP HTTP/SSE endpoint and list available tools."""

        client_context = cls._build_client(mcp_url, mcp_token, timeout)
        client = client_context
        owns_context = False

        try:
            enter = getattr(client_context, "__aenter__", None)
            if enter is not None:
                client = await enter()
                owns_context = True

            tool_schemas = await _maybe_await(client.list_tools())
        except BaseException:
            if owns_context:
                exc_type, exc, traceback = sys.exc_info()
                await client_context.__aexit__(exc_type, exc, traceback)
            raise

        dispatcher = _McpDispatcher(client, timeout=timeout)
        return cls(
            client=client,
            client_context=client_context,
            tool_schemas=list(tool_schemas),
            dispatcher=dispatcher,
            owns_context=owns_context,
        )

    async def crewai_tools(self) -> list[Any]:
        """Return one CrewAI BaseTool instance per MNEMOS tool."""

        if self._closed:
            raise RuntimeError("MnemosCrewAIAdapter is closed.")

        if self._tools is None:
            self._tools = [
                create_crewai_tool(tool_schema, self._dispatcher)()
                for tool_schema in self._tool_schemas
            ]
        return list(self._tools)

    async def aclose(self) -> None:
        if self._closed:
            return

        if self._owns_context and self._client_context is not None:
            await self._client_context.__aexit__(None, None, None)

        self._closed = True

    async def __aenter__(self) -> MnemosCrewAIAdapter:
        if self._closed:
            raise RuntimeError("Cannot enter a closed MnemosCrewAIAdapter.")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    @staticmethod
    def _build_client(mcp_url: str, mcp_token: str, timeout: int) -> Any:
        factory = getattr(McpClient, "from_url", None)
        if factory is not None:
            return factory(mcp_url, token=mcp_token, timeout=timeout)
        return McpClient(mcp_url, token=mcp_token, timeout=timeout)


class _McpDispatcher:
    def __init__(self, client: Any, *, timeout: int) -> None:
        self._client = client
        self._timeout = timeout

    async def call(self, name: str, args: dict[str, Any]) -> str:
        if core_dispatch is not None:
            result = await core_dispatch(self._client, name, args, timeout=self._timeout)
        elif hasattr(self._client, "call_tool"):
            result = await self._client.call_tool(name, args)
        else:
            result = await self._client.call(name, args)
        return _result_to_text(result)


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _result_to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")

    content = getattr(result, "content", None)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if text is not None:
                text_parts.append(str(text))
        if text_parts:
            return "\n".join(text_parts)

    if isinstance(result, dict | list | tuple):
        return json.dumps(result, default=str)
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    return str(result)
