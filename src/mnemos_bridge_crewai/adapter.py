from __future__ import annotations

import asyncio
import json
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import TracebackType
from typing import Any

try:
    from mnemos_bridge_core import McpClient
except ImportError:  # pragma: no cover - exercised when optional runtime dependency is absent.
    McpClient = None  # type: ignore[assignment]

try:
    from mnemos_bridge_core.dispatch import dispatch as core_dispatch
except ImportError:  # pragma: no cover - compatibility fallback for older core builds.
    core_dispatch = None

from .tool_factory import create_crewai_tool
from .tool_policy import filter_tool_schemas, is_destructive_tool_name, tool_schema_name


class _Unset:
    """Sentinel marking "use the policy configured on the adapter"."""


_UNSET: Any = _Unset()


class MnemosCrewAIAdapter:
    def __init__(
        self,
        *,
        client: Any,
        client_context: Any,
        tool_schemas: list[Any],
        dispatcher: Any,
        owns_context: bool,
        readonly: bool = True,
        allowed_tools: Sequence[str] | None = None,
        denied_tools: Sequence[str] | None = None,
    ) -> None:
        self._client = client
        self._client_context = client_context
        self._tool_schemas = tool_schemas
        self._dispatcher = dispatcher
        self._owns_context = owns_context
        self._readonly = readonly
        self._allowed_tools = None if allowed_tools is None else list(allowed_tools)
        self._denied_tools = None if denied_tools is None else list(denied_tools)
        self._tool_classes: dict[str, type[Any]] = {}
        self._tool_instances: dict[str, Any] = {}
        self._cached_dispatcher: Any = None
        self._warned_about_hidden_tools = False
        self._closed = False

    @property
    def tool_schemas(self) -> list[Any]:
        """Return all schemas advertised by the server, before policy filtering."""

        return list(self._tool_schemas)

    @classmethod
    async def connect(
        cls,
        mcp_url: str,
        mcp_token: str,
        *,
        timeout: int = 30,
        readonly: bool = True,
        allowed_tools: Sequence[str] | None = None,
        denied_tools: Sequence[str] | None = None,
    ) -> MnemosCrewAIAdapter:
        """Connect to MNEMOS and configure the default tool-exposure policy."""

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
            readonly=readonly,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
        )

    async def crewai_tools(
        self,
        *,
        readonly: bool = _UNSET,
        allowed_tools: Sequence[str] | None = _UNSET,
        denied_tools: Sequence[str] | None = _UNSET,
    ) -> list[Any]:
        """Return one CrewAI ``BaseTool`` instance per permitted MNEMOS tool."""

        if self._closed:
            raise RuntimeError("MnemosCrewAIAdapter is closed.")

        if self._cached_dispatcher is not self._dispatcher:
            self._tool_classes.clear()
            self._tool_instances.clear()
            self._cached_dispatcher = self._dispatcher

        effective_readonly = self._readonly if readonly is _UNSET else readonly
        effective_allowed = self._allowed_tools if allowed_tools is _UNSET else allowed_tools
        effective_denied = self._denied_tools if denied_tools is _UNSET else denied_tools
        permitted = filter_tool_schemas(
            self._tool_schemas,
            readonly=effective_readonly,
            allowed_tools=effective_allowed,
            denied_tools=effective_denied,
        )
        self._warn_about_hidden_tools(
            permitted,
            readonly=effective_readonly,
            allowed_tools=effective_allowed,
        )

        tools: list[Any] = []
        seen: set[str] = set()
        for tool_schema in permitted:
            name = tool_schema_name(tool_schema)
            if name in seen:
                warnings.warn(
                    f"The MCP server advertised more than one tool named {name!r}; "
                    "only the first is exposed to CrewAI.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            seen.add(name)

            instance = self._tool_instances.get(name)
            if instance is None:
                tool_cls = self._tool_classes.get(name)
                if tool_cls is None:
                    tool_cls = create_crewai_tool(tool_schema, self._dispatcher)
                    self._tool_classes[name] = tool_cls
                instance = tool_cls()
                self._tool_instances[name] = instance
            tools.append(instance)
        return tools

    def _warn_about_hidden_tools(
        self,
        permitted: list[Any],
        *,
        readonly: bool,
        allowed_tools: Sequence[str] | None,
    ) -> None:
        if allowed_tools is not None or not readonly or self._warned_about_hidden_tools:
            return

        permitted_names = {tool_schema_name(schema) for schema in permitted}
        hidden = sorted(
            name
            for name in (tool_schema_name(schema) for schema in self._tool_schemas)
            if name not in permitted_names and is_destructive_tool_name(name)
        )
        if not hidden:
            return

        self._warned_about_hidden_tools = True
        warnings.warn(
            "mnemos-bridge-crewai is running read-only and did not expose "
            f"{len(hidden)} tool(s) outside its explicit safe allowlist: "
            f"{', '.join(hidden)}. Pass readonly=False, or allowed_tools=[...], "
            "to expose them deliberately.",
            UserWarning,
            stacklevel=3,
        )

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
        if McpClient is None:
            raise RuntimeError(
                "mnemos_bridge_core is required to connect. Install mnemos-bridge-core>=0.1.0."
            )
        factory = getattr(McpClient, "from_url", None)
        if factory is not None:
            return factory(mcp_url, token=mcp_token, timeout=timeout)
        return McpClient(mcp_url, token=mcp_token, timeout=timeout)


class _McpDispatcher:
    def __init__(self, client: Any, *, timeout: int) -> None:
        self._client = client
        self._timeout = timeout
        self._loop = asyncio.get_running_loop()

    def call_sync(self, name: str, args: dict[str, Any]) -> str:
        if self._loop.is_closed():
            raise RuntimeError("The event loop that owns the MNEMOS connection is closed.")
        if not self._loop.is_running():
            raise RuntimeError("The event loop that owns the MNEMOS connection is not running.")

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            raise RuntimeError(
                "Synchronous CrewAI tool execution cannot run on the MNEMOS connection's "
                "event loop; use 'await crew.kickoff_async()'."
            )

        future = asyncio.run_coroutine_threadsafe(self.call(name, args), self._loop)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"MNEMOS tool {name!r} did not complete within {self._timeout} seconds."
            ) from None

    async def call(self, name: str, args: dict[str, Any]) -> str:
        if core_dispatch is not None:
            # MCP request IDs do not make domain mutations idempotent. The adapter cannot
            # safely infer that an arbitrary server-provided tool is read-only, so never
            # replay a call after an ambiguous transport failure.
            result = await core_dispatch(
                self._client,
                name,
                args,
                retries=0,
                timeout=self._timeout,
            )
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
    text = _render_result(result)
    if bool(getattr(result, "is_error", False)) or bool(getattr(result, "isError", False)):
        raise RuntimeError(text)
    if _is_structured_error_text(text):
        raise RuntimeError(text)
    return text


def _is_structured_error_text(text: str) -> bool:
    """Recognize the structured failure envelope emitted by MNEMOS MCP transports."""

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and "error" in payload and payload.get("success") is not True


def _render_result(result: Any) -> str:
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
