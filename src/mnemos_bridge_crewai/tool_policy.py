from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "READ_ONLY_TOOL_NAMES",
    "filter_tool_schemas",
    "is_destructive_tool_name",
    "is_read_only_tool_name",
    "tool_schema_name",
]


READ_ONLY_TOOL_NAMES = frozenset(
    {
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
)
"""MNEMOS tools permitted by the default read-only policy.

This explicit allowlist mirrors the non-write entries in the canonical server
contract at ``mnemos/mcp/tools/__init__.py``. Unknown names are intentionally
absent and therefore denied by default. When that server contract changes, this
reviewed snapshot must be updated deliberately.
"""


def is_read_only_tool_name(name: str) -> bool:
    """Return whether ``name`` is explicitly allowed by the read-only contract."""

    return str(name) in READ_ONLY_TOOL_NAMES


def is_destructive_tool_name(name: str) -> bool:
    """Return whether ``name`` must be withheld by the default policy.

    Kept as a compatibility helper for callers of the first policy draft. A
    previously unseen name is treated as unsafe even when its words do not look
    destructive; the default policy fails closed.
    """

    return not is_read_only_tool_name(name)


def tool_schema_name(tool_schema: Any) -> str:
    """Best-effort tool name for an MCP tool schema object or mapping."""

    if isinstance(tool_schema, dict):
        return str(tool_schema.get("name", ""))
    return str(getattr(tool_schema, "name", ""))


def filter_tool_schemas(
    tool_schemas: Iterable[Any],
    *,
    readonly: bool = True,
    allowed_tools: Sequence[str] | None = None,
    denied_tools: Sequence[str] | None = None,
) -> list[Any]:
    """Apply the tool-exposure policy to raw MCP tool schemas.

    An explicit ``allowed_tools`` list is an opt-in and supersedes ``readonly``.
    Otherwise, ``readonly=True`` permits only :data:`READ_ONLY_TOOL_NAMES`.
    ``denied_tools`` is always subtractive and is applied last.
    """

    allowed = None if allowed_tools is None else {str(name) for name in allowed_tools}
    denied = set() if denied_tools is None else {str(name) for name in denied_tools}

    kept: list[Any] = []
    for tool_schema in tool_schemas:
        name = tool_schema_name(tool_schema)
        if allowed is not None:
            if name not in allowed:
                continue
        elif readonly and not is_read_only_tool_name(name):
            continue
        if name in denied:
            continue
        kept.append(tool_schema)
    return kept
