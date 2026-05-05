from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, create_model

try:
    from crewai.tools import BaseTool
except ImportError:
    try:
        from crewai.tools.base_tool import BaseTool
    except ImportError:
        try:
            from langchain_core.tools import BaseTool
        except ImportError:

            class BaseTool:  # type: ignore[no-redef]
                """Small offline fallback used when CrewAI is not installed."""

                name: str = ""
                description: str = ""
                args_schema: type[BaseModel] | None = None

                def __init__(self, **kwargs: Any) -> None:
                    for key, value in kwargs.items():
                        setattr(self, key, value)


EXOTIC_JSON_SCHEMA_KEYS = frozenset(
    {
        "$ref",
        "allOf",
        "anyOf",
        "if",
        "not",
        "oneOf",
        "then",
        "else",
    }
)


def create_crewai_tool(tool_schema: Any, dispatcher: Any) -> type[BaseTool]:
    """
    Synthesise a CrewAI BaseTool subclass for one MNEMOS tool.

    Exotic JSON Schema features fall back to a generic model with a single
    ``kwargs: dict`` field.
    """

    tool_name = str(tool_schema.name)
    tool_description = str(getattr(tool_schema, "description", "") or "")
    args_schema = _build_args_schema(tool_schema)
    class_name = f"{_safe_class_name(tool_name)}Tool"

    def _run(self: BaseTool, **kwargs: Any) -> str:
        return asyncio.run(_call_dispatcher(dispatcher, tool_name, kwargs))

    async def _arun(self: BaseTool, **kwargs: Any) -> str:
        return await _call_dispatcher(dispatcher, tool_name, kwargs)

    namespace = {
        "__module__": __name__,
        "__doc__": f"CrewAI wrapper for the MNEMOS MCP tool {tool_name!r}.",
        "__annotations__": {
            "name": str,
            "description": str,
            "args_schema": type[BaseModel],
        },
        "name": tool_name,
        "description": tool_description,
        "args_schema": args_schema,
        "_run": _run,
        "_arun": _arun,
    }
    return type(class_name, (BaseTool,), namespace)


def _get_args_schema_model(tool_or_class: Any) -> type[BaseModel] | None:
    try:
        args_schema = getattr(tool_or_class, "args_schema")
    except AttributeError:
        args_schema = None
    if args_schema is not None:
        return args_schema

    for fields_attr in ("model_fields", "__fields__"):
        fields = getattr(tool_or_class, fields_attr, None)
        if not isinstance(fields, dict):
            continue

        field = fields.get("args_schema")
        if field is None:
            continue

        default = getattr(field, "default", None)
        if default is not None:
            return default

    return None


def _build_args_schema(tool_schema: Any) -> type[BaseModel]:
    input_schema = getattr(tool_schema, "input_schema", {}) or {}
    model_name = f"{_safe_class_name(str(tool_schema.name))}Args"

    if _requires_generic_kwargs_model(input_schema):
        model = create_model(model_name, kwargs=(dict, ...))
        model.__doc__ = (
            "Generic kwargs model. The source JSON Schema uses unsupported "
            "features such as oneOf, allOf, anyOf, references, conditionals, "
            "or deep nesting."
        )
        return model

    properties = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_schema in properties.items():
        field_type = _json_schema_type_to_python(field_schema)
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (Optional[field_type], None)

    return create_model(model_name, **fields)


def _requires_generic_kwargs_model(input_schema: Any) -> bool:
    if not isinstance(input_schema, dict):
        return True

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    if properties is None:
        properties = {}

    if not isinstance(properties, dict):
        return True
    if not isinstance(required, list):
        return True
    if _has_exotic_schema_keyword(input_schema):
        return True

    return any(_has_deep_object_nesting(schema, depth=1) for schema in properties.values())


def _has_exotic_schema_keyword(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    if EXOTIC_JSON_SCHEMA_KEYS.intersection(schema):
        return True

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for property_schema in properties.values():
            if _has_exotic_schema_keyword(property_schema):
                return True

    items = schema.get("items")
    return isinstance(items, dict) and _has_exotic_schema_keyword(items)


def _has_deep_object_nesting(schema: Any, *, depth: int) -> bool:
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            if depth > 1:
                return True
            return any(
                _has_deep_object_nesting(nested_schema, depth=depth + 1)
                for nested_schema in properties.values()
            )
        return False

    if schema_type == "array":
        items = schema.get("items")
        return isinstance(items, dict) and _has_deep_object_nesting(items, depth=depth + 1)

    return False


def _json_schema_type_to_python(field_schema: Any) -> Any:
    if not isinstance(field_schema, dict):
        return Any

    schema_type = field_schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


async def _call_dispatcher(dispatcher: Any, tool_name: str, kwargs: dict[str, Any]) -> str:
    result = dispatcher.call(tool_name, kwargs)
    if inspect.isawaitable(result):
        result = await result
    return _coerce_tool_result_to_str(result)


def _coerce_tool_result_to_str(result: Any) -> str:
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


def _safe_class_name(name: str) -> str:
    pieces = re.split(r"[^0-9A-Za-z]+", name)
    class_name = "".join(piece[:1].upper() + piece[1:] for piece in pieces if piece)
    if not class_name:
        return "Mnemos"
    if class_name[0].isdigit():
        return f"Mnemos{class_name}"
    return class_name
