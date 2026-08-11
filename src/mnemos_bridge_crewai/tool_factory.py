from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

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

                def run(self, **kwargs: Any) -> Any:
                    return self._run(**kwargs)

                async def arun(self, **kwargs: Any) -> Any:
                    return await self._arun(**kwargs)


def create_crewai_tool(tool_schema: Any, dispatcher: Any) -> type[BaseTool]:
    """
    Synthesise a CrewAI BaseTool subclass for one MNEMOS tool.

    The generated Pydantic model preserves the MCP tool's JSON Schema contract.
    """

    tool_name = str(tool_schema.name)
    tool_description = str(getattr(tool_schema, "description", "") or "")
    args_schema = _build_args_schema(tool_schema)
    class_name = f"{_safe_class_name(tool_name)}Tool"

    def _run(self: BaseTool, **kwargs: Any) -> str:
        call_sync = getattr(dispatcher, "call_sync", None)
        if call_sync is not None:
            return call_sync(tool_name, kwargs)
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

    if not _is_object_schema(input_schema):
        model = create_model(model_name, kwargs=(dict, ...))
        model.__doc__ = (
            "Generic kwargs model. The source JSON Schema uses unsupported "
            "features such as oneOf, allOf, anyOf, references, conditionals, "
            "or deep nesting."
        )
        return model

    return _object_schema_to_model(input_schema, model_name)


class _McpArgsModel(BaseModel):
    """Keep omitted optional MCP arguments omitted during CrewAI validation."""

    model_config = ConfigDict(strict=True)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_unset", True)
        return super().model_dump(*args, **kwargs)


def _is_object_schema(input_schema: Any) -> bool:
    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    return isinstance(properties, dict) and isinstance(required, list)


def _object_schema_to_model(schema: dict[str, Any], model_name: str) -> type[BaseModel]:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_schema in properties.items():
        annotation = _json_schema_type_to_python(
            field_schema,
            nested_model_name=f"{model_name}{_safe_class_name(field_name)}",
        )
        schema_dict = field_schema if isinstance(field_schema, dict) else {}
        default = ... if field_name in required else schema_dict.get("default")
        fields[field_name] = (
            annotation,
            _schema_field(
                schema_dict,
                default,
                hide_implicit_none=field_name not in required and "default" not in schema_dict,
            ),
        )

    model = create_model(model_name, __base__=_McpArgsModel, **fields)
    description = schema.get("description")
    if isinstance(description, str):
        model.__doc__ = description
    return model


def _schema_field(
    schema: dict[str, Any],
    default: Any,
    *,
    hide_implicit_none: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {}
    keyword_map = {
        "description": "description",
        "minimum": "ge",
        "maximum": "le",
        "exclusiveMinimum": "gt",
        "exclusiveMaximum": "lt",
        "multipleOf": "multiple_of",
        "minLength": "min_length",
        "maxLength": "max_length",
        "pattern": "pattern",
        "minItems": "min_length",
        "maxItems": "max_length",
    }
    for json_name, field_name in keyword_map.items():
        if json_name in schema:
            kwargs[field_name] = schema[json_name]
    if hide_implicit_none:
        kwargs["json_schema_extra"] = _remove_implicit_none_default
    return Field(default, **kwargs)


def _remove_implicit_none_default(json_schema: dict[str, Any]) -> None:
    if json_schema.get("default") is None:
        json_schema.pop("default", None)


def _json_schema_type_to_python(
    field_schema: Any,
    *,
    nested_model_name: str = "NestedMcpArgs",
) -> Any:
    if not isinstance(field_schema, dict):
        return Any

    schema_type = field_schema.get("type")
    nullable = isinstance(schema_type, list) and "null" in schema_type
    if isinstance(schema_type, list):
        non_null_types = [item for item in schema_type if item != "null"]
        schema_type = non_null_types[0] if len(non_null_types) == 1 else None

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        annotation = Literal.__getitem__(tuple(enum_values))
        return annotation | None if nullable else annotation

    if schema_type == "string":
        annotation = str
    elif schema_type == "integer":
        annotation = int
    elif schema_type == "number":
        annotation = float
    elif schema_type == "boolean":
        annotation = bool
    elif schema_type == "array":
        item_type = _json_schema_type_to_python(
            field_schema.get("items", {}),
            nested_model_name=f"{nested_model_name}Item",
        )
        annotation = list[item_type]
    elif schema_type == "object" and _is_object_schema(field_schema):
        annotation = _object_schema_to_model(field_schema, nested_model_name)
    elif schema_type == "object":
        additional = field_schema.get("additionalProperties")
        value_type = (
            _json_schema_type_to_python(
                additional,
                nested_model_name=f"{nested_model_name}Value",
            )
            if isinstance(additional, dict)
            else Any
        )
        annotation = dict[str, value_type]
    else:
        annotation = Any

    return annotation | None if nullable and annotation is not Any else annotation


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
