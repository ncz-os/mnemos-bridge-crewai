from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, get_args, get_origin

from mnemos_bridge_crewai.tool_factory import BaseTool, _get_args_schema_model, create_crewai_tool


@dataclass
class ToolSchemaStub:
    name: str
    description: str
    input_schema: dict


class DispatcherStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return '{"ok": true}'


def test_simple_schema_required_string_and_optional_int_fields() -> None:
    schema = ToolSchemaStub(
        name="search_memories",
        description="Search memories.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)
    assert args_schema is not None
    fields = args_schema.model_fields

    assert fields["query"].annotation is str
    assert fields["query"].is_required()
    assert _is_optional_of(fields["limit"].annotation, int)
    assert not fields["limit"].is_required()


def test_array_field_uses_list_annotation() -> None:
    schema = ToolSchemaStub(
        name="tag_memories",
        description="Tag memories.",
        input_schema={
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "required": ["tags"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)

    assert args_schema is not None
    assert args_schema.model_fields["tags"].annotation is list


def test_boolean_field_uses_bool_annotation() -> None:
    schema = ToolSchemaStub(
        name="toggle_memory",
        description="Toggle memory.",
        input_schema={
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)

    assert args_schema is not None
    assert args_schema.model_fields["enabled"].annotation is bool


def test_nested_object_field_one_level_uses_dict_annotation() -> None:
    schema = ToolSchemaStub(
        name="write_memory",
        description="Write memory.",
        input_schema={
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "priority": {"type": "integer"},
                    },
                }
            },
            "required": ["metadata"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)

    assert args_schema is not None
    assert args_schema.model_fields["metadata"].annotation is dict


def test_exotic_schema_one_of_falls_back_to_kwargs_model() -> None:
    schema = ToolSchemaStub(
        name="exotic_memory",
        description="Use exotic schema.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)
    assert args_schema is not None
    fields = args_schema.model_fields

    assert list(fields) == ["kwargs"]
    assert fields["kwargs"].annotation is dict
    assert "Generic kwargs model" in (args_schema.__doc__ or "")


def test_created_tool_metadata_and_args_schema_match_schema() -> None:
    schema = ToolSchemaStub(
        name="search_memories",
        description="Search memories.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)
    tool = tool_cls()

    assert isinstance(tool, BaseTool)
    assert tool.name == schema.name
    assert schema.description in tool.description
    assert tool.args_schema is args_schema


def test_run_is_callable_and_invokes_dispatcher() -> None:
    schema = ToolSchemaStub(
        name="search_memories",
        description="Search memories.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    dispatcher = DispatcherStub()

    tool_cls = create_crewai_tool(schema, dispatcher)
    tool = tool_cls()

    assert callable(tool._run)
    assert tool._run(query="mnemos") == '{"ok": true}'
    assert dispatcher.calls == [("search_memories", {"query": "mnemos"})]


def _is_optional_of(annotation: object, expected: type) -> bool:
    origin = get_origin(annotation)
    return origin is Optional or (
        origin is not None and type(None) in get_args(annotation) and expected in get_args(annotation)
    )
