from __future__ import annotations

from dataclasses import dataclass
import pytest
from pydantic import BaseModel, ValidationError

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
                "limit": {"type": "integer", "default": 10},
                "mode": {"type": "string", "default": "semantic"},
                "cursor": {"type": "string"},
            },
            "required": ["query"],
        },
    )

    dispatcher = DispatcherStub()
    tool_cls = create_crewai_tool(schema, dispatcher)
    args_schema = _get_args_schema_model(tool_cls)
    assert args_schema is not None
    fields = args_schema.model_fields

    assert fields["query"].annotation is str
    assert fields["query"].is_required()
    assert fields["limit"].annotation is int
    assert not fields["limit"].is_required()
    assert fields["limit"].default == 10
    assert fields["mode"].default == "semantic"
    assert args_schema.model_validate({"query": "x"}).model_dump() == {"query": "x"}

    assert tool_cls().run(query="x") == '{"ok": true}'
    assert dispatcher.calls == [("search_memories", {"query": "x"})]


def test_array_field_preserves_item_type_and_size_constraints() -> None:
    schema = ToolSchemaStub(
        name="tag_memories",
        description="Tag memories.",
        input_schema={
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                }
            },
            "required": ["tags"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)

    assert args_schema is not None
    assert args_schema.model_fields["tags"].annotation == list[str]
    args_schema.model_validate({"tags": ["one", "two"]})
    with pytest.raises(ValidationError):
        args_schema.model_validate({"tags": ["one", "two", "three"]})
    with pytest.raises(ValidationError):
        args_schema.model_validate({"tags": [1]})


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


def test_nested_object_preserves_properties_and_required_fields() -> None:
    schema = ToolSchemaStub(
        name="write_memory",
        description="Write memory.",
        input_schema={
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Origin."},
                        "priority": {"type": "integer"},
                    },
                    "required": ["source"],
                }
            },
            "required": ["metadata"],
        },
    )

    tool_cls = create_crewai_tool(schema, DispatcherStub())
    args_schema = _get_args_schema_model(tool_cls)

    assert args_schema is not None
    metadata_model = args_schema.model_fields["metadata"].annotation
    assert issubclass(metadata_model, BaseModel)
    assert metadata_model.model_json_schema()["properties"]["source"]["description"] == "Origin."
    with pytest.raises(ValidationError):
        args_schema.model_validate({"metadata": {"priority": 1}})


def test_unconstrained_objects_preserve_metadata_and_array_item_contents() -> None:
    schema = ToolSchemaStub(
        name="create_and_route",
        description="Create a memory and explain routing.",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "metadata": {"type": "object"},
                "messages": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["content", "messages"],
        },
    )
    args_schema = _get_args_schema_model(create_crewai_tool(schema, DispatcherStub()))
    payload = {
        "content": "x",
        "metadata": {"source": "jira", "priority": 1},
        "messages": [{"role": "user", "content": "hello"}],
    }

    assert args_schema is not None
    assert args_schema.model_validate(payload).model_dump() == payload


def test_structured_objects_preserve_extras_unless_explicitly_forbidden() -> None:
    schema = ToolSchemaStub(
        name="write_memory",
        description="Write memory metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "open_metadata": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
                "closed_metadata": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "additionalProperties": False,
                },
            },
            "required": ["open_metadata", "closed_metadata"],
        },
    )
    args_schema = _get_args_schema_model(create_crewai_tool(schema, DispatcherStub()))

    assert args_schema is not None
    validated = args_schema.model_validate(
        {
            "open_metadata": {"source": "jira", "priority": 1},
            "closed_metadata": {"source": "jira"},
        }
    )
    assert validated.model_dump()["open_metadata"] == {"source": "jira", "priority": 1}
    with pytest.raises(ValidationError):
        args_schema.model_validate(
            {
                "open_metadata": {"source": "jira"},
                "closed_metadata": {"source": "jira", "priority": 1},
            }
        )


def test_exotic_schema_preserves_top_level_argument_names() -> None:
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

    assert list(fields) == ["query"]
    assert not fields["query"].is_required()


def test_deep_array_schema_dispatches_under_original_top_level_name() -> None:
    schema = ToolSchemaStub(
        name="bulk_create_memories",
        description="Create memories.",
        input_schema={
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                    "maxItems": 2,
                }
            },
            "required": ["memories"],
        },
    )
    dispatcher = DispatcherStub()
    tool = create_crewai_tool(schema, dispatcher)()
    memories = [{"content": "remember this"}]

    assert tool.run(memories=memories) == '{"ok": true}'
    assert dispatcher.calls == [("bulk_create_memories", {"memories": memories})]

    args_schema = _get_args_schema_model(type(tool))
    assert args_schema is not None
    with pytest.raises(ValidationError):
        args_schema.model_validate({"memories": [1]})
    with pytest.raises(ValidationError):
        args_schema.model_validate({"memories": [{}]})


def test_optional_non_nullable_enum_and_numeric_bounds_are_preserved() -> None:
    schema = ToolSchemaStub(
        name="search_memories",
        description="Search memories.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum results.",
                },
                "mode": {"type": "string", "enum": ["semantic", "keyword"]},
            },
        },
    )

    args_schema = _get_args_schema_model(create_crewai_tool(schema, DispatcherStub()))
    assert args_schema is not None
    generated = args_schema.model_json_schema()
    assert generated["properties"]["limit"]["description"] == "Maximum results."
    assert generated["properties"]["limit"]["minimum"] == 1
    assert generated["properties"]["limit"]["maximum"] == 100
    assert generated["properties"]["mode"]["enum"] == ["semantic", "keyword"]
    assert "anyOf" not in generated["properties"]["limit"]
    assert "default" not in generated["properties"]["limit"]
    assert args_schema.model_validate({}).model_dump() == {}
    for invalid in ({"limit": None}, {"limit": 0}, {"mode": "invalid"}):
        with pytest.raises(ValidationError):
            args_schema.model_validate(invalid)


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
