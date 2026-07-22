import importlib.metadata
from unittest.mock import patch

import pytest
from packaging.version import Version

if Version(importlib.metadata.version("pydantic_ai")).major < 2:
    pytest.skip("Pydantic AI 2.x compatibility tests", allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel

import mlflow
from mlflow.entities import SpanType
from mlflow.pydantic_ai.autolog import _get_span_type
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


def _span_by_name(spans, name):
    return next(span for span in spans if span.name == name)


def test_auto_instrumentation_enable_disable():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel())

    assert agent.instrument is True

    mlflow.pydantic_ai.autolog(disable=True)
    assert Agent(TestModel()).instrument is None


def test_agent_run_sync_creates_agent_and_llm_spans():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel(model_name="test:model"))

    result = agent.run_sync("hello")

    assert result.output == "success (no tool calls)"
    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans

    root = _span_by_name(spans, "Agent.run_sync")
    nested_agent = _span_by_name(spans, "Agent.run")
    llm = _span_by_name(spans, "TestModel.request")

    assert root.span_type == SpanType.AGENT
    assert nested_agent.span_type == SpanType.AGENT
    assert llm.span_type == SpanType.LLM
    assert nested_agent.parent_id == root.span_id
    assert llm.parent_id == nested_agent.span_id
    assert root.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "pydantic_ai"
    assert llm.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "pydantic_ai"
    assert llm.get_attribute(SpanAttributeKey.MODEL) == "test:model"
    assert llm.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "test"


def test_explicit_instrument_false_is_respected():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel())
    agent.instrument = False

    agent.run_sync("hello")

    spans = get_traces()[0].data.spans
    assert any(span.span_type == SpanType.AGENT for span in spans)
    assert all(span.span_type != SpanType.LLM for span in spans)


@pytest.mark.asyncio
async def test_agent_run_creates_agent_and_llm_spans():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel())

    result = await agent.run("hello")

    assert result.output == "success (no tool calls)"
    spans = get_traces()[0].data.spans
    agent_span = _span_by_name(spans, "Agent.run")
    llm_span = _span_by_name(spans, "TestModel.request")
    assert agent_span.span_type == SpanType.AGENT
    assert llm_span.span_type == SpanType.LLM
    assert llm_span.parent_id == agent_span.span_id


def test_tool_call_creates_tool_span():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel(call_tools=["double"]))

    @agent.tool_plain
    def double(value: int) -> int:
        return value * 2

    agent.run_sync("double a value")

    spans = get_traces()[0].data.spans
    tool_span = _span_by_name(spans, "ToolManager.execute_tool_call")
    assert tool_span.span_type == SpanType.TOOL


@pytest.mark.asyncio
async def test_mcp_toolset_list_tools_creates_tool_span():
    async def list_tools(self):
        return []

    with patch.object(MCPToolset, "list_tools", new=list_tools):
        mlflow.pydantic_ai.autolog(log_traces=True)
        toolset = MCPToolset("http://localhost:8000/mcp")
        assert await toolset.list_tools() == []

    traces = get_traces()
    assert len(traces) == 1
    span = _span_by_name(traces[0].data.spans, "MCPToolset.list_tools")
    assert span.span_type == SpanType.TOOL
    assert _get_span_type(toolset) == SpanType.TOOL


@pytest.mark.asyncio
async def test_mcp_toolset_direct_call_tool_creates_tool_span():
    async def direct_call_tool(self, name, args, *, metadata=None, use_task=False):
        return {"name": name, "args": args, "metadata": metadata, "use_task": use_task}

    with patch.object(MCPToolset, "direct_call_tool", new=direct_call_tool):
        mlflow.pydantic_ai.autolog(log_traces=True)
        toolset = MCPToolset("http://localhost:8000/mcp")
        result = await toolset.direct_call_tool(
            "double",
            {"value": 2},
            metadata={"source": "test"},
            use_task=True,
        )

    assert result == {
        "name": "double",
        "args": {"value": 2},
        "metadata": {"source": "test"},
        "use_task": True,
    }
    traces = get_traces()
    assert len(traces) == 1
    span = _span_by_name(traces[0].data.spans, "MCPToolset.direct_call_tool")
    assert span.span_type == SpanType.TOOL


@pytest.mark.asyncio
async def test_agent_run_stream_creates_agent_and_llm_spans():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel())

    async with agent.run_stream("hello") as result:
        assert await result.get_output() == "success (no tool calls)"

    spans = get_traces()[0].data.spans
    agent_span = _span_by_name(spans, "Agent.run_stream")
    llm_span = _span_by_name(spans, "TestModel.request")
    assert agent_span.span_type == SpanType.AGENT
    assert llm_span.span_type == SpanType.LLM
    assert llm_span.parent_id == agent_span.span_id


def test_agent_run_stream_sync_creates_agent_and_llm_spans():
    mlflow.pydantic_ai.autolog(log_traces=True)
    agent = Agent(TestModel())

    with agent.run_stream_sync("hello") as result:
        assert "".join(result.stream_text()) == "success (no tool calls)"

    spans = get_traces()[0].data.spans
    root = _span_by_name(spans, "Agent.run_stream_sync")
    llm = _span_by_name(spans, "TestModel.request")
    assert root.span_type == SpanType.AGENT
    assert llm.span_type == SpanType.LLM
    assert llm.parent_id == root.span_id
