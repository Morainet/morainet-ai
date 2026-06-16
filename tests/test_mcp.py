from __future__ import annotations

from typing import Any

from morainet import Agent
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.mcp import MCPClient, mcp_tool_to_tool
from morainet.providers import MockProvider


class FakeSession:
    """In-memory stand-in for an MCP server session."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self):
        return [
            {
                "name": "echo",
                "description": "Echo back text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"echoed: {arguments.get('text')}"


def test_mcp_tool_to_tool_builds_schema():
    async def caller(name, args):
        return "ok"

    tool = mcp_tool_to_tool(
        {"name": "f", "description": "d", "inputSchema": {"type": "object", "properties": {}, "required": []}},
        caller,
    )
    assert tool.name == "f"
    assert tool.schema["description"] == "d"
    assert tool.schema["parameters"]["type"] == "object"


async def test_mcp_client_lists_and_invokes_tools():
    session = FakeSession()
    client = MCPClient(session)
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "echo"
    result = await tools[0].invoke({"text": "hi"})
    assert result == "echoed: hi"
    assert session.calls == [("echo", {"text": "hi"})]


class RichSession(FakeSession):
    async def list_resources(self):
        return [{"uri": "mem://profile", "name": "profile", "description": "user profile"}]

    async def read_resource(self, uri):
        return "用户是 Python 开发者" if uri == "mem://profile" else ""

    async def list_prompts(self):
        return [{"name": "greet", "description": "a greeting"}]

    async def get_prompt(self, name, arguments):
        return f"你好，{arguments.get('who', '朋友')}"


async def test_mcp_resources_to_messages():
    client = MCPClient(RichSession())
    resources = await client.list_resources()
    assert resources[0]["uri"] == "mem://profile"
    assert await client.read_resource("mem://profile") == "用户是 Python 开发者"

    msgs = await client.resource_messages()
    assert len(msgs) == 1
    assert "Python 开发者" in msgs[0].content


async def test_mcp_prompts():
    client = MCPClient(RichSession())
    prompts = await client.list_prompts()
    assert prompts[0]["name"] == "greet"
    assert await client.get_prompt("greet", {"who": "Ada"}) == "你好，Ada"


async def test_mcp_capabilities_absent_are_empty():
    # Plain FakeSession has no resources/prompts -> gracefully empty.
    client = MCPClient(FakeSession())
    assert await client.list_resources() == []
    assert await client.list_prompts() == []
    assert await client.resource_messages() == []


async def test_agent_uses_mcp_tools_end_to_end():
    session = FakeSession()
    tools = await MCPClient(session).list_tools()

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "yo"})]
                ),
                usage=Usage(total_tokens=5),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="done")),
        ]
    )
    agent = Agent(provider=provider, tools=tools)
    result = await agent.arun("echo yo")
    assert result.final_answer == "done"
    assert result.steps[0].output == "echoed: yo"
