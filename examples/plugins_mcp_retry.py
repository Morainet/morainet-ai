"""v1.0 features: Plugin registry, MCP tools (via a fake session), retry wrapper.

All offline. Real MCP servers connect via `morainet.mcp.stdio_session`.

Run:
    python examples/plugins_mcp_retry.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from morainet import Agent, MCPClient, PluginRegistry, RetryingProvider, RetryPolicy
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.exceptions import RateLimitError
from morainet.providers import MockProvider
from morainet.providers.base import Provider


def plugin_demo() -> None:
    print("=== Plugin registry ===")
    reg = PluginRegistry()
    reg.register("providers", "my_provider", MockProvider)
    reg.register("tools", "search", lambda q: q)
    print("providers:", reg.names("providers"))
    print("tools:", reg.names("tools"))
    # In real packages these are discovered automatically:
    #   reg.load_entry_points()


class FakeMCPSession:
    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "weather",
                "description": "Get weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return f"{arguments['city']}: sunny 26C"


async def mcp_demo() -> None:
    print("\n=== MCP tools ===")
    tools = await MCPClient(FakeMCPSession()).list_tools()
    print("Discovered MCP tools:", [t.name for t in tools])

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="weather", arguments={"city": "上海"})]
                ),
                usage=Usage(total_tokens=5),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="上海今天晴，26C")),
        ]
    )
    agent = Agent(provider=provider, tools=tools)
    result = await agent.arun("上海天气？")
    print("Answer:", result.final_answer, "| via tool ->", result.steps[0].output)


class FlakyProvider(Provider):
    def __init__(self):
        self.attempts = 0

    async def chat(self, messages, tools=None):
        self.attempts += 1
        if self.attempts < 3:
            raise RateLimitError("429 slow down")
        return ChatResponse(message=Message.assistant(content="recovered"))


async def retry_demo() -> None:
    print("\n=== Retry wrapper ===")

    async def fast_sleep(_):  # don't actually wait in the demo
        return None

    inner = FlakyProvider()
    provider = RetryingProvider(inner, RetryPolicy(max_retries=5), sleep=fast_sleep)
    resp = await provider.chat([Message.user("hi")])
    print(f"Succeeded after {inner.attempts} attempts -> {resp.message.content}")


async def main() -> None:
    plugin_demo()
    await mcp_demo()
    await retry_demo()


if __name__ == "__main__":
    asyncio.run(main())
