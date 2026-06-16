from __future__ import annotations

from morainet import Agent
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider


async def test_agent_as_tool_basic():
    sub = Agent(
        provider=MockProvider(responses=[ChatResponse(message=Message.assistant(content="42"))])
    )
    tool = sub.as_tool("calculator", "Does math.")
    assert tool.name == "calculator"
    assert tool.schema["parameters"]["required"] == ["query"]
    assert await tool.invoke({"query": "6*7"}) == "42"


async def test_orchestrator_delegates_to_subagent():
    # Sub-agent answers any query with a fixed result.
    researcher = Agent(
        provider=MockProvider(
            responses=[ChatResponse(message=Message.assistant(content="上海今天晴，26°C"))]
        )
    )

    # Orchestrator first calls the sub-agent tool, then summarizes.
    orchestrator = Agent(
        provider=MockProvider(
            responses=[
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[
                            ToolCall(id="c1", name="research", arguments={"query": "上海天气"})
                        ]
                    ),
                    usage=Usage(total_tokens=5),
                    finish_reason="tool_calls",
                ),
                ChatResponse(message=Message.assistant(content="根据调研：上海晴，适合短袖")),
            ]
        ),
        tools=[researcher.as_tool("research", "查询信息")],
    )

    result = await orchestrator.arun("上海适合穿什么？")
    assert result.final_answer == "根据调研：上海晴，适合短袖"
    # The orchestrator's step records the sub-agent's output.
    assert result.steps[0].description == "research"
    assert result.steps[0].output == "上海今天晴，26°C"
