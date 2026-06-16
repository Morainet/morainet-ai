"""Quickstart: an agent that calls a tool, runnable with no API key.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，如 "上海"
        unit: 温度单位，celsius 或 fahrenheit
    """
    return f"{city} 今天晴，26°{'C' if unit == 'celsius' else 'F'}"


# Script the mock LLM: first turn requests the tool, second turn answers.
provider = MockProvider(
    responses=[
        ChatResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "上海"})]
            ),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="tool_calls",
        ),
        ChatResponse(
            message=Message.assistant(content="上海今天晴，26°C，适合穿短袖出门。"),
            usage=Usage(prompt_tokens=20, completion_tokens=12, total_tokens=32),
            finish_reason="stop",
        ),
    ]
)

agent = Agent(provider=provider, tools=[get_weather])
result = agent.run("上海今天适合穿什么？")

print("Final answer:", result.final_answer)
print("Steps:", [(s.description, s.status.value, s.output) for s in result.steps])
print("Tokens:", result.usage.total_tokens)
print("Trace:", result.trace_id)


# --- To use a real model, swap the provider: ---------------------------------
# from morainet.providers import OpenAIProvider
# agent = Agent(provider=OpenAIProvider(model="gpt-4o"), tools=[get_weather])
# print(agent.run("上海今天适合穿什么？").final_answer)
