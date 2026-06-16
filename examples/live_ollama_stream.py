"""Streaming with a local Ollama model — print the answer token by token.

`agent.astream` first resolves any tool calls, then streams the final answer.

Prereqs (same as live_ollama.py):
    brew install ollama && ollama serve && ollama pull qwen2.5:3b

Run:
    python examples/live_ollama_stream.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/live_ollama_stream.py
"""

from __future__ import annotations

import asyncio
import os

from morainet import Agent, tool
from morainet.exceptions import ProviderError
from morainet.providers import OllamaProvider


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，例如 "上海"
    """
    fake = {"上海": "晴，26°C", "北京": "多云，22°C"}
    return fake.get(city, f"{city}：数据缺失")


async def main() -> None:
    model = os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b")
    print(f"Using Ollama model: {model}\n")

    agent = Agent(provider=OllamaProvider(model=model), tools=[get_weather])

    print("Answer (streaming): ", end="", flush=True)
    try:
        async for token in agent.astream("上海现在天气怎么样？给一句穿衣建议。"):
            print(token, end="", flush=True)
    except ProviderError as exc:
        print(f"\n[连接 Ollama 失败] 确认 ollama serve 在跑、模型已 pull。原始错误：{exc}")
        return
    print("\n\n[done]")


if __name__ == "__main__":
    asyncio.run(main())
