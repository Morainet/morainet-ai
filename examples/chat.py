"""Interactive chat REPL with a local Ollama model — type your own messages.

Multi-turn memory + tool calling + streaming output.

Prereqs:
    brew install ollama && ollama serve && ollama pull qwen2.5:3b

Run:
    python examples/chat.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/chat.py

Type your question and press Enter. Commands: 'exit' / 'quit' / empty line to leave.
"""

from __future__ import annotations

import asyncio
import os

from morainet import Agent, ShortMemory, tool
from morainet.exceptions import ProviderError
from morainet.providers import OllamaProvider


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，例如 "上海"
    """
    fake = {"上海": "晴，26°C", "北京": "多云，22°C", "广州": "雷阵雨，30°C"}
    return fake.get(city, f"{city}：数据缺失")


async def main() -> None:
    model = os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b")
    agent = Agent(
        provider=OllamaProvider(model=model),
        tools=[get_weather],
        memory=ShortMemory(),  # 跨轮记住对话
        system_prompt="你是一个简洁、友好的中文助手。",
    )

    print(f"== Morainet 本地聊天 (模型: {model}) ==")
    print("输入问题后回车；输入 exit / quit 或空行退出。\n")

    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        if user.lower() in {"exit", "quit", ""}:
            print("再见！")
            return

        print("助手 > ", end="", flush=True)
        try:
            async for token in agent.astream(user):
                print(token, end="", flush=True)
            print("\n")
        except ProviderError as exc:
            print(f"\n[连接 Ollama 失败] 确认 ollama serve 在跑、模型已 pull。\n{exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
