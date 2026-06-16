"""Live debugging with a local Ollama model — see real model input/output.

Prereqs:
    brew install ollama
    ollama serve                 # in another terminal (or it runs as a service)
    ollama pull llama3.1         # a tool-capable model

Run:
    python examples/live_ollama.py
    MORAINET_OLLAMA_MODEL=qwen2.5 python examples/live_ollama.py   # pick a model
"""

from __future__ import annotations

import os

from morainet import Agent, Hook, tool
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


class PrintIO(Hook):
    """Prints what goes into the model and what comes back, each round."""

    def on_llm_end(self, ctx, response):
        print("\n--- 模型输入 (messages) ---")
        for m in ctx.messages:
            payload = m.content if m.content else m.tool_calls
            print(f"  [{m.role.value}] {payload}")
        out = response.message.content or response.message.tool_calls
        print(f"--- 模型输出 ({response.finish_reason}) --- {out}")

    def on_tool_end(self, ctx, step):
        print(f"--- 工具执行 --- {step.description} -> {step.output}")


def main() -> None:
    model = os.getenv("MORAINET_OLLAMA_MODEL", "llama3.1")
    print(f"Using Ollama model: {model}")

    agent = Agent(
        provider=OllamaProvider(model=model),
        tools=[get_weather],
        hooks=[PrintIO()],
    )

    try:
        result = agent.run("上海现在天气怎么样？适合穿什么？")
    except ProviderError as exc:
        print(
            "\n[连接 Ollama 失败] 请确认：\n"
            "  1) 已安装：brew install ollama\n"
            "  2) 服务在跑：ollama serve\n"
            f"  3) 模型已拉取：ollama pull {model}\n"
            f"原始错误：{exc}"
        )
        return

    print("\n================ FINAL ================")
    print(result.final_answer)
    print(f"tokens={result.usage.total_tokens} trace={result.trace_id}")


if __name__ == "__main__":
    main()
