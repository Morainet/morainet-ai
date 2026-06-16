"""方向：编码 / harness —— 带真实工具 + 验证闭环的代码助手。

这是"运行时能撑起 harness"的最小证明：工具（读文件、跑测试）**做真事**，
模型据真实的测试失败输出去定位 bug。即使离线（MockProvider 脚本化工具调用），
工具结果也是真实的——真的读了文件、真的跑了 pytest。

Run:
    python examples/coding_assistant.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/coding_assistant.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider

# 一个带 bug 的临时工程：add 用了减法，测试会失败。
WORKSPACE = Path(tempfile.mkdtemp(prefix="morainet_coding_"))
(WORKSPACE / "calc.py").write_text("def add(a, b):\n    return a - b  # bug\n", encoding="utf-8")
(WORKSPACE / "test_calc.py").write_text(
    "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
)


@tool
def list_files() -> str:
    """列出工程里的文件。"""
    return "\n".join(p.name for p in WORKSPACE.iterdir())


@tool
def read_file(path: str) -> str:
    """读取工程内某个文件的内容。

    Args:
        path: 相对工程根目录的文件名，如 "calc.py"
    """
    target = (WORKSPACE / path).resolve()
    if WORKSPACE.resolve() not in target.parents:
        return "ERROR: 越界访问被拒绝"
    return target.read_text(encoding="utf-8")


@tool
def run_tests() -> str:
    """运行 pytest，返回测试结果（真实执行）。"""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    return (proc.stdout + proc.stderr)[-800:]


def _build_provider() -> Provider:
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    # 离线：脚本化"跑测试→读代码→给结论"，但工具结果是真实的。
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="1", name="run_tests", arguments={})]
                ),
                usage=Usage(total_tokens=10),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="2", name="read_file", arguments={"path": "calc.py"})]
                ),
                usage=Usage(total_tokens=10),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(
                    content="测试失败：add(2,3) 期望 5 实际 -1。calc.py 里 `add` 用了减法，"
                    "应把 `return a - b` 改成 `return a + b`。"
                )
            ),
        ]
    )


async def main() -> None:
    print(f"工程目录：{WORKSPACE}")
    agent = Agent(
        provider=_build_provider(),
        tools=[list_files, read_file, run_tests],
        system_prompt="你是代码助手。可用工具排查并定位测试失败的原因。",
        max_steps=6,
    )
    result = await agent.arun("测试好像挂了，帮我找出原因并说明怎么修。")

    print("\n--- 工具调用轨迹 ---")
    for s in result.steps:
        print(f"  [{s.status.value}] {s.description}")
    print("\n--- 结论 ---")
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
