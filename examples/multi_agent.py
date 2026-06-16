"""Multi-agent orchestration — three topologies, all offline (MockProvider).

1. Hierarchical : orchestrator delegates to sub-agents via as_tool()
2. Sequential   : Pipeline threads each agent's output into the next
3. Routing      : Router dispatches to one specialist

Swap in OllamaProvider/OpenAIProvider for a live run.

Run:
    python examples/multi_agent.py
"""

from __future__ import annotations

from morainet import Agent, Pipeline, Route, Router, Stage
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.providers import MockProvider


def build() -> Agent:
    # Specialist sub-agents (here scripted; in real use give them real providers/tools).
    researcher = Agent(
        provider=MockProvider(
            responses=[ChatResponse(message=Message.assistant(content="上海今天晴，26°C"))]
        )
    )
    writer = Agent(
        provider=MockProvider(
            responses=[ChatResponse(message=Message.assistant(content="出门记得防晒，短袖即可~"))]
        )
    )

    # Orchestrator: its tools are the sub-agents.
    orchestrator = Agent(
        provider=MockProvider(
            responses=[
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[ToolCall(id="r", name="research", arguments={"query": "上海天气"})]
                    ),
                    usage=Usage(total_tokens=5),
                    finish_reason="tool_calls",
                ),
                ChatResponse(
                    message=Message.assistant(
                        tool_calls=[ToolCall(id="w", name="write", arguments={"query": "写穿衣建议"})]
                    ),
                    usage=Usage(total_tokens=5),
                    finish_reason="tool_calls",
                ),
                ChatResponse(message=Message.assistant(content="上海晴26°C，短袖+防晒即可。")),
            ]
        ),
        tools=[
            researcher.as_tool("research", "调研事实信息"),
            writer.as_tool("write", "把要点写成友好的建议"),
        ],
    )
    return orchestrator


def _agent(answer: str) -> Agent:
    return Agent(
        provider=MockProvider(handler=lambda m, t: ChatResponse(
            message=Message.assistant(content=answer)
        ))
    )


def pipeline_demo() -> None:
    print("\n=== 2) Sequential Pipeline ===")
    pipe = Pipeline([
        Stage("research", _agent("上海：晴 26°C，紫外线强")),
        Stage("write", _agent("建议：短袖+防晒，傍晚带薄外套"),
              instruction="基于调研「{research}」，写一句穿衣建议：{query}"),
    ])
    out = pipe.run("上海今天穿什么？")
    print("stages:", out.outputs)
    print("final:", out.final)


def router_demo() -> None:
    print("\n=== 3) Router (分诊) ===")
    routes = [
        Route("billing", _agent("已为您查询账单"), "账单/付款问题"),
        Route("tech", _agent("已帮您排查网络故障"), "技术/设备故障"),
    ]
    router = Router(routes, selector=lambda q: "tech" if "连不上" in q else "billing")
    r = router.run("我的设备连不上网")
    print(f"routed -> {r.route}: {r.final}")


if __name__ == "__main__":
    print("=== 1) Hierarchical (as_tool) ===")
    result = build().run("上海今天适合穿什么？")
    print("Final:", result.final_answer)
    print("Delegations:", [(s.description, s.output) for s in result.steps])

    pipeline_demo()
    router_demo()
