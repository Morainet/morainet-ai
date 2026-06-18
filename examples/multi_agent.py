"""Multi-agent orchestration — five topologies, all offline (MockProvider).

1. Hierarchical : orchestrator delegates to sub-agents via as_tool()
2. Sequential   : Pipeline threads each agent's output into the next
3. Routing      : Router dispatches to one specialist
4. GroupChat    : Free-form multi-agent conversation with speaker selection
5. Debate       : Structured debate with rounds and a judge

Swap in OllamaProvider/OpenAIProvider for a live run.

Run:
    python examples/multi_agent.py
"""

from __future__ import annotations

from morainet import (
    Agent,
    Debate,
    GroupChat,
    GroupChatMember,
    Pipeline,
    Route,
    Router,
    Stage,
)
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


def _member(name: str, answer: str, desc: str = "") -> GroupChatMember:
    return GroupChatMember(name=name, agent=_agent(answer), description=desc)


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


def groupchat_demo() -> None:
    print("\n=== 4) GroupChat (群聊) ===")
    chat = GroupChat(
        members=[
            _member("pm", "从产品角度看，优先级应该是功能A > 功能B > 功能C", "产品经理"),
            _member(
                "engineer",
                "功能A需要2周，功能B需要1周。建议先做B快速上线。TERMINATE",
                "工程师",
            ),
        ],
        speaker_selection="round_robin",
        max_rounds=4,
    )
    result = chat.run("下一个Sprint应该做什么功能？")
    print("发言轮次:", len(result.rounds))
    for r in result.rounds:
        print(f"  [{r['speaker']}]: {r['content'][:50]}...")


def debate_demo() -> None:
    print("\n=== 5) Debate (辩论) ===")
    debate = Debate(
        debaters=[
            _member(
                "remote",
                "远程办公提高员工满意度、减少通勤时间、降低企业办公成本。"
                "GitLab、Basecamp等公司已证明远程模式可行。",
                "支持远程办公",
            ),
            _member(
                "office",
                "办公室促进团队协作和创造力。面对面对话效率更高，"
                "新员工需要现场指导，公司文化需要物理空间维系。",
                "支持办公室办公",
            ),
        ],
        judge=_agent(
            "综合评估：远程办公在提高效率和降低成本方面优势明显，"
            "但办公室在团队建设方面不可替代。建议采用混合办公模式，"
            "每周2-3天在办公室。",
        ),
        rounds=1,
    )
    result = debate.run("远程办公 vs 办公室办公，哪种模式更好？")
    for r in result.rounds:
        speaker = r["speaker"]
        content = r["content"]
        rnd = r.get("round", "")
        label = f"[{speaker}]" + (f" (第{rnd}轮)" if rnd else "")
        print(f"  {label}: {content[:60]}...")
    print(f"\n  最终裁决: {result.final[:80]}...")


if __name__ == "__main__":
    print("=== 1) Hierarchical (as_tool) ===")
    result = build().run("上海今天适合穿什么？")
    print("Final:", result.final_answer)
    print("Delegations:", [(s.description, s.output) for s in result.steps])

    pipeline_demo()
    router_demo()
    groupchat_demo()
    debate_demo()
