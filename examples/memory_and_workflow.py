"""v0.2 features: long-term memory continuity and a Workflow DAG. No API key.

Run:
    python examples/memory_and_workflow.py
"""

from __future__ import annotations

from morainet import Agent
from morainet.core.models import ChatResponse, Message
from morainet.memory import InMemoryVectorStore, LongMemory
from morainet.providers import MockProvider
from morainet.workflow import Workflow


def memory_demo() -> None:
    print("=== Long-term memory ===")
    memory = LongMemory(store=InMemoryVectorStore())

    def handler(messages, tools):
        injected = [m.content for m in messages if m.role.value == "system"]
        return ChatResponse(message=Message.assistant(content=f"(recalled: {injected})"))

    agent = Agent(provider=MockProvider(handler=handler), memory=memory)
    agent.run("记住：我对花生过敏")
    # A later query about 花生 retrieves the stored fact (keyword overlap).
    result = agent.run("关于花生我需要注意什么")
    print("Recalled context ->", result.final_answer)


def workflow_demo() -> None:
    print("\n=== Workflow DAG ===")
    wf = Workflow()
    wf.add_node("fetch", lambda ctx: {"price": 214})
    wf.add_node("analyze", lambda ctx: f"price={ctx['fetch']['price']}, trend=up")
    wf.add_node("report", lambda ctx: f"REPORT: {ctx['analyze']}")
    wf.connect("fetch", "analyze")
    wf.connect("analyze", "report")

    out = wf.run({"symbol": "AAPL"})
    print("Levels ->", wf.topological_levels())
    print("Result ->", out["report"])


if __name__ == "__main__":
    memory_demo()
    workflow_demo()
