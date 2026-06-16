from __future__ import annotations

from morainet.core.models import ChatResponse, Message, Role
from morainet.memory import SummarizingMemory
from morainet.providers import MockProvider


def _summary_provider() -> MockProvider:
    # Every chat() call returns a canned "summary".
    return MockProvider(
        handler=lambda messages, tools: ChatResponse(
            message=Message.assistant(content="SUMMARY")
        )
    )


async def test_no_compression_below_trigger():
    mem = SummarizingMemory(provider=_summary_provider(), keep_recent=2, trigger_messages=5)
    for i in range(4):
        await mem.add(Message.user(f"m{i}"))
    assert mem.summary is None
    assert len(mem) == 4


async def test_compresses_when_over_trigger():
    mem = SummarizingMemory(provider=_summary_provider(), keep_recent=2, trigger_messages=5)
    for i in range(6):  # 6 > trigger 5 -> compress
        await mem.add(Message.user(f"m{i}"))

    assert mem.summary == "SUMMARY"
    assert len(mem) == 2  # only keep_recent retained

    ctx = await mem.get_context("q")
    assert ctx[0].role == Role.SYSTEM
    assert "SUMMARY" in (ctx[0].content or "")
    # recent verbatim messages follow the summary
    assert [m.content for m in ctx[1:]] == ["m4", "m5"]
