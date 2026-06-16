from __future__ import annotations

import pytest

from morainet import Agent, Pipeline, Route, Router, Stage
from morainet.core.models import ChatResponse, Message
from morainet.providers import MockProvider


def _const_agent(answer: str) -> Agent:
    return Agent(provider=MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content=answer)
    )))


# --- Pipeline --------------------------------------------------------------


async def test_pipeline_threads_output_forward():
    captured: list[str] = []

    def stage2_handler(messages, tools):
        captured.append(messages[-1].content or "")
        return ChatResponse(message=Message.assistant(content="OUT2"))

    s1 = Agent(provider=MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="OUT1")
    )))
    s2 = Agent(provider=MockProvider(handler=stage2_handler))

    pipe = Pipeline([Stage("research", s1), Stage("write", s2)])
    result = await pipe.arun("写一篇短文")

    assert result.final == "OUT2"
    assert result.outputs == {"research": "OUT1", "write": "OUT2"}
    # stage 2 saw stage 1's output in its prompt
    assert "OUT1" in captured[0]


async def test_pipeline_instruction_template():
    captured: list[str] = []

    def h2(messages, tools):
        captured.append(messages[-1].content or "")
        return ChatResponse(message=Message.assistant(content="done"))

    s1 = Agent(provider=MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="事实A")
    )))
    s2 = Agent(provider=MockProvider(handler=h2))

    pipe = Pipeline([
        Stage("research", s1),
        Stage("write", s2, instruction="基于调研「{research}」回答：{query}"),
    ])
    await pipe.arun("Q")
    assert captured[0] == "基于调研「事实A」回答：Q"


def test_pipeline_validation():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(ValueError):
        Pipeline([Stage("a", _const_agent("x")), Stage("a", _const_agent("y"))])


def test_pipeline_sync_run():
    pipe = Pipeline([Stage("only", _const_agent("hi"))])
    assert pipe.run("q").final == "hi"


# --- Router ----------------------------------------------------------------


async def test_router_rule_based():
    routes = [Route("math", _const_agent("=42")), Route("chat", _const_agent("hello"))]
    router = Router(routes, selector=lambda q: "math" if "+" in q else "chat")

    r = await router.arun("2+2?")
    assert r.route == "math"
    assert r.final == "=42"

    r2 = await router.arun("hi there")
    assert r2.route == "chat"


async def test_router_unknown_selection_falls_back_to_default():
    routes = [Route("a", _const_agent("A")), Route("b", _const_agent("B"))]
    router = Router(routes, selector=lambda q: "nonexistent", default="b")
    r = await router.arun("x")
    assert r.route == "b"


async def test_router_llm_based():
    routes = [
        Route("billing", _const_agent("账单已处理"), "账单/付款问题"),
        Route("tech", _const_agent("技术已解决"), "技术故障"),
    ]
    # LLM picks "tech".
    selector_provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="tech")
    ))
    router = Router(routes, provider=selector_provider)
    r = await router.arun("我的设备连不上网")
    assert r.route == "tech"
    assert r.final == "技术已解决"


async def test_router_async_selector():
    async def pick(q):
        return "a"

    router = Router([Route("a", _const_agent("A"))], selector=pick)
    assert (await router.arun("x")).route == "a"


def test_router_validation():
    with pytest.raises(ValueError):
        Router([])
    with pytest.raises(ValueError):  # both selector and provider
        Router([Route("a", _const_agent("A"))], selector=lambda q: "a",
               provider=MockProvider())
    with pytest.raises(ValueError):  # neither
        Router([Route("a", _const_agent("A"))])
