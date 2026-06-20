"""Tests for morainet.reasoning.enhanced_react."""

from __future__ import annotations


import pytest

from morainet.core.agent import Agent
from morainet.core.context import Context
from morainet.core.models import (
    ChatResponse,
    Message,
    Usage,
)
from morainet.exceptions import MaxStepsExceededError
from morainet.providers.mock import MockProvider
from morainet.reasoning.enhanced_react import (
    EnhancedReActStrategy,
    _VERIFY_PROMPT,
)
from morainet.tools import tool


# ---------- helpers ----------

@tool
def echo(text: str) -> str:
    """Echo back the input."""
    return text


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ---------- construction ----------

def test_enhanced_react_defaults():
    strategy = EnhancedReActStrategy()
    assert strategy.max_steps is None
    assert strategy.max_decomposition_depth == 3
    assert strategy.max_retry_per_action == 2
    assert strategy.verify_before_answer is True
    assert strategy.compress_after_messages == 30
    assert strategy.tool_cache is None


def test_enhanced_react_custom():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)
    strategy = EnhancedReActStrategy(
        max_steps=10,
        max_decomposition_depth=2,
        max_retry_per_action=1,
        verify_before_answer=False,
        compress_after_messages=50,
        tool_cache=cache,
    )
    assert strategy.max_steps == 10
    assert strategy.max_decomposition_depth == 2
    assert strategy.max_retry_per_action == 1
    assert strategy.verify_before_answer is False
    assert strategy.compress_after_messages == 50
    assert strategy.tool_cache is cache


# ---------- system prompt ----------

def test_system_prompt_default():
    agent = Agent(provider=MockProvider(), tools=[echo])
    strategy = EnhancedReActStrategy()
    prompt = strategy._system_prompt(agent)
    assert "ReAct format" in prompt
    assert "Phase 1" in prompt
    assert "echo" in prompt


def test_system_prompt_from_registry():
    """When 'react_system' template exists, use it."""
    from morainet.prompts.registry import PromptTemplate
    agent = Agent(provider=MockProvider(), tools=[echo])
    agent.prompts.register("react_system", PromptTemplate(name="react_system", template="Custom: {tools}"))
    strategy = EnhancedReActStrategy()
    prompt = strategy._system_prompt(agent)
    assert prompt.startswith("Custom:")
    assert "echo" in prompt


# ---------- _VERIFY_PROMPT ----------

def test_verify_prompt_format():
    prompt = _VERIFY_PROMPT.format(query="what is 2+2?", draft="4")
    assert "2+2" in prompt
    assert "4" in prompt


# ---------- run: simple answer (no tools needed) ----------

async def test_run_direct_answer():
    """Model gives final answer immediately with verify_before_answer=False."""
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="Final Answer: The answer is 42."),
        usage=Usage(total_tokens=10),
    ))
    agent = Agent(provider=provider, tools=[echo])
    strategy = EnhancedReActStrategy(verify_before_answer=False)
    ctx = Context(trace_id="t1", query="what?")
    ctx.messages.append(Message.user("what?"))

    result = await strategy.run(agent, ctx)
    assert "42" in result.final_answer


# ---------- run: tool call then final answer ----------

async def test_run_tool_then_answer():
    call_count = {"n": 0}

    def handler(messages, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ChatResponse(
                message=Message.assistant(content="Thought: I need to echo\nAction: echo\nAction Input: {\"text\": \"hello\"}"),
                usage=Usage(total_tokens=15),
            )
        else:
            return ChatResponse(
                message=Message.assistant(content="Final Answer: echo returned hello"),
                usage=Usage(total_tokens=10),
            )

    agent = Agent(provider=MockProvider(handler=handler), tools=[echo])
    strategy = EnhancedReActStrategy(verify_before_answer=False, compress_after_messages=999)
    ctx = Context(trace_id="t2", query="echo hello")
    ctx.messages.append(Message.user("echo hello"))

    result = await strategy.run(agent, ctx)
    assert "hello" in result.final_answer


# ---------- run: max steps exceeded ----------

async def test_run_max_steps_exceeded():
    agent = Agent(
        provider=MockProvider(handler=lambda m, t: ChatResponse(
            message=Message.assistant(content="Thought: thinking\nAction: echo\nAction Input: {\"text\": \"x\"}"),
            usage=Usage(total_tokens=2),
        )),
        tools=[echo],
    )
    strategy = EnhancedReActStrategy(max_steps=2, verify_before_answer=False, compress_after_messages=999)
    ctx = Context(trace_id="t3", query="loop")
    ctx.messages.append(Message.user("loop"))

    with pytest.raises(MaxStepsExceededError):
        await strategy.run(agent, ctx)


# ---------- run: no action → text as answer ----------

async def test_run_no_action_text_as_answer():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="No tools needed here."),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider, tools=[echo])
    strategy = EnhancedReActStrategy(verify_before_answer=False, compress_after_messages=999)
    ctx = Context(trace_id="t4", query="simple")
    ctx.messages.append(Message.user("simple"))

    result = await strategy.run(agent, ctx)
    assert "No tools needed" in result.final_answer


# ---------- verify answer ----------

async def test_verify_answer_too_short():
    agent = Agent(provider=MockProvider())
    strategy = EnhancedReActStrategy()
    result = await strategy._verify_answer(agent, "what?", "short")
    assert result["ok"] is False


async def test_verify_answer_ok():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="OK, the answer is correct."),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    strategy = EnhancedReActStrategy()
    result = await strategy._verify_answer(agent, "2+2?", "The answer is 4.")
    assert result["ok"] is True


async def test_verify_answer_not_ok():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="The answer is incomplete."),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    strategy = EnhancedReActStrategy()
    result = await strategy._verify_answer(agent, "complex?", "4")
    assert result["ok"] is False


async def test_verify_answer_exception():
    """When verification fails (error), default to OK."""
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    strategy = EnhancedReActStrategy()
    result = await strategy._verify_answer(agent, "q?", "A long enough answer to bypass short check.")
    assert result["ok"] is True


# ---------- reflect on failure ----------

async def test_reflect_on_failure():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="Use a different approach."),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t5", query="test")
    ctx.messages.append(Message.user("test"))
    strategy = EnhancedReActStrategy()
    reflection = await strategy._reflect_on_failure(agent, "broken", {"x": 1}, "boom", ctx)
    assert "different" in reflection or "approach" in reflection


async def test_reflect_on_failure_exception():
    """When reflection fails, return default message."""
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t5", query="test")
    strategy = EnhancedReActStrategy()
    reflection = await strategy._reflect_on_failure(agent, "tool", {}, "error", ctx)
    assert "different approach" in reflection.lower()


# ---------- task analysis ----------

async def test_task_analysis():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="This is a simple calculation task."),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t6", query="compute 2+2")
    ctx.messages.append(Message.user("compute 2+2"))
    strategy = EnhancedReActStrategy()
    await strategy._task_analysis(agent, ctx)
    assert any("Task Analysis" in str(m.content) for m in ctx.messages)


async def test_task_analysis_exception():
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t7", query="test")
    ctx.messages.append(Message.user("test"))
    strategy = EnhancedReActStrategy()
    await strategy._task_analysis(agent, ctx)
    # Should not raise; just passes without analysis


# ---------- _execute_with_reflection: tool cache hit ----------

async def test_execute_cache_hit():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)
    cache.set("echo", {"text": "cached"}, result="CACHED_RESULT")

    agent = Agent(provider=MockProvider(), tools=[echo])
    ctx = Context(trace_id="t8", query="test")
    strategy = EnhancedReActStrategy(tool_cache=cache)
    await strategy._execute_with_reflection(agent, ctx, "echo", {"text": "cached"})

    # Should have added a step and observation from cache
    assert any("cached" in s.description for s in ctx.steps)
    assert any("CACHED_RESULT" in str(m.content) for m in ctx.messages)


# ---------- _execute_with_reflection: success with cache set ----------

async def test_execute_success_with_cache():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)

    agent = Agent(provider=MockProvider(), tools=[echo])
    ctx = Context(trace_id="t9", query="test")
    strategy = EnhancedReActStrategy(tool_cache=cache, max_retry_per_action=0)
    await strategy._execute_with_reflection(agent, ctx, "echo", {"text": "hello"})

    # Result should be cached
    entry = cache.get("echo", {"text": "hello"})
    assert entry is not None
    assert "hello" in str(entry[0])
