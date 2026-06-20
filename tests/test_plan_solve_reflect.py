"""Tests for morainet.reasoning.plan_solve_reflect."""

from __future__ import annotations

import json


from morainet.core.agent import Agent
from morainet.core.context import Context
from morainet.core.models import (
    ChatResponse,
    Message,
    StepStatus,
    ToolCall,
    Usage,
)
from morainet.providers.mock import MockProvider
from morainet.reasoning.plan_solve_reflect import (
    Plan,
    PlanSolveReflectStrategy,
    PlanStep,
)
from morainet.tools import tool


@tool
def echo(text: str) -> str:
    """Echo back."""
    return text


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ---------------------------------------------------------------------------
# PlanStep / Plan dataclasses
# ---------------------------------------------------------------------------

def test_plan_step_defaults():
    s = PlanStep(index=0, description="test")
    assert s.index == 0
    assert s.description == "test"
    assert s.expected_tools == []
    assert s.success_criterion == ""
    assert s.status == StepStatus.PENDING
    assert s.attempts == 0
    assert s.max_attempts == 3
    assert s.result_summary == ""
    assert s.errors == []


def test_plan_step_custom():
    s = PlanStep(
        index=1,
        description="do something",
        expected_tools=["search"],
        success_criterion="found",
        max_attempts=2,
    )
    assert s.expected_tools == ["search"]
    assert s.max_attempts == 2


def test_plan():
    p = Plan(goal="test goal")
    assert p.goal == "test goal"
    assert p.steps == []
    assert p.created_at == 0.0


# ---------------------------------------------------------------------------
# PlanSolveReflectStrategy: construction
# ---------------------------------------------------------------------------

def test_psr_defaults():
    s = PlanSolveReflectStrategy()
    assert s.max_plan_steps == 10
    assert s.max_step_attempts == 3
    assert s.max_reflect_rounds == 3
    assert s.compress_after_steps == 20
    assert s.token_budget is None
    assert s.tool_cache is None


def test_psr_custom():
    from morainet.reasoning.tool_cache import ToolCache
    cache = ToolCache(ttl=None)
    s = PlanSolveReflectStrategy(
        max_plan_steps=5,
        max_step_attempts=2,
        max_reflect_rounds=1,
        compress_after_steps=50,
        token_budget=10000,
        tool_cache=cache,
        planner_prompt="Custom plan: {tools} {query}",
        reflector_prompt="Custom reflect: {query}",
    )
    assert s.max_plan_steps == 5
    assert s.max_step_attempts == 2
    assert s.max_reflect_rounds == 1
    assert s.planner_prompt == "Custom plan: {tools} {query}"
    assert s.reflector_prompt == "Custom reflect: {query}"


# ---------------------------------------------------------------------------
# _format_tools
# ---------------------------------------------------------------------------

def test_format_tools():
    agent = Agent(provider=MockProvider(), tools=[echo, add])
    s = PlanSolveReflectStrategy()
    result = s._format_tools(agent)
    assert "echo" in result
    assert "add" in result
    assert "text" in result  # parameter name


def test_format_tools_empty():
    agent = Agent(provider=MockProvider(), tools=[])
    s = PlanSolveReflectStrategy()
    result = s._format_tools(agent)
    assert "no tools" in result


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

def test_parse_json_direct():
    s = PlanSolveReflectStrategy()
    assert s._parse_json('[{"x": 1}]') == [{"x": 1}]
    assert s._parse_json('{"key": "value"}') == {"key": "value"}


def test_parse_json_code_block():
    s = PlanSolveReflectStrategy()
    text = 'Here is the plan:\n```json\n[{"name": "test"}]\n```\nDone.'
    result = s._parse_json(text)
    assert result == [{"name": "test"}]


def test_parse_json_nested_brackets():
    s = PlanSolveReflectStrategy()
    text = 'xxx [1, 2, 3] yyy'
    result = s._parse_json(text)
    assert result == [1, 2, 3]


def test_parse_json_fallback_to_text():
    s = PlanSolveReflectStrategy()
    text = "no json here at all"
    result = s._parse_json(text)
    assert result == "no json here at all"


# ---------------------------------------------------------------------------
# _compile_final
# ---------------------------------------------------------------------------

def test_compile_final():
    ctx = Context(trace_id="t1", query="test")
    steps = [
        PlanStep(index=0, description="step1", status=StepStatus.SUCCESS, result_summary="done"),
        PlanStep(index=1, description="step2", status=StepStatus.FAILED, result_summary="failed"),
    ]
    plan = Plan(goal="test", steps=steps)
    s = PlanSolveReflectStrategy()
    result = s._compile_final(ctx, plan)
    assert "step1" in result
    assert "step2" in result
    assert "1/2" in result


# ---------------------------------------------------------------------------
# _plan
# ---------------------------------------------------------------------------

async def test_plan_success():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content=json.dumps([
            {"description": "do task 1", "expected_tools": ["echo"], "success_criterion": "done"},
            {"description": "do task 2", "expected_tools": ["add"], "success_criterion": "done"},
        ])),
        usage=Usage(total_tokens=10),
    ))
    agent = Agent(provider=provider, tools=[echo, add])
    ctx = Context(trace_id="t1", query="do things")
    s = PlanSolveReflectStrategy()
    plan = await s._plan(agent, ctx)
    assert len(plan.steps) == 2
    assert plan.steps[0].description == "do task 1"
    assert plan.steps[1].expected_tools == ["add"]


async def test_plan_empty():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="[]"),
        usage=Usage(total_tokens=2),
    ))
    agent = Agent(provider=provider, tools=[echo])
    ctx = Context(trace_id="t1", query="q")
    s = PlanSolveReflectStrategy()
    plan = await s._plan(agent, ctx)
    assert plan.steps == []


async def test_plan_exception():
    """When LLM call fails, return empty plan."""
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    s = PlanSolveReflectStrategy()
    plan = await s._plan(agent, ctx)
    assert plan.steps == []


# ---------------------------------------------------------------------------
# _reflect
# ---------------------------------------------------------------------------

async def test_reflect_done():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content=json.dumps({
            "verdict": "done",
            "final_answer": "All done!",
            "reason": "everything ok",
        })),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    plan = Plan(goal="q", steps=[
        PlanStep(index=0, description="s1", status=StepStatus.SUCCESS, result_summary="ok"),
    ])
    s = PlanSolveReflectStrategy()
    verdict = await s._reflect(agent, ctx, plan)
    assert verdict["verdict"] == "done"
    assert verdict["final_answer"] == "All done!"


async def test_reflect_continue():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content=json.dumps({
            "verdict": "continue",
            "reason": "more work needed",
        })),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    plan = Plan(goal="q", steps=[
        PlanStep(index=0, description="s1", status=StepStatus.PENDING),
    ])
    s = PlanSolveReflectStrategy()
    verdict = await s._reflect(agent, ctx, plan)
    assert verdict["verdict"] == "continue"


async def test_reflect_exception():
    """When LLM fails, defaults to continue if pending steps exist."""
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    plan = Plan(goal="q", steps=[
        PlanStep(index=0, description="s1", status=StepStatus.PENDING),
    ])
    s = PlanSolveReflectStrategy()
    verdict = await s._reflect(agent, ctx, plan)
    assert verdict["verdict"] == "continue"


# ---------------------------------------------------------------------------
# _replan
# ---------------------------------------------------------------------------

async def test_replan():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content=json.dumps([
            {"description": "new step 1"},
            {"description": "new step 2"},
        ])),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider, tools=[echo])
    ctx = Context(trace_id="t1", query="q")
    plan = Plan(goal="q", steps=[
        PlanStep(index=0, description="old", status=StepStatus.FAILED),
    ])
    s = PlanSolveReflectStrategy()
    new_steps = await s._replan(agent, plan, ctx, "fix needed")
    assert len(new_steps) == 2
    assert new_steps[0].description == "new step 1"
    assert new_steps[0].index == 1  # offset by existing steps


async def test_replan_exception():
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    plan = Plan(goal="q")
    s = PlanSolveReflectStrategy()
    new_steps = await s._replan(agent, plan, ctx, "reason")
    assert new_steps == []


# ---------------------------------------------------------------------------
# _check_step_success
# ---------------------------------------------------------------------------

async def test_check_step_success_yes():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="yes"),
        usage=Usage(total_tokens=2),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    ctx.messages.append(Message.user("did the thing"))
    step = PlanStep(index=0, description="step", success_criterion="thing is done")
    s = PlanSolveReflectStrategy()
    result = await s._check_step_success(agent, step, ctx)
    assert result is True


async def test_check_step_success_no():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="no"),
        usage=Usage(total_tokens=2),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    ctx.messages.append(Message.user("tried"))
    step = PlanStep(index=0, description="step", success_criterion="thing is done")
    s = PlanSolveReflectStrategy()
    result = await s._check_step_success(agent, step, ctx)
    assert result is False


async def test_check_step_success_exception():
    provider = MockProvider(handler=lambda m, t: (_ for _ in ()).throw(Exception("fail")))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    step = PlanStep(index=0, description="step", success_criterion="done")
    s = PlanSolveReflectStrategy()
    result = await s._check_step_success(agent, step, ctx)
    assert result is True  # assume success on error


# ---------------------------------------------------------------------------
# _fallback_run
# ---------------------------------------------------------------------------

async def test_fallback_run_direct_answer():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="direct answer"),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider)
    ctx = Context(trace_id="t1", query="q")
    ctx.messages.append(Message.user("q"))
    s = PlanSolveReflectStrategy()
    result = await s._fallback_run(agent, ctx)
    assert "direct answer" in result.final_answer


async def test_fallback_run_with_tool():
    call_count = {"n": 0}

    def handler(messages, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})]
                ),
                usage=Usage(total_tokens=5),
            )
        else:
            return ChatResponse(
                message=Message.assistant(content="done"),
                usage=Usage(total_tokens=3),
            )

    agent = Agent(provider=MockProvider(handler=handler), tools=[echo])
    ctx = Context(trace_id="t1", query="q")
    ctx.messages.append(Message.user("q"))
    s = PlanSolveReflectStrategy()
    result = await s._fallback_run(agent, ctx)
    assert result.final_answer is not None


# ---------------------------------------------------------------------------
# run: empty plan → fallback
# ---------------------------------------------------------------------------

async def test_run_empty_plan_fallback():
    """When planning returns no steps, use fallback."""
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="Final Answer: 42"),
        usage=Usage(total_tokens=5),
    ))
    agent = Agent(provider=provider, tools=[echo])
    ctx = Context(trace_id="t1", query="q")
    ctx.messages.append(Message.user("q"))

    s = PlanSolveReflectStrategy()
    # Override _plan to return a plan with no steps
    async def empty_plan(agent, ctx):
        return Plan(goal="q", steps=[])
    s._plan = empty_plan
    result = await s.run(agent, ctx)
    assert "42" in result.final_answer
