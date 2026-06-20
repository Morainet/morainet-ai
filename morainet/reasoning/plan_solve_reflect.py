"""Plan-Solve-Reflect reasoning strategy for complex, long-running automation.

Three-phase loop designed for tasks that require upfront planning, stepwise
execution with tool use, and quality reflection before returning a final answer.

Phase 1 — **Plan**: LLM decomposes the user's goal into an ordered list of concrete
    substeps, each with expected tool calls and success criteria.

Phase 2 — **Solve**: Execute each planned step in order, calling tools as needed.
    Accumulate results and handle failures with retry + reflection.

Phase 3 — **Reflect**: LLM reviews all step results against the original goal.
    If satisfied → final answer. If not → replan remaining steps and loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from morainet.core.context import Context
from morainet.core.models import AgentResult, Message, Step, StepStatus
from morainet.observability.tracing import logger
from morainet.reasoning.base import (
    ReasoningStrategy,
    enforce_consecutive_errors,
    make_result,
    run_tool_calls,
)
from morainet.reasoning.context_compressor import ContextCompressor
from morainet.reasoning.tool_cache import ToolCache

if TYPE_CHECKING:
    from morainet.core.agent import Agent

# -- data models --------------------------------------------------------------


@dataclass
class PlanStep:
    """One step in a plan: description + execution tracking."""

    index: int
    description: str
    expected_tools: list[str] = field(default_factory=list)
    success_criterion: str = ""
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    result_summary: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = 0.0


# -- strategy -----------------------------------------------------------------


class PlanSolveReflectStrategy(ReasoningStrategy):
    """Three-phase reasoning for long-running complex tasks.

    Parameters
    ----------
    max_plan_steps : int
        Upper bound on how many substeps the planner may produce.
    max_step_attempts : int
        Number of retries allowed per step before giving up.
    max_reflect_rounds : int
        Maximum full plan→solve→reflect loops before returning best-effort.
    compress_after_steps : int
        When the context exceeds this many messages, trigger compression.
    token_budget : int | None
        Compressor budget override. None = use default (90% of model max).
    tool_cache : ToolCache | None
        Optional cache to avoid redundant tool calls.
    planner_prompt : str | None
        Override the built-in planner prompt.
    reflector_prompt : str | None
        Override the built-in reflector prompt.
    """

    _BUILTIN_PLANNER = (
        "You are a task planner. Break the user's goal into an ordered list of "
        "concrete, actionable steps.\n"
        "Each step should be a single action the agent can take using available tools.\n\n"
        "Available tools:\n{tools}\n\n"
        "User goal: {query}\n\n"
        "Return a JSON array of steps. Each step has:\n"
        '- "description": what to do\n'
        '- "expected_tools": [list of tool names likely needed]\n'
        '- "success_criterion": how to know this step is done\n\n'
        "JSON plan:"
    )

    _BUILTIN_REFLECTOR = (
        "You are a progress reviewer. Evaluate whether the steps completed so far "
        "satisfy the original goal.\n\n"
        "Original goal: {query}\n\n"
        "Plan:\n{plan_summary}\n\n"
        "Completed steps:\n{completed}\n\n"
        "Pending steps:\n{pending}\n\n"
        "Reply with a JSON object:\n"
        '- "verdict": "done" if the goal is achieved, "continue" if more work is needed\n'
        '- "final_answer": if done, the complete answer for the user\n'
        '- "reason": brief explanation of the verdict\n\n'
        "JSON:"
    )

    def __init__(
        self,
        max_plan_steps: int = 10,
        max_step_attempts: int = 3,
        max_reflect_rounds: int = 3,
        compress_after_steps: int = 20,
        token_budget: int | None = None,
        tool_cache: ToolCache | None = None,
        planner_prompt: str | None = None,
        reflector_prompt: str | None = None,
    ) -> None:
        self.max_plan_steps = max_plan_steps
        self.max_step_attempts = max_step_attempts
        self.max_reflect_rounds = max_reflect_rounds
        self.compress_after_steps = compress_after_steps
        self.token_budget = token_budget
        self.tool_cache = tool_cache
        self.planner_prompt = planner_prompt or self._BUILTIN_PLANNER
        self.reflector_prompt = reflector_prompt or self._BUILTIN_REFLECTOR

    # -- main loop -------------------------------------------------------------

    async def run(self, agent: "Agent", ctx: Context) -> AgentResult:
        compressor = ContextCompressor(
            provider=agent.provider,
            token_budget=self.token_budget,
        )

        # Phase 1: Plan
        plan = await self._plan(agent, ctx)
        if not plan.steps:
            # Fallback: run as normal tool-calling loop
            return await self._fallback_run(agent, ctx)

        for reflect_round in range(self.max_reflect_rounds):
            logger.debug(
                f"[{ctx.trace_id}] Plan-Solve-Reflect round {reflect_round + 1}/{self.max_reflect_rounds}"
            )

            # Phase 2: Solve — execute each pending step
            for step in plan.steps:
                if step.status == StepStatus.SUCCESS:
                    continue

                step.status = StepStatus.RUNNING
                await self._execute_step(agent, ctx, step)

                # Check context size; compress if needed
                if len(ctx.messages) > self.compress_after_steps:
                    compressed = await compressor.compress(ctx.messages, ctx.usage)
                    ctx.messages = compressed.messages
                    logger.debug(
                        f"[{ctx.trace_id}] context compressed: "
                        f"{compressed.stats['before_count']} → {compressed.stats['after_count']} msgs"
                    )

            # Phase 3: Reflect
            pending = [s for s in plan.steps if s.status != StepStatus.SUCCESS]
            if not pending:
                # All steps done — build final answer from results
                final = self._compile_final(ctx, plan)
                return make_result(ctx, final)

            verdict = await self._reflect(agent, ctx, plan)
            if verdict["verdict"] == "done":
                return make_result(ctx, verdict.get("final_answer", "Task completed."))

            # Replan remaining steps if reflection says continue
            if reflect_round < self.max_reflect_rounds - 1:
                new_steps = await self._replan(agent, plan, ctx, verdict.get("reason", ""))
                for s in new_steps:
                    plan.steps.append(s)

        # Exhausted reflect rounds — best-effort answer
        best = self._compile_final(ctx, plan)
        return make_result(ctx, best)

    # -- phase implementations -------------------------------------------------

    async def _plan(self, agent: "Agent", ctx: Context) -> Plan:
        """Phase 1: decompose the user's goal into concrete steps."""
        tools_text = self._format_tools(agent)
        prompt = self.planner_prompt.format(tools=tools_text, query=ctx.query)

        try:
            response = await agent.provider.chat([Message.user(prompt)])
            text = response.message.content or ""
            plan_data = self._parse_json(text)
        except Exception:
            plan_data = []

        if not isinstance(plan_data, list) or not plan_data:
            return Plan(goal=ctx.query)

        steps = []
        for i, step_data in enumerate(plan_data[: self.max_plan_steps]):
            if isinstance(step_data, dict):
                steps.append(
                    PlanStep(
                        index=i,
                        description=step_data.get("description", f"Step {i + 1}"),
                        expected_tools=step_data.get("expected_tools", []),
                        success_criterion=step_data.get("success_criterion", ""),
                        max_attempts=self.max_step_attempts,
                    )
                )

        logger.debug(
            f"[{ctx.trace_id}] planned {len(steps)} steps: "
            f"{[s.description[:40] for s in steps]}"
        )
        return Plan(goal=ctx.query, steps=steps)

    async def _execute_step(
        self, agent: "Agent", ctx: Context, step: PlanStep
    ) -> None:
        """Phase 2: execute one planned step with tool calling and retry."""

        # Prompt the LLM to execute this specific step
        exec_prompt = (
            f"Execute this step using available tools:\n\n"
            f"Step: {step.description}\n"
            f"Success criterion: {step.success_criterion}\n\n"
            f"When finished, summarize the result clearly."
        )
        ctx.add_message(Message.user(exec_prompt))

        for attempt in range(step.max_attempts):
            step.attempts = attempt + 1
            try:
                response = await agent.provider.chat(
                    ctx.messages, agent.registry.schemas() or None
                )
                ctx.add_usage(response.usage)
                ctx.add_message(response.message)
                await agent.hooks.llm_end(ctx, response)

                if response.message.tool_calls:
                    cached_call = None
                    if self.tool_cache:
                        for tc in response.message.tool_calls:
                            cached_call = self.tool_cache.get(tc.name, tc.arguments)

                    if cached_call:
                        cached_result, cached_error = cached_call
                        if cached_error is None:
                            ctx.add_message(
                                Message.tool(
                                    content=json.dumps(cached_result, default=str),
                                    tool_call_id=response.message.tool_calls[0].id,
                                )
                            )
                            step_rec = Step(
                                index=len(ctx.steps),
                                description=f"{step.description} [cached]",
                                status=StepStatus.SUCCESS,
                                output=cached_result,
                            )
                            ctx.add_step(step_rec)
                            step.status = StepStatus.SUCCESS
                            step.result_summary = str(cached_result)[:200]
                            return

                    await run_tool_calls(
                        agent.registry, ctx, response.message.tool_calls,
                        agent.hooks, agent.approve_tool,
                    )
                    enforce_consecutive_errors(agent.max_consecutive_errors, ctx)

                    # Cache results
                    for i, tc in enumerate(response.message.tool_calls):
                        last_step = ctx.steps[-len(response.message.tool_calls) + i] if i < len(ctx.steps) else None
                        if last_step and self.tool_cache:
                            self.tool_cache.set(
                                tc.name, tc.arguments,
                                result=last_step.output,
                                error=last_step.error,
                            )
                else:
                    # No tool calls — model answered directly
                    step.status = StepStatus.SUCCESS
                    step.result_summary = (response.message.content or "")[:200]
                    return

                # Check if success criterion is met
                if step.success_criterion:
                    check = await self._check_step_success(agent, step, ctx)
                    if check:
                        step.status = StepStatus.SUCCESS
                        return

            except Exception as exc:
                step.errors.append(str(exc))
                logger.warning(
                    f"[{ctx.trace_id}] step '{step.description[:30]}' attempt {attempt + 1} failed: {exc}"
                )
                if attempt < step.max_attempts - 1:
                    ctx.add_message(
                        Message.user(
                            f"Step failed: {exc}. Please try a different approach "
                            f"for: {step.description}"
                        )
                    )
                else:
                    step.status = StepStatus.FAILED

        if step.status != StepStatus.SUCCESS:
            step.status = StepStatus.FAILED
            step.result_summary = f"Failed after {step.attempts} attempts: {'; '.join(step.errors[-3:])}"

    async def _check_step_success(self, agent: "Agent", step: PlanStep, ctx: Context) -> bool:
        """Lightweight check: does the latest output satisfy the success criterion?"""
        recent = ctx.messages[-3:]
        content = "\n".join(str(m.content or "") for m in recent)
        prompt = (
            f"Does the following output satisfy this success criterion?\n\n"
            f"Criterion: {step.success_criterion}\n\n"
            f"Recent output:\n{content}\n\n"
            f"Reply ONLY 'yes' or 'no'."
        )
        try:
            resp = await agent.provider.chat([Message.user(prompt)])
            return "yes" in (resp.message.content or "").strip().lower()
        except Exception:
            return True  # Assume success on check failure

    async def _reflect(
        self, agent: "Agent", ctx: Context, plan: Plan
    ) -> dict[str, str]:
        """Phase 3: reflect on progress and decide whether the goal is achieved."""
        completed = [s for s in plan.steps if s.status == StepStatus.SUCCESS]
        pending = [s for s in plan.steps if s.status != StepStatus.SUCCESS]

        plan_summary = "\n".join(
            f"  [{s.status.value.upper()}] {s.description}" for s in plan.steps
        )
        completed_text = "\n".join(
            f"- {s.description}: {s.result_summary or 'done'}" for s in completed
        )
        pending_text = "\n".join(
            f"- {s.description} (failed: {'; '.join(s.errors[-2:])})"
            if s.status == StepStatus.FAILED
            else f"- {s.description}"
            for s in pending
        )

        prompt = self.reflector_prompt.format(
            query=plan.goal,
            plan_summary=plan_summary,
            completed=completed_text or "(none)",
            pending=pending_text or "(none)",
        )

        try:
            response = await agent.provider.chat([Message.user(prompt)])
            verdict = self._parse_json(response.message.content or "")
        except Exception:
            verdict = {}

        if not isinstance(verdict, dict):
            verdict = {"verdict": "continue" if pending else "done"}

        logger.debug(
            f"[{ctx.trace_id}] reflect verdict: {verdict.get('verdict', 'unknown')} "
            f"({verdict.get('reason', '')[:60]})"
        )
        return {"verdict": str(verdict.get("verdict", "done")), "final_answer": str(verdict.get("final_answer", "")), "reason": str(verdict.get("reason", ""))}

    async def _replan(
        self, agent: "Agent", plan: Plan, ctx: Context, reason: str
    ) -> list[PlanStep]:
        """Create new steps based on reflection feedback."""
        pending = [s for s in plan.steps if s.status != StepStatus.SUCCESS]
        pending_desc = "\n".join(f"- {s.description}" for s in pending)

        prompt = (
            f"Some steps are still pending:\n{pending_desc}\n\n"
            f"Reviewer feedback: {reason}\n\n"
            f"Generate additional steps to complete the goal: {plan.goal}\n\n"
            f"Available tools:\n{self._format_tools(agent)}\n\n"
            f"Return a JSON array of steps."
        )

        try:
            response = await agent.provider.chat([Message.user(prompt)])
            steps_data = self._parse_json(response.message.content or "")
        except Exception:
            return []

        if not isinstance(steps_data, list):
            return []

        offset = len(plan.steps)
        new_steps: list[PlanStep] = []
        for i, sd in enumerate(steps_data[: self.max_plan_steps]):
            if isinstance(sd, dict):
                new_steps.append(
                    PlanStep(
                        index=offset + i,
                        description=sd.get("description", f"Step {offset + i + 1}"),
                        expected_tools=sd.get("expected_tools", []),
                        success_criterion=sd.get("success_criterion", ""),
                        max_attempts=self.max_step_attempts,
                    )
                )
        return new_steps

    async def _fallback_run(self, agent: "Agent", ctx: Context) -> AgentResult:
        """When planning fails, degrade to a simple tool-calling loop."""
        from morainet.exceptions import MaxStepsExceededError
        from morainet.reasoning.base import enforce_budget

        for _ in range(agent.max_steps):
            response = await agent.provider.chat(ctx.messages, agent.registry.schemas() or None)
            ctx.add_usage(response.usage)
            ctx.add_message(response.message)
            await agent.hooks.llm_end(ctx, response)
            enforce_budget(agent.token_budget, ctx)

            if not response.message.tool_calls:
                return make_result(ctx, response.message.content or "")

            await run_tool_calls(
                agent.registry, ctx, response.message.tool_calls, agent.hooks, agent.approve_tool,
            )
            enforce_consecutive_errors(agent.max_consecutive_errors, ctx)

        raise MaxStepsExceededError(f"Fallback did not converge within max_steps={agent.max_steps}")

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _format_tools(agent: "Agent") -> str:
        lines = []
        for schema in agent.registry.schemas():
            params = ", ".join(schema["parameters"]["properties"].keys())
            lines.append(f"- {schema['name']}({params}): {schema['description']}")
        return "\n".join(lines) or "(no tools available)"

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract and parse the first JSON array/object from LLM output."""
        text = text.strip().strip("`").strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON block
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find first [ or {
        for start, end in [("[", "]"), ("{", "}")]:
            si = text.find(start)
            if si != -1:
                ei = text.rfind(end)
                if ei > si:
                    try:
                        return json.loads(text[si : ei + 1])
                    except json.JSONDecodeError:
                        pass
        return text

    @staticmethod
    def _compile_final(ctx: Context, plan: Plan) -> str:
        """Build a final answer from the plan step results."""
        lines = []
        for s in plan.steps:
            status = "✓" if s.status == StepStatus.SUCCESS else "✗"
            lines.append(f"{status} {s.description}")
            if s.result_summary:
                lines.append(f"  → {s.result_summary}")
        success_count = sum(1 for s in plan.steps if s.status == StepStatus.SUCCESS)
        header = f"Completed {success_count}/{len(plan.steps)} steps:\n\n"
        return header + "\n".join(lines)
