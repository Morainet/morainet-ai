"""Enhanced ReAct strategy with multi-layer task decomposition, failure reflection,
retry with corrected approach, and multi-round self-verification.

Extends the base ReAct loop with:

1. **Task decomposition** — before acting, break complex goals into subtasks
2. **Failure reflection** — on tool error, analyze the cause and retry with a
   corrected approach (not blind retry)
3. **Self-verification** — before declaring a Final Answer, the model verifies
   its own output against the original query
4. **Context compression** — automatic compression when the message history
   grows too large
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from morainet.core.context import Context
from morainet.core.models import AgentResult, Message, Step, StepStatus
from morainet.exceptions import MaxStepsExceededError
from morainet.observability.tracing import logger
from morainet.reasoning.base import (
    ReasoningStrategy,
    enforce_budget,
    enforce_consecutive_errors,
    execute_tool,
    make_result,
    stringify,
)
from morainet.reasoning.context_compressor import ContextCompressor
from morainet.reasoning.tool_cache import ToolCache
from morainet.reasoning.react import (
    parse_action,
    parse_final_answer,
)

if TYPE_CHECKING:
    from morainet.core.agent import Agent

_ENHANCED_PROMPT = """\
You are an advanced agent that solves tasks using the ReAct format.

You have access to these tools:
{tools}

Follow this structured workflow:

## Phase 1 — Analyze
Before taking action, briefly analyze the task:
- What is the user really asking?
- Can this be broken into subtasks?
- Which tools are relevant?

## Phase 2 — Act (repeat as needed)
Use EXACTLY this format for each step:
Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <a JSON object of arguments>

After each Action you receive an Observation.

## Phase 3 — Verify
Before giving your final answer, verify:
- Did you answer the user's original question?
- Are the tool results consistent?
- Is anything missing?

When fully satisfied, respond with:
Thought: <final reasoning>
Final Answer: <the complete answer to the user>
"""

_VERIFY_PROMPT = (
    "Before giving a final answer, verify the following:\n"
    "1. Does this directly answer the user's original question?\n"
    "2. Are the tool results consistent with the answer?\n"
    "3. Is anything missing or incomplete?\n\n"
    "User's original query: {query}\n\n"
    "Your proposed answer: {draft}\n\n"
    "If satisfied, reply 'OK'. If not, reply with what needs to be fixed."
)


class EnhancedReActStrategy(ReasoningStrategy):
    """Enhanced ReAct with task decomposition, failure reflection, and verification.

    Parameters
    ----------
    max_steps : int | None
        Maximum reasoning loop iterations. Falls back to agent.max_steps.
    max_decomposition_depth : int
        Maximum depth for recursive task decomposition.
    max_retry_per_action : int
        Number of retries per tool call on failure (with reflection).
    verify_before_answer : bool
        Whether to run a self-verification step before returning the final answer.
    compress_after_messages : int
        Trigger context compression when message count exceeds this.
    tool_cache : ToolCache | None
        Optional cache for tool call results.
    """

    def __init__(
        self,
        max_steps: int | None = None,
        max_decomposition_depth: int = 3,
        max_retry_per_action: int = 2,
        verify_before_answer: bool = True,
        compress_after_messages: int = 30,
        tool_cache: ToolCache | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_decomposition_depth = max_decomposition_depth
        self.max_retry_per_action = max_retry_per_action
        self.verify_before_answer = verify_before_answer
        self.compress_after_messages = compress_after_messages
        self.tool_cache = tool_cache

    # -- system prompt ---------------------------------------------------------

    def _system_prompt(self, agent: "Agent") -> str:
        if "react_system" in agent.prompts._templates:
            return agent.prompts.get("react_system").render(tools=self._render_tools(agent))
        return _ENHANCED_PROMPT.format(tools=self._render_tools(agent))

    @staticmethod
    def _render_tools(agent: "Agent") -> str:
        from morainet.reasoning.react import _render_tools
        return _render_tools(agent)

    # -- main loop -------------------------------------------------------------

    async def run(self, agent: "Agent", ctx: Context) -> AgentResult:
        max_steps = self.max_steps if self.max_steps is not None else agent.max_steps
        compressor = ContextCompressor(provider=agent.provider)

        # Inject enhanced system prompt
        ctx.messages.insert(0, Message.system(self._system_prompt(agent)))

        # Phase 1: Task analysis (optional decomposition)
        await self._task_analysis(agent, ctx)

        for step_no in range(max_steps):
            response = await agent.provider.chat(ctx.messages)
            ctx.add_usage(response.usage)
            text = response.message.content or ""
            ctx.add_message(Message.assistant(content=text))
            await agent.hooks.llm_end(ctx, response)
            enforce_budget(agent.token_budget, ctx)

            # Check for Final Answer
            final = parse_final_answer(text)
            if final is not None:
                # Phase 3: Self-verification
                if self.verify_before_answer:
                    verified = await self._verify_answer(agent, ctx.query, final)
                    if not verified["ok"] and step_no < max_steps - 1:
                        ctx.add_message(
                            Message.user(
                                f"Verification found issues: {verified.get('feedback', '')}\n"
                                f"Please fix and try again."
                            )
                        )
                        continue
                return make_result(ctx, final)

            # Parse action
            action = parse_action(text)
            if action is None:
                return make_result(ctx, text.strip())

            name, args = action
            await self._execute_with_reflection(agent, ctx, name, args)

            enforce_consecutive_errors(agent.max_consecutive_errors, ctx)

            # Periodic context compression
            if len(ctx.messages) > self.compress_after_messages:
                compressed = await compressor.compress(ctx.messages, ctx.usage)
                ctx.messages = compressed.messages
                logger.debug(
                    f"[{ctx.trace_id}] ReAct context compressed: "
                    f"{compressed.stats['before_count']} → {compressed.stats['after_count']} msgs"
                )

        raise MaxStepsExceededError(
            f"EnhancedReAct did not converge within max_steps={max_steps}"
        )

    # -- reflection / retry logic ----------------------------------------------

    async def _execute_with_reflection(
        self, agent: "Agent", ctx: Context, tool_name: str, args: dict[str, Any]
    ) -> None:
        """Execute a tool with failure reflection and retry.

        On failure, the model is prompted to reflect on why it failed and how
        to correct the call, then retries with the corrected approach.
        """
        for attempt in range(1 + self.max_retry_per_action):
            # Check cache first
            if self.tool_cache and attempt == 0:
                cached = self.tool_cache.get(tool_name, args)
                if cached is not None:
                    cached_result, cached_error = cached
                    if cached_error is None:
                        step = Step(
                            index=len(ctx.steps),
                            description=f"{tool_name} [cached]",
                            status=StepStatus.SUCCESS,
                            output=cached_result,
                        )
                        ctx.add_step(step)
                        ctx.add_message(
                            Message.tool(
                                content=stringify(cached_result),
                                tool_call_id=f"cached_{tool_name}",
                            )
                        )
                        await agent.hooks.tool_end(ctx, step)
                        logger.debug(f"[{ctx.trace_id}] ReAct cached hit: {tool_name}")
                        return

            result, error = await execute_tool(
                agent.registry, tool_name, args, agent.approve_tool
            )

            step = Step(
                index=len(ctx.steps),
                description=tool_name,
                status=StepStatus.SUCCESS if error is None else StepStatus.FAILED,
                output=result,
                error=error,
            )
            ctx.add_step(step)
            await agent.hooks.tool_end(ctx, step)

            if error is None:
                # Cache successful result
                if self.tool_cache:
                    self.tool_cache.set(tool_name, args, result=result)
                observation = stringify(result)
                ctx.add_message(Message.user(f"Observation: {observation}"))
                return

            # Reflection on failure
            if attempt < self.max_retry_per_action:
                logger.warning(
                    f"[{ctx.trace_id}] ReAct tool '{tool_name}' failed (attempt {attempt + 1}): {error}"
                )
                reflection = await self._reflect_on_failure(
                    agent, tool_name, args, error, ctx
                )
                ctx.add_message(
                    Message.user(
                        f"The tool '{tool_name}' failed: {error}\n\n"
                        f"Reflection: {reflection}\n\n"
                        f"Please correct your approach and try a different way "
                        f"to achieve the same goal. You may need to use a different "
                        f"tool or different arguments."
                    )
                )
            else:
                # All retries exhausted
                ctx.add_message(
                    Message.user(
                        f"Observation: ERROR: {error}\n"
                        f"(Tool '{tool_name}' failed after {1 + self.max_retry_per_action} attempts. "
                        f"Please work around this and continue with other approaches.)"
                    )
                )
                if self.tool_cache:
                    self.tool_cache.set(tool_name, args, result=None, error=error)

    async def _reflect_on_failure(
        self,
        agent: "Agent",
        tool_name: str,
        args: dict[str, Any],
        error: str,
        ctx: Context,
    ) -> str:
        """Ask the LLM to analyze why a tool call failed and suggest a fix."""
        recent = ctx.messages[-4:]
        context_text = "\n".join(
            f"[{m.role.value}] {m.content or ''}" for m in recent
        )
        prompt = (
            f"A tool call failed. Analyze why and suggest a fix.\n\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {args}\n"
            f"Error: {error}\n\n"
            f"Recent context:\n{context_text}\n\n"
            f"Brief reflection (1-2 sentences):"
        )
        try:
            response = await agent.provider.chat([Message.user(prompt)])
            return (response.message.content or "Try a different approach.").strip()
        except Exception:
            return "Try a different approach or tool."

    # -- task analysis ---------------------------------------------------------

    async def _task_analysis(self, agent: "Agent", ctx: Context) -> None:
        """Phase 1: Analyze task complexity and optionally decompose."""
        prompt = (
            f"Briefly analyze this task (1-2 sentences): what does the user need, "
            f"and what approach should you take? Consider whether the task can be "
            f"broken into subtasks.\n\nTask: {ctx.query}"
        )
        try:
            response = await agent.provider.chat(ctx.messages + [Message.user(prompt)])
            analysis = response.message.content or ""
            ctx.add_message(
                Message.user(
                    f"[Task Analysis]\n{analysis.strip()}\n\n"
                    f"Now proceed with the task. Remember to verify your answer "
                    f"before giving it."
                )
            )
            logger.debug(f"[{ctx.trace_id}] task analysis: {analysis[:100]}")
        except Exception:
            pass  # Non-critical; proceed without analysis

    # -- self-verification -----------------------------------------------------

    async def _verify_answer(
        self, agent: "Agent", query: str, draft_answer: str
    ) -> dict[str, Any]:
        """Phase 3: Verify the answer quality before returning."""
        # Fast check: if the answer is very short, it might be incomplete
        if len(draft_answer.strip()) < 10:
            return {"ok": False, "feedback": "Answer is too short."}

        prompt = _VERIFY_PROMPT.format(query=query, draft=draft_answer)
        try:
            resp = await agent.provider.chat([Message.user(prompt)])
            text = (resp.message.content or "").strip()
            is_ok = text.upper().startswith("OK") or text.lower().startswith("ok")
            logger.debug(f"self-verify: {'PASS' if is_ok else 'FAIL'} ({text[:80]})")
            return {"ok": is_ok, "feedback": text if not is_ok else ""}
        except Exception:
            return {"ok": True, "feedback": ""}
