"""ReAct strategy: text-based Reason + Act loop.

Works with models that lack native tool calling. The model emits a
Thought / Action / Action Input trace; we parse it, run the tool, feed back an
Observation, and repeat until it produces a Final Answer.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from morainet.core.context import Context
from morainet.core.models import AgentResult, Message, Step, StepStatus
from morainet.exceptions import MaxStepsExceededError
from morainet.reasoning.base import (
    ReasoningStrategy,
    enforce_budget,
    enforce_consecutive_errors,
    execute_tool,
    make_result,
    stringify,
)

if TYPE_CHECKING:
    from morainet.core.agent import Agent

_FINAL_RE = re.compile(r"Final Answer:\s*(?P<answer>.+)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(?P<name>[\w.-]+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(?P<input>.+?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL)

_PROMPT = """\
You are an agent that solves tasks step by step using the ReAct format.

You have access to the following tools:
{tools}

Use EXACTLY this format:
Thought: <your reasoning>
Action: <one tool name from the list above>
Action Input: <a JSON object of arguments>

After each Action you will be given an Observation. Continue with more
Thought/Action steps as needed. When you can answer, respond with:
Thought: <reasoning>
Final Answer: <the answer to the user>"""


def parse_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    return m.group("answer").strip() if m else None


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract ``(tool_name, arguments)`` from a ReAct step, or None."""
    name_m = _ACTION_RE.search(text)
    if not name_m:
        return None
    name = name_m.group("name").strip()

    args: dict[str, Any] = {}
    input_m = _INPUT_RE.search(text)
    if input_m:
        raw = input_m.group("input").strip().strip("`").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            args = {}
    return name, args


def _render_tools(agent: "Agent") -> str:
    lines = []
    for schema in agent.registry.schemas():
        params = ", ".join(schema["parameters"]["properties"].keys())
        lines.append(f"- {schema['name']}({params}): {schema['description']}")
    return "\n".join(lines) or "(no tools available)"


class ReActStrategy(ReasoningStrategy):
    def __init__(self, max_steps: int | None = None) -> None:
        self.max_steps = max_steps

    def _system_prompt(self, agent: "Agent") -> str:
        if "react_system" in agent.prompts._templates:  # allow user override
            return agent.prompts.get("react_system").render(tools=_render_tools(agent))
        return _PROMPT.format(tools=_render_tools(agent))

    async def run(self, agent: "Agent", ctx: Context) -> AgentResult:
        ctx.messages.insert(0, Message.system(self._system_prompt(agent)))
        max_steps = self.max_steps if self.max_steps is not None else agent.max_steps

        for _ in range(max_steps):
            response = await agent.provider.chat(ctx.messages)
            ctx.add_usage(response.usage)
            text = response.message.content or ""
            ctx.add_message(Message.assistant(content=text))
            await agent.hooks.llm_end(ctx, response)
            enforce_budget(agent.token_budget, ctx)

            final = parse_final_answer(text)
            if final is not None:
                return make_result(ctx, final)

            action = parse_action(text)
            if action is None:
                # Model answered without the ReAct scaffold; take its text as-is.
                return make_result(ctx, text.strip())

            name, args = action
            result, error = await execute_tool(agent.registry, name, args, agent.approve_tool)
            step = Step(
                index=len(ctx.steps),
                description=name,
                status=StepStatus.SUCCESS if error is None else StepStatus.FAILED,
                output=result,
                error=error,
            )
            ctx.add_step(step)
            await agent.hooks.tool_end(ctx, step)
            enforce_consecutive_errors(agent.max_consecutive_errors, ctx)
            observation = stringify(result) if error is None else f"ERROR: {error}"
            ctx.add_message(Message.user(f"Observation: {observation}"))

        raise MaxStepsExceededError(
            f"ReAct did not converge within max_steps={max_steps}"
        )
