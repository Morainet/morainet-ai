"""Prompt templates and registry.

Templates are versioned, overridable, and rendered via structured substitution
(no f-string/eval), so user-supplied values can't inject template syntax.
"""

from __future__ import annotations

import string

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    name: str
    version: str = "v1"
    template: str

    def variables(self) -> set[str]:
        """Placeholder names referenced by the template."""
        return {field for _, field, _, _ in string.Formatter().parse(self.template) if field}

    def render(self, **kwargs: object) -> str:
        missing = self.variables() - set(kwargs)
        if missing:
            raise ValueError(
                f"Prompt '{self.name}' missing variables: {sorted(missing)}"
            )
        # format_map substitutes after parsing, so braces inside values are literal.
        return self.template.format_map(kwargs)


BUILTIN_TEMPLATES: dict[str, PromptTemplate] = {
    "planner": PromptTemplate(
        name="planner",
        template=(
            "You are a planner. Break the user's goal into ordered, concrete steps.\n"
            "Available tools:\n{tools}\n\nGoal: {query}"
        ),
    ),
    "executor": PromptTemplate(
        name="executor",
        template=(
            "Execute the current step. Call a tool if needed, otherwise answer directly.\n"
            "Step: {step}"
        ),
    ),
    "reflector": PromptTemplate(
        name="reflector",
        template=(
            "Review progress so far and decide whether the goal is satisfied.\n"
            "Reply 'finish' with the final answer, or 'continue' with a reason.\n"
            "Progress:\n{progress}"
        ),
    ),
    "summarizer": PromptTemplate(
        name="summarizer",
        template="Summarize the following conversation concisely:\n{history}",
    ),
    "episode_compressor": PromptTemplate(
        name="episode_compressor",
        template=(
            "你是一个记忆压缩助手。将以下对话压缩成一段简洁的摘要，"
            "保留关键决策、用户偏好、重要事实和未完成的任务。\n\n"
            "{prior}\n\n需要压缩的新对话：\n{history}\n\n摘要："
        ),
    ),
    "fact_extractor": PromptTemplate(
        name="fact_extractor",
        template=(
            "从以下对话摘要中提取关键事实和决策。每条事实用一行表示，格式：\n"
            "  topic: value\n\n"
            "只提取客观事实、用户信息和重要决定。忽略闲聊和过渡性内容。\n\n"
            "对话摘要：\n{history}\n\n提取的事实："
        ),
    ),
    "preference_detector": PromptTemplate(
        name="preference_detector",
        template=(
            "从以下对话中检测用户偏好。对每个偏好，输出一行：\n"
            "  CATEGORY | key: value\n\n"
            "CATEGORY 可以是: style(回答风格), domain(专业领域), "
            "language(语言), format(格式), identity(身份), other\n\n"
            "对话：\n{history}\n\n检测到的偏好："
        ),
    ),
    "conflict_resolver": PromptTemplate(
        name="conflict_resolver",
        template=(
            "以下知识库中存在矛盾的事实。对于每组矛盾，判断哪个事实更可信，并说明理由。\n\n"
            "矛盾事实：\n{conflicts}\n\n裁决（每行一个裁决）："
        ),
    ),
    "context_compressor": PromptTemplate(
        name="context_compressor",
        template=(
            "You are a context compressor. Extract the key facts, decisions, and "
            "important context from this conversation. List each fact as a bullet point. "
            "Exclude greetings, small talk, and redundant rephrasing. "
            "Focus on what would be needed to resume the task later.\n\n"
            "Conversation:\n{history}\n\nKey facts:"
        ),
    ),
    "failure_reflector": PromptTemplate(
        name="failure_reflector",
        template=(
            "A tool call failed. Analyze why and suggest a fix.\n\n"
            "Tool: {tool_name}\nArguments: {arguments}\nError: {error}\n\n"
            "Recent context:\n{context}\n\nBrief reflection (1-2 sentences):"
        ),
    ),
    "self_verifier": PromptTemplate(
        name="self_verifier",
        template=(
            "Before giving a final answer, verify:\n"
            "1. Does this directly answer the user's original question?\n"
            "2. Are the tool results consistent with the answer?\n"
            "3. Is anything missing or incomplete?\n\n"
            "User's query: {query}\nYour proposed answer: {draft}\n\n"
            "Reply 'OK' if satisfied, or explain what needs fixing."
        ),
    ),
}


class PromptRegistry:
    """Holds built-in prompts plus optional per-Agent overrides."""

    def __init__(self, overrides: dict[str, PromptTemplate | str] | None = None) -> None:
        self._templates: dict[str, PromptTemplate] = dict(BUILTIN_TEMPLATES)
        for name, tpl in (overrides or {}).items():
            self.register(name, tpl)

    def register(self, name: str, template: PromptTemplate | str) -> PromptTemplate:
        tpl = template if isinstance(template, PromptTemplate) else PromptTemplate(
            name=name, template=template
        )
        self._templates[name] = tpl
        return tpl

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        try:
            tpl = self._templates[name]
        except KeyError:
            raise KeyError(f"Prompt '{name}' is not registered") from None
        if version is not None and tpl.version != version:
            raise KeyError(f"Prompt '{name}' version '{version}' not found (have '{tpl.version}')")
        return tpl
