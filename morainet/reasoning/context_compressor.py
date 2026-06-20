"""Intelligent context compression: distinguishes key facts from redundant dialogue.

Provides progressive compression strategies — full → summarized → key-facts-only —
with dynamic token budget control to prevent context overflow during long-running tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from morainet.core.models import Message, Role, Usage
from morainet.observability.tracing import logger

if TYPE_CHECKING:
    from morainet.providers.base import Provider


@dataclass
class CompressionResult:
    """Output of a context compression pass."""

    messages: list[Message]
    """Compressed message list ready for the next LLM call."""

    stats: dict[str, Any] = field(default_factory=dict)
    """Metadata about what was kept / discarded / summarized."""


class ContextCompressor:
    """Compresses conversation context by identifying and retaining key facts while
    collapsing redundant or low-signal dialogue.

    Three tiers of compression (applied progressively as budget tightens):

    1. **summarize** — old turns are LLM-summarized into a single system message
    2. **key_facts** — only tool results, decisions, and task-relevant facts survive
    3. **truncate** — last-resort hard cut, keeping the system prompt + latest N turns
    """

    def __init__(
        self,
        provider: Provider | None = None,
        keep_recent: int = 4,
        token_budget: int | None = None,
        model_max_tokens: int = 128_000,
        safe_margin: float = 0.90,
    ) -> None:
        self.provider = provider
        self.keep_recent = keep_recent
        self._budget = token_budget
        self.model_max_tokens = model_max_tokens
        self.safe_margin = safe_margin
        self._key_facts: list[str] = []
        self._summary: str | None = None

    # -- public API ------------------------------------------------------------

    @property
    def token_budget(self) -> int:
        if self._budget is not None:
            return self._budget
        return int(self.model_max_tokens * self.safe_margin)

    def set_token_budget(self, budget: int) -> None:
        self._budget = budget

    @property
    def summary(self) -> str | None:
        return self._summary

    @property
    def key_facts(self) -> list[str]:
        return list(self._key_facts)

    async def compress(
        self,
        messages: list[Message],
        usage: Usage | None = None,
    ) -> CompressionResult:
        """Compress messages to fit within the current token budget.

        Returns the compressed message list and statistics about what was retained.
        """
        budget = self.token_budget
        stats: dict[str, Any] = {
            "before_count": len(messages),
            "before_tokens": _estimate_list(messages),
            "budget": budget,
        }

        # Fast path: already within budget
        if stats["before_tokens"] <= budget:
            stats["after_count"] = len(messages)
            stats["after_tokens"] = stats["before_tokens"]
            stats["strategy"] = "none"
            return CompressionResult(messages=messages, stats=stats)

        # Separate system prompt from conversation turns
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        conversation = [m for m in messages if m.role != Role.SYSTEM]

        # Tier 1: Summarize old turns via LLM
        if self.provider is not None and len(conversation) > self.keep_recent * 2:
            result = await self._summarize_compress(system_msgs, conversation, budget)
            if result is not None:
                stats.update(result.stats)
                return result

        # Tier 2: Key-fact extraction (keep only high-signal messages)
        result = await self._key_fact_compress(system_msgs, conversation, budget)
        if result is not None:
            stats.update(result.stats)
            return result

        # Tier 3: Hard truncation — keep system + latest N turns
        compact = system_msgs + conversation[-self.keep_recent :]
        stats["after_count"] = len(compact)
        stats["after_tokens"] = _estimate_list(compact)
        stats["strategy"] = "truncate"
        logger.debug(
            f"Context compressor: truncate {stats['before_count']} → {stats['after_count']} messages "
            f"({stats['before_tokens']} → {stats['after_tokens']} tokens)"
        )
        return CompressionResult(messages=compact, stats=stats)

    def extract_key_facts(self, messages: list[Message]) -> list[Message]:
        """Extract high-signal messages: tool results, decisions, error corrections.

        Low-signal messages (greetings, acknowledgments, redundant rephrasing)
        are identified and can be dropped when budget is tight.
        """
        facts: list[Message] = []
        for m in messages:
            if m.role == Role.TOOL:
                facts.append(m)
                continue
            if m.role == Role.ASSISTANT and (m.tool_calls or _has_decision_signal(m.content)):
                facts.append(m)
            elif m.role == Role.USER:
                # Keep user messages that contain substantive information
                facts.append(m)
        return facts

    async def build_fact_index(self, messages: list[Message]) -> list[str]:
        """Use the LLM to build a structured fact index from conversation history.

        Only possible when a provider is available; otherwise returns raw extraction.
        """
        if self.provider is None or not messages:
            return []

        history = _format_history(messages)
        prompt = (
            "Extract the key facts, decisions, and important context from this conversation. "
            "List each fact as a bullet point. Exclude greetings, small talk, and redundant "
            "rephrasing. Focus on what would be needed to resume the task later.\n\n"
            f"Conversation:\n{history}\n\nKey facts:"
        )
        try:
            resp = await self.provider.chat([Message.user(prompt)])
            self._key_facts = [
                line.strip("- •\n\r ").strip()
                for line in (resp.message.content or "").splitlines()
                if line.strip("- •\n\r ") and len(line.strip("- •\n\r ")) > 3
            ]
        except Exception:
            pass  # Non-critical; fall back to absent index
        return self._key_facts

    # -- internals ------------------------------------------------------------

    async def _summarize_compress(
        self, system_msgs: list[Message], conversation: list[Message], budget: int
    ) -> CompressionResult | None:
        """Tier 1: LLM-summarize old turns, keep recent ones verbatim."""
        if self.provider is None:
            return None

        old = conversation[: -self.keep_recent]
        recent = conversation[-self.keep_recent :]
        if not old:
            return None

        history_text = _format_history(old)
        prompt = (
            "Summarize the following conversation concisely. Focus on:\n"
            "- Key decisions made\n"
            "- Tool results and findings\n"
            "- Task progress and remaining steps\n"
            "- Important context that must not be lost\n\n"
            "Ignore: greetings, small talk, redundant rephrasing, filler.\n\n"
            f"Conversation:\n{history_text}"
        )
        existing_summary = f"Previous summary: {self._summary}\n\n" if self._summary else ""
        try:
            resp = await self.provider.chat([Message.user(existing_summary + prompt)])
            self._summary = resp.message.content or self._summary
        except Exception:
            return None  # LLM unavailable; fall through to next tier

        summary_msg = Message.system(f"[Compressed context] {self._summary}")
        compact = system_msgs + [summary_msg] + recent
        tokens = _estimate_list(compact)
        if tokens <= budget:
            logger.debug(
                f"Context compressor: summarize {len(conversation)} → {len(compact)} messages "
                f"({tokens} tokens, budget={budget})"
            )
            return CompressionResult(
                messages=compact,
                stats={
                    "strategy": "summarize",
                    "after_count": len(compact),
                    "after_tokens": tokens,
                },
            )
        return None  # Still over budget; fall through

    async def _key_fact_compress(
        self, system_msgs: list[Message], conversation: list[Message], budget: int
    ) -> CompressionResult | None:
        """Tier 2: Keep only high-signal messages — tool results, decisions, user facts."""
        facts = self.extract_key_facts(conversation)
        compact = system_msgs + facts
        tokens = _estimate_list(compact)
        if tokens <= budget:
            logger.debug(
                f"Context compressor: key-fact {len(conversation)} → {len(compact)} messages "
                f"({tokens} tokens, budget={budget})"
            )
            return CompressionResult(
                messages=compact,
                stats={
                    "strategy": "key_facts",
                    "after_count": len(compact),
                    "after_tokens": tokens,
                },
            )
        return None  # Still over budget


# -- helpers ------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (conservative)."""
    return max(1, len(text) // 3)


def _estimate_list(messages: list[Message]) -> int:
    return sum(_estimate_tokens(m.content or "") for m in messages) + len(messages) * 2


def _format_history(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.role.value
        if m.tool_calls:
            calls = ", ".join(f"{tc.name}(…)" for tc in m.tool_calls)
            lines.append(f"[{role}] {m.content or ''} [calls: {calls}]")
        elif m.tool_call_id:
            lines.append(f"[{role}] call_id={m.tool_call_id}: {m.content or ''}")
        else:
            lines.append(f"[{role}] {m.content or ''}")
    return "\n".join(lines)


def _has_decision_signal(content: str | None) -> bool:
    """Check if the assistant message contains decision-making language."""
    if not content:
        return False
    signals = [
        "decided",
        "决定",
        "最终",
        "conclusion",
        "结论",
        "therefore",
        "因此",
        "result",
        "结果",
        "found",
        "发现",
        "completed",
        "完成",
        "error",
        "错误",
        "need to",
        "需要",
    ]
    low = content.lower()
    return any(s in low for s in signals)
