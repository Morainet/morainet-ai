"""Memory that compresses old turns into a running summary via the LLM."""

from __future__ import annotations

from morainet.core.models import Message
from morainet.memory.base import Memory
from morainet.prompts.registry import BUILTIN_TEMPLATES, PromptTemplate
from morainet.providers.base import Provider


class SummarizingMemory(Memory):
    """Keeps recent turns verbatim; folds older turns into a running summary.

    When the buffer exceeds ``trigger_messages``, everything except the last
    ``keep_recent`` messages is summarized (via ``provider`` + the ``summarizer``
    prompt) and merged into a single summary, bounding context growth.
    """

    def __init__(
        self,
        provider: Provider,
        keep_recent: int = 6,
        trigger_messages: int = 12,
        prompt: PromptTemplate | None = None,
    ) -> None:
        self.provider = provider
        self.keep_recent = keep_recent
        self.trigger_messages = trigger_messages
        self.prompt = prompt or BUILTIN_TEMPLATES["summarizer"]
        self._summary: str | None = None
        self._messages: list[Message] = []

    async def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.trigger_messages:
            await self._compress()

    async def _compress(self) -> None:
        old = self._messages[: -self.keep_recent] if self.keep_recent else self._messages
        recent = self._messages[-self.keep_recent :] if self.keep_recent else []
        if not old:
            return

        history = "\n".join(f"{m.role.value}: {m.content or ''}" for m in old)
        if self._summary:
            history = f"已有摘要：{self._summary}\n\n新对话：\n{history}"

        response = await self.provider.chat([Message.user(self.prompt.render(history=history))])
        self._summary = response.message.content or self._summary
        self._messages = recent

    async def get_context(self, query: str, limit: int = 10) -> list[Message]:
        context: list[Message] = []
        if self._summary:
            context.append(Message.system(f"[对话摘要] {self._summary}"))
        context.extend(self._messages[-limit:])
        return context

    @property
    def summary(self) -> str | None:
        return self._summary

    def __len__(self) -> int:
        return len(self._messages)
