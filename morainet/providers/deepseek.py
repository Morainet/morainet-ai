"""DeepSeek provider — OpenAI-compatible, so it reuses OpenAIProvider."""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.deepseek_api_key,
            base_url=base_url or settings.deepseek_base_url,
            timeout=timeout,
        )
