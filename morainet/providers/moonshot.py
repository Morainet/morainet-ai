"""Moonshot AI (Kimi) provider — OpenAI-compatible.

Moonshot's API is fully OpenAI-compatible.
API docs: https://platform.moonshot.cn/docs
"""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class MoonshotProvider(OpenAIProvider):
    """月之暗面 Moonshot (Kimi) via OpenAI-compatible API.

    Requires ``MORAINET_MOONSHOT_API_KEY`` env var.
    Default endpoint: ``https://api.moonshot.cn/v1``
    """

    def __init__(
        self,
        model: str = "moonshot-v1-8k",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.moonshot_api_key,
            base_url=base_url or settings.moonshot_base_url,
            timeout=timeout,
        )
