"""MiniMax provider — OpenAI-compatible.

MiniMax's API is OpenAI-compatible.
API docs: https://platform.minimaxi.com/document
"""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class MiniMaxProvider(OpenAIProvider):
    """MiniMax (abab / MiniMax-Text) via OpenAI-compatible API.

    Requires ``MORAINET_MINIMAX_API_KEY`` env var.
    Default endpoint: ``https://api.minimax.chat/v1``
    """

    def __init__(
        self,
        model: str = "MiniMax-Text-01",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.minimax_api_key,
            base_url=base_url or settings.minimax_base_url,
            timeout=timeout,
        )
