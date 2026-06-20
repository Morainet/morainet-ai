"""Tongyi Qwen (DashScope) provider — OpenAI-compatible.

DashScope provides an OpenAI-compatible endpoint for Qwen models.
API docs: https://help.aliyun.com/zh/model-studio/
"""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class QwenProvider(OpenAIProvider):
    """通义千问 (Qwen) via DashScope OpenAI-compatible API.

    Requires ``MORAINET_QWEN_API_KEY`` env var or dashscope API key.
    Default endpoint: ``https://dashscope.aliyuncs.com/compatible-mode/v1``
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.qwen_api_key,
            base_url=base_url or settings.qwen_base_url,
            timeout=timeout,
        )
