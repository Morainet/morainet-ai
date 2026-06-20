"""Zhipu AI (GLM / ChatGLM) provider — OpenAI-compatible.

ZhipuAI's API is fully OpenAI-compatible.
API docs: https://open.bigmodel.cn/dev/api
"""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class ZhipuProvider(OpenAIProvider):
    """智谱 AI (GLM-4 / ChatGLM) via OpenAI-compatible API.

    Requires ``MORAINET_ZHIPU_API_KEY`` env var.
    Default endpoint: ``https://open.bigmodel.cn/api/paas/v4``
    """

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.zhipu_api_key,
            base_url=base_url or settings.zhipu_base_url,
            timeout=timeout,
        )
