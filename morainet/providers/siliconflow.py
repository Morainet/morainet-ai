"""SiliconFlow provider — OpenAI-compatible, for local/cloud inference.

SiliconFlow provides a serverless inference platform with OpenAI-compatible
APIs, supporting many open-source models (Qwen, DeepSeek, Llama, etc.).
API docs: https://docs.siliconflow.cn/
"""

from __future__ import annotations

from morainet.config import settings
from morainet.providers.openai import OpenAIProvider


class SiliconFlowProvider(OpenAIProvider):
    """SiliconFlow (serverless open-source model inference).

    Requires ``MORAINET_SILICONFLOW_API_KEY`` env var.
    Default endpoint: ``https://api.siliconflow.cn/v1``
    Default model: ``Qwen/Qwen2.5-7B-Instruct``

    Supports many open-source models: Qwen, DeepSeek, Llama, Yi, etc.
    See https://docs.siliconflow.cn/docs/model-names for the full list.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key or settings.siliconflow_api_key,
            base_url=base_url or settings.siliconflow_base_url,
            timeout=timeout,
        )
