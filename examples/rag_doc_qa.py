"""方向：知识 / RAG —— 文档问答。

把知识库写入 LongMemory，Agent 在回答前自动检索相关文档并注入上下文
（这是 Agent + LongMemory 的内置集成）。

离线可跑：**检索是真实的**（HashEmbedder 向量相似度）；模型部分用 MockProvider，
也可切到本地 Ollama 让它真的"读资料作答"。

Run:
    python examples/rag_doc_qa.py
    # 真模型（本地）：先 ollama pull qwen2.5:3b，再
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/rag_doc_qa.py
"""

from __future__ import annotations

import asyncio
import os

from morainet import Agent
from morainet.core.models import ChatResponse, Message, Role
from morainet.memory import InMemoryVectorStore, LongMemory
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider

KNOWLEDGE = [
    "退款政策：商品签收后 7 天内可无理由退款，生鲜类除外。",
    "配送时效：一线城市次日达，其他地区 2-3 个工作日。",
    "会员权益：付费会员享受免运费和专属客服。",
    "营业时间：客服在线时间为周一至周五 9:00-18:00。",
]


def _build_provider() -> Provider:
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    # 离线：从注入的 [memory] 资料中作答，证明检索确实喂给了模型。
    def answer(messages: list[Message], tools: object) -> ChatResponse:
        docs = [
            (m.content or "").replace("[memory] ", "")
            for m in messages
            if m.role is Role.SYSTEM and "[memory]" in (m.content or "")
        ]
        question = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        if docs:
            return ChatResponse(message=Message.assistant(content=f"（根据资料）{docs[0]}"))
        return ChatResponse(message=Message.assistant(content=f"未检索到与「{question}」相关的资料。"))

    return MockProvider(handler=answer)


async def main() -> None:
    # score_threshold 过滤弱匹配。离线 HashEmbedder 是关键词级（共享字符 bigram 才命中）；
    # 想要"语义检索"（问法不同也能召回）请用 OllamaEmbedder("nomic-embed-text")。
    embedder = None
    if os.getenv("MORAINET_OLLAMA_MODEL"):
        from morainet.memory import OllamaEmbedder

        embedder = OllamaEmbedder(os.getenv("MORAINET_EMBED_MODEL", "nomic-embed-text"))

    memory = LongMemory(store=InMemoryVectorStore(), embedder=embedder, score_threshold=0.1)
    for doc in KNOWLEDGE:
        await memory.add(Message.assistant(content=doc))

    agent = Agent(provider=_build_provider(), memory=memory)

    for q in ["怎么申请退款？", "营业时间是几点到几点？", "我想买只恐龙"]:
        result = await agent.arun(q)
        print(f"Q: {q}\nA: {result.final_answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
