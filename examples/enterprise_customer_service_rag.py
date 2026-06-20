"""企业级实战：智能客服 RAG 系统。

演示一个完整的智能客服系统，包含：
- 知识库 RAG 检索（LongMemory + 语义检索）
- 多轮对话上下文（ShortMemory）
- 意图识别与分流（Router 分诊到售前/售后/技术）
- 工具调用（查订单、创建工单、查询知识库）
- 人工升级兜底（危险操作审批）
- 会话摘要与满意度收集

离线可跑：检索是真实的，模型用 MockProvider 脚本化；
也可切到本地 Ollama 真模型。

Run:
    python examples/enterprise_customer_service_rag.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/enterprise_customer_service_rag.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from morainet import Agent, Router, Route, ShortMemory, tool
from morainet.core.models import ChatResponse, Message, Role, ToolCall, Usage
from morainet.memory import InMemoryVectorStore, LongMemory
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider

# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    # 售前
    "产品 A 标准版 ¥999/月，含 10 用户、100GB 存储、邮件支持。",
    "产品 A 企业版 ¥2999/月，含无限用户、1TB 存储、7×24 专属支持。",
    "产品 B 基础版 ¥199/月，轻量级项目管理工具，适合 5 人以下团队。",
    "新用户注册即享 14 天免费试用，无需绑定信用卡。",
    # 售后
    "退款政策：标准版购买 7 天内可无理由退款；企业版购买 30 天内可退款。",
    "发票申请：登录控制台 → 费用中心 → 发票管理 → 填写企业信息，3 个工作日内开具电子发票。",
    "续费：到期前 7 天会发送邮件提醒，支持自动续费（可随时关闭）。",
    # 技术
    "API 调用报 429：触发频率限制，默认 100 次/分钟。可在控制台 → API 管理中调整配额。",
    "SSO 配置：支持 SAML 2.0 和 OIDC。进入管理后台 → 安全 → 单点登录，按指引填写 IdP 信息。",
    "数据导出：控制台 → 数据中心 → 导出，支持 CSV/JSON 格式，单次最多导出 10 万条。",
]

# 模拟订单数据库
ORDERS_DB: dict[str, dict] = {
    "ORD-2024-001": {
        "status": "已发货",
        "product": "产品 A 企业版",
        "amount": 2999,
        "created": "2024-01-15",
        "tracking": "SF1234567890",
    },
    "ORD-2024-002": {
        "status": "待支付",
        "product": "产品 A 标准版",
        "amount": 999,
        "created": "2024-01-20",
        "tracking": "",
    },
}

# 模拟工单系统
TICKETS: list[dict] = []


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


@tool
def query_order(order_id: str) -> str:
    """查询订单信息。

    Args:
        order_id: 订单号，格式如 ORD-2024-001
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return f"未找到订单 {order_id}。请确认订单号是否正确。"
    return (
        f"订单 {order_id}：{order['product']}，金额 ¥{order['amount']}，"
        f"状态 {order['status']}，创建于 {order['created']}"
        + (f"，物流单号 {order['tracking']}" if order["tracking"] else "")
    )


@tool
def create_ticket(category: str, title: str, description: str) -> str:
    """创建工单，转交人工处理。用于无法自动解决的复杂问题。

    Args:
        category: 工单分类，如 技术/售后/投诉
        title: 工单标题
        description: 详细描述
    """
    ticket = {
        "id": f"TK-{len(TICKETS) + 1:04d}",
        "category": category,
        "title": title,
        "description": description,
        "status": "待处理",
        "created": datetime.now(timezone.utc).isoformat(),
    }
    TICKETS.append(ticket)
    return f"工单 {ticket['id']} 已创建（{category}），预计 2 小时内回复。"


@tool(dangerous=True)
def refund_order(order_id: str) -> str:
    """执行订单退款操作（危险：涉及资金变动，需审批）。

    Args:
        order_id: 要退款的订单号
    """
    order = ORDERS_DB.get(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    if order["status"] == "已退款":
        return f"订单 {order_id} 已经退款过。"
    order["status"] = "已退款"
    return f"订单 {order_id}（{order['product']}）已退款 ¥{order['amount']}，3-5 个工作日到账。"


# ---------------------------------------------------------------------------
# Provider 构建
# ---------------------------------------------------------------------------


def _build_rag_provider() -> Provider:
    """构建 RAG 客服 Provider。"""
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    def handler(messages: list[Message], tools: object) -> ChatResponse:
        user_msg = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        memory_docs = [
            (m.content or "").replace("[memory] ", "")
            for m in messages
            if m.role is Role.SYSTEM and "[memory]" in (m.content or "")
        ]

        if user_msg and memory_docs:
            relevant = memory_docs[0][:100]
            return ChatResponse(
                message=Message.assistant(content=f"根据知识库「{relevant}...」，为您解答。")
            )
        return ChatResponse(
            message=Message.assistant(
                content="您好！我是智能客服小莫。请问有什么可以帮您？\n"
                "- 产品咨询（价格/功能/试用）\n- 订单查询\n"
                "- 退款/发票\n- 技术问题"
            )
        )

    return MockProvider(handler=handler)


def _build_router_provider() -> Provider:
    """构建分流 Router 的 Provider。"""
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)

    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="1", name="query_order", arguments={"order_id": "ORD-2024-001"})
                    ]
                ),
                usage=Usage(total_tokens=10),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(
                    content="您的订单 ORD-2024-001 已发货，物流单号 SF1234567890。预计 2-3 天送达。"
                )
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 客服子 Agent
# ---------------------------------------------------------------------------


@dataclass
class CustomerServiceSystem:
    """完整的智能客服系统。"""

    rag_agent: Agent
    router: Router
    chat_agent: Agent

    @classmethod
    async def create(cls, use_live_model: bool = False) -> "CustomerServiceSystem":
        """初始化客服系统。"""
        embedder = None
        if use_live_model:
            from morainet.memory import OllamaEmbedder

            embedder = OllamaEmbedder(os.getenv("MORAINET_EMBED_MODEL", "nomic-embed-text"))

        # 长期记忆：知识库
        long_mem = LongMemory(
            store=InMemoryVectorStore(), embedder=embedder, score_threshold=0.1
        )
        for doc in KNOWLEDGE_BASE:
            await long_mem.add(Message.assistant(content=doc))

        # 短期记忆：会话上下文
        short_mem = ShortMemory()

        rag_agent = Agent(
            provider=_build_rag_provider(),
            memory=long_mem,
            system_prompt=(
                "你是企业智能客服。根据知识库资料回答用户问题。"
                "语当简洁专业、态度友好。不确定时请建议转人工。"
            ),
        )

        # 售前 Agent
        sales_agent = Agent(
            provider=_build_rag_provider(),
            memory=long_mem,
            system_prompt="你是售前顾问。负责产品介绍、价格咨询、试用申请。态度热情专业。",
        )

        # 售后 Agent
        support_agent = Agent(
            provider=_build_router_provider(),
            tools=[query_order, create_ticket, refund_order],
            memory=short_mem,
            system_prompt=(
                "你是售后专员。处理订单查询、退款、发票、投诉。"
                "退款操作需确认用户身份后执行。复杂投诉请创建工单转人工。"
            ),
            approve_tool=lambda name, args: input(f"⚠ 危险操作 [{name}]{args} — 是否执行? [y/N] ") == "y",
        )

        # 技术 Agent
        tech_agent = Agent(
            provider=_build_rag_provider(),
            memory=long_mem,
            system_prompt=(
                "你是技术支持工程师。负责 API 配置、SSO、数据导出等技术问题。"
                "提供分步骤的操作指引。无法解决时创建技术工单。"
            ),
            tools=[create_ticket],
        )

        # Router 分诊
        router = Router(
            [
                Route("sales", sales_agent, "产品咨询、价格、功能、试用"),
                Route("support", support_agent, "订单、退款、发票、投诉、续费"),
                Route("tech", tech_agent, "API、SSO、配置、数据导出、技术故障"),
            ],
            selector=lambda q: _classify_intent(q),
        )

        chat_agent = Agent(
            provider=_build_rag_provider(),
            memory=short_mem,
            system_prompt="你是小莫，友好的客服助手。用中文回复。",
        )

        return cls(rag_agent=rag_agent, router=router, chat_agent=chat_agent)

    async def handle(self, query: str) -> str:
        """处理一条用户消息。"""
        route = await self.router.arun(query)
        return f"[{route.route}] {route.final}"


def _classify_intent(query: str) -> str:
    """简单的意图分类。"""
    sales_keywords = ["价格", "多少钱", "试用", "功能", "介绍", "对比"]
    support_keywords = ["订单", "退款", "发票", "续费", "投诉", "支付"]
    tech_keywords = ["API", "SSO", "配置", "导出", "报错", "技术", "接入"]

    for kw in sales_keywords:
        if kw in query:
            return "sales"
    for kw in support_keywords:
        if kw in query:
            return "support"
    for kw in tech_keywords:
        if kw in query:
            return "tech"
    return "sales"  # 默认售前


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    use_live = bool(os.getenv("MORAINET_OLLAMA_MODEL"))
    print("=" * 60)
    print("企业智能客服 RAG 系统")
    print("=" * 60)

    system = await CustomerServiceSystem.create(use_live_model=use_live)

    # 场景 1：产品咨询 → 售前
    print("\n--- 场景 1：产品咨询 ---")
    result = await system.handle("产品 A 标准版多少钱？有什么功能？")
    print(result)

    # 场景 2：订单查询 → 售后
    print("\n--- 场景 2：订单查询 ---")
    result = await system.handle("帮我查一下订单 ORD-2024-001 到哪了？")
    print(result)

    # 场景 3：RAG 知识检索
    print("\n--- 场景 3：知识库检索 ---")
    ra = await system.rag_agent.arun("怎么申请退款？")
    print(f"RAG 回答: {ra.final_answer}")

    # 场景 4：技术问题
    print("\n--- 场景 4：技术问题 ---")
    result = await system.handle("API 调用返回 429 错误怎么办？")
    print(result)

    # 场景 5：多轮对话
    print("\n--- 场景 5：多轮对话 ---")
    q1 = await system.chat_agent.arun("我叫张三")
    q2 = await system.chat_agent.arun("我刚才说我叫什么？")
    print(f"Q1: {q1.final_answer}")
    print(f"Q2: {q2.final_answer}")

    print("\n" + "=" * 60)
    print("客服系统运行完毕")
    if TICKETS:
        print(f"本次会话创建工单: {len(TICKETS)} 个")
        for t in TICKETS:
            print(f"  {t['id']} [{t['category']}] {t['title']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
