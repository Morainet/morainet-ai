"""企业级实战：多部门协同流水线。

演示跨部门协作的 Agent 编排模式：
- 需求评审 → 技术设计 → 编码实现 → 测试 → 上线 的五阶段流水线
- 每个阶段由专属 Agent 负责，不同部门角色协作
- GroupChat：产品/研发/测试 三方讨论需求优先级
- Debate：架构选型辩论（微服务 vs 单体）
- Pipeline + Workflow 混合编排
- Checkpoint 持久化：长时间运转的任务支持断点恢复

离线可跑，MockProvider 脚本化；也可切到本地 Ollama。

Run:
    python examples/enterprise_cross_dept_pipeline.py
    MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/enterprise_cross_dept_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from morainet import (
    Agent,
    Debate,
    GroupChat,
    GroupChatMember,
    Pipeline,
    Stage,
    Workflow,
    tool,
)
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.memory import LongMemory, InMemoryVectorStore, ShortMemory
from morainet.persistence import InMemoryCheckpointStore
from morainet.providers import MockProvider, OllamaProvider
from morainet.providers.base import Provider


# ---------------------------------------------------------------------------
# 部门 Agent 工厂
# ---------------------------------------------------------------------------


def _agent(name: str, answer: str, system_prompt: str = "") -> Agent:
    """快速创建脚本化 Agent。"""
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return Agent(
            provider=OllamaProvider(model=model),
            system_prompt=system_prompt,
        )
    return Agent(
        provider=MockProvider(
            handler=lambda m, t: ChatResponse(message=Message.assistant(content=answer))
        ),
        system_prompt=system_prompt,
    )


def _member(name: str, answer: str, desc: str, sys_prompt: str = "") -> GroupChatMember:
    """快速创建群聊成员。"""
    return GroupChatMember(name=name, agent=_agent(name, answer, sys_prompt), description=desc)


# ---------------------------------------------------------------------------
# 共享知识库（项目上下文）
# ---------------------------------------------------------------------------

PROJECT_DOCS = [
    "项目代号 Phoenix，为电商平台构建智能推荐系统。",
    "技术栈：Python 3.11 + FastAPI + PostgreSQL + Redis + Kafka。",
    "非功能性需求：QPS 1000+，P99 延迟 < 200ms，可用性 99.95%。",
    "已有团队：产品 2 人、后端 3 人、前端 2 人、测试 1 人、SRE 1 人。",
    "排期：2 个月完成 MVP，第 1 个月核心推荐引擎，第 2 个月集成与压测。",
]


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------

TASKS: list[dict] = []
DESIGNS: list[dict] = []


@tool
def create_task(title: str, assignee: str, priority: str = "medium", description: str = "") -> str:
    """创建任务并分配。

    Args:
        title: 任务标题
        assignee: 负责人/团队
        priority: 优先级 low/medium/high/critical
        description: 任务描述
    """
    task = {
        "id": f"TASK-{len(TASKS) + 1:03d}",
        "title": title,
        "assignee": assignee,
        "priority": priority,
        "description": description,
        "status": "todo",
        "created": datetime.now(timezone.utc).isoformat(),
    }
    TASKS.append(task)
    return f"任务 {task['id']} 已创建（{assignee}, {priority}）"


@tool
def save_design(title: str, content: str, author: str) -> str:
    """保存技术设计方案。

    Args:
        title: 设计标题
        content: 设计内容
        author: 作者
    """
    design = {
        "id": f"DES-{len(DESIGNS) + 1:03d}",
        "title": title,
        "content": content,
        "author": author,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    DESIGNS.append(design)
    return f"设计文档 {design['id']} 已保存（{author}）"


@tool
def get_project_status() -> str:
    """获取项目当前状态。"""
    todo = sum(1 for t in TASKS if t["status"] == "todo")
    in_progress = sum(1 for t in TASKS if t["status"] == "in_progress")
    done = sum(1 for t in TASKS if t["status"] == "done")
    return f"任务: {todo} 待办 / {in_progress} 进行中 / {done} 已完成 | 设计: {len(DESIGNS)} 篇"


# ---------------------------------------------------------------------------
# 场景 1：五阶段交付流水线
# ---------------------------------------------------------------------------


def build_delivery_pipeline() -> Pipeline:
    """构建需求 → 设计 → 编码 → 测试 → 上线 流水线。"""
    pm = _agent(
        "pm",
        "需求：推荐引擎需支持协同过滤 + 内容推荐双策略，ABTest 框架，日处理 1000 万条行为数据。"
        "MVP 范围：基于物品的协同过滤、实时行为采集、离线召回 + 在线排序。",
        "你是产品经理，负责需求分析和 PRD 编写。",
    )
    architect = _agent(
        "architect",
        "技术设计：微服务架构，推荐服务独立部署。Kafka 消费行为日志 → Flink 实时特征工程 "
        "→ TF Serving 模型推理 → Redis 缓存召回结果。API Gateway 统一接入。",
        "你是架构师，负责系统设计与技术选型。",
    )
    developer = _agent(
        "developer",
        "编码计划：Week1 搭建项目骨架+数据库设计，Week2 协同过滤核心逻辑，"
        "Week3 实时特征管道，Week4 在线服务+API。使用 faiss 向量检索。",
        "你是后端开发工程师。",
    )
    qa = _agent(
        "qa",
        "测试计划：单元测试覆盖率 > 80%，集成测试覆盖核心链路，"
        "压测验证 QPS 1000+ P99 < 200ms，混沌工程验证容错。",
        "你是测试工程师。",
    )
    sre = _agent(
        "sre",
        "上线计划：K8s 部署，HPA 自动伸缩，Prometheus + Grafana 监控，"
        "灰度发布 10%→50%→100%，告警阈值 P99 > 500ms trigger 回滚。",
        "你是 SRE 运维工程师。",
    )

    return Pipeline(
        [
            Stage("requirements", pm, instruction="分析并输出需求：{query}"),
            Stage("design", architect, instruction="基于需求「{requirements}」做技术设计：{query}"),
            Stage(
                "coding",
                developer,
                instruction="基于需求「{requirements}」和设计「{design}」制定编码计划：{query}",
            ),
            Stage("testing", qa, instruction="基于需求和技术栈制定测试计划：{query}"),
            Stage("deploy", sre, instruction="制定上线与运维方案：{query}"),
        ]
    )


# ---------------------------------------------------------------------------
# 场景 2：三方群聊讨论
# ---------------------------------------------------------------------------


def build_requirement_discussion() -> GroupChat:
    """产品/研发/测试三方讨论需求优先级。"""
    return GroupChat(
        members=[
            _member(
                "pm",
                "从业务价值看，协同过滤对用户体验提升最大，建议 P0。"
                "内容推荐可以 P1，ABTest 框架 P2。",
                "产品经理",
                "你是产品经理，从用户价值和商业目标角度评估需求优先级。",
            ),
            _member(
                "engineer",
                "协同过滤实现复杂度中等（矩阵计算+相似度），预估 3 周。"
                "内容推荐依赖 NLP 模型，需要额外的 GPU 资源，目前没有。"
                "可以先做协同过滤，内容推荐用简单的标签匹配代替。TERMINATE",
                "后端工程师",
                "你是后端工程师，从技术可行性和工期角度评估。",
            ),
            _member(
                "qa",
                "协同过滤需要验证推荐准确率和覆盖率，测试数据至少需要 10 万条行为日志。"
                "建议先造一批模拟数据跑通测试框架。"
                "同意 PM 的优先级，协同过滤 P0。TERMINATE",
                "测试工程师",
                "你是测试工程师。",
            ),
        ],
        speaker_selection="round_robin",
        max_rounds=5,
    )


# ---------------------------------------------------------------------------
# 场景 3：架构选型辩论
# ---------------------------------------------------------------------------


def build_architecture_debate() -> Debate:
    """架构选型辩论：微服务 vs 单体。"""
    return Debate(
        debaters=[
            _member(
                "microservice",
                "微服务优势明显：独立部署、技术栈自由、故障隔离、团队自治。"
                "推荐引擎作为独立服务，可以单独扩缩容，不影响主商城。"
                "阿里巴巴、Netflix 都验证了微服务在大规模场景下的价值。"
                "虽然运维复杂一些，但 K8s + Service Mesh 已经成熟。",
                "微服务派",
                "你支持微服务架构。",
            ),
            _member(
                "monolith",
                "对于 5 人团队、2 个月 MVP，微服务是过度设计。"
                "单体应用开发效率高、调试简单、部署方便。"
                "先用模块化单体（Modular Monolith）快速验证，"
                "等用户量上来、边界清晰后再拆分。"
                "过早拆分微服务是初创公司的常见陷阱。",
                "单体派",
                "你支持模块化单体架构。",
            ),
        ],
        judge=_agent(
            "architect",
            "综合考虑团队规模（5人）、时间（2个月MVP）、技术能力："
            "建议采用模块化单体起步，但代码结构按领域划分（DDD），"
            "预留未来拆分为微服务的接口边界。推荐引擎模块独立设计，"
            "通过内部 API 调用，但可以作为独立进程部署。"
            "这样既保持了开发效率，又不锁死架构演进路径。",
            "你是首席架构师，做最终仲裁。",
        ),
        rounds=2,
    )


# ---------------------------------------------------------------------------
# 场景 4：Workflow DAG 混合编排
# ---------------------------------------------------------------------------


def build_project_kickoff_workflow() -> Workflow:
    """项目启动工作流 DAG。"""
    wf = Workflow()

    wf.add_node("init", lambda ctx: {"status": "项目启动", "time": datetime.now(timezone.utc).isoformat()})
    wf.add_node("create_requirements", lambda ctx: create_task.func(
        "PRD 编写", "产品经理", "high", "编写智能推荐系统 PRD"
    ))
    wf.add_node("create_design", lambda ctx: create_task.func(
        "技术方案设计", "架构师", "high", "系统架构与接口设计"
    ))
    wf.add_node("create_dev_tasks", lambda ctx: "\n".join([
        create_task.func("协同过滤引擎", "后端", "critical", ""),
        create_task.func("实时特征管道", "后端", "high", ""),
        create_task.func("在线服务 API", "后端", "high", ""),
    ]))
    wf.add_node("create_test_tasks", lambda ctx: "\n".join([
        create_task.func("测试用例编写", "测试", "medium", ""),
        create_task.func("压测方案", "测试", "high", ""),
    ]))
    wf.add_node("summary", lambda ctx: get_project_status.func())

    # DAG 拓扑：初始化后并行创建需求和设计
    wf.connect("init", "create_requirements")
    wf.connect("init", "create_design")
    # 设计完成后并行创建开发任务和测试任务
    wf.connect("create_requirements", "create_dev_tasks")
    wf.connect("create_design", "create_dev_tasks")
    wf.connect("create_requirements", "create_test_tasks")
    wf.connect("create_design", "create_test_tasks")
    # 汇总
    wf.connect("create_dev_tasks", "summary")
    wf.connect("create_test_tasks", "summary")

    return wf


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("企业级多部门协同流水线")
    print("=" * 60)

    # --- 场景 1：交付流水线 ---
    print("\n" + "=" * 60)
    print("场景 1：五阶段交付流水线（Pipeline）")
    print("=" * 60)
    pipe = build_delivery_pipeline()
    out = pipe.run("构建智能推荐系统 Phoenix 项目")
    print(f"\n阶段输出：")
    for stage_name, output in out.outputs.items():
        print(f"  [{stage_name}]\n    {output[:80]}...")
    print(f"\n最终上线方案：\n  {out.final[:120]}...")

    # --- 场景 2：需求讨论 ---
    print("\n" + "=" * 60)
    print("场景 2：需求优先级讨论（GroupChat）")
    print("=" * 60)
    chat = build_requirement_discussion()
    chat_out = chat.run("Phoenix 项目第一个 Sprint 做什么功能？")
    print(f"讨论轮次：{len(chat_out.rounds)}")
    for r in chat_out.rounds:
        print(f"  [{r['speaker']}] {r['content'][:80]}...")

    # --- 场景 3：架构辩论 ---
    print("\n" + "=" * 60)
    print("场景 3：架构选型辩论（Debate）")
    print("=" * 60)
    debate = build_architecture_debate()
    debate_out = debate.run("推荐系统选微服务还是单体架构？")
    for r in debate_out.rounds:
        print(f"  [{r['speaker']}] (第{r.get('round', '?')}轮) {r['content'][:100]}...")
    print(f"\n架构仲裁：\n  {debate_out.final[:150]}...")

    # --- 场景 4：项目启动 Workflow ---
    print("\n" + "=" * 60)
    print("场景 4：项目启动 DAG Workflow")
    print("=" * 60)
    wf = build_project_kickoff_workflow()
    wf_out = wf.run({})
    print(f"最终状态：{wf_out.get('summary', '')}")
    print(f"任务列表：")
    for t in TASKS:
        print(f"  {t['id']} [{t['priority']}] {t['title']} → {t['assignee']}")
    print(f"\n流程图：\n{wf.to_mermaid()}")

    # --- Checkpoint 持久化 ---
    print("\n" + "=" * 60)
    print("场景 5：Checkpoint 持久化")
    print("=" * 60)
    store = InMemoryCheckpointStore()
    agent = Agent(
        provider=_build_provider(),
        tools=[create_task, get_project_status],
        memory=ShortMemory(),
        checkpoint_store=store,
        system_prompt="你是项目经理。跟踪任务状态，更新进度。",
    )
    result = await agent.arun("更新项目状态")
    cp = await store.load(result.trace_id)
    print(f"Checkpoint 保存成功（trace_id: {result.trace_id[:8]}...）")
    print(f"  cursor: {cp.cursor if cp else 'N/A'}")
    print(f"  messages: {len(cp.messages) if cp else 0} 条")

    # 模拟断点恢复
    if cp:
        resume_result = await agent.resume(cp)
        print(f"恢复执行完成：{resume_result.final_answer[:80]}...")

    print("\n" + "=" * 60)
    print("多部门协同流水线运行完毕")
    print(f"总计创建 {len(TASKS)} 个任务, {len(DESIGNS)} 篇设计文档")
    print("=" * 60)


def _build_provider() -> Provider:
    model = os.getenv("MORAINET_OLLAMA_MODEL")
    if model:
        return OllamaProvider(model=model)
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[
                        ToolCall(id="1", name="get_project_status", arguments={})
                    ]
                ),
                usage=Usage(total_tokens=10),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(
                    content="项目状态更新：所有任务按计划推进，无阻塞。"
                    "协同过滤引擎开发进度 60%，实时特征管道 40%。"
                    "下周目标：完成协同过滤模块单元测试。"
                ),
            ),
        ]
    )


if __name__ == "__main__":
    asyncio.run(main())
