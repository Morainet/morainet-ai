# Examples — 多方向参考 Agent

Morainet 是**通用 Agent Runtime**。下面这些示例用**同一套内核**搭出不同方向的 agent，
证明运行时的通用性，也作为上手模板。

> 全部**离线可跑**（不需 API key）：框架机制（检索 / 工具 / 路由 / 编排 / 验证）做的是**真事**，
> 语言部分用 `MockProvider`；多数示例可用 `MORAINET_OLLAMA_MODEL=qwen2.5:3b` 切到本地真模型。
>
> **成熟度说明**：基础示例为 **demo 级**参考实现，企业级示例为**接近生产**的参考架构。

## 基础示例

| 方向 | 示例 | 复用的内核能力 |
| --- | --- | --- |
| 工具调用 | `quickstart.py` | `@tool` + 工具调用循环 |
| 知识 / RAG | `rag_doc_qa.py` | `LongMemory` 检索 + Agent 自动注入 |
| 编码 / harness | `coding_assistant.py` | 真实工具(读文件/跑测试) + 验证闭环 |
| 客服 / 分诊 | `multi_agent.py`（Router 部分） | `Router` 路由 |
| 研究 / 写作 | `multi_agent.py`（Pipeline 部分） | `Pipeline` 顺序编排 |
| 多 Agent 协作 | `multi_agent.py`（as_tool 部分） | `Agent.as_tool()` 层级 |
| 流程自动化 | `memory_and_workflow.py` | `Workflow` DAG |
| 长期记忆 | `memory_and_workflow.py` | `LongMemory` |
| 本地 / 隐私 | `live_ollama.py` · `chat.py` | `OllamaProvider`（本机、零成本） |
| 流式对话 | `live_ollama_stream.py` · `chat.py` | `agent.astream()` |
| 推理策略 | `react_and_providers.py` | `ReActStrategy` + 多 Provider |
| 调试 / 持久化 | `debug_and_checkpoint.py` | `Debugger` · `Checkpoint` · `resume` |
| 扩展 | `plugins_mcp_retry.py` | Plugin · MCP · 重试 |

## 企业级实战

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 智能客服 RAG | `enterprise_customer_service_rag.py` | LongMemory + Router + ShortMemory + 工具审批 + 工单系统 |
| 代码工程助手 | `enterprise_code_assistant.py` | 真实工具(AST/Lint/测试) + Workflow DAG + Agent 审查 |
| 自动化运维 | `enterprise_ops_agent.py` | 告警处理 + 预案 Runbook + 巡检 + 部署验证 |
| 多部门协同 | `enterprise_cross_dept_pipeline.py` | Pipeline + GroupChat + Debate + Workflow DAG + Checkpoint |

## 运行

```bash
# 离线（默认）
python examples/rag_doc_qa.py
python examples/coding_assistant.py
python examples/multi_agent.py

# 本地真模型（需 ollama serve + ollama pull qwen2.5:3b）
MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/coding_assistant.py
```

## 多模态

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 多模态基础 | `multimodal_basic.py` | ContentPart 类型 + Message 构建器 + 多模态工具注册 + Provider 适配器路由 |
| 多模态 RAG | `multimodal_rag.py` | MultimodalDocument + ImageCaptioner + MultimodalRAG + VisionReasoningChain + Agent 集成 |

## 分布式 Agent 调度

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 分布式工作流 | `distributed_workflow.py` | Task Queue (Redis/RabbitMQ) + DistributedParallelScheduler + TaskEnvelope 序列化 + 进度追踪 |
| 分布式集群 | `distributed_cluster.py` | ConsistentHashRing 会话分片 + LoadBalancer + HybridRouter 边缘/云端路由 + DistributedRunTrace + 分布式断点恢复 |

## 可视化配套工具

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| Debug Web 面板 | `debug_panel_demo.py` | PanelHook 事件流 + 实时 Web 面板 + Token 消耗图表 + 工具调用时间线 + 记忆检索日志 |
| Mermaid 工作流导出 | `debug_panel_demo.py` | 交互式 HTML 图表 + Mermaid/SVG/PNG 导出 + 执行状态着色 + 暗色模式切换 |
| CLI 命令行工具 | `debug_panel_demo.py` | 批量执行 Agent + trace 导出/合并 + memory 清理 + tool Schema 调试 + workflow viz |

### 启动 Debug Web 面板

```bash
# 终端 1：启动面板
python -m morainet.debug_panel.server --port 8080

# 终端 2：运行带 PanelHook 的 Agent（事件自动推送到面板）
python examples/debug_panel_demo.py
# 浏览器访问 http://127.0.0.1:8080
```

### CLI 快速参考

```bash
python -m morainet.cli run "What is AI?"                          # 执行单次查询
python -m morainet.cli batch queries.txt -o results.json           # 批量执行
python -m morainet.cli trace export ./traces/                      # 导出 trace
python -m morainet.cli trace merge t1.json t2.json                 # 合并多节点 trace
python -m morainet.cli trace inspect trace.json                    # 查看 trace 详情
python -m morainet.cli memory clean                                # 清理记忆
python -m morainet.cli tool schema -m my_tools.py                  # 调试工具 Schema
python -m morainet.cli workflow viz -m my_wf.py -o output.html     # 可视化工作流
```

## 主动式长期记忆系统

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 分层记忆自动归纳 | `hierarchical_memory_demo.py` | 短期缓冲 → 中期摘要 → 长期事实知识库自动沉淀 |
| 事实冲突检测 & 时效性 | `hierarchical_memory_demo.py` | 同主题事实冲突检测、TTL 过期自动失效、新鲜度评分 |
| 用户偏好 & 任务目标持久化 | `hierarchical_memory_demo.py` | 跨会话保持 Agent 人设、偏好 Pin 锁定、目标子任务树 |
| 时序记忆检索 | `hierarchical_memory_demo.py` | 时间线记录决策/事件/里程碑、`review_history()` 回顾历史决策 |
| Agent 集成 | `hierarchical_memory_demo.py` | HierarchicalMemory 作为 Agent memory 直接使用、自动上下文注入 |

### 📦 核心模块速查

```python
from morainet import (
    HierarchicalMemory,     # 三层记忆：缓冲→摘要→知识库
    FactStore,              # 事实 CRUD + 冲突检测 + TTL 过期
    UserPreferencesStore,   # 用户偏好持久化 + Pin 锁定
    TaskGoalStore,          # 任务目标树 + 状态追踪
    TemporalMemory,         # 决策/事件时间线 + 关键字回顾
)
```

### 三层记忆架构

```
Level 1 — Episodic Buffer (短期缓冲)
  └─ 最近 N 条原始消息，溢出后触发压缩
       │
       ▼
Level 2 — Episode Summaries (中期摘要)
  └─ LLM 压缩旧消息为结构化摘要，保留决策/偏好/事实
       │
       ▼
Level 3 — Factual Knowledge Base (长期知识库)
  ├─ FactStore     → 主题-值对 + 冲突检测 + TTL 过期
  ├─ Preferences   → 用户偏好 + Pin 锁定 (跨会话人设)
  ├─ TaskGoals     → 长期目标 + 子任务 + 状态追踪
  └─ Temporal      → 决策时间线 + 回顾历史
```

## 多智能体协作高阶编排

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| A2A 原生通信协议 | `multiagent_collaboration_demo.py` | AgentIdentity + A2AChannel + A2ABus 共享消息总线 + handshake/query/delegate/event |
| 辩论拓扑 | `multiagent_collaboration_demo.py` | DebateTeam 多轮辩论 + Arbiter 仲裁 + 开场/反驳/总结 |
| 评审拓扑 | `multiagent_collaboration_demo.py` | ReviewTeam 产出→审查→修改循环 + 多审查员 |
| 分层委托拓扑 | `multiagent_collaboration_demo.py` | HierarchicalTeam 主 Agent 分解→委托专家→汇总 |
| 共享内存池 | `multiagent_collaboration_demo.py` | SharedMemoryPool 多 Agent 共享记忆总线 |
| 管道 & 路由 | `multiagent_collaboration_demo.py` | Pipeline 顺序编排 + Router 条件路由 |
| 群聊协作 | `multiagent_collaboration_demo.py` | GroupChat 多 Agent 轮流发言 + 共享对话历史 |
| 动态 Agent 生成 | `multiagent_collaboration_demo.py` | AgentFactory spawn/destroy + AgentBlueprint 模板 |
| 资源 & 权限隔离 | `multiagent_collaboration_demo.py` | AgentSandbox + ResourceQuota + PermissionProfile + MemoryNamespace |
| Agent 池化复用 | `multiagent_collaboration_demo.py` | AgentPool prewarm + acquire/release + 多策略调度 |

### 核心模块速查

```python
from morainet import (
    # A2A 原生协议
    A2AChannel, A2ABus, A2AMessage, AgentIdentity,
    # 协作拓扑
    Debate, DebateTeam, ReviewTeam, HierarchicalTeam,
    SharedMemoryPool, Pipeline, Stage, Router, Route,
    GroupChat, GroupChatMember, TeamOrchestrator,
    # 动态 Agent 工厂
    AgentFactory, AgentBlueprint, AgentPool,
    # 资源 & 权限隔离
    AgentSandbox, ResourceQuota, PermissionProfile, MemoryNamespace,
)
```

### 多 Agent 拓扑架构

```
A2A Protocol (原生通信，无需工具中转)
├─ A2AChannel      — 点对点双向通道 (handshake, query, delegate)
└─ A2ABus          — 多对多消息总线 (broadcast, topic filter)

Topologies (协作模式)
├─ Debate / DebateTeam       — 多轮辩论 + Arbiter 仲裁
├─ ReviewTeam                — 产出→审查→修改循环
├─ HierarchicalTeam          — 任务分解→委托专家→汇总
├─ SharedMemoryPool          — 共享记忆池隐式协作
├─ Pipeline / Stage          — 顺序流水线编排
├─ Router / Route            — 条件路由到最佳 Agent
└─ GroupChat                 — 多 Agent 轮流群聊

Sandbox (资源 & 权限隔离)
├─ ResourceQuota     — token/step/time 三向量限制
├─ PermissionProfile — LIMITED/STANDARD/ELEVATED/FULL 四级权限
└─ MemoryNamespace   — 每 Agent 独立记忆空间，互不可读

Agent Lifecycle (动态生命周期)
├─ AgentFactory      — Blueprint 模板 → spawn 生成 → destroy 销毁
└─ AgentPool         — 预热 + acquire/release 复用 + 自动回收
```

> macOS + Homebrew Python：命令行需 `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`
> （编辑器 F5 已在 `.vscode` 配好）。
