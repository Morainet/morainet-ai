# Morainet AI 实现说明与路线 (v1.4)

> **版本**：v1.4 · **状态**：代码已落地，文档待完善 · **更新**：2026-08-29
>
> 本文是 [`architecture.md`](architecture.md)（v1.2 设计稿）与 [`architecture-v1.3.md`](architecture-v1.3.md) 的**实现版续篇**：
> 记录 v1.1–v1.4 真实落地的模块（工程治理 / 多模态 / 分布式 / 调试工具），以及未来迭代方向。
> 文中类名/签名均与 `morainet/` 代码一致；标注 *(Planned)* 的为尚未实现的规划项。

---

## 目录

1. [实现状态总览](#1-实现状态总览)
2. [v1.1–v1.4 新增模块](#2-v11v14-新增模块)
3. [实际 API 速览（新增部分）](#3-实际-api-速览新增部分)
4. [已知差距与风险](#4-已知差距与风险)
5. [未来迭代方向](#5-未来迭代方向)

---

## 1. 实现状态总览

截至 v1.4，代码约 1.2 万行（不含测试），单测 650+，`ruff` + `mypy --strict` 全绿，覆盖率门禁暂定为 44%（因新增模块测试滞后，见 §4）。

| 模块 | 实现 | 状态 |
| --- | --- | --- |
| Agent Core / 推理策略 / Tool System | v1.0 基础上增强（`EnhancedReActStrategy` / `PlanSolveReflectStrategy` / `ContextCompressor` / `ToolCache`） | ✅ |
| Provider | 13 家厂商 + `ModelRouter` + `OllamaScheduler` + `multi_model_query` | ✅ |
| Memory / VectorStore | 6 种向量库后端 + `RAGPipeline` / `KnowledgeBase` / `HierarchicalMemory` / `TemporalMemory` / `FactStore` / `TaskGoalStore` / `UserPreferencesStore` | ✅ |
| Multi-agent | A2A 协议 + 群聊/辩论/评审/委托 + 工厂/池/沙箱 | ✅ |
| 工程治理 | `engineering/`：限流（TokenBucket / SlidingWindow）· 并发 `ConcurrencyLimiter` · 计费 `BillingTracker` · 熔断 `CircuitBreaker` | ✅ |
| 工具安全 | `PermissionEnforcer` / `ApprovalFlow` / `AuditLogger`（InMemory / File / SQLite 存储） | ✅ |
| 持久化 | Checkpoint：InMemory / File / SQLite / Redis / PostgreSQL | ✅ |
| 多模态 | `multimodal/`：内容模型 · 视觉/OCR/图表/语音工具 · 多模态 RAG · Provider 适配器 | ✅ |
| 分布式 | `distributed/`：任务队列 · DAG 分布式调度 · 集群/分片 · 负载均衡 · 分布式 Checkpoint | ✅ |
| 调试工具 | `cli/` + `debug_panel/`：`morainet chat` · 本地调试面板 · Mermaid 导出 | ✅ |
| MCP / Plugin | 连接池 + 资源缓存 + 插件市场 | ✅ |
| 工程 | GitHub Actions CI · 发布（PyPI，含预发布工作流）· `tests/live` 联调脚手架 | ✅ |

---

## 2. v1.1–v1.4 新增模块

### 2.1 工程治理 `morainet/engineering/`

面向生产环境的横切能力，全部与 Provider 解耦、可独立使用：

- **限流**：`TokenBucketRateLimiter`（令牌桶）/ `SlidingWindowRateLimiter`（滑动窗口），线程安全、支持异步 acquire。
- **并发控制**：`ConcurrencyLimiter` 信号量式并发上限。
- **计费**：`BillingTracker` / `BillingStats` 按模型累计 token 与请求数，可导出统计。
- **熔断**：`CircuitBreaker`（CLOSED / OPEN / HALF_OPEN 三态），失败阈值 + 冷却时间自动熔断。

### 2.2 多模态 `morainet/multimodal/`

在保留"统一 `Content` 中间表示"的基础上，新增结构化多模态内容模型：

- **内容模型**：`TextPart` / `ImageUrlPart` / `ImageBase64Part` / `AudioPart` / `FilePart`（`ContentType` 枚举），支持：
  - 序列化：`content_to_str` / `content_to_openai_blocks`（转 OpenAI 视觉消息格式）
  - 判定：`content_has_text` / `content_has_images` / `split_text_and_images`
  - 解析：`parse_multimodal_content` / `sanitize_content`（防注入清理）/ `source_to_part`（本地路径或 URL → Part）
- **工具**：`image_understand` / `describe_image` / `ocr` / `chart_parse` / `speech_to_text`，返回统一 `ImageAnalysisResult`。
- **多模态 RAG**：`MultimodalDocument`（文字 + 图片）→ `ImageCaptioner` / `ImageTextEncoder` 抽象（默认 `SimpleImageCaptioner` / `SimpleImageTextEncoder`，可替换为真实视觉模型）→ `MultimodalRetriever` → `MultimodalRAG`（检索 + 生成闭环）；`VisionReasoningChain` 实现"看图-推理"链路。
- **Provider 适配器**：`MultimodalAdapter` 抽象 + `register_adapter` / `get_adapter` / `default_adapter` 注册表，便于对接各家视觉模型。

> 设计约束：工具默认不发起真实网络请求；模型能力通过适配器注入，便于离线单测与示例运行。

### 2.3 分布式 `morainet/distributed/`

将 Agent 从"单进程"扩展到"多节点"：

- **任务队列**：`Task` / `TaskResult` / `TaskStatus`，`TaskBackend` 抽象，实现 `RedisBackend`（`[redis]`）/ `RabbitMQBackend`；`TaskProducer` / `TaskConsumer` 完成异步任务投递与消费。
- **DAG 分布式调度**：`TaskEnvelope` 携带任务元数据，`DistributedScheduler` / `DistributedParallelScheduler` / `DistributedProgressScheduler` 复用 v1.0 的 Workflow DAG 语义做跨节点执行，`DistributedNodeExecutor` 负责节点内执行。
- **集群与分片**：`ConsistentHashRing`（一致性哈希）→ `SessionShard` 会话分片 → `SessionShardRouter` 将 Agent 会话稳定路由到节点；`AgentCluster` / `ClusterMember` / `EdgeNode` + `MemberStatus` / `ClusterRole` 描述集群拓扑；`cloud_or_edge` 给出云边决策启发式。
- **负载均衡**：`Endpoint`（带权重/健康状态）→ `RoundRobinBalancer` / `WeightedRoundRobinBalancer` / `HybridRouter`（混合路由）/ `ProviderShard`（按 Provider 分片）。
- **分布式 Checkpoint**：`ClusterCheckpointStore` / `HeartbeatCheckpointStore`（心跳续租）/ `DistributeCheckpointHook`（自动挂钩到 Agent 执行）。

### 2.4 调试工具 `morainet/cli/` + `morainet/debug_panel/`

- **CLI**：`morainet chat`（`morainet.cli.main:main`）——交互式聊天入口。
- **调试面板**：`PanelStore` / `PanelHook` / `get_panel_store` 收集运行轨迹，`morainet-debug` 启动本地 Web 面板；`export_mermaid_html` / `export_mermaid_svg` / `export_mermaid_png` 将 Workflow / Trace 导出为可视化图表。

### 2.5 工具安全 `morainet/tools/`（安全子模块）

- `PermissionRegistry` / `PermissionEnforcer`：按工具声明权限，未授权抛 `ToolPermissionError`。
- `ApprovalFlow` / `ApprovalRequest` / `ApprovalResponse`：审批流（`CallbackApprover` 回调式 / `InteractiveApprover` 交互式）。
- `AuditLogger` / `AuditEntry`：操作审计（`InMemoryAuditStore` / `FileAuditStore` / `SQLiteAuditStore`）。

---

## 3. 实际 API 速览（新增部分）

### 工程治理

```python
from morainet import (
    TokenBucketRateLimiter, SlidingWindowRateLimiter, ConcurrencyLimiter,
    BillingTracker, CircuitBreaker, CircuitState,
)
limiter = TokenBucketRateLimiter(rate=10, capacity=20)
async with limiter: ...
```

### 多模态

```python
from morainet import parse_multimodal_content, content_to_openai_blocks
from morainet import MultimodalRAG, VisionReasoningChain, ImageUrlPart, TextPart
parts = parse_multimodal_content("看图：https://example.com/a.png")
blocks = content_to_openai_blocks(parts)   # -> [{"type": "image_url", ...}, ...]
```

### 分布式

```python
from morainet import (
    ConsistentHashRing, SessionShardRouter, RoundRobinBalancer,
    DistributedScheduler, Task, ClusterCheckpointStore,
)
ring = ConsistentHashRing(nodes=["n1", "n2", "n3"])
scheduler = DistributedScheduler(executor=..., backend=...)
```

### 调试工具

```python
from morainet import PanelHook, get_panel_store, export_mermaid_html
hook = PanelHook()                      # 挂到 Agent(hooks=[hook])
html = export_mermaid_html(workflow)
```

---

## 4. 已知差距与风险

| 差距 | 说明 | 影响 |
| --- | --- | --- |
| 新模块测试滞后 | `multimodal/`、`distributed/`、`cli/`、`debug_panel/` 尚无专项单测 | 覆盖率 84% → 44%；新模块行为无回归保护 |
| 文档滞后 | 上述模块缺少 wiki / 教程 | 用户上手成本高 |
| 分布式依赖外部服务 | Redis / RabbitMQ 后端、集群拓扑需真实服务验证 | 离线只能覆盖纯内存/单机路径 |
| 多模态真实模型联调 | 视觉/OCR/语音工具默认 stub，需接真实模型 key 验证 | 示例可跑但产出受限于 stub |
| 版本与文档漂移 | 此前的 v1.0 文档与代码量级不匹配 | 已通过本次 v1.4 对齐缓解 |

---

## 5. 未来迭代方向

按"投入产出"分组。

### 5.1 上线就绪（投入小、价值高）
- **新模块补测**：`tests/test_distributed.py` / `tests/test_multimodal.py` / `tests/test_cli.py`，覆盖率门禁 44 → 55 → 70 阶梯回升。
- **真端点联调全绿**：用各家 key 跑 `pytest -m live`，验证多模态适配器 + 分布式真实后端。
- **发布验证**：走通 PyPI 稳定版 + 预发布（`publish-pre-release.yml`）双链路。

### 5.2 能力深化（抬高天花板）
- **分布式生产加固**：任务队列持久化 / 重试 / 死信、集群节点故障转移的真实场景测试与文档。
- **上下文工程闭环**：把 `ContextCompressor` 接入默认推理循环 + token 预算动态估算。
- **成本治理闭环**：`ModelRouter` + `BillingTracker` + 限流/熔断组合成"路由→限流→计费→熔断"一体化治理示例。

### 5.3 生态 / 体验
- **CLI + 调试面板打磨**：离线 trace 浏览、单 Agent 试跑，作为本地调试的差异化体验。
- **文档站重构**：按「核心 / 记忆 / 多智能体 / 分布式 / 多模态 / 工程化」组织，配典型示例。
- **更多 embedding / rerank 服务集成**：接真实 embedding 服务，完善 `rerank` extra。

> **建议优先级**：5.1 → 5.2。先把"新模块测试 + 真实联调 + 发布"补齐（可信度），再投分布式加固与成本治理（差异化）；生态/体验待真实用户反馈后再做。
