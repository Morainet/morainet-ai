# Morainet AI 实现说明与路线 (v1.3)

> **版本**：v1.3 · **状态**：已实现（对应代码 v1.0.0）· **更新**：2026-06-16
>
> 本文是 [`architecture.md`](architecture.md)（v1.2 设计稿）的**实现版补充**：
> 记录真实落地的 API、与设计稿的差异，以及未来迭代方向。
> 设计理念与整体架构仍以 `architecture.md` 为准；本文中类名/签名均与 `morainet/` 代码一致。
> 标注 *(Planned)* 的为尚未实现的规划项。

---

## 目录

1. [实现状态总览](#1-实现状态总览)
2. [与 v1.2 设计稿的差异](#2-与-v12-设计稿的差异)
3. [实际 API 速览](#3-实际-api-速览)
4. [v1.0 已落地的能力](#4-v10-已落地的能力)
5. [未来迭代方向](#5-未来迭代方向)

---

## 1. 实现状态总览

代码约 3000 行（不含测试），单测 80+，覆盖率 ~84%，`ruff` + `mypy --strict` 全绿。

| 模块 | 实现 | 状态 |
| --- | --- | --- |
| Agent Core | `core/agent.py` | ✅ |
| 推理策略 | `reasoning/`：`ToolCallingStrategy`（默认）+ `ReActStrategy` | ✅ |
| Tool System | `tools/`：`@tool` / `ToolRegistry` / `Tool.from_schema` | ✅ |
| Provider | OpenAI / Claude / Gemini / Ollama / DeepSeek / Mock + `RetryingProvider` | ✅ |
| 流式 | OpenAI(SSE) / Ollama(NDJSON) | ✅ |
| Memory | `ShortMemory` / `LongMemory` / `SummarizingMemory` | ✅ |
| Embedder | `HashEmbedder`（默认）/ `OllamaEmbedder` / `OpenAIEmbedder` | ✅ |
| VectorStore | `InMemoryVectorStore`（默认）/ `ChromaStore` | ✅ |
| Workflow | `Workflow`（DAG + 分层并行 + `to_mermaid`/`to_dot`） | ✅ |
| Prompt | `PromptTemplate` / `PromptRegistry` / 内置模板 | ✅ |
| 可观测 | `Hook` / `HookManager` / `TraceCollector` / `Debugger` | ✅ |
| 持久化 | `Checkpoint` / `InMemory`·`File`·`SQLite` Store / `CheckpointHook` / `resume` | ✅ |
| 多 Agent | `Agent.as_tool()`（层级）/ `Pipeline`（顺序）/ `Router`（路由） | ✅ |
| 生产化 | `RetryPolicy` / `token_budget` / `max_consecutive_errors` / 危险工具审批 | ✅ |
| 扩展 | `PluginRegistry` / MCP（工具 + 资源 + 提示，`MCPClient` + `stdio_session`） | ✅ |
| OTel | `OTelHook`（可选 `[otel]`） | ✅ |
| 工程 | GitHub Actions CI · 覆盖率门禁 80% · `tests/live` 联调脚手架 | ✅ |

---

## 2. 与 v1.2 设计稿的差异

设计稿是蓝图，实现时做了取舍。以下是值得注意的差异（**以代码为准**）：

| 设计稿（v1.2） | 实际实现 | 说明 |
| --- | --- | --- |
| Reasoning = Planner/Executor/Reflector 三层 | `ToolCallingStrategy`（默认）+ `ReActStrategy` | 三层流水线未实现；改为工具调用循环，更简单、与原生 function-calling 对齐 |
| `ReasoningStrategy.step()` | `ReasoningStrategy.run(agent, ctx)` | 接口更内聚，直接驱动整个循环并返回 `AgentResult` |
| `ProviderError.TimeoutError` | `ProviderTimeoutError` | 避免与内置 `TimeoutError` 冲突 |
| `MemoryError` | `MemoryStoreError` | 同理避免与内置异常重名；并新增 `WorkflowError`/`CycleError`/`NodeNotFoundError`/`BudgetExceededError` |
| 默认向量库 = ChromaDB | 默认 `InMemoryVectorStore`，Chroma 为可选 `[chroma]` | 保持内核零依赖 |
| Plugin：`registry.register_provider(...)` | `PluginRegistry.register(kind, name, obj)` / `load_entry_points()` | 统一注册表 API |
| MCP：`MCPClient.connect("stdio", ...)` | `stdio_session(cmd, args)` 上下文管理器 + `MCPClient(session)` | session 可注入，便于测试 |
| `prompts/templates/` 目录 | 内置模板写在 `prompts/registry.py` 的 `BUILTIN_TEMPLATES` | 无单独模板目录 |
| Workflow「节点级重试」 | 暂未实现 | 仅环检测 + 拓扑分层并行 |

> 这些差异已在实现过程中通过单元测试验证；设计稿保留作为意图说明。

---

## 3. 实际 API 速览

### Agent

```python
Agent(
    provider: Provider,
    tools=None, memory=None,
    strategy=None,                 # 默认 ToolCallingStrategy
    hooks=None, checkpoint_store=None,
    retry=None,                    # RetryPolicy；自动包装 RetryingProvider
    max_steps=None, token_budget=None,
    approve_tool=None,             # 危险工具审批回调（同步/异步）
    system_prompt=None, prompts=None,
)
agent.run(q) / await agent.arun(q)
async for tok in agent.astream(q): ...
agent.resume(cp) / await agent.aresume(cp)
agent.as_tool(name, description) -> Tool      # 多 Agent
```

### 推理策略

```python
from morainet import ToolCallingStrategy, ReActStrategy
class ReasoningStrategy(ABC):
    async def run(self, agent, ctx) -> AgentResult: ...
```

### Memory / Embedder / VectorStore

```python
from morainet.memory import (
    ShortMemory, LongMemory, SummarizingMemory,
    HashEmbedder, OllamaEmbedder, OpenAIEmbedder,
    InMemoryVectorStore, ChromaStore,
)
ShortMemory(max_messages=50, max_tokens=None, token_counter=estimate_tokens)
LongMemory(store=None, embedder=None, score_threshold=0.0)
SummarizingMemory(provider, keep_recent=6, trigger_messages=12, prompt=None)
```

### Provider

```python
from morainet.providers import (
    OpenAIProvider, ClaudeProvider, GeminiProvider,
    OllamaProvider, DeepSeekProvider, MockProvider,
    RetryingProvider, RetryPolicy,
)
```

### 可观测 / 持久化 / 扩展

```python
from morainet import Hook, TraceCollector, Debugger
from morainet import Checkpoint, InMemoryCheckpointStore, FileCheckpointStore
from morainet import PluginRegistry, plugins
from morainet.mcp import MCPClient, stdio_session
```

### 异常（实际层级）

```text
MorainetError
├── ConfigError
├── ProviderError → RateLimitError / ProviderTimeoutError / AuthError / ContextLengthError
├── ToolError → ToolNotFoundError / ToolValidationError / ToolExecutionError
├── ReasoningError → MaxStepsExceededError / BudgetExceededError
├── MemoryStoreError
└── WorkflowError → CycleError / NodeNotFoundError
```

---

## 4. v1.0 已落地的能力

按交付批次（A/B/C）记录，便于追溯。

**A 组 · 补齐设计稿承诺**
- `ShortMemory(max_tokens=...)` 按 token 预算裁剪
- `Agent(token_budget=...)` 超限抛 `BudgetExceededError`
- `@tool(dangerous=True)` + `Agent(approve_tool=...)` 人工审批

**B 组 · 工程化**
- GitHub Actions CI（Python 3.11/3.12，ruff + mypy + pytest）
- 覆盖率门禁 80%
- `Agent(retry=RetryPolicy(...))` 自动重试瞬时错误
- `tests/live/` 真端点联调脚手架（默认排除，`pytest -m live` 运行，无凭证自动跳过）
- `CONTRIBUTING.md`

**C 组 · 能力深化**
- 多 Agent：`agent.as_tool()`（orchestrator / 子 agent 协作）
- 真实 embedding：`OllamaEmbedder` / `OpenAIEmbedder`（Ollama 已本机验证语义检索）
- 上下文压缩：`SummarizingMemory`（启用一直空置的 `summarizer` 模板）

**已验证联调**：Ollama（含工具调用 + 流式 + embedding）已对真实本地端点验证；OpenAI/DeepSeek 复用同一 httpx 路径。

---

## 5. 未来迭代方向

按"投入产出"分组，供后续选择。

### 5.1 上线就绪（投入小、价值高）
- **真端点联调全绿**：用各家 key 跑 `pytest -m live`，验证 Claude / Gemini / MCP `stdio_session`。
- **发布 PyPI**：打 tag、生成 wheel、发布 `morainet-ai`。
- **覆盖率提升**：为 HTTP provider 路径补 `httpx.MockTransport` 单测，门禁逐步抬到 85%。

### 5.2 能力深化（抬高天花板）
- **上下文压缩进推理循环**：当前 `SummarizingMemory` 仅在 Memory 层；可在长对话中自动触发，并接 `ContextLengthError` 兜底。
- **多 Agent 进阶**：已有层级(`as_tool`)/顺序(`Pipeline`)/路由(`Router`)；可再加 GroupChat（manager 选发言者）、子 agent 并行、共享记忆。
- **真实 embedding 缓存**：embedding 结果缓存，减少重复请求。
- **流式工具调用**：增量解析 `tool_calls`；补 Claude / Gemini 的 SSE 流式。

### 5.3 可观测 / 运维
- **成本统计与预算告警**：基于 `Usage` 做按模型计价、预算阈值告警。
- **更多 VectorStore**：Qdrant / pgvector / Milvus 适配（需运行服务端，待联调）。

> 本批已落地（原 Planned → ✅）：`max_consecutive_errors` 终止、`SQLiteCheckpointStore`、
> MCP 资源/提示集成、`OTelHook`（OpenTelemetry 导出）。
> 仍 Planned 的向量库后端（Qdrant/Redis/pgvector/Milvus）需运行真实服务端，离线无法验证，故暂留。

### 5.4 生态 / 体验
- CLI（`morainet chat`）、Web Playground / Debugger UI。
- 记忆持久化（`LongMemory` 默认落盘）。
- 更多内置工具（计算、HTTP、文件、时间）。

> **建议优先级**：5.1 → 5.2。先把"能连真端点 + 能发布"补齐（可信度），再投多 Agent 编排与上下文压缩（差异化）；运维/生态待真实用户反馈后再做，避免过早为假想需求设计。
