# 入门教程：30 分钟搭一个 Agent

> 本页面向 GitHub Wiki。跟着做完，你会用 Morainet AI 搭出一个能调工具、有记忆、
> 能流式输出的 Agent，并了解多 Agent、调试与生产化的用法。
> 全程**不需要 API key**——用本地 Ollama 即可。

---

## 0. 准备

### 安装

```bash
git clone <your-repo-url> morainet-ai && cd morainet-ai
python -m venv .venv && source .venv/bin/activate     # 需 Python 3.11+
pip install -e ".[dev]"
```

### 准备一个本地模型（免费、无需 key）

```bash
brew install ollama          # macOS；其他平台见 ollama.com
ollama serve                 # 单开一个终端保持运行
ollama pull qwen2.5:3b       # 支持工具调用的小模型（约 1.9GB）
```

> **macOS + Homebrew Python 提示**：若 `import` 报 `pyexpat` / `libexpat` 错误，
> 命令前加 `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`。

> 想用云端模型？把下文的 `OllamaProvider` 换成 `OpenAIProvider(model="gpt-4o")`，
> 并设 `export MORAINET_OPENAI_API_KEY=sk-...` 即可，其余代码不变。

---

## 1. 第一个 Agent + 工具

新建 `my_agent.py`：

```python
from morainet import Agent, tool
from morainet.providers import OllamaProvider


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称，如 "上海"
    """
    fake = {"上海": "晴，26°C", "北京": "多云，22°C"}
    return fake.get(city, f"{city}：暂无数据")


agent = Agent(provider=OllamaProvider(model="qwen2.5:3b"), tools=[get_weather])
result = agent.run("上海今天适合穿什么？")
print(result.final_answer)
```

运行：

```bash
python my_agent.py
```

**发生了什么？**
1. `@tool` 自动从函数签名 + docstring 生成 JSON Schema，模型据此知道能调用 `get_weather`。
2. Agent 把问题 + 工具列表发给模型；模型决定调用 `get_weather(city="上海")`。
3. 框架执行工具，把结果回灌给模型，模型生成最终答案。

查看过程：

```python
print(result.usage.total_tokens)                          # token 消耗
print([(s.description, s.output) for s in result.steps])  # 工具调用轨迹
```

> 工具可以是同步或 `async def`，框架自动识别。

---

## 2. 加记忆

### 短期记忆（多轮对话）

```python
from morainet import Agent, ShortMemory
from morainet.providers import OllamaProvider

agent = Agent(provider=OllamaProvider(model="qwen2.5:3b"), memory=ShortMemory())
agent.run("我叫小明")
print(agent.run("我叫什么？").final_answer)   # 记得住
```

### 长期记忆 / RAG（知识库问答）

把文档存进 `LongMemory`，Agent 回答前会自动检索相关内容并注入上下文：

```python
import asyncio
from morainet import Agent
from morainet.core.models import Message
from morainet.memory import LongMemory, InMemoryVectorStore, OllamaEmbedder
from morainet.providers import OllamaProvider


async def main():
    memory = LongMemory(
        store=InMemoryVectorStore(),
        embedder=OllamaEmbedder("nomic-embed-text"),  # 真实语义检索；需 ollama pull nomic-embed-text
        score_threshold=0.1,
    )
    for doc in ["退款政策：签收 7 天内可退", "营业时间：周一至周五 9-18 点"]:
        await memory.add(Message.assistant(content=doc))

    agent = Agent(provider=OllamaProvider(model="qwen2.5:3b"), memory=memory)
    print((await agent.arun("怎么退款？")).final_answer)


asyncio.run(main())
```

> 不想装 embedding 模型？默认 `HashEmbedder` 离线可用，但只是关键词级；语义检索请用 `OllamaEmbedder` 或 `OpenAIEmbedder`。

---

## 3. 流式输出

逐 token 打印，体验更顺滑：

```python
import asyncio

async def chat():
    async for token in agent.astream("写一句关于秋天的话"):
        print(token, end="", flush=True)

asyncio.run(chat())
```

---

## 4. 多 Agent 编排

三种拓扑，按需选用。

### 层级：一个 agent 把另一个当工具

```python
researcher = Agent(provider=..., tools=[...])
orchestrator = Agent(provider=..., tools=[researcher.as_tool("research", "调研事实")])
orchestrator.run("帮我查并总结上海天气")
```

### 顺序：Pipeline（调研 → 撰写）

```python
from morainet import Pipeline, Stage

pipe = Pipeline([
    Stage("research", researcher),
    Stage("write", writer, instruction="基于调研「{research}」写一段话：{query}"),
])
print(pipe.run("介绍上海").final)
```

### 路由：Router（分诊到专家）

```python
from morainet import Router, Route

router = Router(
    [Route("billing", billing_agent, "账单问题"),
     Route("tech", tech_agent, "技术故障")],
    selector=lambda q: "tech" if "连不上" in q else "billing",   # 规则路由
    # 或 provider=OllamaProvider(...) 让模型来路由
)
print(router.run("设备连不上网").route)   # -> tech
```

---

## 5. 切换模型 / 推理策略

```python
from morainet import Agent, ReActStrategy
from morainet.providers import OpenAIProvider, ClaudeProvider, DeepSeekProvider

Agent(provider=OpenAIProvider(model="gpt-4o"))
Agent(provider=ClaudeProvider(model="claude-sonnet-4-6"))
Agent(provider=DeepSeekProvider())                       # OpenAI 兼容
Agent(provider=..., strategy=ReActStrategy())            # 模型不支持原生工具调用时
```

---

## 6. 调试与持久化

### 看运行时间线

```python
from morainet import Agent, Debugger

dbg = Debugger()
agent = Agent(provider=..., tools=[...], hooks=[dbg])
agent.run("...")
print(dbg.timeline())     # run_start / llm / tool / run_end 逐步带耗时
```

### Checkpoint：崩溃后恢复

```python
from morainet import Agent, FileCheckpointStore

store = FileCheckpointStore("./.ckpt")
agent = Agent(provider=..., checkpoint_store=store)   # 每步自动快照
result = agent.run("长任务……")

# 之后从某次 trace 恢复
import asyncio
cp = asyncio.run(store.load(result.trace_id))
agent.resume(cp)
```

---

## 7. 走向生产

```python
from morainet import Agent
from morainet.providers import RetryPolicy

agent = Agent(
    provider=...,
    retry=RetryPolicy(max_retries=3),     # 限流/超时自动指数退避重试
    token_budget=20_000,                  # 超预算抛 BudgetExceededError
    max_consecutive_errors=3,             # 连续失败 3 次中止
    approve_tool=lambda name, args: input(f"运行 {name}? [y/N] ") == "y",  # 危险工具人工审批
)
```

危险工具用 `@tool(dangerous=True)` 标记后，才会触发 `approve_tool` 审批。

---

## 8. Workflow（流程已知时）

当步骤固定、不需要模型自主决策，用 DAG 更可控：

```python
from morainet import Workflow

wf = Workflow()
wf.add_node("fetch", lambda ctx: {"price": 214})
wf.add_node("report", lambda ctx: f"价格={ctx['fetch']['price']}")
wf.connect("fetch", "report")

print(wf.run()["report"])
print(wf.to_mermaid())        # 导出流程图
```

---

## 下一步

- 浏览 `examples/`：RAG、编码助手、多 Agent、调试等可运行示例。
- 架构与设计：`docs/architecture.md`、`docs/architecture-v1.3.md`。
- 自定义扩展（Provider / Tool / Memory / Strategy / Hook）：见 `CONTRIBUTING.md`。

遇到问题欢迎提 Issue。
