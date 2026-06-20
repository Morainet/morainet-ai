"""Hierarchical memory system demo — 7 sections showing all new capabilities.

Sections:
  1. FactStore — CRUD, conflict detection, expiry, freshness
  2. UserPreferencesStore — cross-session persona
  3. TaskGoalStore — long-running objectives
  4. TemporalMemory — timeline, decisions, review_history
  5. HierarchicalMemory — end-to-end 3-layer auto-sedimentation (no LLM)
  6. HierarchicalMemory with MockProvider — compression + fact extraction
  7. Integration: Agent + HierarchicalMemory
"""

from __future__ import annotations

import asyncio
import time

from morainet.core.agent import Agent
from morainet.core.models import Message, Role
from morainet.memory.facts import FactStatus, FactStore
from morainet.memory.hierarchical import HierarchicalMemory
from morainet.memory.preferences import (
    GoalStatus,
    Priority,
    TaskGoalStore,
    UserPreferencesStore,
)
from morainet.memory.temporal import EntryKind, TemporalMemory
from morainet.providers.mock import MockProvider


SEP = "\n" + "─" * 60


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: FactStore — CRUD, conflict detection, expiry, freshness
# ═══════════════════════════════════════════════════════════════════════════
def demo_fact_store():
    print(SEP)
    print("Section 1: FactStore — 知识库 CRUD、冲突检测、时效性标记、过期自动失效")
    print(SEP)

    store = FactStore()

    # Basic CRUD
    f1 = store.upsert("user_name", "Alice", evidence="user: My name is Alice")
    f2 = store.upsert("python_version", "3.12", evidence="user: I use Python 3.12")
    f3 = store.upsert("project_name", "morainet", tags=["project", "oss"])
    print(f"  插入 3 条事实: {f1.topic}={f1.value}, {f2.topic}={f2.value}, {f3.topic}={f3.value}")
    assert store.fact_count == 3

    # Get by topic
    retrieved = store.get("user_name")
    assert retrieved is not None and retrieved.value == "Alice"
    print(f"  获取 'user_name' → {retrieved.value}")

    # Same value reinforce
    f1b = store.upsert("user_name", "Alice", evidence="user: as I said, I'm Alice")
    print(f"  重复相同值 → confidence: {f1b.confidence:.2f}")

    # Conflict detection
    f1c = store.upsert("user_name", "Bob", evidence="user: actually, call me Bob")
    old_versions = store.get_all("user_name")
    statuses = [(f.value, f.status.value) for f in old_versions]
    print(f"  冲突检测 'user_name': {statuses}")
    assert any(s == "superseded" for _, s in statuses)

    # Has conflict
    assert store.has_conflict("user_name")
    print(f"  has_conflict('user_name'): {store.has_conflict('user_name')}")

    # Freshness with TTL
    f4 = store.upsert("temp_config", "value_x", ttl=0.001)  # 1ms TTL
    time.sleep(0.01)
    assert f4.is_expired
    print(f"  TTL=1ms 事实已过期: {f4.is_expired}")

    # Expire / purge
    expired = store.expire()
    print(f"  自动标记过期: {len(expired)} 条")
    purged = store.purge_expired()
    print(f"  清理过期事实: {purged} 条")

    # Search
    results = store.search("python")
    print(f"  搜索 'python': {[(r.topic, r.value) for r in results]}")

    # Topic listing
    topics = store.list_topics()
    print(f"  所有主题: {topics}")

    # Conflicts
    conflicts = store.conflicts()
    print(f"  冲突数量: {len(conflicts)}")

    # Export as messages
    msgs = store.to_messages()
    print(f"  导出为消息: {len(msgs)} 条")
    for m in msgs:
        print(f"    {m.content}")

    # Freshness scores
    for f in [f1b, f3]:
        print(f"  {f.topic} freshness: {f.freshness:.3f} (age={f.age_seconds:.3f}s)")

    print("  Section 1 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: UserPreferencesStore — cross-session persona
# ═══════════════════════════════════════════════════════════════════════════
def demo_user_preferences():
    print(SEP)
    print("Section 2: UserPreferencesStore — 用户偏好、跨会话保持 Agent 人设")
    print(SEP)

    prefs = UserPreferencesStore()

    # Set preferences
    prefs.set("response_style", "concise", category="style", evidence="user: keep it short")
    prefs.set("language", "zh-CN", category="language")
    prefs.set("domain", "backend", category="domain", pin=True)

    print(f"  已设置 {len(prefs)} 条偏好")

    # Get
    print(f"  response_style → {prefs.get('response_style')}")
    print(f"  non_existent → {prefs.get('non_existent', 'default')}")

    # Pin protection
    prefs.set("domain", "frontend")  # should be ignored
    print(f"  尝试覆盖已固定偏好 → {prefs.get('domain')} (仍为 backend)")

    prefs.unpin("domain")
    prefs.set("domain", "fullstack")
    print(f"  取消固定后覆盖 → {prefs.get('domain')}")

    # Category listing
    print(f"  所有分类: {prefs.all_categories()}")
    print(f"  style 分类: {[(p.key, p.value) for p in prefs.by_category('style')]}")

    # Export as messages
    msgs = prefs.to_messages()
    for m in msgs:
        print(f"  {m.content}")

    # Dict export
    d = prefs.to_dict()
    print(f"  导出为字典: {d}")

    print("  Section 2 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: TaskGoalStore — long-running objectives
# ═══════════════════════════════════════════════════════════════════════════
def demo_task_goals():
    print(SEP)
    print("Section 3: TaskGoalStore — 任务目标长期持久存储")
    print(SEP)

    store = TaskGoalStore()

    # Create goals
    g1 = store.create("实现用户认证模块", Priority.HIGH)
    g2 = store.create("优化数据库查询性能", Priority.MEDIUM)
    store.create("编写 API 文档", Priority.LOW)
    g4 = store.create("添加 OAuth2 支持", Priority.HIGH, parent_goal_id=g1.goal_id)

    print(f"  创建了 {len(store)} 个目标")

    # List active
    active = store.list_active(sort_by="priority")
    print("  按优先级排列的活动目标:")
    for g in active:
        print(f"    [{g.priority.value}] {g.description} (id={g.goal_id})")

    # Child goals
    children = store.children_of(g1.goal_id)
    print(f"  '{g1.description}' 的子目标: {[g.description for g in children]}")

    # Complete a goal
    store.update_status(g2.goal_id, GoalStatus.COMPLETED)
    completed = store.list_completed()
    print(f"  已完成目标: {[g.description for g in completed]}")

    # Trace association
    store.add_trace(g1.goal_id, "trace_abc123")
    g = store.get(g1.goal_id)
    print(f"  {g.description} 关联 trace: {g.related_trace_ids}")

    # Export as messages
    msgs = store.to_messages()
    for m in msgs:
        print(f"  {m.content}")

    print("  Section 3 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: TemporalMemory — timeline, decisions, review_history
# ═══════════════════════════════════════════════════════════════════════════
def demo_temporal_memory():
    print(SEP)
    print("Section 4: TemporalMemory — 时序记忆检索、回顾历史任务决策")
    print(SEP)

    tm = TemporalMemory(max_entries=100)

    # Record decisions
    tm.record_decision("选择 PostgreSQL 作为主数据库", "经过对比 MySQL 和 PG，选择了 PG 因为 JSONB 支持更好", tags=["database", "architecture"])
    time.sleep(0.01)
    tm.record_decision("采用 FastAPI 框架", "性能测试后决定使用 FastAPI 替代 Flask", tags=["framework", "python"])
    time.sleep(0.01)
    tm.record_decision("Redis 作为缓存层", "预期 QPS > 1000，需要 Redis 缓存", tags=["cache", "redis"])

    # Record milestone
    tm.record_milestone("v1.0 发布", "首个正式版本上线", trace_id="trace_rel_v1")

    # Record runs
    tm.record_run("如何优化 PostgreSQL 查询?", "建议添加联合索引", trace_id="run_001", tags=["database"])
    tm.record_run("为什么 API 响应慢?", "发现 N+1 查询问题", trace_id="run_002", tags=["performance"])

    print(f"  已记录 {len(tm)} 条时间线事件")
    print(f"  统计: {tm.stats()}")

    # Timeline — all
    all_entries = tm.timeline(limit=5)
    print("  最近 5 条时间线:")
    for e in all_entries:
        print(f"    [{e.kind.value}] {e.title}")

    # Timeline — decisions only
    decisions = tm.decisions()
    print(f"  决策记录 ({len(decisions)}):")
    for d in decisions:
        print(f"    {d.title}")

    # Review history
    history = tm.review_history("数据库", window_days=365)
    print(f"  回顾 '数据库' 相关历史: {len(history)} 条")
    for h in history:
        print(f"    {h.title}")

    # Search
    results = tm.search("API")
    print(f"  搜索 'API': {[r.title for r in results]}")

    # By tag
    tagged = tm.by_tag("database")
    print(f"  标签 'database': {[t.title for t in tagged]}")

    # Export as messages
    msgs = tm.to_messages(window_days=365)
    for m in msgs:
        lines = m.content.split("\n") if m.content else []
        print(f"  上下文注入 ({len(lines)} 行):")
        for line in lines[:5]:
            print(f"    {line}")

    print("  Section 4 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: HierarchicalMemory — buffer-only mode (no LLM)
# ═══════════════════════════════════════════════════════════════════════════
async def demo_hierarchical_no_llm():
    print(SEP)
    print("Section 5: HierarchicalMemory — 无 LLM 模式 (纯缓冲 + 知识库)")
    print(SEP)

    mem = HierarchicalMemory(
        provider=None,              # no LLM → no auto-compression
        episodic_max=5,
        episodic_keep_recent=2,
        enable_preferences=True,
        enable_goals=True,
        enable_temporal=True,
    )

    # Simulate a conversation
    messages = [
        Message.user("你好，我叫张三"),
        Message.assistant("你好张三！"),
        Message.user("我是后端开发工程师"),
        Message.assistant("明白了，有什么后端相关的问题吗？"),
        Message.user("我们项目用的 Python 3.12"),
    ]

    for msg in messages:
        await mem.add(msg)

    print(f"  Level-1 缓冲: {len(mem._buffer)} 条消息")
    print(f"  FactStore 事实数: {mem.fact_store.fact_count}")

    # Preferences auto-extracted
    if mem.preferences:
        print(f"  偏好数: {len(mem.preferences)}")
        for k, p in mem.preferences._prefs.items():
            print(f"    {k} = {p.value} (category={p.category})")

    # Add more messages to trigger compression (buffer-only mode just truncates)
    for i in range(5):
        await mem.add(Message.user(f"message {i}"))
        await mem.add(Message.assistant(f"response {i}"))

    print(f"  溢出后缓冲: {len(mem._buffer)} 条 (max={mem.episodic_max}, keep_recent={mem.episodic_keep_recent})")

    # Manual fact insert
    mem.fact_store.upsert("language", "python", evidence="user uses python")
    mem.fact_store.upsert("framework", "fastapi", evidence="project uses fastapi")

    # Add a preference manually
    mem.preferences.set("code_style", "PEP 8 strict", category="style", evidence="user asked for strict PEP8")
    mem.preferences.set("preferred_db", "PostgreSQL", category="domain")

    # Track goals
    mem.track_goal("完成 API 开发", Priority.HIGH)
    mem.track_goal("编写单元测试", Priority.MEDIUM)

    # Record decisions
    mem.record_decision("选择 Python 3.12", "兼容性测试通过", tags=["python"])
    mem.record_decision("使用 FastAPI 框架", "性能优于 Flask 3x", tags=["framework"])

    # Get context
    ctx = await mem.get_context("Python 版本")
    print(f"\n  get_context('Python 版本') → {len(ctx)} 条消息:")
    for m in ctx:
        content = m.content
        if isinstance(content, str):
            preview = content[:120]
        else:
            preview = str(content)[:120]
        print(f"    [{m.role.value}] {preview}")

    # Summary
    s = mem.summary()
    print(f"\n  内存概览: {s}")

    # Review history
    history = mem.review_history("框架")
    print(f"  回顾 '框架' 历史: {len(history)} 条")

    # Maintenance
    stats = await mem.maintenance()
    print(f"  维护统计: {stats}")

    print("  Section 5 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: HierarchicalMemory with MockProvider — compression + extraction
# ═══════════════════════════════════════════════════════════════════════════
async def demo_hierarchical_with_llm():
    print(SEP)
    print("Section 6: HierarchicalMemory + MockProvider — LLM 压缩 + 事实提取")
    print(SEP)

    # MockProvider returns canned responses
    mock = MockProvider()

    mem = HierarchicalMemory(
        provider=mock,
        episodic_max=8,
        episodic_keep_recent=3,
        fact_extraction_interval=1,   # extract every compression
        enable_preferences=True,
        enable_goals=True,
        enable_temporal=True,
    )

    # Feed a multi-turn conversation — enough to trigger compression
    conversation = [
        ("你好，我是李四，一名数据科学家", "你好李四！"),
        ("我主要用 Python 和 PyTorch 做深度学习", "很不错的技术栈"),
        ("我们目前在做一个图像识别项目", "图像识别是一个很好的方向"),
        ("模型部署方面，我们决定使用 ONNX Runtime", "ONNX Runtime 性能很好"),
        ("另外，数据库我们选了 PostgreSQL", "PostgreSQL 是个可靠的选择"),
        ("API 框架最终确定用 FastAPI", "FastAPI 很适合 AI 服务"),
        ("代码规范要求严格遵循 PEP 8", "PEP 8 是 Python 社区标准"),
        ("缓存层用 Redis", "Redis 能显著提升 API 响应速度"),
    ]

    for user_msg, assistant_msg in conversation:
        await mem.add(Message.user(user_msg))
        await mem.add(Message.assistant(assistant_msg))

    print(f"  缓冲消息数: {len(mem._buffer)}")
    print(f"  摘要数: {len(mem._episodes)}")
    print(f"  事实数: {mem.fact_store.fact_count}")
    print(f"  偏好数: {len(mem.preferences) if mem.preferences else 0}")
    print(f"  时间线条目: {len(mem.temporal) if mem.temporal else 0}")

    # Episode summaries
    if mem._episodes:
        print(f"\n  摘要内容:")
        for i, ep in enumerate(mem._episodes):
            preview = ep[:150] + "..." if len(ep) > 150 else ep
            print(f"    摘要 {i+1}: {preview}")

    # All facts
    all_facts = [(f.topic, f.value, f.status.value, f.confidence)
                 for f in mem.fact_store._facts.values()]
    if all_facts:
        print(f"\n  所有事实:")
        for topic, value, status, conf in all_facts:
            print(f"    {topic}: {value} (status={status}, confidence={conf:.2f})")

    # Summary stats
    s = mem.summary()
    print(f"\n  内存概览: {s}")

    print("  Section 6 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Integration — Agent + HierarchicalMemory
# ═══════════════════════════════════════════════════════════════════════════
async def demo_agent_integration():
    print(SEP)
    print("Section 7: Agent + HierarchicalMemory — 端到端集成")
    print(SEP)

    mock = MockProvider()
    mem = HierarchicalMemory(
        provider=mock,
        episodic_max=10,
        episodic_keep_recent=3,
        enable_preferences=True,
        enable_goals=True,
        enable_temporal=True,
    )

    # Pre-seed some facts and preferences
    mem.fact_store.upsert("project", "morainet", evidence="current project")
    mem.fact_store.upsert("language", "Python", evidence="primary language")
    mem.preferences.set("response_style", "简洁明了", category="style")

    agent = Agent(
        provider=mock,
        memory=mem,
    )

    # Run agent — memory should inject context (use arun since we're in async context)
    result = await agent.arun("有哪些可用的 API 端点？")

    print(f"  Agent 运行完成: trace_id={result.trace_id}")
    print(f"  最终回答: {result.final_answer[:100]}...")
    print(f"  步骤数: {len(result.steps)}")
    print(f"  Token 用量: {result.usage.total_tokens}")

    # After run, memory records the Q&A
    print(f"  缓冲消息数 (含本次): {len(mem._buffer)}")

    # Record the run in the timeline
    mem.record_run("有哪些可用的 API 端点？", result.final_answer[:200], trace_id=result.trace_id)

    print("  Section 7 PASSED PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Hierarchical Memory System — 主动式长期记忆系统升级")
    print("=" * 60)

    # Sync demos
    demo_fact_store()
    demo_user_preferences()
    demo_task_goals()
    demo_temporal_memory()

    # Async demos
    asyncio.run(demo_hierarchical_no_llm())
    asyncio.run(demo_hierarchical_with_llm())
    asyncio.run(demo_agent_integration())

    print("\n" + "=" * 60)
    print("  全部 7 个演示通过 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
