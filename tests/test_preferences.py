from __future__ import annotations

import time
from unittest.mock import patch

from morainet.memory.preferences import (
    GoalStatus,
    Preference,
    Priority,
    TaskGoal,
    TaskGoalStore,
    UserPreferencesStore,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_goal_status_enum():
    assert GoalStatus.ACTIVE.value == "active"
    assert GoalStatus.COMPLETED.value == "completed"
    assert GoalStatus.ABANDONED.value == "abandoned"
    assert GoalStatus.ON_HOLD.value == "on_hold"


def test_priority_enum():
    assert Priority.LOW.value == "low"
    assert Priority.MEDIUM.value == "medium"
    assert Priority.HIGH.value == "high"
    assert Priority.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# Preference
# ---------------------------------------------------------------------------

def test_preference_defaults():
    pref = Preference(key="style", value="concise")
    assert pref.key == "style"
    assert pref.value == "concise"
    assert pref.category == "general"
    assert pref.confidence == 1.0
    assert pref.pinned is False
    assert pref.evidence == ""
    assert isinstance(pref.created_at, float)
    assert isinstance(pref.updated_at, float)


def test_preference_custom():
    pref = Preference(key="language", value="zh", category="language", confidence=0.8, pinned=True)
    assert pref.pinned is True
    assert pref.category == "language"


# ---------------------------------------------------------------------------
# TaskGoal
# ---------------------------------------------------------------------------

def test_task_goal_defaults():
    goal = TaskGoal(goal_id="g1", description="Finish the report")
    assert goal.goal_id == "g1"
    assert goal.description == "Finish the report"
    assert goal.status == GoalStatus.ACTIVE
    assert goal.priority == Priority.MEDIUM
    assert goal.completed_at is None
    assert goal.parent_goal_id is None
    assert goal.related_trace_ids == []
    assert goal.notes == ""


# ---------------------------------------------------------------------------
# UserPreferencesStore
# ---------------------------------------------------------------------------

class TestUserPreferencesStore:
    def setup_method(self):
        self.store = UserPreferencesStore()

    def test_set_new_preference(self):
        pref = self.store.set("style", "concise")
        assert pref.key == "style"
        assert pref.value == "concise"
        assert len(self.store) == 1

    def test_set_overwrites_non_pinned(self):
        self.store.set("style", "concise")
        pref = self.store.set("style", "verbose")
        assert pref.value == "verbose"
        assert len(self.store) == 1

    def test_set_cannot_overwrite_pinned(self):
        self.store.set("style", "concise", pin=True)
        pref = self.store.set("style", "verbose")
        assert pref.value == "concise"
        assert pref.pinned is True

    def test_get_existing(self):
        self.store.set("style", "concise")
        assert self.store.get("style") == "concise"

    def test_get_nonexistent_with_default(self):
        assert self.store.get("missing", "default") == "default"

    def test_get_nonexistent_no_default(self):
        assert self.store.get("missing") is None

    def test_pin_existing(self):
        self.store.set("style", "concise")
        assert self.store.pin("style") is True
        assert self.store.get("style") == "concise"

    def test_pin_nonexistent(self):
        assert self.store.pin("missing") is False

    def test_unpin(self):
        self.store.set("style", "concise", pin=True)
        assert self.store.unpin("style") is True
        # Now can be overwritten
        pref = self.store.set("style", "verbose")
        assert pref.value == "verbose"

    def test_unpin_nonexistent(self):
        assert self.store.unpin("missing") is False

    def test_delete_existing(self):
        self.store.set("style", "concise")
        assert self.store.delete("style") is True
        assert len(self.store) == 0

    def test_delete_nonexistent(self):
        assert self.store.delete("missing") is False

    def test_by_category(self):
        self.store.set("style", "concise", category="style")
        self.store.set("tone", "formal", category="style")
        self.store.set("lang", "zh", category="language")
        results = self.store.by_category("style")
        assert len(results) == 2
        assert all(p.category == "style" for p in results)

    def test_all_categories(self):
        self.store.set("style", "concise", category="style")
        self.store.set("lang", "zh", category="language")
        self.store.set("tone", "formal", category="style")
        cats = self.store.all_categories()
        assert cats == ["language", "style"]

    def test_to_dict(self):
        self.store.set("style", "concise")
        self.store.set("lang", "zh")
        d = self.store.to_dict()
        assert d == {"style": "concise", "lang": "zh"}

    def test_to_messages_empty(self):
        messages = self.store.to_messages()
        assert messages == []

    def test_to_messages_with_preferences(self):
        self.store.set("style", "concise")
        messages = self.store.to_messages()
        assert len(messages) == 1
        assert "style" in messages[0].content

    def test_len(self):
        assert len(self.store) == 0
        self.store.set("a", 1)
        self.store.set("b", 2)
        assert len(self.store) == 2

    def test_contains(self):
        self.store.set("key", "value")
        assert "key" in self.store
        assert "missing" not in self.store


# ---------------------------------------------------------------------------
# TaskGoalStore
# ---------------------------------------------------------------------------

class TestTaskGoalStore:
    def setup_method(self):
        self.store = TaskGoalStore()

    def test_create_goal(self):
        goal = self.store.create("Write tests")
        assert goal.description == "Write tests"
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == Priority.MEDIUM
        assert goal.goal_id.startswith("goal_")
        assert len(self.store) == 1

    def test_create_goal_with_priority_and_parent(self):
        goal = self.store.create("Sub-task", priority=Priority.HIGH, parent_goal_id="parent_1")
        assert goal.priority == Priority.HIGH
        assert goal.parent_goal_id == "parent_1"

    def test_get_existing(self):
        goal = self.store.create("Task")
        retrieved = self.store.get(goal.goal_id)
        assert retrieved is goal

    def test_get_nonexistent(self):
        assert self.store.get("does_not_exist") is None

    def test_update_status(self):
        goal = self.store.create("Task")
        result = self.store.update_status(goal.goal_id, GoalStatus.ON_HOLD)
        assert result is True
        assert goal.status == GoalStatus.ON_HOLD

    def test_update_status_completed_sets_timestamp(self):
        goal = self.store.create("Task")
        self.store.update_status(goal.goal_id, GoalStatus.COMPLETED)
        assert goal.status == GoalStatus.COMPLETED
        assert goal.completed_at is not None

    def test_update_status_nonexistent(self):
        assert self.store.update_status("no_exist", GoalStatus.COMPLETED) is False

    def test_add_trace(self):
        goal = self.store.create("Task")
        result = self.store.add_trace(goal.goal_id, "trace_1")
        assert result is True
        assert "trace_1" in goal.related_trace_ids

    def test_add_trace_no_duplicate(self):
        goal = self.store.create("Task")
        self.store.add_trace(goal.goal_id, "trace_1")
        self.store.add_trace(goal.goal_id, "trace_1")
        assert len(goal.related_trace_ids) == 1

    def test_add_trace_nonexistent(self):
        assert self.store.add_trace("no_exist", "trace_1") is False

    def test_delete_existing(self):
        goal = self.store.create("Task")
        assert self.store.delete(goal.goal_id) is True
        assert len(self.store) == 0

    def test_delete_nonexistent(self):
        assert self.store.delete("no_exist") is False

    def test_list_active_sorted_by_priority(self):
        with patch.object(time, "time", side_effect=[1.0, 2.0, 3.0]):
            self.store.create("Low task", priority=Priority.LOW)
            self.store.create("Critical task", priority=Priority.CRITICAL)
            self.store.create("Medium task", priority=Priority.MEDIUM)
        active = self.store.list_active()
        priorities = [g.priority for g in active]
        assert priorities == [Priority.CRITICAL, Priority.MEDIUM, Priority.LOW]

    def test_list_active_sorted_by_created_at(self):
        self.store.create("First")
        self.store.create("Second")
        self.store.create("Third")
        active = self.store.list_active(sort_by="created_at")
        assert len(active) == 3
        timestamps = [g.created_at for g in active]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_completed_with_limit(self):
        with patch.object(time, "time", return_value=1.0):
            g1 = self.store.create("Task 1")
            g2 = self.store.create("Task 2")
            g3 = self.store.create("Task 3")
        self.store.update_status(g1.goal_id, GoalStatus.COMPLETED)
        self.store.update_status(g2.goal_id, GoalStatus.COMPLETED)
        self.store.update_status(g3.goal_id, GoalStatus.COMPLETED)
        completed = self.store.list_completed(limit=2)
        assert len(completed) == 2

    def test_list_completed_excludes_active(self):
        g1 = self.store.create("Task 1")
        self.store.create("Task 2")
        self.store.update_status(g1.goal_id, GoalStatus.COMPLETED)
        completed = self.store.list_completed()
        assert len(completed) == 1

    def test_children_of(self):
        parent = self.store.create("Parent")
        child = self.store.create("Child", parent_goal_id=parent.goal_id)
        children = self.store.children_of(parent.goal_id)
        assert len(children) == 1
        assert children[0].goal_id == child.goal_id

    def test_children_of_no_children(self):
        assert self.store.children_of("no_exist") == []

    def test_to_messages_empty(self):
        messages = self.store.to_messages()
        assert messages == []

    def test_to_messages_with_active_goals(self):
        self.store.create("Finish project")
        messages = self.store.to_messages()
        assert len(messages) == 1
        assert "Finish project" in messages[0].content
