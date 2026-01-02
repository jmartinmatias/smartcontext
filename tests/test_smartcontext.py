"""
SmartContext Test Suite

Tests for all modules:
- Session protocol and trimming
- Compression strategies
- Branching
- Cache optimization
- Task checklist
- Error memory
- Storage backends
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session import (
    SessionItem, BaseSession, TrimConfig, TrimStrategy
)
from compression import (
    ContextCompressor, CompressionConfig, CompressionStrategy,
    HeuristicCompressor
)
from branching import BranchManager, Branch
from cache import CacheOptimizer, CacheConfig, DiversityInjector
from checklist import TaskChecklist, Task, TaskStatus
from errors import ErrorMemory, ErrorEntry, ErrorCategory


# ============================================================================
# Session Tests
# ============================================================================

class TestSessionItem:
    def test_create_item(self):
        item = SessionItem(role="user", content="Hello")
        assert item.role == "user"
        assert item.content == "Hello"
        assert item.timestamp  # Should have default timestamp

    def test_estimate_tokens(self):
        item = SessionItem(role="user", content="a" * 100)
        # ~4 chars per token
        assert item.estimate_tokens() == 25

    def test_to_dict_and_back(self):
        item = SessionItem(role="user", content="Test", metadata={"key": "value"})
        data = item.to_dict()
        restored = SessionItem.from_dict(data)
        assert restored.role == item.role
        assert restored.content == item.content
        assert restored.metadata == item.metadata


class TestBaseSession:
    def test_add_and_get_items(self):
        session = BaseSession(session_id="test")
        session.add_item("user", "Hello")
        session.add_item("assistant", "Hi there")

        items = session.get_items()
        assert len(items) == 2
        assert items[0].role == "user"
        assert items[1].role == "assistant"

    def test_pop_item(self):
        session = BaseSession()
        session.add_item("user", "Hello")
        session.add_item("assistant", "Hi")

        popped = session.pop_item()
        assert popped.role == "assistant"
        assert len(session.get_items()) == 1

    def test_clear_session(self):
        session = BaseSession()
        session.add_item("user", "Hello")
        session.clear_session()
        assert len(session.get_items()) == 0

    def test_trimming_last_n(self):
        config = TrimConfig(strategy=TrimStrategy.LAST_N, max_items=3)
        session = BaseSession(trim_config=config)

        for i in range(5):
            session.add_item("user", f"Message {i}")

        items = session.get_items()
        assert len(items) == 3
        assert items[0].content == "Message 2"

    def test_trimming_first_last(self):
        config = TrimConfig(
            strategy=TrimStrategy.FIRST_LAST,
            max_items=4,
            keep_first=2
        )
        session = BaseSession(trim_config=config)

        for i in range(6):
            session.add_item("user", f"Message {i}")

        items = session.get_items()
        assert len(items) == 4
        # First 2 preserved
        assert items[0].content == "Message 0"
        assert items[1].content == "Message 1"
        # Last 2 preserved
        assert items[2].content == "Message 4"
        assert items[3].content == "Message 5"

    def test_get_stats(self):
        session = BaseSession(session_id="test-session")
        session.add_item("user", "Hello world")

        stats = session.get_stats()
        assert stats["session_id"] == "test-session"
        assert stats["item_count"] == 1
        assert "estimated_tokens" in stats


# ============================================================================
# Compression Tests
# ============================================================================

class TestHeuristicCompressor:
    def test_no_compression_for_small_input(self):
        compressor = HeuristicCompressor()
        config = CompressionConfig(min_items_to_compress=10)

        items = [SessionItem(role="user", content=f"Msg {i}") for i in range(5)]
        result = compressor.compress(items, config)

        assert result.strategy_used == CompressionStrategy.NONE
        assert result.original_items == 5
        assert result.compressed_items == 5

    def test_compression_reduces_items(self):
        compressor = HeuristicCompressor()
        config = CompressionConfig(
            min_items_to_compress=5,
            preserve_recent=2
        )

        # Create 10 items with substantial content
        items = [
            SessionItem(role="user", content=f"This is a longer message number {i} with some content")
            for i in range(10)
        ]
        result = compressor.compress(items, config)

        assert result.strategy_used == CompressionStrategy.HEURISTIC
        assert result.compressed_items < result.original_items
        assert len(result.preserved_items) == 2  # Last 2 preserved


class TestContextCompressor:
    def test_should_compress(self):
        compressor = ContextCompressor()
        config = CompressionConfig(
            min_items_to_compress=5,
            token_threshold=100
        )

        # Few items, low tokens - should not compress
        small_items = [SessionItem(role="user", content="Hi") for _ in range(3)]
        assert not compressor.should_compress(small_items, config)

        # Many items, high tokens - should compress
        large_items = [
            SessionItem(role="user", content="x" * 100)
            for _ in range(10)
        ]
        assert compressor.should_compress(large_items, config)


# ============================================================================
# Branching Tests
# ============================================================================

class TestBranchManager:
    def test_create_branch(self):
        manager = BranchManager()
        branch = manager.create_branch("feature", "Test feature branch")

        assert branch.name == "feature"
        assert branch.description == "Test feature branch"
        assert branch.parent_id == "main"

    def test_switch_branch(self):
        manager = BranchManager()
        manager.create_branch("feature")

        manager.switch_branch("feature")
        assert manager.current_branch_id == "feature"

        manager.switch_branch("main")
        assert manager.current_branch_id == "main"

    def test_delete_branch(self):
        manager = BranchManager()
        manager.create_branch("temp")

        assert manager.delete_branch("temp")
        assert "temp" not in manager.branches

    def test_cannot_delete_main(self):
        manager = BranchManager()
        with pytest.raises(ValueError):
            manager.delete_branch("main")

    def test_list_branches(self):
        manager = BranchManager()
        manager.create_branch("feature1")
        manager.create_branch("feature2")

        branches = manager.list_branches()
        names = [b.name for b in branches]

        assert "main" in names
        assert "feature1" in names
        assert "feature2" in names

    def test_branch_comparison(self):
        manager = BranchManager()

        # Add items to main
        manager.session.add_item("user", "Main message 1")
        manager.session.add_item("assistant", "Response 1")

        # Create branch and add different items
        manager.create_branch("feature")
        manager.switch_branch("feature")
        manager.session.add_item("user", "Feature message")

        comparison = manager.compare_branches("main", "feature")
        assert comparison["common_ancestor_items"] >= 0


# ============================================================================
# Cache Tests
# ============================================================================

class TestCacheOptimizer:
    def test_stable_prefix(self):
        optimizer = CacheOptimizer()
        optimizer.set_stable_prefix("System: You are a helpful assistant.")

        assert optimizer.get_stable_prefix() == "System: You are a helpful assistant."

    def test_cache_breakpoint(self):
        optimizer = CacheOptimizer()

        assert not optimizer.has_pending_breakpoint()
        optimizer.mark_cache_breakpoint()
        assert optimizer.has_pending_breakpoint()

    def test_deterministic_serialization(self):
        optimizer = CacheOptimizer()
        items = [
            SessionItem(role="user", content="Hello"),
            SessionItem(role="assistant", content="Hi there")
        ]

        result1 = optimizer.serialize_deterministic(items)
        result2 = optimizer.serialize_deterministic(items)

        # Should be identical (no timestamps in output)
        assert result1 == result2

    def test_stats_tracking(self):
        optimizer = CacheOptimizer()
        items = [SessionItem(role="user", content="Test")]

        optimizer.optimize_context(items)
        optimizer.optimize_context(items)

        stats = optimizer.get_stats()
        assert stats["stats"]["total_prompts"] == 2


class TestDiversityInjector:
    def test_template_rotation(self):
        injector = DiversityInjector()
        items = [SessionItem(role="user", content="Hello")]

        result1 = injector.serialize_with_diversity(items, rotate=True)
        result2 = injector.serialize_with_diversity(items, rotate=True)

        # Different templates should produce different output
        assert result1 != result2

    def test_stats(self):
        injector = DiversityInjector()
        items = [SessionItem(role="user", content="Hello")]

        injector.serialize_with_diversity(items)
        injector.serialize_with_diversity(items)

        stats = injector.get_stats()
        assert stats["rotations"] == 2


# ============================================================================
# Checklist Tests
# ============================================================================

class TestTaskChecklist:
    def test_add_task(self):
        checklist = TaskChecklist()
        task = checklist.add_task("Implement feature")

        assert task.description == "Implement feature"
        assert task.status == TaskStatus.IN_PROGRESS  # First task auto-starts

    def test_complete_task(self):
        checklist = TaskChecklist()
        task = checklist.add_task("Test task")

        checklist.complete_task(task.id)
        assert task.status == TaskStatus.COMPLETED

    def test_progress_tracking(self):
        checklist = TaskChecklist()
        checklist.add_task("Task 1")
        checklist.add_task("Task 2")
        checklist.add_task("Task 3")

        # Complete first task
        checklist.complete_task("1")

        progress = checklist.get_progress()
        assert progress["total"] == 3
        assert progress["completed"] == 1
        assert progress["percent_complete"] == pytest.approx(33.33, rel=0.1)

    def test_recitation_output(self):
        checklist = TaskChecklist(title="Test Tasks")
        checklist.add_task("First task")
        checklist.add_task("Second task")

        recitation = checklist.get_recitation()

        assert "Test Tasks" in recitation
        assert "First task" in recitation
        assert "Second task" in recitation

    def test_compact_recitation(self):
        checklist = TaskChecklist()
        checklist.add_task("Current task")
        checklist.add_task("Next task")

        compact = checklist.get_compact_recitation()

        assert "CURRENT" in compact
        assert "Current task" in compact

    def test_is_complete(self):
        checklist = TaskChecklist()
        t1 = checklist.add_task("Task 1")
        t2 = checklist.add_task("Task 2")

        assert not checklist.is_complete()

        checklist.complete_task(t1.id)
        checklist.complete_task(t2.id)

        assert checklist.is_complete()


# ============================================================================
# Error Memory Tests
# ============================================================================

class TestErrorMemory:
    def test_log_error(self):
        memory = ErrorMemory()
        entry = memory.log_error(
            action="Run tests",
            error="AssertionError: expected True",
            context="test_example.py:42"
        )

        assert entry.action == "Run tests"
        # "expected" keyword triggers VALIDATION category
        assert entry.category == ErrorCategory.VALIDATION  # Auto-detected

    def test_resolve_error(self):
        memory = ErrorMemory()
        entry = memory.log_error("Action", "Error")

        memory.resolve_error(entry.id, "Fixed by adding null check")

        assert entry.resolved
        assert entry.resolution == "Fixed by adding null check"

    def test_get_unresolved(self):
        memory = ErrorMemory()
        memory.log_error("Action 1", "Error 1")
        e2 = memory.log_error("Action 2", "Error 2")

        memory.resolve_error(e2.id)

        unresolved = memory.get_unresolved()
        assert len(unresolved) == 1

    def test_error_context_output(self):
        memory = ErrorMemory()
        memory.log_error("Run build", "SyntaxError: invalid syntax")

        context = memory.get_error_context()

        assert "Recent Errors" in context
        assert "Run build" in context
        assert "SyntaxError" in context

    def test_max_errors_limit(self):
        memory = ErrorMemory(max_errors=3)

        for i in range(5):
            memory.log_error(f"Action {i}", f"Error {i}")

        assert len(memory.errors) <= 3

    def test_auto_categorization(self):
        memory = ErrorMemory()

        # Syntax error
        e1 = memory.log_error("Parse", "SyntaxError: unexpected token")
        assert e1.category == ErrorCategory.SYNTAX

        # Network error
        e2 = memory.log_error("Fetch", "ConnectionError: refused")
        assert e2.category == ErrorCategory.NETWORK

        # Permission error
        e3 = memory.log_error("Write", "PermissionError: access denied")
        assert e3.category == ErrorCategory.PERMISSION


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    def test_session_with_compression(self):
        """Test session + compression working together."""
        session = BaseSession(
            trim_config=TrimConfig(strategy=TrimStrategy.LAST_N, max_items=20)
        )

        # Add items
        for i in range(15):
            session.add_item("user", f"Message {i} with some content")
            session.add_item("assistant", f"Response {i}")

        # Compress
        compressor = ContextCompressor()
        config = CompressionConfig(
            strategy=CompressionStrategy.HEURISTIC,
            min_items_to_compress=5
        )

        items = session.get_items()
        result = compressor.compress(items, config)

        assert result.tokens_saved > 0 or result.original_tokens == result.compressed_tokens

    def test_checklist_with_errors(self):
        """Test checklist + error memory working together."""
        checklist = TaskChecklist()
        errors = ErrorMemory()

        # Add tasks
        task = checklist.add_task("Run tests")

        # Simulate error
        errors.log_error("Run tests", "Test failed: assertion error")

        # Get combined context
        task_context = checklist.get_compact_recitation()
        error_context = errors.get_error_context()

        assert "Run tests" in task_context
        assert "Run tests" in error_context


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
