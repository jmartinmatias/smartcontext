"""
SmartContext Task Checklist (Recitation Pattern)

Implements the "recitation for attention control" pattern from Manus:
"Creating and updating a task checklist pushes the global plan
into Claude's recent attention window."

This solves the "lost in the middle" problem by ensuring the current
task context is always in the most recent tokens.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# Task States
# ============================================================================

class TaskStatus(Enum):
    """Status of a task in the checklist."""
    PENDING = "pending"         # Not started
    IN_PROGRESS = "in_progress" # Currently working on
    COMPLETED = "completed"     # Done
    BLOCKED = "blocked"         # Waiting on something
    SKIPPED = "skipped"         # Intentionally skipped


# ============================================================================
# Task Model
# ============================================================================

@dataclass
class Task:
    """A single task in the checklist."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    notes: str = ""
    subtasks: List["Task"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, notes: str = "") -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        if notes:
            self.notes = notes

    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS

    def block(self, reason: str = "") -> None:
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        if reason:
            self.notes = reason

    def skip(self, reason: str = "") -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED
        if reason:
            self.notes = reason

    def add_subtask(self, description: str) -> "Task":
        """Add a subtask."""
        subtask = Task(
            id=f"{self.id}.{len(self.subtasks) + 1}",
            description=description
        )
        self.subtasks.append(subtask)
        return subtask

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        task = cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {})
        )
        task.subtasks = [cls.from_dict(st) for st in data.get("subtasks", [])]
        return task


# ============================================================================
# Task Checklist
# ============================================================================

class TaskChecklist:
    """
    Manages a checklist of tasks for attention control.

    The checklist serves two purposes:
    1. Tracks progress on multi-step tasks
    2. When serialized, pushes the plan into recent context
       (solving "lost in the middle" problem)

    Usage:
        checklist = TaskChecklist()
        checklist.add_task("Implement feature X")
        checklist.add_task("Write tests for X")
        checklist.add_task("Update documentation")

        # Start working
        checklist.start_next()

        # Include in context to maintain focus
        context = checklist.get_recitation()
    """

    def __init__(self, title: str = "Task Checklist"):
        self.title = title
        self.tasks: List[Task] = []
        self.current_index: int = -1
        self.created_at = datetime.now().isoformat()

    def add_task(
        self,
        description: str,
        metadata: Dict[str, Any] = None
    ) -> Task:
        """Add a new task to the checklist."""
        task = Task(
            id=str(len(self.tasks) + 1),
            description=description,
            metadata=metadata or {}
        )
        self.tasks.append(task)

        # Auto-start first task
        if len(self.tasks) == 1:
            self.current_index = 0
            task.start()

        return task

    def add_tasks(self, descriptions: List[str]) -> List[Task]:
        """Add multiple tasks at once."""
        return [self.add_task(desc) for desc in descriptions]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
            for subtask in task.subtasks:
                if subtask.id == task_id:
                    return subtask
        return None

    def complete_task(self, task_id: str, notes: str = "") -> bool:
        """Mark a task as completed."""
        task = self.get_task(task_id)
        if task:
            task.complete(notes)
            return True
        return False

    def complete_current(self, notes: str = "") -> Optional[Task]:
        """Complete the current task and move to next."""
        if 0 <= self.current_index < len(self.tasks):
            current = self.tasks[self.current_index]
            current.complete(notes)
            return self.start_next()
        return None

    def start_next(self) -> Optional[Task]:
        """Start the next pending task."""
        for i, task in enumerate(self.tasks):
            if task.status == TaskStatus.PENDING:
                if self.current_index >= 0 and self.current_index < len(self.tasks):
                    # Mark current as completed if still in progress
                    current = self.tasks[self.current_index]
                    if current.status == TaskStatus.IN_PROGRESS:
                        current.complete()

                self.current_index = i
                task.start()
                return task
        return None

    def get_current_task(self) -> Optional[Task]:
        """Get the currently active task."""
        if 0 <= self.current_index < len(self.tasks):
            return self.tasks[self.current_index]
        return None

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    def get_progress(self) -> Dict[str, Any]:
        """Get progress statistics."""
        total = len(self.tasks)
        completed = len(self.get_completed_tasks())
        pending = len(self.get_pending_tasks())
        in_progress = len([t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS])
        blocked = len([t for t in self.tasks if t.status == TaskStatus.BLOCKED])

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "in_progress": in_progress,
            "blocked": blocked,
            "percent_complete": (completed / total * 100) if total > 0 else 0
        }

    def get_recitation(self, include_completed: bool = True) -> str:
        """
        Generate checklist string for context injection.

        This is the key method for attention control - including this
        in the recent context keeps the task plan in focus.
        """
        lines = [f"## {self.title}"]
        lines.append("")

        progress = self.get_progress()
        lines.append(f"Progress: {progress['completed']}/{progress['total']} ({progress['percent_complete']:.0f}%)")
        lines.append("")

        for i, task in enumerate(self.tasks):
            # Status indicators
            if task.status == TaskStatus.COMPLETED:
                status = "[x]"
                marker = ""
            elif task.status == TaskStatus.IN_PROGRESS:
                status = "[>]"
                marker = " <-- CURRENT"
            elif task.status == TaskStatus.BLOCKED:
                status = "[!]"
                marker = " (blocked)"
            elif task.status == TaskStatus.SKIPPED:
                status = "[-]"
                marker = " (skipped)"
            else:
                status = "[ ]"
                marker = ""

            # Skip completed if not requested
            if not include_completed and task.status == TaskStatus.COMPLETED:
                continue

            lines.append(f"{status} {task.id}. {task.description}{marker}")

            # Include subtasks
            for subtask in task.subtasks:
                st_status = "[x]" if subtask.status == TaskStatus.COMPLETED else "[ ]"
                lines.append(f"    {st_status} {subtask.id}. {subtask.description}")

            # Include notes if blocked
            if task.status == TaskStatus.BLOCKED and task.notes:
                lines.append(f"       Note: {task.notes}")

        return "\n".join(lines)

    def get_compact_recitation(self) -> str:
        """
        Generate compact checklist for minimal context usage.

        Only includes current and next tasks.
        """
        lines = [f"## {self.title}"]

        current = self.get_current_task()
        if current:
            lines.append(f"CURRENT: {current.description}")

        pending = self.get_pending_tasks()
        if pending and pending[0] != current:
            lines.append(f"NEXT: {pending[0].description}")

        progress = self.get_progress()
        lines.append(f"({progress['completed']}/{progress['total']} complete)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checklist to dictionary."""
        return {
            "title": self.title,
            "created_at": self.created_at,
            "current_index": self.current_index,
            "tasks": [t.to_dict() for t in self.tasks]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskChecklist":
        """Deserialize checklist from dictionary."""
        checklist = cls(title=data.get("title", "Task Checklist"))
        checklist.created_at = data.get("created_at", datetime.now().isoformat())
        checklist.current_index = data.get("current_index", -1)
        checklist.tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return checklist

    def clear(self) -> None:
        """Clear all tasks."""
        self.tasks = []
        self.current_index = -1
