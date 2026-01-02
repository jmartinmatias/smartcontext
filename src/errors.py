"""
SmartContext Error Memory (Manus Pattern)

Implements error preservation for learning:
"Maintain failed actions and error traces in context.
This helps the model learn and avoid repeating mistakes."

Keeping errors in context is a form of "agentic learning" -
the model can see what went wrong and adjust.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import re


# ============================================================================
# Error Categories
# ============================================================================

class ErrorCategory(Enum):
    """Categories of errors for better organization."""
    SYNTAX = "syntax"           # Code syntax errors
    RUNTIME = "runtime"         # Runtime exceptions
    LOGIC = "logic"             # Logical errors
    TOOL = "tool"               # Tool/command failures
    NETWORK = "network"         # Network/API errors
    PERMISSION = "permission"   # Access/permission errors
    VALIDATION = "validation"   # Input validation errors
    UNKNOWN = "unknown"         # Uncategorized


# ============================================================================
# Error Entry
# ============================================================================

@dataclass
class ErrorEntry:
    """A recorded error with context."""
    id: str
    action: str                 # What was attempted
    error: str                  # Error message/traceback
    context: str                # Surrounding context
    category: ErrorCategory
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False
    resolution: str = ""        # How it was fixed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, resolution: str = "") -> None:
        """Mark error as resolved."""
        self.resolved = True
        self.resolution = resolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "error": self.error,
            "context": self.context,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            action=data["action"],
            error=data["error"],
            context=data.get("context", ""),
            category=ErrorCategory(data.get("category", "unknown")),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            resolved=data.get("resolved", False),
            resolution=data.get("resolution", ""),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# Error Memory
# ============================================================================

class ErrorMemory:
    """
    Preserves errors for learning and context.

    From Manus: "Don't sanitize errors from context. Keeping them
    helps the model avoid repeating the same mistakes."

    This class:
    1. Logs errors with full context
    2. Categorizes errors automatically
    3. Tracks resolutions
    4. Provides formatted context for injection
    """

    # Patterns for auto-categorization
    CATEGORY_PATTERNS = {
        ErrorCategory.SYNTAX: [
            r'SyntaxError', r'IndentationError', r'unexpected token',
            r'parsing error', r'invalid syntax'
        ],
        ErrorCategory.RUNTIME: [
            r'TypeError', r'ValueError', r'AttributeError',
            r'NameError', r'IndexError', r'KeyError',
            r'RuntimeError', r'Exception'
        ],
        ErrorCategory.NETWORK: [
            r'ConnectionError', r'TimeoutError', r'HTTPError',
            r'Network', r'API', r'request failed', r'ECONNREFUSED'
        ],
        ErrorCategory.PERMISSION: [
            r'PermissionError', r'AccessDenied', r'Forbidden',
            r'EACCES', r'permission denied', r'unauthorized'
        ],
        ErrorCategory.VALIDATION: [
            r'ValidationError', r'invalid', r'required field',
            r'must be', r'expected'
        ],
        ErrorCategory.TOOL: [
            r'command not found', r'tool failed', r'exit code',
            r'subprocess', r'command error'
        ],
    }

    def __init__(self, max_errors: int = 10):
        self.errors: List[ErrorEntry] = []
        self.max_errors = max_errors
        self._error_counter = 0

    def log_error(
        self,
        action: str,
        error: str,
        context: str = "",
        category: Optional[ErrorCategory] = None,
        metadata: Dict[str, Any] = None
    ) -> ErrorEntry:
        """
        Log a failed action with full trace.

        Args:
            action: What was attempted (e.g., "Run pytest")
            error: Error message or traceback
            context: Surrounding code/command context
            category: Error category (auto-detected if None)
            metadata: Additional metadata

        Returns:
            The created ErrorEntry
        """
        self._error_counter += 1

        # Auto-categorize if not specified
        if category is None:
            category = self._categorize_error(error)

        entry = ErrorEntry(
            id=f"err-{self._error_counter}",
            action=action,
            error=error,
            context=context,
            category=category,
            metadata=metadata or {}
        )

        self.errors.append(entry)

        # Trim old errors
        if len(self.errors) > self.max_errors:
            # Keep resolved errors longer (they're learning opportunities)
            unresolved = [e for e in self.errors if not e.resolved]
            resolved = [e for e in self.errors if e.resolved]

            if len(unresolved) > self.max_errors:
                unresolved = unresolved[-self.max_errors:]

            self.errors = resolved[-(self.max_errors // 2):] + unresolved

        return entry

    def resolve_error(self, error_id: str, resolution: str = "") -> bool:
        """Mark an error as resolved."""
        for error in self.errors:
            if error.id == error_id:
                error.resolve(resolution)
                return True
        return False

    def resolve_last(self, resolution: str = "") -> bool:
        """Resolve the most recent error."""
        if self.errors:
            self.errors[-1].resolve(resolution)
            return True
        return False

    def get_unresolved(self) -> List[ErrorEntry]:
        """Get all unresolved errors."""
        return [e for e in self.errors if not e.resolved]

    def get_by_category(self, category: ErrorCategory) -> List[ErrorEntry]:
        """Get errors by category."""
        return [e for e in self.errors if e.category == category]

    def get_recent(self, limit: int = 5) -> List[ErrorEntry]:
        """Get most recent errors."""
        return self.errors[-limit:]

    def get_error_context(self, include_resolved: bool = False) -> str:
        """
        Get errors formatted for context injection.

        This is the key method - including this in context helps
        the model learn from previous mistakes.
        """
        errors = self.errors if include_resolved else self.get_unresolved()

        if not errors:
            return ""

        lines = ["## Recent Errors (learn from these)"]
        lines.append("")

        for error in errors[-5:]:  # Last 5 errors
            status = "[RESOLVED]" if error.resolved else "[UNRESOLVED]"
            lines.append(f"### {status} {error.action}")
            lines.append(f"**Category**: {error.category.value}")
            lines.append(f"**Error**: {self._truncate(error.error, 200)}")

            if error.context:
                lines.append(f"**Context**: {self._truncate(error.context, 100)}")

            if error.resolved and error.resolution:
                lines.append(f"**Resolution**: {error.resolution}")

            lines.append("")

        return "\n".join(lines)

    def get_compact_context(self) -> str:
        """Get compact error context for minimal token usage."""
        unresolved = self.get_unresolved()

        if not unresolved:
            return ""

        lines = ["## Active Errors"]
        for error in unresolved[-3:]:
            lines.append(f"- {error.action}: {self._truncate(error.error, 50)}")

        return "\n".join(lines)

    def get_learning_summary(self) -> str:
        """
        Get summary of resolved errors for learning.

        This shows what went wrong and how it was fixed -
        valuable for preventing future mistakes.
        """
        resolved = [e for e in self.errors if e.resolved and e.resolution]

        if not resolved:
            return ""

        lines = ["## Lessons Learned"]
        lines.append("")

        for error in resolved[-5:]:
            lines.append(f"- **{error.action}**: {self._truncate(error.error, 50)}")
            lines.append(f"  Fix: {error.resolution}")
            lines.append("")

        return "\n".join(lines)

    def _categorize_error(self, error: str) -> ErrorCategory:
        """Auto-categorize error based on patterns."""
        error_lower = error.lower()

        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error, re.IGNORECASE):
                    return category

        return ErrorCategory.UNKNOWN

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def clear(self) -> None:
        """Clear all errors."""
        self.errors = []

    def clear_resolved(self) -> None:
        """Clear only resolved errors."""
        self.errors = [e for e in self.errors if not e.resolved]

    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        by_category = {}
        for cat in ErrorCategory:
            by_category[cat.value] = len(self.get_by_category(cat))

        return {
            "total": len(self.errors),
            "unresolved": len(self.get_unresolved()),
            "resolved": len([e for e in self.errors if e.resolved]),
            "by_category": by_category,
            "max_errors": self.max_errors
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "errors": [e.to_dict() for e in self.errors],
            "counter": self._error_counter,
            "max_errors": self.max_errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorMemory":
        """Deserialize from dictionary."""
        memory = cls(max_errors=data.get("max_errors", 10))
        memory._error_counter = data.get("counter", 0)
        memory.errors = [ErrorEntry.from_dict(e) for e in data.get("errors", [])]
        return memory
