"""
SmartContext Session System

OpenAI-compatible session protocol with trimming support.
Based on OpenAI Agents SDK session memory patterns.
"""

from dataclasses import dataclass, field
from typing import Protocol, List, Optional, Dict, Any, runtime_checkable
from datetime import datetime
from enum import Enum
import json


# ============================================================================
# Session Items
# ============================================================================

@dataclass
class SessionItem:
    """A single item in the session history."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionItem":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )

    def estimate_tokens(self) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(self.content) // 4


# ============================================================================
# Session Protocol (OpenAI-compatible)
# ============================================================================

@runtime_checkable
class SessionProtocol(Protocol):
    """
    Standard session interface compatible with OpenAI Agents SDK.

    Methods:
        get_items: Retrieve session history
        add_items: Add new items to session
        pop_item: Remove and return last item (undo)
        clear_session: Reset session
    """

    def get_items(self, limit: Optional[int] = None) -> List[SessionItem]:
        """Get session items, optionally limited to last N."""
        ...

    def add_items(self, items: List[SessionItem]) -> None:
        """Add items to session."""
        ...

    def pop_item(self) -> Optional[SessionItem]:
        """Remove and return the last item (undo operation)."""
        ...

    def clear_session(self) -> None:
        """Clear all session items."""
        ...


# ============================================================================
# Trimming Strategies
# ============================================================================

class TrimStrategy(Enum):
    """Available trimming strategies."""
    NONE = "none"                    # No trimming
    LAST_N = "last_n"               # Keep last N items
    FIRST_LAST = "first_last"       # Keep first M + last N
    TOKEN_BUDGET = "token_budget"   # Keep within token limit


@dataclass
class TrimConfig:
    """Configuration for session trimming."""
    strategy: TrimStrategy = TrimStrategy.FIRST_LAST
    max_items: int = 50             # Max items to keep
    keep_first: int = 2             # Items to preserve from start (context)
    token_budget: int = 4000        # Max tokens (for TOKEN_BUDGET strategy)

    def should_trim(self, items: List[SessionItem]) -> bool:
        """Check if trimming is needed."""
        if self.strategy == TrimStrategy.NONE:
            return False
        if self.strategy == TrimStrategy.TOKEN_BUDGET:
            total_tokens = sum(item.estimate_tokens() for item in items)
            return total_tokens > self.token_budget
        return len(items) > self.max_items


class TrimmingMixin:
    """
    Mixin that adds auto-trimming to session implementations.

    Trimming strategies:
    - NONE: Keep everything
    - LAST_N: Keep only the last N items
    - FIRST_LAST: Keep first M items (context) + last N items
    - TOKEN_BUDGET: Keep items that fit within token limit
    """

    trim_config: TrimConfig = TrimConfig()

    def trim_items(self, items: List[SessionItem]) -> List[SessionItem]:
        """Apply trimming strategy to items."""
        if not self.trim_config.should_trim(items):
            return items

        strategy = self.trim_config.strategy

        if strategy == TrimStrategy.NONE:
            return items

        elif strategy == TrimStrategy.LAST_N:
            return items[-self.trim_config.max_items:]

        elif strategy == TrimStrategy.FIRST_LAST:
            keep_first = self.trim_config.keep_first
            keep_last = self.trim_config.max_items - keep_first
            if len(items) <= self.trim_config.max_items:
                return items
            return items[:keep_first] + items[-keep_last:]

        elif strategy == TrimStrategy.TOKEN_BUDGET:
            # Keep items from end until budget exceeded
            result = []
            total_tokens = 0
            for item in reversed(items):
                item_tokens = item.estimate_tokens()
                if total_tokens + item_tokens > self.trim_config.token_budget:
                    break
                result.insert(0, item)
                total_tokens += item_tokens
            return result

        return items


# ============================================================================
# Base Session Implementation
# ============================================================================

class BaseSession(TrimmingMixin):
    """
    Base session implementation with in-memory storage.

    Provides:
    - OpenAI-compatible interface
    - Auto-trimming
    - Token estimation
    - Serialization
    """

    def __init__(
        self,
        session_id: str = "default",
        trim_config: Optional[TrimConfig] = None
    ):
        self.session_id = session_id
        self.trim_config = trim_config or TrimConfig()
        self._items: List[SessionItem] = []
        self._created_at = datetime.now().isoformat()

    def get_items(self, limit: Optional[int] = None) -> List[SessionItem]:
        """Get session items, optionally limited."""
        items = self._items
        if limit is not None:
            items = items[-limit:]
        return items

    def add_items(self, items: List[SessionItem]) -> None:
        """Add items and auto-trim if needed."""
        self._items.extend(items)
        self._items = self.trim_items(self._items)

    def add_item(self, role: str, content: str, metadata: Dict[str, Any] = None) -> SessionItem:
        """Convenience method to add a single item."""
        item = SessionItem(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.add_items([item])
        return item

    def pop_item(self) -> Optional[SessionItem]:
        """Remove and return the last item."""
        if self._items:
            return self._items.pop()
        return None

    def clear_session(self) -> None:
        """Clear all items."""
        self._items = []

    def estimate_total_tokens(self) -> int:
        """Estimate total tokens in session."""
        return sum(item.estimate_tokens() for item in self._items)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "item_count": len(self._items),
            "estimated_tokens": self.estimate_total_tokens(),
            "created_at": self._created_at,
            "trim_strategy": self.trim_config.strategy.value,
            "max_items": self.trim_config.max_items
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "trim_config": {
                "strategy": self.trim_config.strategy.value,
                "max_items": self.trim_config.max_items,
                "keep_first": self.trim_config.keep_first,
                "token_budget": self.trim_config.token_budget
            },
            "items": [item.to_dict() for item in self._items]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSession":
        """Deserialize session from dictionary."""
        trim_config = TrimConfig(
            strategy=TrimStrategy(data.get("trim_config", {}).get("strategy", "first_last")),
            max_items=data.get("trim_config", {}).get("max_items", 50),
            keep_first=data.get("trim_config", {}).get("keep_first", 2),
            token_budget=data.get("trim_config", {}).get("token_budget", 4000)
        )
        session = cls(
            session_id=data.get("session_id", "default"),
            trim_config=trim_config
        )
        session._created_at = data.get("created_at", datetime.now().isoformat())
        session._items = [
            SessionItem.from_dict(item) for item in data.get("items", [])
        ]
        return session


# ============================================================================
# Conversation Format Helpers
# ============================================================================

def items_to_messages(items: List[SessionItem]) -> List[Dict[str, str]]:
    """Convert session items to Claude API message format."""
    return [
        {"role": item.role, "content": item.content}
        for item in items
        if item.role in ("user", "assistant")
    ]


def messages_to_items(messages: List[Dict[str, str]]) -> List[SessionItem]:
    """Convert Claude API messages to session items."""
    return [
        SessionItem(role=msg["role"], content=msg["content"])
        for msg in messages
    ]


def format_as_transcript(items: List[SessionItem]) -> str:
    """Format session items as readable transcript."""
    lines = []
    for item in items:
        role_display = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
            "tool": "Tool"
        }.get(item.role, item.role.title())
        lines.append(f"[{role_display}]: {item.content}")
    return "\n\n".join(lines)
