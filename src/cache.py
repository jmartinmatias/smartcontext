"""
SmartContext KV-Cache Optimization

Optimizes context for maximum KV-cache hit rate.
Based on Manus blog post: "KV-cache hit rate is the single most
important metric for a production AI agent."

Key principles:
- Stable prefixes: Don't inject timestamps at start
- Append-only context: Only add, never reorder
- Deterministic serialization: Consistent output format
- Explicit cache breakpoints: Mark when cache should reset
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import hashlib
import json

from session import SessionItem


# ============================================================================
# Cache Configuration
# ============================================================================

class CacheStrategy(Enum):
    """Cache optimization strategies."""
    NONE = "none"               # No optimization
    STABLE_PREFIX = "stable"    # Optimize prefix stability
    APPEND_ONLY = "append"      # Strict append-only
    FULL = "full"               # All optimizations


@dataclass
class CacheConfig:
    """Configuration for cache optimization."""
    strategy: CacheStrategy = CacheStrategy.FULL
    prefix_tokens: int = 500        # Tokens reserved for stable prefix
    enable_breakpoints: bool = True # Allow explicit cache breaks
    track_stats: bool = True        # Track cache statistics


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    total_prompts: int = 0
    prefix_changes: int = 0
    reorders: int = 0
    breakpoints: int = 0
    estimated_hit_rate: float = 1.0

    def record_prompt(self, prefix_changed: bool = False, reordered: bool = False):
        """Record a prompt and update stats."""
        self.total_prompts += 1
        if prefix_changed:
            self.prefix_changes += 1
        if reordered:
            self.reorders += 1

        # Estimate hit rate based on prefix stability
        if self.total_prompts > 0:
            disruptions = self.prefix_changes + self.reorders + self.breakpoints
            self.estimated_hit_rate = max(0, 1 - (disruptions / self.total_prompts))

    def record_breakpoint(self):
        """Record an explicit cache breakpoint."""
        self.breakpoints += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_prompts": self.total_prompts,
            "prefix_changes": self.prefix_changes,
            "reorders": self.reorders,
            "breakpoints": self.breakpoints,
            "estimated_hit_rate": f"{self.estimated_hit_rate:.2%}"
        }


# ============================================================================
# Cache Optimizer
# ============================================================================

class CacheOptimizer:
    """
    Optimizes context for KV-cache hit rate.

    The key insight from Manus is that LLM inference is dominated by
    KV-cache hits. Every time the prefix changes, the cache is invalidated.

    This optimizer:
    1. Maintains a stable prefix that never changes during a session
    2. Ensures items are only appended, never reordered
    3. Uses deterministic serialization (no timestamps in keys)
    4. Provides explicit cache breakpoints for intentional resets
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()

        # Stable prefix components
        self._stable_prefix: Optional[str] = None
        self._prefix_hash: Optional[str] = None

        # Content tracking for append-only verification
        self._content_hashes: List[str] = []

        # Breakpoint tracking
        self._breakpoint_pending = False

    def set_stable_prefix(self, prefix: str) -> None:
        """
        Set the immutable prefix for this session.

        This prefix should contain:
        - System instructions
        - Role definitions
        - Constant context (project info, etc.)

        Do NOT include:
        - Timestamps
        - Session IDs that change
        - Dynamic content
        """
        self._stable_prefix = prefix
        self._prefix_hash = self._hash(prefix)

    def get_stable_prefix(self) -> Optional[str]:
        """Get the current stable prefix."""
        return self._stable_prefix

    def mark_cache_breakpoint(self) -> None:
        """
        Explicitly mark that the cache should reset.

        Use this when:
        - Switching to a completely different task
        - After a major error recovery
        - When context needs full refresh
        """
        self._breakpoint_pending = True
        self.stats.record_breakpoint()

    def clear_breakpoint(self) -> None:
        """Clear pending breakpoint after it's been handled."""
        self._breakpoint_pending = False

    def has_pending_breakpoint(self) -> bool:
        """Check if a cache breakpoint is pending."""
        return self._breakpoint_pending

    def serialize_deterministic(
        self,
        items: List[SessionItem],
        include_prefix: bool = True
    ) -> str:
        """
        Serialize items in a deterministic way for cache optimization.

        Key principles:
        - No timestamps in output (they cause cache misses)
        - Consistent ordering
        - Stable formatting
        """
        parts = []

        # Add stable prefix first (if set and requested)
        if include_prefix and self._stable_prefix:
            parts.append(self._stable_prefix)
            parts.append("")  # Separator

        # Serialize items without timestamps
        for item in items:
            serialized = self._serialize_item(item)
            parts.append(serialized)

        return "\n".join(parts)

    def _serialize_item(self, item: SessionItem) -> str:
        """Serialize a single item deterministically."""
        # Role mapping for consistent output
        role_display = {
            "user": "Human",
            "assistant": "Assistant",
            "system": "System",
            "tool": "Tool"
        }.get(item.role, item.role.title())

        # Don't include timestamp in output
        return f"[{role_display}]\n{item.content}"

    def verify_append_only(self, items: List[SessionItem]) -> bool:
        """
        Verify that items are append-only (no reordering).

        Returns True if the order is preserved.
        """
        new_hashes = [self._hash(item.content) for item in items]

        # Check if existing hashes are preserved in order
        if len(new_hashes) < len(self._content_hashes):
            # Items were removed - not append-only
            return False

        for i, old_hash in enumerate(self._content_hashes):
            if i >= len(new_hashes) or new_hashes[i] != old_hash:
                # Order changed
                return False

        # Update tracked hashes
        self._content_hashes = new_hashes
        return True

    def optimize_context(
        self,
        items: List[SessionItem],
        include_prefix: bool = True
    ) -> str:
        """
        Optimize and serialize context for maximum cache hits.

        Returns the optimized context string.
        """
        # Check for prefix changes
        prefix_changed = False
        if include_prefix and self._stable_prefix:
            current_hash = self._hash(self._stable_prefix)
            if self._prefix_hash and current_hash != self._prefix_hash:
                prefix_changed = True
            self._prefix_hash = current_hash

        # Check for reordering
        reordered = not self.verify_append_only(items)

        # Record stats
        self.stats.record_prompt(prefix_changed, reordered)

        # Handle breakpoint
        if self._breakpoint_pending:
            self._content_hashes = [self._hash(item.content) for item in items]
            self._breakpoint_pending = False

        return self.serialize_deterministic(items, include_prefix)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache optimization statistics."""
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "prefix_tokens": self.config.prefix_tokens,
                "enable_breakpoints": self.config.enable_breakpoints
            },
            "stats": self.stats.to_dict(),
            "has_prefix": self._stable_prefix is not None,
            "tracked_items": len(self._content_hashes),
            "breakpoint_pending": self._breakpoint_pending
        }

    def _hash(self, content: str) -> str:
        """Create a hash for content tracking."""
        return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================================
# Diversity Injection (Manus Pattern)
# ============================================================================

class DiversityInjector:
    """
    Injects structural variation to break pattern matching.

    From Manus: "Batch tasks benefit from diversity injection to
    prevent rhythmic repetition in Claude's outputs."

    This provides multiple serialization templates that rotate
    to create variation while maintaining semantic equivalence.
    """

    TEMPLATES = [
        # Template 1: Standard format
        lambda item: f"[{item.role.title()}]: {item.content}",

        # Template 2: Markdown headers
        lambda item: f"### {item.role.title()}\n{item.content}",

        # Template 3: XML-like
        lambda item: f"<{item.role}>\n{item.content}\n</{item.role}>",

        # Template 4: Minimal
        lambda item: f"{item.role.upper()}: {item.content}",
    ]

    def __init__(self):
        self._template_index = 0
        self._rotation_count = 0

    def serialize_with_diversity(
        self,
        items: List[SessionItem],
        rotate: bool = True
    ) -> str:
        """
        Serialize items with structural variation.

        Args:
            items: Items to serialize
            rotate: Whether to rotate template on each call
        """
        template = self.TEMPLATES[self._template_index]

        result = []
        for item in items:
            result.append(template(item))

        if rotate:
            self._rotate_template()

        return "\n\n".join(result)

    def _rotate_template(self):
        """Rotate to next template."""
        self._template_index = (self._template_index + 1) % len(self.TEMPLATES)
        self._rotation_count += 1

    def reset(self):
        """Reset to first template."""
        self._template_index = 0

    def get_current_template_index(self) -> int:
        """Get current template index."""
        return self._template_index

    def get_stats(self) -> Dict[str, Any]:
        """Get diversity injection statistics."""
        return {
            "current_template": self._template_index,
            "template_count": len(self.TEMPLATES),
            "rotations": self._rotation_count
        }
