"""
SmartContext Attention System

Manages attention allocation across different context types.
Provides policies for different task modes (coding, debugging, etc.)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class AttentionMode(Enum):
    """Available attention modes."""
    CODING = "coding"
    DEBUGGING = "debugging"
    EXPLORING = "exploring"
    PLANNING = "planning"
    BALANCED = "balanced"


@dataclass
class AttentionPolicy:
    """Defines how to allocate attention across context types."""
    name: str
    mode: AttentionMode
    description: str
    allocation: Dict[str, float]  # Category -> percentage (0.0 to 1.0)

    def get_tokens(self, category: str, total_budget: int) -> int:
        """Get token allocation for a category."""
        percentage = self.allocation.get(category, 0.0)
        return int(total_budget * percentage)

    def format_allocation(self, total_budget: int = 4000) -> str:
        """Format allocation as visual bars."""
        lines = []
        for category, percentage in sorted(self.allocation.items(),
                                           key=lambda x: x[1], reverse=True):
            tokens = self.get_tokens(category, total_budget)
            bar = "█" * int(percentage * 20) + "░" * (20 - int(percentage * 20))
            lines.append(f"  {category:12} {bar} {percentage:>5.0%} ({tokens:>4} tokens)")
        return "\n".join(lines)


# Pre-defined policies
POLICIES: Dict[AttentionMode, AttentionPolicy] = {
    AttentionMode.CODING: AttentionPolicy(
        name="Task-Focused",
        mode=AttentionMode.CODING,
        description="Focus on the current coding task",
        allocation={
            "goal": 0.40,
            "memories": 0.20,
            "observations": 0.15,
            "artifacts": 0.10,
            "strategy": 0.10,
            "system": 0.05
        }
    ),
    AttentionMode.DEBUGGING: AttentionPolicy(
        name="Debugging",
        mode=AttentionMode.DEBUGGING,
        description="Focus on errors and recent observations",
        allocation={
            "observations": 0.50,
            "memories": 0.20,
            "goal": 0.15,
            "artifacts": 0.10,
            "system": 0.05,
            "strategy": 0.00
        }
    ),
    AttentionMode.EXPLORING: AttentionPolicy(
        name="Exploration",
        mode=AttentionMode.EXPLORING,
        description="Focus on understanding and learning",
        allocation={
            "memories": 0.30,
            "artifacts": 0.25,
            "goal": 0.20,
            "observations": 0.15,
            "system": 0.05,
            "strategy": 0.05
        }
    ),
    AttentionMode.PLANNING: AttentionPolicy(
        name="Planning",
        mode=AttentionMode.PLANNING,
        description="Focus on architecture and design",
        allocation={
            "goal": 0.30,
            "memories": 0.25,
            "artifacts": 0.15,
            "strategy": 0.15,
            "observations": 0.10,
            "system": 0.05
        }
    ),
    AttentionMode.BALANCED: AttentionPolicy(
        name="Balanced",
        mode=AttentionMode.BALANCED,
        description="Even distribution across all categories",
        allocation={
            "goal": 0.20,
            "memories": 0.20,
            "observations": 0.20,
            "artifacts": 0.15,
            "strategy": 0.15,
            "system": 0.10
        }
    )
}


class AttentionManager:
    """
    Manages attention mode and auto-detection.
    """

    # Keywords for auto-detection
    MODE_KEYWORDS = {
        AttentionMode.DEBUGGING: [
            "bug", "error", "broken", "fix", "crash", "fail",
            "issue", "wrong", "not working", "exception", "stack trace"
        ],
        AttentionMode.PLANNING: [
            "plan", "architect", "design", "structure", "approach",
            "strategy", "how should", "what's the best way"
        ],
        AttentionMode.EXPLORING: [
            "what is", "how does", "explain", "understand", "explore",
            "show me", "tell me about", "where is", "find"
        ]
    }

    def __init__(self, default_mode: AttentionMode = AttentionMode.CODING):
        self.current_mode = default_mode
        self.auto_detect = True

    def get_policy(self) -> AttentionPolicy:
        """Get the current attention policy."""
        return POLICIES[self.current_mode]

    def set_mode(self, mode: AttentionMode):
        """Manually set the attention mode."""
        self.current_mode = mode

    def detect_mode(self, message: str) -> Optional[AttentionMode]:
        """
        Auto-detect the appropriate mode from message content.
        Returns None if no specific mode is detected.
        """
        message_lower = message.lower()

        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                return mode

        return None

    def maybe_switch_mode(self, message: str) -> bool:
        """
        Auto-switch mode if detected and auto_detect is enabled.
        Returns True if mode was switched.
        """
        if not self.auto_detect:
            return False

        detected = self.detect_mode(message)
        if detected and detected != self.current_mode:
            self.current_mode = detected
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current attention status."""
        policy = self.get_policy()
        return {
            "mode": self.current_mode.value,
            "policy_name": policy.name,
            "description": policy.description,
            "auto_detect": self.auto_detect,
            "allocation": policy.allocation
        }


@dataclass
class CompiledContext:
    """The result of context compilation."""
    mode: AttentionMode
    policy: AttentionPolicy
    sections: Dict[str, str]  # Category -> content
    token_usage: Dict[str, int]  # Category -> tokens used
    total_tokens: int
    budget: int

    def to_prompt(self) -> str:
        """Convert to a prompt string."""
        lines = []
        for category in ["system", "goal", "memories", "observations", "artifacts", "strategy"]:
            if category in self.sections and self.sections[category]:
                lines.append(f"[{category.upper()}]")
                lines.append(self.sections[category])
                lines.append("")
        return "\n".join(lines)

    def get_usage_summary(self) -> str:
        """Get a summary of token usage."""
        lines = [f"Token Usage: {self.total_tokens}/{self.budget}"]
        for category, tokens in sorted(self.token_usage.items(),
                                        key=lambda x: x[1], reverse=True):
            if tokens > 0:
                lines.append(f"  {category}: {tokens}")
        return "\n".join(lines)


class ContextCompiler:
    """
    Compiles optimal context from memory using attention policies.
    """

    def __init__(self, attention: AttentionManager):
        self.attention = attention

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(text) // 4

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def compile(
        self,
        goal: str = "",
        memories: List[str] = None,
        observations: List[str] = None,
        artifacts: List[str] = None,
        strategy: str = "",
        system: str = "",
        token_budget: int = 4000
    ) -> CompiledContext:
        """
        Compile context according to current attention policy.

        Args:
            goal: Current task/goal
            memories: Relevant memories from long-term storage
            observations: Recent observations/outputs
            artifacts: Referenced artifact summaries
            strategy: Approach/constraints
            system: System-level instructions
            token_budget: Total token budget

        Returns:
            CompiledContext with optimally allocated sections
        """
        policy = self.attention.get_policy()

        # Prepare raw content
        raw = {
            "goal": goal,
            "memories": "\n".join(memories or []),
            "observations": "\n".join(observations or []),
            "artifacts": "\n".join(artifacts or []),
            "strategy": strategy,
            "system": system
        }

        # Allocate tokens and truncate
        sections = {}
        token_usage = {}
        total_used = 0

        for category, content in raw.items():
            allocated = policy.get_tokens(category, token_budget)
            truncated = self.truncate_to_tokens(content, allocated)
            sections[category] = truncated
            used = self.estimate_tokens(truncated)
            token_usage[category] = used
            total_used += used

        return CompiledContext(
            mode=self.attention.current_mode,
            policy=policy,
            sections=sections,
            token_usage=token_usage,
            total_tokens=total_used,
            budget=token_budget
        )
