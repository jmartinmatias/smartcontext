#!/usr/bin/env python3
"""
SmartContext - Intelligent Context Engineering for AI Assistants

A unified MCP server that provides:
- Tiered memory (working/session/long-term/artifacts)
- Attention management with auto-detection
- Session trimming and compression
- Conversation branching
- Cache optimization (Manus patterns)
- Task checklists (attention control)
- Error preservation (learning)
- Transparency modes for debugging

Usage:
    python smartcontext.py [--namespace NAME] [--storage-dir PATH] [--backend json|sqlite]

The server exposes tools that Claude Code can use automatically,
or users can invoke directly with commands like:
    /remember, /forget, /recall, /mode, /transparency
"""

import argparse
from pathlib import Path
from typing import Optional, List
from enum import Enum

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

from memory import MemoryStore, Memory
from attention import AttentionManager, AttentionMode, ContextCompiler
from branching import BranchManager, Branch
from session import BaseSession, TrimConfig


# ============================================================================
# Transparency Levels
# ============================================================================

class TransparencyLevel(Enum):
    OFF = "off"          # Invisible (default)
    MINIMAL = "minimal"  # Brief summary
    NORMAL = "normal"    # Show key details
    FULL = "full"        # Show everything (debug)


# ============================================================================
# SmartContext Server
# ============================================================================

class SmartContext:
    """
    The main SmartContext engine.
    Coordinates memory, attention, branching, and transparency.
    """

    def __init__(
        self,
        namespace: str = "default",
        storage_dir: Path = None,
        storage_backend: str = "json"
    ):
        self.memory = MemoryStore(
            storage_dir,
            namespace,
            storage_backend=storage_backend
        )
        self.attention = AttentionManager()
        self.compiler = ContextCompiler(self.attention)
        self.transparency = TransparencyLevel.OFF

        # Branch manager for conversation exploration
        self._branch_manager: Optional[BranchManager] = None

    @property
    def branch_manager(self) -> BranchManager:
        """Lazy-load branch manager."""
        if self._branch_manager is None:
            self._branch_manager = BranchManager()
        return self._branch_manager

    def prepare_context(self, user_message: str, token_budget: int = 4000) -> dict:
        """
        Prepare optimal context for a user message.
        This is the main "magic" function.

        1. Auto-detect and maybe switch attention mode
        2. Search relevant memories
        3. Compile context according to attention policy
        4. Return context + transparency info
        """
        # Auto-detect mode
        mode_switched = self.attention.maybe_switch_mode(user_message)

        # Search relevant memories
        relevant_memories = self.memory.search(user_message, limit=5)
        memory_contents = [m.content for m in relevant_memories]

        # Get recent session observations
        recent_events = self.memory.get_session_log(limit=5)
        observations = [
            f"{e['type']}: {e['data']}" for e in recent_events if e.get('data')
        ]

        # Get artifact references (just names, not content)
        artifacts = [f"#{a.id}: {a.name}" for a in self.memory.list_artifacts()[:5]]

        # Get working context
        working = self.memory.get_working_summary(max_tokens=200)

        # Get checklist for attention control
        checklist = self.memory.get_checklist(compact=True)
        if checklist:
            observations.append(checklist)

        # Get error context for learning
        errors = self.memory.get_errors(include_resolved=False)
        if errors:
            observations.append(errors)

        # Compile context
        compiled = self.compiler.compile(
            goal=working or user_message[:200],
            memories=memory_contents,
            observations=observations,
            artifacts=artifacts,
            token_budget=token_budget
        )

        # Log this interaction
        self.memory.log_event("message", {"content": user_message[:100]})

        # Update working context
        self.memory.set_working("last_message", user_message[:100])
        self.memory.set_working("current_mode", self.attention.current_mode.value)

        return {
            "compiled_context": compiled.to_prompt(),
            "mode": self.attention.current_mode.value,
            "mode_switched": mode_switched,
            "memories_used": len(relevant_memories),
            "token_usage": compiled.total_tokens,
            "token_budget": token_budget,
            "transparency": self._format_transparency(
                mode_switched, relevant_memories, compiled
            )
        }

    def _format_transparency(
        self,
        mode_switched: bool,
        memories: List[Memory],
        compiled
    ) -> str:
        """Format transparency output based on current level."""
        if self.transparency == TransparencyLevel.OFF:
            return ""

        lines = []

        if self.transparency in [TransparencyLevel.MINIMAL, TransparencyLevel.NORMAL, TransparencyLevel.FULL]:
            lines.append(f"📎 {len(memories)} memories loaded")
            if mode_switched:
                lines.append(f"🔄 Mode switched to {self.attention.current_mode.value}")

        if self.transparency in [TransparencyLevel.NORMAL, TransparencyLevel.FULL]:
            lines.append(f"🎯 Mode: {self.attention.get_policy().name}")
            for mem in memories[:3]:
                lines.append(f"   • {mem.content[:50]}...")

        if self.transparency == TransparencyLevel.FULL:
            lines.append(f"📊 Tokens: {compiled.total_tokens}/{compiled.budget}")
            lines.append(compiled.get_usage_summary())

        if lines:
            border = "─" * 50
            header = f"┌─ SMARTCONTEXT {border[:35]}┐"
            footer = f"└{border}┘"
            body = "\n".join(f"│ {line:<49}│" for line in lines)
            return f"\n{header}\n{body}\n{footer}\n"

        return ""


# ============================================================================
# MCP Server
# ============================================================================

# Initialize
mcp = FastMCP("smartcontext")
ctx: SmartContext = None  # Initialized in main()


# ============================================================================
# Memory Tools
# ============================================================================

@mcp.tool()
def remember(content: str, tags: str = "") -> str:
    """
    Save something to long-term memory.

    Args:
        content: What to remember
        tags: Comma-separated tags (optional)

    Example: remember("Project uses PostgreSQL", "database,tech-stack")
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    memory = ctx.memory.remember(content, tags=tag_list)
    return f"✓ Remembered: \"{content[:50]}{'...' if len(content) > 50 else ''}\" (ID: {memory.id})"


@mcp.tool()
def forget(query: str) -> str:
    """
    Remove something from memory.

    Args:
        query: Search query to find what to forget
    """
    matches = ctx.memory.search(query, limit=1)
    if not matches:
        return f"No memories found matching: \"{query}\""

    memory = matches[0]
    ctx.memory.forget(memory.id)
    return f"✓ Forgot: \"{memory.content[:50]}...\""


@mcp.tool()
def recall(topic: str, limit: int = 5) -> str:
    """
    Search memories for a topic.

    Args:
        topic: What to search for
        limit: Maximum results (default 5)
    """
    matches = ctx.memory.search(topic, limit=limit)

    if not matches:
        return f"No memories found for: \"{topic}\""

    lines = [f"Found {len(matches)} memories for \"{topic}\":\n"]
    for i, mem in enumerate(matches, 1):
        lines.append(f"{i}. {mem.content}")
        if mem.tags:
            lines.append(f"   Tags: {', '.join(mem.tags)}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Attention Tools
# ============================================================================

@mcp.tool()
def set_mode(mode: str) -> str:
    """
    Set the attention mode.

    Args:
        mode: One of 'coding', 'debugging', 'exploring', 'planning', 'balanced'
    """
    try:
        attention_mode = AttentionMode(mode)
    except ValueError:
        modes = [m.value for m in AttentionMode]
        return f"Unknown mode: {mode}. Available: {', '.join(modes)}"

    ctx.attention.set_mode(attention_mode)
    policy = ctx.attention.get_policy()

    lines = [
        f"✓ Switched to {policy.name} mode",
        f"  {policy.description}",
        "",
        "Attention allocation:",
        policy.format_allocation()
    ]
    return "\n".join(lines)


@mcp.tool()
def set_transparency(level: str) -> str:
    """
    Set transparency level.

    Args:
        level: One of 'off', 'minimal', 'normal', 'full'
    """
    try:
        ctx.transparency = TransparencyLevel(level)
    except ValueError:
        levels = [l.value for l in TransparencyLevel]
        return f"Unknown level: {level}. Available: {', '.join(levels)}"

    descriptions = {
        "off": "Context engineering is invisible (default)",
        "minimal": "Shows brief summary (e.g., '3 memories loaded')",
        "normal": "Shows memory names and current mode",
        "full": "Shows everything (debug mode)"
    }
    return f"✓ Transparency: {level}\n  {descriptions[level]}"


@mcp.tool()
def status() -> str:
    """Show current SmartContext status."""
    stats = ctx.memory.stats()
    attention = ctx.attention.get_status()

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║              SMARTCONTEXT STATUS                  ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Namespace: {stats['namespace']:<36} ║",
        f"║  Mode: {attention['mode']:<12} Transparency: {ctx.transparency.value:<10} ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Long-term memories: {stats['long_term_memories']:<4}                        ║",
        f"║  Artifacts stored:   {stats['artifacts']:<4}                        ║",
        f"║  Session items:      {stats.get('session_items', 0):<4}                        ║",
        f"║  Working context:    {stats['working_context_keys']:<4} keys                   ║",
        "╠══════════════════════════════════════════════════╣",
        f"║  Storage backend:    {stats.get('storage_backend', 'json'):<26} ║",
        f"║  Errors logged:      {stats.get('errors_logged', 0):<4}                        ║",
        f"║  Checklist tasks:    {stats.get('checklist_tasks', 0):<4}                        ║",
        "╚══════════════════════════════════════════════════╝"
    ]
    return "\n".join(lines)


@mcp.tool()
def prepare(message: str, token_budget: int = 4000) -> str:
    """
    Prepare optimal context for a message.
    This is called automatically before responses.

    Args:
        message: The user's message
        token_budget: Available tokens (default 4000)
    """
    result = ctx.prepare_context(message, token_budget)

    output = []
    if result["transparency"]:
        output.append(result["transparency"])

    if result["compiled_context"]:
        output.append("[Relevant Context]")
        output.append(result["compiled_context"])

    return "\n".join(output) if output else "(No additional context)"


# ============================================================================
# Artifact Tools
# ============================================================================

@mcp.tool()
def store_artifact(name: str, content: str) -> str:
    """
    Store large content as an artifact.

    Args:
        name: Name for the artifact (e.g., 'api-routes.ts')
        content: The content to store
    """
    artifact = ctx.memory.store_artifact(name, content)
    return f"✓ Stored: {name} (#{artifact.id}, {artifact.size} chars)"


@mcp.tool()
def get_artifact(name_or_id: str) -> str:
    """
    Retrieve an artifact by name or ID.

    Args:
        name_or_id: Artifact name or ID
    """
    artifact = ctx.memory.get_artifact(name_or_id.lstrip("#"))
    if not artifact:
        return f"Artifact not found: {name_or_id}"

    return f"=== {artifact.name} ===\n\n{artifact.content}"


@mcp.tool()
def list_artifacts() -> str:
    """List all stored artifacts."""
    artifacts = ctx.memory.list_artifacts()
    if not artifacts:
        return "No artifacts stored."

    lines = ["Stored Artifacts:", ""]
    for a in artifacts:
        lines.append(f"  #{a.id}: {a.name} ({a.size} chars)")

    return "\n".join(lines)


# ============================================================================
# Session Tools
# ============================================================================

@mcp.tool()
def end_session(summary: str = "") -> str:
    """
    End session and save summary to long-term memory.

    Args:
        summary: Optional session summary
    """
    if not summary:
        # Auto-generate from session log
        events = ctx.memory.get_session_log(limit=10)
        if events:
            summary = f"Session with {len(events)} events"
        else:
            summary = "Empty session"

    memory = ctx.memory.end_session(summary)
    return f"✓ Session ended.\n  Summary saved: \"{summary[:60]}...\""


@mcp.tool()
def set_goal(goal: str) -> str:
    """
    Set the current working goal.

    Args:
        goal: What you're working on
    """
    ctx.memory.set_working("current_goal", goal)
    ctx.memory.log_event("goal_set", {"goal": goal})
    return f"✓ Goal set: {goal}"


@mcp.tool()
def get_goal() -> str:
    """Get the current working goal."""
    goal = ctx.memory.get_working("current_goal")
    if goal:
        return f"Current goal: {goal}"
    return "No goal set. Use set_goal() to set one."


@mcp.tool()
def set_max_turns(n: int) -> str:
    """
    Set maximum session turns (for trimming).

    Args:
        n: Maximum number of turns to keep
    """
    ctx.memory.set_max_turns(n)
    return f"✓ Max session turns set to {n}"


@mcp.tool()
def get_session_stats() -> str:
    """Get session statistics."""
    stats = ctx.memory.get_session_stats()
    lines = [
        "Session Statistics:",
        f"  Items: {stats['item_count']}",
        f"  Tokens (est): {stats['estimated_tokens']}",
        f"  Trim strategy: {stats['trim_strategy']}",
        f"  Max items: {stats['max_items']}"
    ]
    return "\n".join(lines)


@mcp.tool()
def pop_last() -> str:
    """Remove the last session item (undo)."""
    item = ctx.memory.pop_session_item()
    if item:
        return f"✓ Removed: [{item.role}] {item.content[:50]}..."
    return "No items to remove"


# ============================================================================
# Compression Tools
# ============================================================================

@mcp.tool()
def compress_context(strategy: str = "hybrid") -> str:
    """
    Compress session context to save tokens.

    Args:
        strategy: 'none', 'heuristic', 'llm', or 'hybrid'
    """
    result = ctx.memory.compress_session(strategy)

    lines = [
        f"✓ Context compressed",
        f"  Strategy: {result.strategy_used.value}",
        f"  Original: {result.original_items} items ({result.original_tokens} tokens)",
        f"  Compressed: {result.compressed_items} items ({result.compressed_tokens} tokens)",
        f"  Saved: {result.tokens_saved} tokens ({(1 - result.compression_ratio) * 100:.1f}%)"
    ]
    return "\n".join(lines)


@mcp.tool()
def set_compression_strategy(strategy: str) -> str:
    """
    Set the compression strategy.

    Args:
        strategy: 'none', 'heuristic', 'llm', or 'hybrid'
    """
    ctx.memory.set_compression_strategy(strategy)
    return f"✓ Compression strategy set to: {strategy}"


@mcp.tool()
def get_compression_stats() -> str:
    """Get compression statistics."""
    stats = ctx.memory.get_compression_stats()
    lines = [
        "Compression Statistics:",
        f"  Current items: {stats['current_items']}",
        f"  Current tokens: {stats['current_tokens']}",
        f"  Strategy: {stats['strategy']}",
        f"  Should compress: {stats['should_compress']}",
        f"  Threshold: {stats['threshold']} tokens"
    ]
    return "\n".join(lines)


# ============================================================================
# Branching Tools
# ============================================================================

@mcp.tool()
def branch_session(name: str, description: str = "") -> str:
    """
    Fork current session into a new branch.

    Args:
        name: Branch name
        description: Optional description
    """
    try:
        branch = ctx.branch_manager.create_branch(name, description)
        return f"✓ Created branch: {name}\n  {description or 'No description'}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def switch_branch(name: str) -> str:
    """
    Switch to a different conversation branch.

    Args:
        name: Branch name to switch to
    """
    try:
        branch = ctx.branch_manager.switch_branch(name)
        return f"✓ Switched to branch: {name}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def list_branches() -> str:
    """List all conversation branches."""
    branches = ctx.branch_manager.list_branches()

    lines = ["Conversation Branches:", ""]
    for b in branches:
        marker = "→ " if b.is_active else "  "
        lines.append(f"{marker}{b.name}: {b.description or 'No description'}")

    return "\n".join(lines)


@mcp.tool()
def delete_branch(name: str) -> str:
    """
    Delete a conversation branch.

    Args:
        name: Branch name to delete
    """
    try:
        if ctx.branch_manager.delete_branch(name):
            return f"✓ Deleted branch: {name}"
        return f"Branch not found: {name}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def compare_branches(branch1: str, branch2: str) -> str:
    """
    Compare two conversation branches.

    Args:
        branch1: First branch name
        branch2: Second branch name
    """
    try:
        comparison = ctx.branch_manager.compare_branches(branch1, branch2)
        lines = [
            f"Branch Comparison: {branch1} vs {branch2}",
            "",
            f"{branch1}:",
            f"  Items: {comparison['branch1']['item_count']}",
            f"  Unique: {comparison['branch1']['unique_items']}",
            f"  Tokens: {comparison['branch1']['tokens']}",
            "",
            f"{branch2}:",
            f"  Items: {comparison['branch2']['item_count']}",
            f"  Unique: {comparison['branch2']['unique_items']}",
            f"  Tokens: {comparison['branch2']['tokens']}",
            "",
            f"Common history: {comparison['common_ancestor_items']} items"
        ]
        return "\n".join(lines)
    except ValueError as e:
        return f"Error: {e}"


# ============================================================================
# Cache Optimization Tools (Manus)
# ============================================================================

@mcp.tool()
def set_stable_prefix(prefix: str) -> str:
    """
    Set immutable context prefix for cache optimization.

    Args:
        prefix: The stable prefix (system instructions, role, etc.)
    """
    ctx.memory.set_stable_prefix(prefix)
    return f"✓ Stable prefix set ({len(prefix)} chars)"


@mcp.tool()
def mark_cache_break() -> str:
    """Signal that the KV-cache should reset."""
    ctx.memory.mark_cache_break()
    return "✓ Cache breakpoint marked"


@mcp.tool()
def get_cache_stats() -> str:
    """Get cache optimization statistics."""
    stats = ctx.memory.get_cache_stats()

    lines = [
        "Cache Optimization:",
        f"  Strategy: {stats['config']['strategy']}",
        f"  Has prefix: {stats['has_prefix']}",
        f"  Tracked items: {stats['tracked_items']}",
        f"  Breakpoint pending: {stats['breakpoint_pending']}",
        "",
        "Stats:",
        f"  Total prompts: {stats['stats']['total_prompts']}",
        f"  Prefix changes: {stats['stats']['prefix_changes']}",
        f"  Reorders: {stats['stats']['reorders']}",
        f"  Estimated hit rate: {stats['stats']['estimated_hit_rate']}"
    ]
    return "\n".join(lines)


# ============================================================================
# Task Checklist Tools (Manus)
# ============================================================================

@mcp.tool()
def add_task(description: str) -> str:
    """
    Add a task to the checklist (attention control).

    Args:
        description: Task description
    """
    task = ctx.memory.add_task(description)
    return f"✓ Added task {task.id}: {description}"


@mcp.tool()
def complete_task(task_id: str, notes: str = "") -> str:
    """
    Mark a task as complete.

    Args:
        task_id: Task ID to complete
        notes: Optional completion notes
    """
    if ctx.memory.complete_task(task_id, notes):
        return f"✓ Completed task {task_id}"
    return f"Task not found: {task_id}"


@mcp.tool()
def get_checklist(compact: bool = False) -> str:
    """
    Get current task checklist.

    Args:
        compact: If True, show minimal version
    """
    checklist = ctx.memory.get_checklist(compact=compact)
    if not checklist:
        return "No tasks in checklist"
    return checklist


@mcp.tool()
def clear_checklist() -> str:
    """Clear all tasks from checklist."""
    ctx.memory.clear_checklist()
    return "✓ Checklist cleared"


# ============================================================================
# Error Memory Tools (Manus)
# ============================================================================

@mcp.tool()
def log_error(action: str, error: str, context: str = "") -> str:
    """
    Log a failed action for learning.

    Args:
        action: What was attempted
        error: Error message
        context: Optional context
    """
    entry = ctx.memory.log_error(action, error, context)
    return f"✓ Logged error {entry.id}: {action}"


@mcp.tool()
def resolve_error(error_id: str, resolution: str = "") -> str:
    """
    Mark an error as resolved.

    Args:
        error_id: Error ID to resolve
        resolution: How it was fixed
    """
    if ctx.memory.resolve_error(error_id, resolution):
        return f"✓ Resolved error {error_id}"
    return f"Error not found: {error_id}"


@mcp.tool()
def get_errors(include_resolved: bool = False) -> str:
    """
    Get error context for learning.

    Args:
        include_resolved: Include resolved errors
    """
    errors = ctx.memory.get_errors(include_resolved=include_resolved)
    if not errors:
        return "No errors logged"
    return errors


@mcp.tool()
def clear_errors() -> str:
    """Clear error memory."""
    ctx.memory.clear_errors()
    return "✓ Error memory cleared"


# ============================================================================
# Strategy Evolution Tools (from mcp_server_agentic_memory.py)
# ============================================================================

@mcp.tool()
def initialize_strategy(domain: str = "general", template: str = "") -> str:
    """
    Initialize agent strategy/playbook.

    Args:
        domain: Domain context (e.g., 'security', 'code_analysis')
        template: Optional template ('security_audit')
    """
    strategy = ctx.memory.initialize_strategy(domain, template or None)
    return f"✓ Strategy initialized\n  Domain: {strategy.domain}\n  Heuristics: {len(strategy.heuristics)}"


@mcp.tool()
def update_strategy(update_type: str, content: str) -> str:
    """
    Update agent's evolving strategy based on execution feedback.

    Args:
        update_type: 'add_heuristic', 'add_pattern', 'record_failure', 'record_success'
        content: Description of the learning
    """
    result = ctx.memory.update_strategy(update_type, content)
    if result['success']:
        return f"✓ Strategy updated (v{result['strategy_version']})\n  Type: {update_type}\n  Heuristics: {result['total_heuristics']}, Patterns: {result['total_patterns']}"
    return f"Error: {result['error']}"


@mcp.tool()
def get_strategy() -> str:
    """Get current agent strategy/playbook."""
    strategy = ctx.memory.get_strategy()
    if not strategy:
        return "No strategy initialized. Use initialize_strategy() first."

    lines = [
        f"Agent Strategy (v{strategy.version})",
        f"Domain: {strategy.domain}",
        "",
        f"Heuristics ({len(strategy.heuristics)}):"
    ]
    for h in strategy.heuristics[:5]:
        lines.append(f"  • {h['content']}")

    lines.append(f"\nPatterns ({len(strategy.learned_patterns)}):")
    for p in strategy.learned_patterns[:3]:
        lines.append(f"  • {p['content']}")

    return "\n".join(lines)


# ============================================================================
# Schema-Driven Compaction Tools (from mcp_server_agentic_memory.py)
# ============================================================================

@mcp.tool()
def compact_session_schema(preserve_recent: int = 20, extract_insights: bool = True) -> str:
    """
    Compact session using schema-driven summarization.
    Groups events by type and extracts insights to long-term memory.

    Args:
        preserve_recent: Keep this many recent events uncompacted
        extract_insights: Extract patterns to long-term memory
    """
    result = ctx.memory.compact_session_schema(preserve_recent, extract_insights)

    if 'message' in result:
        return result['message']

    lines = [
        "✓ Session compacted",
        f"  Original size: {result['original_size']}",
        f"  Compacted size: {result['compacted_size']}",
        f"  Compression: {result['compression_ratio']}",
        f"  Insights extracted: {result['insights_extracted']}"
    ]
    return "\n".join(lines)


# ============================================================================
# Agentic Working Context Tools (from mcp_server_agentic_memory.py)
# ============================================================================

@mcp.tool()
def get_agentic_context(include_strategy: bool = False) -> str:
    """
    Get minimal working context for immediate action.
    Only what's relevant NOW - not a transcript.

    Args:
        include_strategy: Include current strategy/playbook
    """
    import json
    result = ctx.memory.get_agentic_working_context(include_strategy)

    lines = [
        "Working Context:",
        f"  Goal: {result['current_goal'] or '(none set)'}",
        f"  Constraints: {len(result['constraints'])}",
        f"  Artifacts: {len(result['artifacts'])}",
        f"  Recent observations: {len(result['recent_observations'])}",
        f"  Applicable memories: {len(result['applicable_memories'])}",
        f"  Token estimate: ~{result['token_estimate']}"
    ]

    if include_strategy and 'strategy' in result:
        lines.append(f"\n  Strategy v{result['strategy']['version']}")

    return "\n".join(lines)


@mcp.tool()
def add_constraint(constraint: str) -> str:
    """
    Add a constraint to working context.

    Args:
        constraint: Constraint description
    """
    ctx.memory.add_constraint(constraint)
    return f"✓ Added constraint: {constraint}"


@mcp.tool()
def set_active_artifacts(artifact_ids: str) -> str:
    """
    Set which artifacts are active in working context.

    Args:
        artifact_ids: Comma-separated artifact IDs
    """
    ids = [id.strip() for id in artifact_ids.split(",") if id.strip()]
    ctx.memory.set_active_artifacts(ids)
    return f"✓ Set {len(ids)} active artifacts"


# ============================================================================
# Context Compilation Tools (from mcp_server_context_compiler.py)
# ============================================================================

@mcp.tool()
def get_compilation_stats(last_n: int = 10) -> str:
    """
    Get statistics about context compilation quality over time.

    Args:
        last_n: Number of recent compilations to analyze
    """
    stats = ctx.compiler.get_compilation_stats(last_n)

    if not stats['success']:
        return stats['error']

    qs = stats['quality_scores']
    lines = [
        "Compilation Statistics:",
        f"  Compilations: {stats['total_compilations']}",
        f"  Quality: avg={qs['mean']:.1f}, min={qs['min']:.1f}, max={qs['max']:.1f}",
        f"  Trend: {stats['trend']}",
        "",
        "Strategies used:"
    ]
    for strategy, count in stats['strategies_used'].items():
        lines.append(f"  {strategy}: {count}")

    return "\n".join(lines)


@mcp.tool()
def explain_compilation() -> str:
    """Explain the last context compilation decision."""
    result = ctx.compiler.explain_compilation()

    if not result['success']:
        return result['error']

    exp = result['explanation']
    lines = [
        "Last Compilation:",
        f"  Strategy: {exp['strategy_used']}",
        f"  Quality: {exp['quality_score']}",
        "",
        "Decisions:"
    ]
    for d in exp['decisions']:
        lines.append(f"  • {d['decision']}")
        lines.append(f"    Rationale: {d['rationale']}")

    return "\n".join(lines)


# ============================================================================
# Storage Tools
# ============================================================================

@mcp.tool()
def export_session(format: str = "json") -> str:
    """
    Export current session data.

    Args:
        format: Export format ('json')
    """
    import json
    data = {
        "namespace": ctx.memory.namespace,
        "stats": ctx.memory.stats(),
        "session_items": [
            item.to_dict() for item in ctx.memory.get_session_items()
        ],
        "working_context": ctx.memory.working_context
    }
    return json.dumps(data, indent=2, default=str)


# ============================================================================
# Main
# ============================================================================

def main():
    global ctx

    parser = argparse.ArgumentParser(description="SmartContext MCP Server")
    parser.add_argument("--namespace", default="default", help="Memory namespace")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory")
    parser.add_argument("--backend", default="json", choices=["json", "sqlite"],
                        help="Storage backend")
    args = parser.parse_args()

    ctx = SmartContext(
        namespace=args.namespace,
        storage_dir=args.storage_dir,
        storage_backend=args.backend
    )

    print(f"SmartContext starting...")
    print(f"  Namespace: {args.namespace}")
    print(f"  Storage: {ctx.memory.storage_dir}")
    print(f"  Backend: {args.backend}")
    print(f"  Memories: {len(ctx.memory.long_term)}")
    print(f"  Artifacts: {len(ctx.memory.artifacts)}")

    mcp.run()


if __name__ == "__main__":
    main()
