#!/usr/bin/env python3
"""
SmartContext - Intelligent Context Engineering for AI Assistants

A unified MCP server that provides:
- Tiered memory (working/session/long-term/artifacts)
- Attention management with auto-detection
- Transparency modes for debugging
- Simple, invisible operation by default

Usage:
    python smartcontext.py [--namespace NAME] [--storage-dir PATH]

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
    Coordinates memory, attention, and transparency.
    """

    def __init__(self, namespace: str = "default", storage_dir: Path = None):
        self.memory = MemoryStore(storage_dir, namespace)
        self.attention = AttentionManager()
        self.compiler = ContextCompiler(self.attention)
        self.transparency = TransparencyLevel.OFF

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
        f"║  Session events:     {stats['session_events']:<4}                        ║",
        f"║  Working context:    {stats['working_context_keys']:<4} keys                   ║",
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


# ============================================================================
# Main
# ============================================================================

def main():
    global ctx

    parser = argparse.ArgumentParser(description="SmartContext MCP Server")
    parser.add_argument("--namespace", default="default", help="Memory namespace")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory")
    args = parser.parse_args()

    ctx = SmartContext(
        namespace=args.namespace,
        storage_dir=args.storage_dir
    )

    print(f"SmartContext starting...")
    print(f"  Namespace: {args.namespace}")
    print(f"  Storage: {ctx.memory.storage_dir}")
    print(f"  Memories: {len(ctx.memory.long_term)}")
    print(f"  Artifacts: {len(ctx.memory.artifacts)}")

    mcp.run()


if __name__ == "__main__":
    main()
