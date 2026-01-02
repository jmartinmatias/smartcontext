"""
SmartContext - Intelligent Context Engineering for AI Assistants

Modules:
    memory: Tiered memory system (working/session/long-term/artifacts)
    attention: Attention management and context compilation
    smartcontext: Unified MCP server
"""

from .memory import MemoryStore, Memory, Artifact
from .attention import AttentionManager, AttentionMode, ContextCompiler, AttentionPolicy

__version__ = "1.0.0"
__all__ = [
    "MemoryStore",
    "Memory",
    "Artifact",
    "AttentionManager",
    "AttentionMode",
    "ContextCompiler",
    "AttentionPolicy"
]
