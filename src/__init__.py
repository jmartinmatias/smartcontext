"""
SmartContext - Intelligent Context Engineering for AI Assistants

A comprehensive context engineering toolkit combining:
- OpenAI Agents SDK session memory patterns
- Manus context engineering production patterns

Modules:
    memory: Tiered memory system (working/session/long-term/artifacts)
    attention: Attention management and context compilation
    session: OpenAI-compatible session protocol with trimming
    compression: Context compression (heuristic/LLM/hybrid)
    branching: Conversation branching for exploration
    cache: KV-cache optimization (Manus pattern)
    checklist: Task recitation for attention control (Manus pattern)
    errors: Error preservation for learning (Manus pattern)
    storage: Multiple storage backends (JSON/SQLite)
    smartcontext: Unified MCP server

Usage:
    # As MCP server
    python src/smartcontext.py --namespace myproject

    # Programmatic
    from smartcontext import MemoryStore, AttentionManager
    memory = MemoryStore(namespace="myproject")
    memory.remember("Important fact", tags=["info"])
"""

# Core modules
from .memory import MemoryStore, Memory, Artifact, WorkingContext, AgentStrategy
from .attention import (
    AttentionManager, AttentionMode, ContextCompiler, AttentionPolicy,
    CompiledSection, CompiledContext
)

# Session management
from .session import (
    SessionItem,
    SessionProtocol,
    BaseSession,
    TrimConfig,
    TrimStrategy
)

# Compression
from .compression import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    CompressionResult,
    HeuristicCompressor
)

# Branching
from .branching import BranchManager, Branch

# Manus patterns
from .cache import CacheOptimizer, CacheConfig, CacheStats, DiversityInjector
from .checklist import TaskChecklist, Task, TaskStatus
from .errors import ErrorMemory, ErrorEntry, ErrorCategory

# Version
__version__ = "2.0.0"

__all__ = [
    # Memory
    "MemoryStore",
    "Memory",
    "Artifact",
    "WorkingContext",
    "AgentStrategy",

    # Attention
    "AttentionManager",
    "AttentionMode",
    "ContextCompiler",
    "AttentionPolicy",
    "CompiledSection",
    "CompiledContext",

    # Session
    "SessionItem",
    "SessionProtocol",
    "BaseSession",
    "TrimConfig",
    "TrimStrategy",

    # Compression
    "ContextCompressor",
    "CompressionConfig",
    "CompressionStrategy",
    "CompressionResult",
    "HeuristicCompressor",

    # Branching
    "BranchManager",
    "Branch",

    # Cache (Manus)
    "CacheOptimizer",
    "CacheConfig",
    "CacheStats",
    "DiversityInjector",

    # Checklist (Manus)
    "TaskChecklist",
    "Task",
    "TaskStatus",

    # Errors (Manus)
    "ErrorMemory",
    "ErrorEntry",
    "ErrorCategory",
]
