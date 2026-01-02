"""
SmartContext Memory System

Tiered memory architecture for long-running autonomous agents:
- Working Context: Minimal per-call view (what's relevant NOW)
- Session Memory: Structured event logs for current task trajectory
- Long-term Memory: Searchable insights across multiple runs
- Artifacts: Large objects referenced by handle, not inlined

Enhanced with features distilled from mcp-tools-servers:
- Session protocol (OpenAI-compatible)
- Storage backends (JSON, SQLite)
- Context compression
- Error preservation (Manus pattern)
- Task checklist (Manus pattern)
- Schema-driven compaction (reversible summarization)
- Strategy/playbook evolution
- Insight extraction from sessions

Core Principles (from Google ADK, Anthropic ACE, Manus):
1. Context as compiled view, not transcript
2. Tiered memory: Working Context → Session → Memory → Artifacts
3. Retrieval over pinning (query on-demand, don't accumulate)
4. Schema-driven summarization (preserve structure, not just content)
5. Offload heavy state to file system
6. Minimal default context ("scope by default")
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict

# Import new modules
from session import (
    SessionItem, BaseSession, TrimConfig, TrimStrategy,
    SessionProtocol
)
from compression import (
    ContextCompressor, CompressionConfig, CompressionStrategy,
    CompressionResult
)
from checklist import TaskChecklist, Task, TaskStatus
from errors import ErrorMemory, ErrorEntry, ErrorCategory
from cache import CacheOptimizer, CacheConfig

# Try to import storage backends
try:
    from storage import StorageBackend, StorageConfig, JSONStorage, SQLiteStorage
    HAS_STORAGE_BACKENDS = True
except ImportError:
    HAS_STORAGE_BACKENDS = False


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    tags: List[str]
    created: str
    importance: str = "normal"  # low, normal, high, critical
    source: str = "user"  # user, auto, system

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        # Handle field name mapping from storage
        if 'created_at' in data and 'created' not in data:
            data['created'] = data.pop('created_at')
        # Remove any extra fields not in the dataclass
        valid_fields = {'id', 'content', 'tags', 'created', 'importance', 'source'}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class Artifact:
    """Large content stored by reference."""
    id: str
    name: str
    content: str
    size: int
    created: str
    content_type: str = "text"  # text, code, json, markdown

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkingContext:
    """
    Minimal per-call context - only what's relevant NOW.
    (Distilled from mcp_server_agentic_memory.py)
    """
    agent_id: str
    current_goal: str
    relevant_constraints: List[str]
    active_artifacts: List[str]  # References, not content
    recent_observations: List[Dict]  # Last 3-5 events only
    applicable_memories: List[str]  # Retrieved, not pinned

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingContext":
        return cls(**data)


@dataclass
class AgentStrategy:
    """
    Evolving strategy/playbook for an agent.
    (Distilled from mcp_server_agentic_memory.py)
    """
    domain: str = "general"
    heuristics: List[Dict] = None
    learned_patterns: List[Dict] = None
    failure_modes: List[Dict] = None
    success_patterns: List[Dict] = None
    version: int = 1

    def __post_init__(self):
        self.heuristics = self.heuristics or []
        self.learned_patterns = self.learned_patterns or []
        self.failure_modes = self.failure_modes or []
        self.success_patterns = self.success_patterns or []

    def add_heuristic(self, content: str, metadata: Dict = None) -> None:
        """Add a heuristic to the strategy."""
        self.heuristics.append({
            'content': content,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self.version += 1

    def add_pattern(self, content: str, metadata: Dict = None) -> None:
        """Add a learned pattern."""
        self.learned_patterns.append({
            'content': content,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self.version += 1

    def record_failure(self, content: str, metadata: Dict = None) -> None:
        """Record a failure mode."""
        self.failure_modes.append({
            'content': content,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self.version += 1

    def record_success(self, content: str, metadata: Dict = None) -> None:
        """Record a success pattern."""
        self.success_patterns.append({
            'content': content,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self.version += 1

    def to_dict(self) -> dict:
        return {
            'domain': self.domain,
            'heuristics': self.heuristics,
            'learned_patterns': self.learned_patterns,
            'failure_modes': self.failure_modes,
            'success_patterns': self.success_patterns,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentStrategy":
        return cls(**data)


class MemoryStore:
    """
    Persistent memory storage with tiered architecture.

    Enhanced features:
    - Session protocol with trimming
    - Multiple storage backends
    - Context compression
    - Error tracking
    - Task checklists
    - KV-cache optimization
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        namespace: str = "default",
        storage_backend: str = "json",  # "json" or "sqlite"
        max_session_turns: int = 50,
        compression_strategy: str = "hybrid"
    ):
        self.namespace = namespace
        self.storage_dir = storage_dir or Path.home() / ".smartcontext" / namespace
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage backend
        self._storage: Optional[StorageBackend] = None
        if HAS_STORAGE_BACKENDS:
            config = StorageConfig(
                base_dir=self.storage_dir.parent,
                namespace=namespace
            )
            if storage_backend == "sqlite":
                self._storage = SQLiteStorage(config)
            else:
                self._storage = JSONStorage(config)

        # Session with trimming
        self._session = BaseSession(
            session_id=namespace,
            trim_config=TrimConfig(
                strategy=TrimStrategy.FIRST_LAST,
                max_items=max_session_turns,
                keep_first=2
            )
        )

        # Compression
        self._compressor = ContextCompressor()
        self._compression_config = CompressionConfig(
            strategy=CompressionStrategy(compression_strategy)
        )

        # Error memory (Manus pattern)
        self._error_memory = ErrorMemory()

        # Task checklist (Manus pattern)
        self._checklist = TaskChecklist()

        # Cache optimization (Manus pattern)
        self._cache_optimizer = CacheOptimizer()

        # Load existing data
        self.working_context: Dict[str, Any] = self._load("working_context")
        self.long_term: Dict[str, Memory] = self._load_memories("long_term")
        self.artifacts: Dict[str, Artifact] = self._load_artifacts("artifacts")
        self.session_log: List[Dict] = []

    def _load(self, name: str) -> dict:
        """Load a JSON file."""
        if self._storage:
            return self._storage.get_all_working() if name == "working_context" else {}
        path = self.storage_dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save(self, name: str, data: Any):
        """Save to a JSON file."""
        if self._storage and name == "working_context":
            for key, value in data.items():
                self._storage.set_working(key, value)
            return
        path = self.storage_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load_memories(self, name: str) -> Dict[str, Memory]:
        """Load memories from file."""
        if self._storage:
            memories = self._storage.list_memories()
            return {m["id"]: Memory.from_dict(m) for m in memories}
        data = self._load(name)
        return {k: Memory.from_dict(v) for k, v in data.items()}

    def _load_artifacts(self, name: str) -> Dict[str, Artifact]:
        """Load artifacts from file."""
        if self._storage:
            artifacts = self._storage.list_artifacts()
            return {
                a["id"]: Artifact(
                    id=a["id"],
                    name=a["name"],
                    content=a["content"],
                    size=a.get("size", len(a["content"])),
                    created=a.get("created_at", datetime.now().isoformat()),
                    content_type=a.get("content_type", "text")
                )
                for a in artifacts
            }
        data = self._load(name)
        return {k: Artifact(**v) for k, v in data.items()}

    def _generate_id(self, text: str) -> str:
        """Generate a short ID."""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    # =========================================================================
    # Working Context (Current Focus)
    # =========================================================================

    def set_working(self, key: str, value: Any):
        """Set a value in working context."""
        self.working_context[key] = value
        if self._storage:
            self._storage.set_working(key, value)
        else:
            self._save("working_context", self.working_context)

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working context."""
        return self.working_context.get(key, default)

    def clear_working(self):
        """Clear working context."""
        self.working_context = {}
        if self._storage:
            self._storage.clear_working()
        else:
            self._save("working_context", self.working_context)

    def get_working_summary(self, max_tokens: int = 500) -> str:
        """Get a token-limited summary of working context."""
        if not self.working_context:
            return ""

        lines = []
        for key, value in self.working_context.items():
            line = f"- {key}: {str(value)[:100]}"
            lines.append(line)

        summary = "\n".join(lines)
        # Rough token estimate (4 chars per token)
        if len(summary) > max_tokens * 4:
            summary = summary[:max_tokens * 4] + "..."

        return summary

    # =========================================================================
    # Long-term Memory (Persistent)
    # =========================================================================

    def remember(self, content: str, tags: List[str] = None,
                 importance: str = "normal", source: str = "user") -> Memory:
        """Save to long-term memory."""
        memory_id = self._generate_id(content)
        memory = Memory(
            id=memory_id,
            content=content,
            tags=tags or [],
            created=datetime.now().isoformat(),
            importance=importance,
            source=source
        )
        self.long_term[memory_id] = memory

        if self._storage:
            self._storage.save_memory(memory_id, {
                "id": memory_id,
                "content": content,
                "tags": tags or [],
                "created_at": memory.created,
                "importance": importance,
                "source": source
            })
        else:
            self._save("long_term", {k: v.to_dict() for k, v in self.long_term.items()})

        return memory

    def forget(self, memory_id: str) -> bool:
        """Remove from long-term memory."""
        if memory_id in self.long_term:
            del self.long_term[memory_id]
            if self._storage:
                self._storage.delete_memory(memory_id)
            else:
                self._save("long_term", {k: v.to_dict() for k, v in self.long_term.items()})
            return True
        return False

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search long-term memory.
        Uses simple word overlap; replace with embeddings for production.
        """
        # Try storage backend search first
        if self._storage:
            results = self._storage.search_memories(query, limit)
            return [Memory.from_dict(r) for r in results]

        query_words = set(query.lower().split())

        scored = []
        for memory in self.long_term.values():
            content_words = set(memory.content.lower().split())
            tag_words = set(" ".join(memory.tags).lower().split())
            all_words = content_words | tag_words

            if not all_words:
                continue

            overlap = len(query_words & all_words)
            score = overlap / len(query_words) if query_words else 0

            # Boost by importance
            importance_boost = {"low": 0.8, "normal": 1.0, "high": 1.2, "critical": 1.5}
            score *= importance_boost.get(memory.importance, 1.0)

            if score > 0.1:
                scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def get_all_memories(self) -> List[Memory]:
        """Get all long-term memories."""
        return list(self.long_term.values())

    # =========================================================================
    # Artifacts (Large Content)
    # =========================================================================

    def store_artifact(self, name: str, content: str,
                       content_type: str = "text") -> Artifact:
        """Store large content as an artifact."""
        artifact_id = self._generate_id(name + content[:100])
        artifact = Artifact(
            id=artifact_id,
            name=name,
            content=content,
            size=len(content),
            created=datetime.now().isoformat(),
            content_type=content_type
        )
        self.artifacts[artifact_id] = artifact

        if self._storage:
            self._storage.save_artifact(artifact_id, {
                "id": artifact_id,
                "name": name,
                "content": content,
                "content_type": content_type,
                "created_at": artifact.created
            })
        else:
            self._save("artifacts", {k: v.to_dict() for k, v in self.artifacts.items()})

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID or name."""
        # Try direct ID lookup
        if artifact_id in self.artifacts:
            return self.artifacts[artifact_id]

        # Try name lookup
        for artifact in self.artifacts.values():
            if artifact.name == artifact_id:
                return artifact

        # Try storage backend
        if self._storage:
            data = self._storage.load_artifact(artifact_id)
            if data:
                return Artifact(
                    id=data["id"],
                    name=data["name"],
                    content=data["content"],
                    size=data.get("size", len(data["content"])),
                    created=data.get("created_at", datetime.now().isoformat()),
                    content_type=data.get("content_type", "text")
                )

        return None

    def list_artifacts(self) -> List[Artifact]:
        """List all artifacts (metadata only, not content)."""
        return list(self.artifacts.values())

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
            if self._storage:
                self._storage.delete_artifact(artifact_id)
            else:
                self._save("artifacts", {k: v.to_dict() for k, v in self.artifacts.items()})
            return True
        return False

    # =========================================================================
    # Session Memory (Current Conversation)
    # =========================================================================

    def log_event(self, event_type: str, data: Dict = None):
        """Log an event to session memory."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {}
        }
        self.session_log.append(event)

        # Also add to session protocol
        self._session.add_item(
            role="system",
            content=f"[{event_type}] {json.dumps(data or {})}",
            metadata={"event_type": event_type}
        )

    def get_session_log(self, limit: int = None) -> List[Dict]:
        """Get session log entries."""
        if limit:
            return self.session_log[-limit:]
        return self.session_log

    def get_session_items(self, limit: int = None) -> List[SessionItem]:
        """Get session items (protocol-based)."""
        return self._session.get_items(limit)

    def add_session_item(self, role: str, content: str) -> SessionItem:
        """Add an item to the session."""
        return self._session.add_item(role, content)

    def pop_session_item(self) -> Optional[SessionItem]:
        """Remove and return the last session item (undo)."""
        return self._session.pop_item()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self._session.get_stats()

    def set_max_turns(self, max_turns: int):
        """Set the maximum number of session turns."""
        self._session.trim_config.max_items = max_turns

    def end_session(self, summary: str = None) -> Memory:
        """
        End session and optionally save summary to long-term memory.
        """
        if summary:
            memory = self.remember(
                content=summary,
                tags=["session-summary", "auto-generated"],
                importance="normal",
                source="system"
            )
        else:
            memory = None

        # Clear session state
        self.session_log = []
        self._session.clear_session()
        self.clear_working()

        return memory

    # =========================================================================
    # Compression
    # =========================================================================

    def compress_session(
        self,
        strategy: str = None
    ) -> CompressionResult:
        """
        Compress session context.

        Args:
            strategy: Compression strategy (none/heuristic/llm/hybrid)

        Returns:
            CompressionResult with stats
        """
        if strategy:
            self._compression_config.strategy = CompressionStrategy(strategy)

        items = self._session.get_items()
        result = self._compressor.compress(items, self._compression_config)

        # Update session with compressed items
        if result.strategy_used != CompressionStrategy.NONE:
            self._session._items = result.preserved_items
            if result.summary:
                self._session.add_item(
                    role="system",
                    content=f"[Compressed History]\n{result.summary}",
                    metadata={"compressed": True}
                )

        return result

    def should_compress(self) -> bool:
        """Check if compression is recommended."""
        items = self._session.get_items()
        return self._compressor.should_compress(items, self._compression_config)

    def set_compression_strategy(self, strategy: str):
        """Set the compression strategy."""
        self._compression_config.strategy = CompressionStrategy(strategy)

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        items = self._session.get_items()
        total_tokens = sum(item.estimate_tokens() for item in items)
        return {
            "current_items": len(items),
            "current_tokens": total_tokens,
            "strategy": self._compression_config.strategy.value,
            "should_compress": self.should_compress(),
            "threshold": self._compression_config.token_threshold
        }

    # =========================================================================
    # Error Memory (Manus Pattern)
    # =========================================================================

    def log_error(
        self,
        action: str,
        error: str,
        context: str = ""
    ) -> ErrorEntry:
        """Log an error for learning."""
        return self._error_memory.log_error(action, error, context)

    def resolve_error(self, error_id: str, resolution: str = "") -> bool:
        """Mark an error as resolved."""
        return self._error_memory.resolve_error(error_id, resolution)

    def get_errors(self, include_resolved: bool = False) -> str:
        """Get errors formatted for context."""
        return self._error_memory.get_error_context(include_resolved)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self._error_memory.get_stats()

    def clear_errors(self):
        """Clear error memory."""
        self._error_memory.clear()

    # =========================================================================
    # Task Checklist (Manus Pattern)
    # =========================================================================

    def add_task(self, description: str) -> Task:
        """Add a task to the checklist."""
        return self._checklist.add_task(description)

    def complete_task(self, task_id: str, notes: str = "") -> bool:
        """Complete a task."""
        return self._checklist.complete_task(task_id, notes)

    def get_checklist(self, compact: bool = False) -> str:
        """Get checklist for context injection."""
        if compact:
            return self._checklist.get_compact_recitation()
        return self._checklist.get_recitation()

    def get_checklist_progress(self) -> Dict[str, Any]:
        """Get checklist progress."""
        return self._checklist.get_progress()

    def clear_checklist(self):
        """Clear the checklist."""
        self._checklist.clear()

    # =========================================================================
    # Cache Optimization (Manus Pattern)
    # =========================================================================

    def set_stable_prefix(self, prefix: str):
        """Set immutable context prefix for cache optimization."""
        self._cache_optimizer.set_stable_prefix(prefix)

    def get_stable_prefix(self) -> Optional[str]:
        """Get the current stable prefix."""
        return self._cache_optimizer.get_stable_prefix()

    def mark_cache_break(self):
        """Mark that the cache should reset."""
        self._cache_optimizer.mark_cache_breakpoint()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache optimization statistics."""
        return self._cache_optimizer.get_stats()

    def get_optimized_context(self, include_prefix: bool = True) -> str:
        """Get context optimized for KV-cache hits."""
        items = self._session.get_items()
        return self._cache_optimizer.optimize_context(items, include_prefix)

    # =========================================================================
    # Strategy Evolution (from mcp_server_agentic_memory.py)
    # =========================================================================

    def initialize_strategy(
        self,
        domain: str = "general",
        template: str = None
    ) -> AgentStrategy:
        """
        Initialize or load agent strategy.
        (Distilled from mcp_server_agentic_memory.py)
        """
        if template == "security_audit":
            self._strategy = AgentStrategy(
                domain="security",
                heuristics=[
                    {'content': 'Check authentication before authorization'},
                    {'content': 'Validate all user inputs'},
                    {'content': 'Use parameterized queries for database access'}
                ]
            )
        else:
            self._strategy = AgentStrategy(domain=domain)

        return self._strategy

    def update_strategy(
        self,
        update_type: str,  # "add_heuristic", "add_pattern", "record_failure", "record_success"
        content: str,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Update agent's evolving strategy based on execution feedback.
        (Distilled from mcp_server_agentic_memory.py)
        """
        if not hasattr(self, '_strategy') or self._strategy is None:
            self.initialize_strategy()

        if update_type == 'add_heuristic':
            self._strategy.add_heuristic(content, metadata)
        elif update_type == 'add_pattern':
            self._strategy.add_pattern(content, metadata)
        elif update_type == 'record_failure':
            self._strategy.record_failure(content, metadata)
        elif update_type == 'record_success':
            self._strategy.record_success(content, metadata)
        else:
            return {'success': False, 'error': f'Unknown update type: {update_type}'}

        return {
            'success': True,
            'strategy_version': self._strategy.version,
            'update_type': update_type,
            'total_heuristics': len(self._strategy.heuristics),
            'total_patterns': len(self._strategy.learned_patterns)
        }

    def get_strategy(self) -> Optional[AgentStrategy]:
        """Get the current strategy."""
        return getattr(self, '_strategy', None)

    # =========================================================================
    # Schema-Driven Compaction (from mcp_server_agentic_memory.py)
    # =========================================================================

    def compact_session_schema(
        self,
        preserve_recent: int = 20,
        extract_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Compact session memory using schema-driven summarization.
        Reduces session size while preserving essential structure.
        (Distilled from mcp_server_agentic_memory.py)
        """
        events = self.session_log
        original_size = len(events)

        if original_size <= preserve_recent:
            return {
                'success': True,
                'message': 'Session too small to compact',
                'session_size': original_size
            }

        # Split into old (to compact) and recent (preserve)
        old_events = events[:-preserve_recent]
        recent_events = events[-preserve_recent:]

        # Schema-driven compaction: group by type and summarize
        compacted = self._schema_driven_compact(old_events)

        insights_extracted = []
        if extract_insights:
            insights_extracted = self._extract_insights_from_events(old_events)

        # Replace session with compacted + recent
        self.session_log = compacted + recent_events

        return {
            'success': True,
            'original_size': original_size,
            'compacted_size': len(self.session_log),
            'compression_ratio': f'{len(self.session_log) / original_size:.2%}',
            'insights_extracted': len(insights_extracted),
            'insights': insights_extracted
        }

    def _schema_driven_compact(self, events: List[Dict]) -> List[Dict]:
        """
        Compact events using schema preservation.
        (Distilled from mcp_server_agentic_memory.py)
        """
        from collections import defaultdict

        # Group by type
        by_type = defaultdict(list)
        for event in events:
            by_type[event.get('type', 'unknown')].append(event)

        # Create compact summaries per type
        compacted = []
        for event_type, type_events in by_type.items():
            if len(type_events) > 5:
                # Summarize multiple events of same type
                compacted.append({
                    'type': f'{event_type}_summary',
                    'timestamp': type_events[0].get('timestamp', ''),
                    'data': {
                        'original_count': len(type_events),
                        'summary': f'Compacted {len(type_events)} {event_type} events',
                        'time_range': [
                            type_events[0].get('timestamp', ''),
                            type_events[-1].get('timestamp', '')
                        ]
                    }
                })
            else:
                # Keep individual events if few
                compacted.extend(type_events)

        return compacted

    def _extract_insights_from_events(self, events: List[Dict]) -> List[Dict]:
        """
        Extract patterns and insights from events to long-term memory.
        (Distilled from mcp_server_agentic_memory.py)
        """
        insights = []

        # Pattern detection: repeated errors
        error_events = [e for e in events if e.get('type') == 'error']
        if len(error_events) > 3:
            insight_content = f'Session had {len(error_events)} errors'
            insight = {
                'type': 'pattern',
                'domain': 'error_handling',
                'content': insight_content,
                'confidence': 0.8,
                'source_events': [e.get('timestamp', '') for e in error_events[:5]]
            }
            insights.append(insight)

            # Store to long-term memory
            self.remember(
                content=insight_content,
                tags=['insight', 'auto-extracted', 'error-pattern'],
                importance='normal',
                source='system'
            )

        # Pattern detection: successful actions
        success_events = [e for e in events if 'success' in str(e.get('data', {})).lower()]
        if len(success_events) > 5:
            insight_content = f'Session had {len(success_events)} successful operations'
            insight = {
                'type': 'pattern',
                'domain': 'success_tracking',
                'content': insight_content,
                'confidence': 0.7,
                'source_events': [e.get('timestamp', '') for e in success_events[:5]]
            }
            insights.append(insight)

        return insights

    # =========================================================================
    # Agentic Working Context (from mcp_server_agentic_memory.py)
    # =========================================================================

    def get_agentic_working_context(
        self,
        include_strategy: bool = False
    ) -> Dict[str, Any]:
        """
        Get minimal working context for immediate action.
        (Distilled from mcp_server_agentic_memory.py)
        """
        # Resolve active artifact references to content
        active_artifacts = {}
        for artifact_id in self.working_context.get('active_artifacts', []):
            artifact = self.get_artifact(artifact_id)
            if artifact:
                active_artifacts[artifact_id] = {
                    'name': artifact.name,
                    'type': artifact.content_type,
                    'size': artifact.size
                }

        # Get applicable memories from working context
        applicable_memories = []
        for memory_id in self.working_context.get('applicable_memories', []):
            if memory_id in self.long_term:
                memory = self.long_term[memory_id]
                applicable_memories.append({
                    'id': memory_id,
                    'content': memory.content,
                    'tags': memory.tags
                })

        result = {
            'current_goal': self.working_context.get('goal', ''),
            'constraints': self.working_context.get('constraints', []),
            'artifacts': active_artifacts,
            'recent_observations': self.session_log[-5:],  # Last 5 only
            'applicable_memories': applicable_memories,
            'token_estimate': len(json.dumps(self.working_context)) // 4
        }

        if include_strategy:
            strategy = self.get_strategy()
            if strategy:
                result['strategy'] = strategy.to_dict()

        return result

    def set_goal(self, goal: str):
        """Set the current goal in working context."""
        self.set_working('goal', goal)
        self.log_event('goal_set', {'goal': goal})

    def get_goal(self) -> str:
        """Get the current goal."""
        return self.get_working('goal', '')

    def add_constraint(self, constraint: str):
        """Add a constraint to working context."""
        constraints = self.working_context.get('constraints', [])
        constraints.append(constraint)
        self.set_working('constraints', constraints)

    def set_active_artifacts(self, artifact_ids: List[str]):
        """Set active artifact references."""
        self.set_working('active_artifacts', artifact_ids)

    def set_applicable_memories(self, memory_ids: List[str]):
        """Set applicable memory references."""
        self.set_working('applicable_memories', memory_ids)

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        session_stats = self._session.get_stats()

        base_stats = {
            "namespace": self.namespace,
            "working_context_keys": len(self.working_context),
            "long_term_memories": len(self.long_term),
            "artifacts": len(self.artifacts),
            "session_events": len(self.session_log),
            "storage_dir": str(self.storage_dir),
            "storage_backend": "storage" if self._storage else "json",
        }

        # Add session protocol stats
        base_stats.update({
            "session_items": session_stats["item_count"],
            "session_tokens": session_stats["estimated_tokens"],
            "trim_strategy": session_stats["trim_strategy"],
            "max_items": session_stats["max_items"]
        })

        # Add feature stats
        base_stats["errors_logged"] = self._error_memory.get_stats()["total"]
        base_stats["checklist_tasks"] = len(self._checklist.tasks)
        base_stats["cache_optimized"] = self._cache_optimizer.get_stable_prefix() is not None

        # Add strategy stats
        strategy = self.get_strategy()
        if strategy:
            base_stats["strategy_version"] = strategy.version
            base_stats["strategy_heuristics"] = len(strategy.heuristics)
            base_stats["strategy_patterns"] = len(strategy.learned_patterns)

        # Compliance with agentic principles
        base_stats["principle_compliance"] = {
            "minimal_context": len(self.session_log[-5:]) <= 5,
            "retrieval_based": len(self.working_context.get('applicable_memories', [])) > 0,
            "artifact_offloading": len(self.working_context.get('active_artifacts', [])) > 0,
            "evolving_strategy": strategy is not None and strategy.version > 1 if strategy else False
        }

        return base_stats
