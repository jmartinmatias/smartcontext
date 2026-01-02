"""
SmartContext Memory System

Three-tier memory architecture:
- Working Context: Current focus (~500 tokens)
- Session Memory: This conversation's history
- Long-term Memory: Persistent, searchable knowledge
- Artifacts: Large content stored by reference
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


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
        return cls(**data)


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


class MemoryStore:
    """
    Persistent memory storage with tiered architecture.
    """

    def __init__(self, storage_dir: Optional[Path] = None, namespace: str = "default"):
        self.namespace = namespace
        self.storage_dir = storage_dir or Path.home() / ".smartcontext" / namespace
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.working_context: Dict[str, Any] = self._load("working_context")
        self.long_term: Dict[str, Memory] = self._load_memories("long_term")
        self.artifacts: Dict[str, Artifact] = self._load_artifacts("artifacts")
        self.session_log: List[Dict] = []

    def _load(self, name: str) -> dict:
        """Load a JSON file."""
        path = self.storage_dir / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save(self, name: str, data: Any):
        """Save to a JSON file."""
        path = self.storage_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))

    def _load_memories(self, name: str) -> Dict[str, Memory]:
        """Load memories from file."""
        data = self._load(name)
        return {k: Memory.from_dict(v) for k, v in data.items()}

    def _load_artifacts(self, name: str) -> Dict[str, Artifact]:
        """Load artifacts from file."""
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
        self._save("working_context", self.working_context)

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a value from working context."""
        return self.working_context.get(key, default)

    def clear_working(self):
        """Clear working context."""
        self.working_context = {}
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
        self._save("long_term", {k: v.to_dict() for k, v in self.long_term.items()})
        return memory

    def forget(self, memory_id: str) -> bool:
        """Remove from long-term memory."""
        if memory_id in self.long_term:
            del self.long_term[memory_id]
            self._save("long_term", {k: v.to_dict() for k, v in self.long_term.items()})
            return True
        return False

    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Search long-term memory.
        Uses simple word overlap; replace with embeddings for production.
        """
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

        return None

    def list_artifacts(self) -> List[Artifact]:
        """List all artifacts (metadata only, not content)."""
        return list(self.artifacts.values())

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
            self._save("artifacts", {k: v.to_dict() for k, v in self.artifacts.items()})
            return True
        return False

    # =========================================================================
    # Session Memory (Current Conversation)
    # =========================================================================

    def log_event(self, event_type: str, data: Dict = None):
        """Log an event to session memory."""
        self.session_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {}
        })

    def get_session_log(self, limit: int = None) -> List[Dict]:
        """Get session log entries."""
        if limit:
            return self.session_log[-limit:]
        return self.session_log

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
        self.clear_working()

        return memory

    # =========================================================================
    # Stats
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "namespace": self.namespace,
            "working_context_keys": len(self.working_context),
            "long_term_memories": len(self.long_term),
            "artifacts": len(self.artifacts),
            "session_events": len(self.session_log),
            "storage_dir": str(self.storage_dir)
        }
