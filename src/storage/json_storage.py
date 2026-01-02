"""
SmartContext JSON Storage Backend

File-based JSON storage implementation.
Default backend - simple, portable, human-readable.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from .base import StorageBackend, StorageConfig


class JSONStorage(StorageBackend):
    """
    JSON file-based storage backend.

    Structure:
        ~/.smartcontext/{namespace}/
            memories.json       # Long-term memories
            artifacts.json      # Artifact metadata + content
            sessions/           # Session files
                {session_id}.json
            working.json        # Working context
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        super().__init__(config)
        self._working_cache: Dict[str, Any] = {}
        self._load_working()

    def _ensure_storage(self) -> None:
        """Create storage directories if they don't exist."""
        base = self.get_storage_path()
        base.mkdir(parents=True, exist_ok=True)
        (base / "sessions").mkdir(exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        """Get full path for a storage file."""
        return self.get_storage_path() / filename

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file, return None if doesn't exist."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _write_json(self, path: Path, data: Any) -> None:
        """Write JSON file with backup support."""
        if self.config.backup_enabled and path.exists():
            self._backup_file(path)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _backup_file(self, path: Path) -> None:
        """Create backup of file before overwriting."""
        backup_dir = self.get_storage_path() / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.stem}_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_path)

        # Cleanup old backups
        backups = sorted(backup_dir.glob(f"{path.stem}_*{path.suffix}"))
        for old_backup in backups[:-self.config.backup_count]:
            old_backup.unlink()

    # ========================================================================
    # Memory Operations
    # ========================================================================

    def _memories_path(self) -> Path:
        return self._get_path("memories.json")

    def _load_all_memories(self) -> Dict[str, Dict[str, Any]]:
        """Load all memories from file."""
        data = self._read_json(self._memories_path())
        return data.get("memories", {}) if data else {}

    def _save_all_memories(self, memories: Dict[str, Dict[str, Any]]) -> None:
        """Save all memories to file."""
        self._write_json(self._memories_path(), {"memories": memories})

    def save_memory(self, memory_id: str, data: Dict[str, Any]) -> None:
        memories = self._load_all_memories()
        memories[memory_id] = data
        self._save_all_memories(memories)

    def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        memories = self._load_all_memories()
        return memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        memories = self._load_all_memories()
        if memory_id in memories:
            del memories[memory_id]
            self._save_all_memories(memories)
            return True
        return False

    def list_memories(self) -> List[Dict[str, Any]]:
        memories = self._load_all_memories()
        return list(memories.values())

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword search in memory content."""
        memories = self._load_all_memories()
        query_lower = query.lower()
        query_words = query_lower.split()

        results = []
        for memory in memories.values():
            content = memory.get("content", "").lower()
            tags = [t.lower() for t in memory.get("tags", [])]

            # Score based on word matches
            score = 0
            for word in query_words:
                if word in content:
                    score += 2
                if any(word in tag for tag in tags):
                    score += 3

            if score > 0:
                results.append((score, memory))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results[:limit]]

    # ========================================================================
    # Artifact Operations
    # ========================================================================

    def _artifacts_path(self) -> Path:
        return self._get_path("artifacts.json")

    def _load_all_artifacts(self) -> Dict[str, Dict[str, Any]]:
        data = self._read_json(self._artifacts_path())
        return data.get("artifacts", {}) if data else {}

    def _save_all_artifacts(self, artifacts: Dict[str, Dict[str, Any]]) -> None:
        self._write_json(self._artifacts_path(), {"artifacts": artifacts})

    def save_artifact(self, artifact_id: str, data: Dict[str, Any]) -> None:
        artifacts = self._load_all_artifacts()
        artifacts[artifact_id] = data
        self._save_all_artifacts(artifacts)

    def load_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        artifacts = self._load_all_artifacts()
        return artifacts.get(artifact_id)

    def delete_artifact(self, artifact_id: str) -> bool:
        artifacts = self._load_all_artifacts()
        if artifact_id in artifacts:
            del artifacts[artifact_id]
            self._save_all_artifacts(artifacts)
            return True
        return False

    def list_artifacts(self) -> List[Dict[str, Any]]:
        artifacts = self._load_all_artifacts()
        return list(artifacts.values())

    # ========================================================================
    # Session Operations
    # ========================================================================

    def _session_path(self, session_id: str) -> Path:
        return self._get_path(f"sessions/{session_id}.json")

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        self._write_json(self._session_path(session_id), data)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self._session_path(session_id))

    def delete_session(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[str]:
        sessions_dir = self._get_path("sessions")
        if not sessions_dir.exists():
            return []
        return [p.stem for p in sessions_dir.glob("*.json")]

    # ========================================================================
    # Working Context Operations
    # ========================================================================

    def _working_path(self) -> Path:
        return self._get_path("working.json")

    def _load_working(self) -> None:
        """Load working context from file."""
        data = self._read_json(self._working_path())
        self._working_cache = data.get("working", {}) if data else {}

    def _save_working(self) -> None:
        """Save working context to file."""
        self._write_json(self._working_path(), {"working": self._working_cache})

    def set_working(self, key: str, value: Any) -> None:
        self._working_cache[key] = value
        if self.config.auto_save:
            self._save_working()

    def get_working(self, key: str) -> Optional[Any]:
        return self._working_cache.get(key)

    def clear_working(self) -> None:
        self._working_cache = {}
        if self.config.auto_save:
            self._save_working()

    def get_all_working(self) -> Dict[str, Any]:
        return self._working_cache.copy()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        memories = self._load_all_memories()
        artifacts = self._load_all_artifacts()
        sessions = self.list_sessions()

        return {
            "backend": "json",
            "namespace": self.config.namespace,
            "storage_path": str(self.get_storage_path()),
            "memory_count": len(memories),
            "artifact_count": len(artifacts),
            "session_count": len(sessions),
            "working_keys": len(self._working_cache),
            "total_size_bytes": self._get_total_size()
        }

    def _get_total_size(self) -> int:
        """Calculate total storage size in bytes."""
        total = 0
        for path in self.get_storage_path().rglob("*.json"):
            total += path.stat().st_size
        return total

    def export_all(self) -> Dict[str, Any]:
        """Export all data for backup/migration."""
        return {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "namespace": self.config.namespace,
            "memories": self._load_all_memories(),
            "artifacts": self._load_all_artifacts(),
            "sessions": {
                sid: self.load_session(sid)
                for sid in self.list_sessions()
            },
            "working": self.get_all_working()
        }

    def import_all(self, data: Dict[str, Any]) -> None:
        """Import data from backup/migration."""
        # Memories
        if "memories" in data:
            self._save_all_memories(data["memories"])

        # Artifacts
        if "artifacts" in data:
            self._save_all_artifacts(data["artifacts"])

        # Sessions
        if "sessions" in data:
            for session_id, session_data in data["sessions"].items():
                if session_data:
                    self.save_session(session_id, session_data)

        # Working
        if "working" in data:
            self._working_cache = data["working"]
            self._save_working()
