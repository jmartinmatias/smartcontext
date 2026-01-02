"""
SmartContext Storage Backend Interface

Abstract base class for all storage implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    base_dir: Path = field(default_factory=lambda: Path.home() / ".smartcontext")
    namespace: str = "default"
    auto_save: bool = True
    backup_enabled: bool = False
    backup_count: int = 3


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    All storage implementations must provide these methods for:
    - Memories (long-term storage)
    - Artifacts (large content)
    - Sessions (conversation history)
    - Working context (ephemeral key-value)
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._ensure_storage()

    @abstractmethod
    def _ensure_storage(self) -> None:
        """Ensure storage location exists."""
        pass

    # ========================================================================
    # Memory Operations
    # ========================================================================

    @abstractmethod
    def save_memory(self, memory_id: str, data: Dict[str, Any]) -> None:
        """Save a memory entry."""
        pass

    @abstractmethod
    def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load a memory entry by ID."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    def list_memories(self) -> List[Dict[str, Any]]:
        """List all memory entries."""
        pass

    @abstractmethod
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by content."""
        pass

    # ========================================================================
    # Artifact Operations
    # ========================================================================

    @abstractmethod
    def save_artifact(self, artifact_id: str, data: Dict[str, Any]) -> None:
        """Save an artifact."""
        pass

    @abstractmethod
    def load_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Load an artifact by ID."""
        pass

    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        pass

    @abstractmethod
    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all artifacts."""
        pass

    # ========================================================================
    # Session Operations
    # ========================================================================

    @abstractmethod
    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        pass

    # ========================================================================
    # Working Context Operations
    # ========================================================================

    @abstractmethod
    def set_working(self, key: str, value: Any) -> None:
        """Set a working context value."""
        pass

    @abstractmethod
    def get_working(self, key: str) -> Optional[Any]:
        """Get a working context value."""
        pass

    @abstractmethod
    def clear_working(self) -> None:
        """Clear all working context."""
        pass

    @abstractmethod
    def get_all_working(self) -> Dict[str, Any]:
        """Get all working context as dict."""
        pass

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

    @abstractmethod
    def export_all(self) -> Dict[str, Any]:
        """Export all data for backup."""
        pass

    @abstractmethod
    def import_all(self, data: Dict[str, Any]) -> None:
        """Import data from backup."""
        pass

    def get_storage_path(self) -> Path:
        """Get the base storage path for this namespace."""
        return self.config.base_dir / self.config.namespace
