"""
SmartContext SQLite Storage Backend

SQLite database storage implementation.
Better for concurrency, querying, and larger datasets.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager
from .base import StorageBackend, StorageConfig


class SQLiteStorage(StorageBackend):
    """
    SQLite database storage backend.

    Features:
    - Better concurrency than JSON files
    - Full-text search support
    - Transaction support
    - Better performance for large datasets

    Schema:
        memories: id, content, tags, importance, source, created_at, metadata
        artifacts: id, name, content, content_type, size, created_at
        sessions: id, data, created_at, updated_at
        session_items: id, session_id, role, content, timestamp, metadata
        working: key, value, updated_at
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        super().__init__(config)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _ensure_storage(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.get_storage_path().mkdir(parents=True, exist_ok=True)

    def _db_path(self) -> Path:
        """Get database file path."""
        return self.get_storage_path() / "smartcontext.db"

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path()),
                check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Memories table
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT,
                    importance TEXT DEFAULT 'normal',
                    source TEXT DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    metadata TEXT
                );

                -- Full-text search for memories
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags,
                    content='memories',
                    content_rowid='rowid'
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (NEW.rowid, NEW.content, NEW.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES('delete', OLD.rowid, OLD.content, OLD.tags);
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (NEW.rowid, NEW.content, NEW.tags);
                END;

                -- Artifacts table
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    size INTEGER,
                    created_at TEXT NOT NULL
                );

                -- Sessions table
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- Session items for efficient querying
                CREATE TABLE IF NOT EXISTS session_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_session_items_session
                ON session_items(session_id);

                -- Working context table
                CREATE TABLE IF NOT EXISTS working (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance);

                CREATE INDEX IF NOT EXISTS idx_artifacts_name
                ON artifacts(name);
            """)
            conn.commit()

    # ========================================================================
    # Memory Operations
    # ========================================================================

    def save_memory(self, memory_id: str, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, tags, importance, source, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                data.get("content", ""),
                json.dumps(data.get("tags", [])),
                data.get("importance", "normal"),
                data.get("source", "user"),
                data.get("created_at", datetime.now().isoformat()),
                json.dumps(data.get("metadata", {}))
            ))
            conn.commit()

    def load_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                return self._row_to_memory(row)
            return None

    def delete_memory(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_memories(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Full-text search using FTS5."""
        with self._get_connection() as conn:
            # Use FTS5 for efficient full-text search
            rows = conn.execute("""
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()

            if rows:
                return [self._row_to_memory(row) for row in rows]

            # Fallback to LIKE if FTS returns nothing
            like_query = f"%{query}%"
            rows = conn.execute("""
                SELECT * FROM memories
                WHERE content LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (like_query, like_query, limit)).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def _row_to_memory(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to memory dict."""
        return {
            "id": row["id"],
            "content": row["content"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "importance": row["importance"],
            "source": row["source"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }

    # ========================================================================
    # Artifact Operations
    # ========================================================================

    def save_artifact(self, artifact_id: str, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            content = data.get("content", "")
            conn.execute("""
                INSERT OR REPLACE INTO artifacts
                (id, name, content, content_type, size, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                artifact_id,
                data.get("name", artifact_id),
                content,
                data.get("content_type", "text"),
                len(content),
                data.get("created_at", datetime.now().isoformat())
            ))
            conn.commit()

    def load_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            # Try by ID first
            row = conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()

            # Try by name if not found
            if not row:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE name = ?", (artifact_id,)
                ).fetchone()

            if row:
                return self._row_to_artifact(row)
            return None

    def delete_artifact(self, artifact_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM artifacts WHERE id = ? OR name = ?",
                (artifact_id, artifact_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_artifacts(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_artifact(row) for row in rows]

    def _row_to_artifact(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to artifact dict."""
        return {
            "id": row["id"],
            "name": row["name"],
            "content": row["content"],
            "content_type": row["content_type"],
            "size": row["size"],
            "created_at": row["created_at"]
        }

    # ========================================================================
    # Session Operations
    # ========================================================================

    def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            now = datetime.now().isoformat()
            conn.execute("""
                INSERT OR REPLACE INTO sessions (id, data, created_at, updated_at)
                VALUES (?, ?, COALESCE(
                    (SELECT created_at FROM sessions WHERE id = ?), ?
                ), ?)
            """, (session_id, json.dumps(data), session_id, now, now))
            conn.commit()

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                return json.loads(row["data"])
            return None

    def delete_session(self, session_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_sessions(self) -> List[str]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [row["id"] for row in rows]

    # ========================================================================
    # Working Context Operations
    # ========================================================================

    def set_working(self, key: str, value: Any) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO working (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), datetime.now().isoformat()))
            conn.commit()

    def get_working(self, key: str) -> Optional[Any]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM working WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
            return None

    def clear_working(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM working")
            conn.commit()

    def get_all_working(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM working").fetchall()
            return {row["key"]: json.loads(row["value"]) for row in rows}

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            memory_count = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            artifact_count = conn.execute(
                "SELECT COUNT(*) FROM artifacts"
            ).fetchone()[0]
            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            working_count = conn.execute(
                "SELECT COUNT(*) FROM working"
            ).fetchone()[0]

            db_size = self._db_path().stat().st_size if self._db_path().exists() else 0

            return {
                "backend": "sqlite",
                "namespace": self.config.namespace,
                "storage_path": str(self._db_path()),
                "memory_count": memory_count,
                "artifact_count": artifact_count,
                "session_count": session_count,
                "working_keys": working_count,
                "total_size_bytes": db_size
            }

    def export_all(self) -> Dict[str, Any]:
        """Export all data for backup/migration."""
        with self._get_connection() as conn:
            memories = {
                row["id"]: self._row_to_memory(row)
                for row in conn.execute("SELECT * FROM memories").fetchall()
            }
            artifacts = {
                row["id"]: self._row_to_artifact(row)
                for row in conn.execute("SELECT * FROM artifacts").fetchall()
            }
            sessions = {}
            for row in conn.execute("SELECT id, data FROM sessions").fetchall():
                sessions[row["id"]] = json.loads(row["data"])

            return {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "namespace": self.config.namespace,
                "backend": "sqlite",
                "memories": memories,
                "artifacts": artifacts,
                "sessions": sessions,
                "working": self.get_all_working()
            }

    def import_all(self, data: Dict[str, Any]) -> None:
        """Import data from backup/migration."""
        with self._get_connection() as conn:
            # Memories
            for memory_id, memory_data in data.get("memories", {}).items():
                self.save_memory(memory_id, memory_data)

            # Artifacts
            for artifact_id, artifact_data in data.get("artifacts", {}).items():
                self.save_artifact(artifact_id, artifact_data)

            # Sessions
            for session_id, session_data in data.get("sessions", {}).items():
                if session_data:
                    self.save_session(session_id, session_data)

            # Working
            for key, value in data.get("working", {}).items():
                self.set_working(key, value)

            conn.commit()

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
