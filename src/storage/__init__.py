"""
SmartContext Storage Backends

Provides multiple storage implementations:
- JSONStorage: File-based JSON storage (default)
- SQLiteStorage: SQLite database storage
- EncryptedStorage: Encryption wrapper for any backend
"""

from .base import StorageBackend, StorageConfig
from .json_storage import JSONStorage
from .sqlite_storage import SQLiteStorage

__all__ = [
    "StorageBackend",
    "StorageConfig",
    "JSONStorage",
    "SQLiteStorage",
]
