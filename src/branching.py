"""
SmartContext Conversation Branching

Enables "what if" exploration by forking conversation state.
Based on OpenAI AdvancedSQLiteSession patterns.

Use cases:
- Explore alternative approaches
- Compare different solutions
- Save checkpoints before risky operations
- A/B testing prompts
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
import copy

from session import SessionItem, BaseSession, TrimConfig

if TYPE_CHECKING:
    from storage.base import StorageBackend


# ============================================================================
# Branch Model
# ============================================================================

@dataclass
class Branch:
    """Represents a conversation branch."""
    id: str
    name: str
    parent_id: Optional[str] = None  # ID of parent branch (None for main)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    is_active: bool = False
    checkpoint_index: int = 0  # Item index at branch point
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "description": self.description,
            "is_active": self.is_active,
            "checkpoint_index": self.checkpoint_index,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Branch":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            description=data.get("description", ""),
            is_active=data.get("is_active", False),
            checkpoint_index=data.get("checkpoint_index", 0),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# Branch Manager
# ============================================================================

class BranchManager:
    """
    Manages conversation branches for exploration.

    Example usage:
        manager = BranchManager(session)

        # Create a branch to try a different approach
        manager.create_branch("try-recursion", "Try recursive solution")

        # Work on the branch...

        # Switch back to main
        manager.switch_branch("main")

        # Compare results
        comparison = manager.compare_branches("main", "try-recursion")
    """

    def __init__(
        self,
        session: Optional[BaseSession] = None,
        storage: Optional["StorageBackend"] = None
    ):
        self.session = session or BaseSession()
        self.storage = storage

        # Branch registry
        self.branches: Dict[str, Branch] = {}
        self.branch_sessions: Dict[str, BaseSession] = {}

        # Create main branch
        self._init_main_branch()

        self.current_branch_id = "main"

    def _init_main_branch(self) -> None:
        """Initialize the main branch."""
        main = Branch(
            id="main",
            name="main",
            is_active=True,
            description="Main conversation branch"
        )
        self.branches["main"] = main
        self.branch_sessions["main"] = self.session

    def create_branch(
        self,
        name: str,
        description: str = "",
        from_branch: Optional[str] = None
    ) -> Branch:
        """
        Create a new branch from current state.

        Args:
            name: Branch name (must be unique)
            description: Optional description
            from_branch: Branch to fork from (default: current)

        Returns:
            The created Branch
        """
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")

        # Determine parent branch
        parent_id = from_branch or self.current_branch_id
        parent_session = self.branch_sessions.get(parent_id, self.session)

        # Create branch metadata
        branch = Branch(
            id=name,
            name=name,
            parent_id=parent_id,
            description=description,
            checkpoint_index=len(parent_session.get_items())
        )

        # Copy session state
        branch_session = self._copy_session(parent_session)

        # Register
        self.branches[name] = branch
        self.branch_sessions[name] = branch_session

        return branch

    def switch_branch(self, name: str) -> Branch:
        """
        Switch to a different branch.

        Args:
            name: Branch name to switch to

        Returns:
            The activated Branch
        """
        if name not in self.branches:
            raise ValueError(f"Branch '{name}' not found")

        # Deactivate current
        if self.current_branch_id in self.branches:
            self.branches[self.current_branch_id].is_active = False

        # Activate new
        self.current_branch_id = name
        self.branches[name].is_active = True
        self.session = self.branch_sessions[name]

        return self.branches[name]

    def delete_branch(self, name: str) -> bool:
        """
        Delete a branch.

        Args:
            name: Branch name to delete

        Returns:
            True if deleted, False if not found
        """
        if name == "main":
            raise ValueError("Cannot delete main branch")

        if name not in self.branches:
            return False

        # Switch away if current
        if self.current_branch_id == name:
            self.switch_branch("main")

        del self.branches[name]
        del self.branch_sessions[name]

        return True

    def merge_branch(
        self,
        source: str,
        target: str = "main",
        strategy: str = "append"
    ) -> int:
        """
        Merge one branch into another.

        Args:
            source: Branch to merge from
            target: Branch to merge into
            strategy: Merge strategy ("append" or "replace")

        Returns:
            Number of items merged
        """
        if source not in self.branches:
            raise ValueError(f"Source branch '{source}' not found")
        if target not in self.branches:
            raise ValueError(f"Target branch '{target}' not found")

        source_session = self.branch_sessions[source]
        target_session = self.branch_sessions[target]

        source_items = source_session.get_items()
        checkpoint = self.branches[source].checkpoint_index

        # Get items added after the branch point
        new_items = source_items[checkpoint:]

        if strategy == "replace":
            # Replace target items from checkpoint onwards
            target_items = target_session.get_items()[:checkpoint]
            target_session._items = target_items + new_items
        else:
            # Append new items
            target_session.add_items(new_items)

        return len(new_items)

    def get_current_branch(self) -> Branch:
        """Get the currently active branch."""
        return self.branches[self.current_branch_id]

    def get_current_session(self) -> BaseSession:
        """Get the session for the current branch."""
        return self.branch_sessions[self.current_branch_id]

    def list_branches(self) -> List[Branch]:
        """List all branches."""
        return list(self.branches.values())

    def get_branch(self, name: str) -> Optional[Branch]:
        """Get a branch by name."""
        return self.branches.get(name)

    def compare_branches(
        self,
        branch1: str,
        branch2: str
    ) -> Dict[str, Any]:
        """
        Compare two branches.

        Returns statistics about differences.
        """
        if branch1 not in self.branches:
            raise ValueError(f"Branch '{branch1}' not found")
        if branch2 not in self.branches:
            raise ValueError(f"Branch '{branch2}' not found")

        session1 = self.branch_sessions[branch1]
        session2 = self.branch_sessions[branch2]

        items1 = session1.get_items()
        items2 = session2.get_items()

        # Find common ancestor (if any)
        common_count = 0
        for i, (a, b) in enumerate(zip(items1, items2)):
            if a.content == b.content and a.role == b.role:
                common_count = i + 1
            else:
                break

        return {
            "branch1": {
                "name": branch1,
                "item_count": len(items1),
                "unique_items": len(items1) - common_count,
                "tokens": session1.estimate_total_tokens()
            },
            "branch2": {
                "name": branch2,
                "item_count": len(items2),
                "unique_items": len(items2) - common_count,
                "tokens": session2.estimate_total_tokens()
            },
            "common_ancestor_items": common_count,
            "divergence_point": common_count
        }

    def get_branch_tree(self) -> Dict[str, Any]:
        """Get branch hierarchy as tree structure."""
        def build_tree(branch_id: str) -> Dict[str, Any]:
            branch = self.branches[branch_id]
            children = [
                b for b in self.branches.values()
                if b.parent_id == branch_id
            ]
            return {
                "id": branch.id,
                "name": branch.name,
                "is_active": branch.is_active,
                "item_count": len(self.branch_sessions[branch_id].get_items()),
                "children": [build_tree(c.id) for c in children]
            }

        return build_tree("main")

    def _copy_session(self, session: BaseSession) -> BaseSession:
        """Deep copy a session."""
        new_session = BaseSession(
            session_id=f"{session.session_id}-branch-{len(self.branches)}",
            trim_config=copy.copy(session.trim_config)
        )
        new_session._items = [
            SessionItem(
                role=item.role,
                content=item.content,
                timestamp=item.timestamp,
                metadata=copy.copy(item.metadata)
            )
            for item in session.get_items()
        ]
        return new_session

    def to_dict(self) -> Dict[str, Any]:
        """Serialize branch manager state."""
        return {
            "current_branch": self.current_branch_id,
            "branches": {
                name: branch.to_dict()
                for name, branch in self.branches.items()
            },
            "sessions": {
                name: session.to_dict()
                for name, session in self.branch_sessions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchManager":
        """Deserialize branch manager state."""
        manager = cls(session=None)

        # Clear default state
        manager.branches = {}
        manager.branch_sessions = {}

        # Restore branches
        for name, branch_data in data.get("branches", {}).items():
            manager.branches[name] = Branch.from_dict(branch_data)

        # Restore sessions
        for name, session_data in data.get("sessions", {}).items():
            manager.branch_sessions[name] = BaseSession.from_dict(session_data)

        # Set current
        manager.current_branch_id = data.get("current_branch", "main")
        if manager.current_branch_id in manager.branch_sessions:
            manager.session = manager.branch_sessions[manager.current_branch_id]

        return manager
