"""
SmartContext Attention System

Manages attention allocation across different context types.
Provides policies for different task modes (coding, debugging, etc.)

Enhanced with Context Compiler features from mcp-tools-servers:
- Attention Budget Allocation - Distribute limited context window optimally
- Adaptive Resolution Pyramid - Critical content gets full detail, background gets summary
- Semantic Chunking - Break content at logical boundaries, not arbitrary splits
- Cross-Reference Graph - Show relationships between context pieces
- Temporal Decay Weighting - Recent content gets more detail than old
- Contextual Retrieval - Add provenance to each chunk
- Attention Sinks - Put most important content at start and end

Based on Research:
- Anthropic ACE: Context as computed view, not accumulation
- Google ADK: Resolution pyramids for multi-level detail
- Anthropic Contextual Retrieval: Chunk context matters
- Meta In-Context Pretraining: Ordering and adjacency matter
- Attention Sinks research: First tokens get disproportionate attention
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal, Tuple
from enum import Enum
from datetime import datetime
from collections import defaultdict
import re
import math


class AttentionMode(Enum):
    """Available attention modes."""
    CODING = "coding"
    DEBUGGING = "debugging"
    EXPLORING = "exploring"
    PLANNING = "planning"
    BALANCED = "balanced"


@dataclass
class AttentionPolicy:
    """Defines how to allocate attention across context types."""
    name: str
    mode: AttentionMode
    description: str
    allocation: Dict[str, float]  # Category -> percentage (0.0 to 1.0)

    def get_tokens(self, category: str, total_budget: int) -> int:
        """Get token allocation for a category."""
        percentage = self.allocation.get(category, 0.0)
        return int(total_budget * percentage)

    def format_allocation(self, total_budget: int = 4000) -> str:
        """Format allocation as visual bars."""
        lines = []
        for category, percentage in sorted(self.allocation.items(),
                                           key=lambda x: x[1], reverse=True):
            tokens = self.get_tokens(category, total_budget)
            bar = "█" * int(percentage * 20) + "░" * (20 - int(percentage * 20))
            lines.append(f"  {category:12} {bar} {percentage:>5.0%} ({tokens:>4} tokens)")
        return "\n".join(lines)


# Pre-defined policies
POLICIES: Dict[AttentionMode, AttentionPolicy] = {
    AttentionMode.CODING: AttentionPolicy(
        name="Task-Focused",
        mode=AttentionMode.CODING,
        description="Focus on the current coding task",
        allocation={
            "goal": 0.40,
            "memories": 0.20,
            "observations": 0.15,
            "artifacts": 0.10,
            "strategy": 0.10,
            "system": 0.05
        }
    ),
    AttentionMode.DEBUGGING: AttentionPolicy(
        name="Debugging",
        mode=AttentionMode.DEBUGGING,
        description="Focus on errors and recent observations",
        allocation={
            "observations": 0.50,
            "memories": 0.20,
            "goal": 0.15,
            "artifacts": 0.10,
            "system": 0.05,
            "strategy": 0.00
        }
    ),
    AttentionMode.EXPLORING: AttentionPolicy(
        name="Exploration",
        mode=AttentionMode.EXPLORING,
        description="Focus on understanding and learning",
        allocation={
            "memories": 0.30,
            "artifacts": 0.25,
            "goal": 0.20,
            "observations": 0.15,
            "system": 0.05,
            "strategy": 0.05
        }
    ),
    AttentionMode.PLANNING: AttentionPolicy(
        name="Planning",
        mode=AttentionMode.PLANNING,
        description="Focus on architecture and design",
        allocation={
            "goal": 0.30,
            "memories": 0.25,
            "artifacts": 0.15,
            "strategy": 0.15,
            "observations": 0.10,
            "system": 0.05
        }
    ),
    AttentionMode.BALANCED: AttentionPolicy(
        name="Balanced",
        mode=AttentionMode.BALANCED,
        description="Even distribution across all categories",
        allocation={
            "goal": 0.20,
            "memories": 0.20,
            "observations": 0.20,
            "artifacts": 0.15,
            "strategy": 0.15,
            "system": 0.10
        }
    )
}


class AttentionManager:
    """
    Manages attention mode and auto-detection.
    """

    # Keywords for auto-detection
    MODE_KEYWORDS = {
        AttentionMode.DEBUGGING: [
            "bug", "error", "broken", "fix", "crash", "fail",
            "issue", "wrong", "not working", "exception", "stack trace"
        ],
        AttentionMode.PLANNING: [
            "plan", "architect", "design", "structure", "approach",
            "strategy", "how should", "what's the best way"
        ],
        AttentionMode.EXPLORING: [
            "what is", "how does", "explain", "understand", "explore",
            "show me", "tell me about", "where is", "find"
        ]
    }

    def __init__(self, default_mode: AttentionMode = AttentionMode.CODING):
        self.current_mode = default_mode
        self.auto_detect = True

    def get_policy(self) -> AttentionPolicy:
        """Get the current attention policy."""
        return POLICIES[self.current_mode]

    def set_mode(self, mode: AttentionMode):
        """Manually set the attention mode."""
        self.current_mode = mode

    def detect_mode(self, message: str) -> Optional[AttentionMode]:
        """
        Auto-detect the appropriate mode from message content.
        Returns None if no specific mode is detected.
        """
        message_lower = message.lower()

        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                return mode

        return None

    def maybe_switch_mode(self, message: str) -> bool:
        """
        Auto-switch mode if detected and auto_detect is enabled.
        Returns True if mode was switched.
        """
        if not self.auto_detect:
            return False

        detected = self.detect_mode(message)
        if detected and detected != self.current_mode:
            self.current_mode = detected
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current attention status."""
        policy = self.get_policy()
        return {
            "mode": self.current_mode.value,
            "policy_name": policy.name,
            "description": policy.description,
            "auto_detect": self.auto_detect,
            "allocation": policy.allocation
        }


@dataclass
class CompiledContext:
    """The result of context compilation."""
    mode: AttentionMode
    policy: AttentionPolicy
    sections: Dict[str, str]  # Category -> content
    token_usage: Dict[str, int]  # Category -> tokens used
    total_tokens: int
    budget: int

    def to_prompt(self) -> str:
        """Convert to a prompt string."""
        lines = []
        for category in ["system", "goal", "memories", "observations", "artifacts", "strategy"]:
            if category in self.sections and self.sections[category]:
                lines.append(f"[{category.upper()}]")
                lines.append(self.sections[category])
                lines.append("")
        return "\n".join(lines)

    def get_usage_summary(self) -> str:
        """Get a summary of token usage."""
        lines = [f"Token Usage: {self.total_tokens}/{self.budget}"]
        for category, tokens in sorted(self.token_usage.items(),
                                        key=lambda x: x[1], reverse=True):
            if tokens > 0:
                lines.append(f"  {category}: {tokens}")
        return "\n".join(lines)


@dataclass
class CompiledSection:
    """
    A compiled section of the prompt with metadata.
    (Distilled from mcp_server_context_compiler.py)
    """
    section_type: str
    content: str
    token_count: int
    resolution_level: str  # "full", "summary", "reference"
    provenance: Dict[str, Any]
    importance_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class ContextCompiler:
    """
    Compiles optimal context from memory using attention policies.

    Enhanced with features from mcp_server_context_compiler.py:
    - Semantic chunking at logical boundaries
    - Temporal decay for observations
    - Context graph for relationships
    - Quality scoring
    - Attention sinks optimization
    """

    def __init__(self, attention: AttentionManager):
        self.attention = attention
        self.compilation_history: List[Dict] = []
        self.quality_metrics: List[float] = []

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(str(text)) // 4

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def summarize(self, text: str, max_tokens: int) -> str:
        """Simple summarization by truncation with sentence boundary."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Try to break at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.7:
            return truncated[:last_period + 1]

        return truncated + "..."

    def semantic_chunk(self, content: str, content_type: str = "text") -> List[str]:
        """
        Chunk content at semantic boundaries.
        (Distilled from mcp_server_context_compiler.py)
        """
        chunks = []

        if content_type == 'code':
            # Split at function boundaries
            patterns = [r'\ndef ', r'\nfunction ', r'\nclass ', r'\nconst ', r'\nlet ']
            current_chunk = []
            lines = content.split('\n')

            for line in lines:
                current_chunk.append(line)
                # Check if this line starts a new semantic unit
                if any(re.search(pattern, '\n' + line) for pattern in patterns):
                    if len(current_chunk) > 1:
                        chunks.append('\n'.join(current_chunk[:-1]))
                        current_chunk = [line]

            if current_chunk:
                chunks.append('\n'.join(current_chunk))

        elif content_type in ['document', 'markdown']:
            # Split at paragraph boundaries
            paragraphs = content.split('\n\n')
            chunks = [p for p in paragraphs if p.strip()]

        else:
            # Default: keep as single chunk
            chunks = [content]

        return chunks if chunks else [content]

    def apply_temporal_decay(
        self,
        items: List[Dict],
        decay_rate: float = 0.15
    ) -> List[Tuple[float, Dict]]:
        """
        Apply temporal decay weighting to items.
        More recent items get higher weight.
        (Distilled from mcp_server_context_compiler.py)
        """
        # Sort by timestamp (most recent first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )

        weighted = []
        for i, item in enumerate(sorted_items):
            decay_factor = max(0.3, 1.0 - (i * decay_rate))
            weighted.append((decay_factor, item))

        return weighted

    def build_context_graph(
        self,
        sections: List[CompiledSection]
    ) -> Dict[str, Any]:
        """
        Build cross-reference graph showing relationships.
        (Distilled from mcp_server_context_compiler.py)
        """
        graph = {
            'nodes': [],
            'edges': []
        }

        # Add nodes
        for section in sections:
            graph['nodes'].append({
                'id': section.section_type,
                'type': section.section_type,
                'importance': section.importance_score,
                'tokens': section.token_count
            })

        section_types = {s.section_type for s in sections}

        # Add edges (relationships)
        if 'goal' in section_types:
            if 'memories' in section_types:
                graph['edges'].append({
                    'from': 'goal',
                    'to': 'memories',
                    'relationship': 'informed_by'
                })
            if 'observations' in section_types:
                graph['edges'].append({
                    'from': 'goal',
                    'to': 'observations',
                    'relationship': 'guides'
                })

        if 'observations' in section_types and 'artifacts' in section_types:
            graph['edges'].append({
                'from': 'observations',
                'to': 'artifacts',
                'relationship': 'references'
            })

        return graph

    def optimize_section_order(
        self,
        sections: List[CompiledSection]
    ) -> List[CompiledSection]:
        """
        Optimize ordering using attention sinks principle.
        Most important content at START and END.
        (Distilled from mcp_server_context_compiler.py)
        """
        priority_order = {
            'system': 0,      # Very beginning (attention sink #1)
            'goal': 1,        # Right after system
            'observations': 2,
            'memories': 3,
            'artifacts': 4,
            'strategy': 5     # End (attention sink #2)
        }

        return sorted(
            sections,
            key=lambda s: priority_order.get(s.section_type, 99)
        )

    def calculate_quality_score(
        self,
        sections: List[CompiledSection],
        max_tokens: int
    ) -> float:
        """
        Calculate quality score for compilation.
        (Distilled from mcp_server_context_compiler.py)
        """
        if not sections:
            return 0.0

        total_tokens = sum(s.token_count for s in sections)

        # Factor 1: Budget utilization
        utilization = min(1.0, total_tokens / max_tokens) if max_tokens > 0 else 0

        # Factor 2: Coverage (6 possible sections)
        coverage = len(sections) / 6

        # Factor 3: Resolution quality
        full_detail_ratio = len([
            s for s in sections if s.resolution_level == 'full'
        ]) / len(sections)

        # Weighted average
        score = (utilization * 0.4 + coverage * 0.3 + full_detail_ratio * 0.3) * 100

        return round(score, 2)

    def compile_section(
        self,
        section_type: str,
        content: Any,
        token_budget: int,
        temporal_decay: bool = False,
        include_provenance: bool = True
    ) -> Optional[CompiledSection]:
        """
        Compile a single section with adaptive resolution.
        (Distilled from mcp_server_context_compiler.py)
        """
        if not content or token_budget == 0:
            return None

        # Handle different content types
        if isinstance(content, list):
            if temporal_decay and content:
                # Apply temporal decay to list items
                weighted = self.apply_temporal_decay(
                    [{'content': c, 'timestamp': ''} for c in content]
                    if isinstance(content[0], str) else content
                )
                # Build content with decay-based resolution
                compiled_items = []
                tokens_used = 0
                for decay_factor, item in weighted:
                    if tokens_used >= token_budget:
                        break
                    item_content = item.get('content', str(item)) if isinstance(item, dict) else str(item)
                    if decay_factor > 0.7:
                        resolution = 'full'
                        text = item_content
                    elif decay_factor > 0.4:
                        resolution = 'summary'
                        text = self.summarize(item_content, 50)
                    else:
                        resolution = 'reference'
                        text = self.summarize(item_content, 20)

                    item_tokens = self.estimate_tokens(text)
                    if tokens_used + item_tokens <= token_budget:
                        compiled_items.append(text)
                        tokens_used += item_tokens

                content_str = "\n".join(compiled_items)
                resolution_level = 'full'  # Mixed
            else:
                content_str = "\n".join(str(c) for c in content)
                content_str = self.truncate_to_tokens(content_str, token_budget)
                resolution_level = 'full'
        else:
            content_str = self.truncate_to_tokens(str(content), token_budget)
            resolution_level = 'full'

        token_count = self.estimate_tokens(content_str)

        # Determine importance score
        importance_map = {
            'system': 1.0,
            'goal': 1.0,
            'observations': 0.8,
            'memories': 0.7,
            'artifacts': 0.6,
            'strategy': 0.5
        }

        provenance = {'source': section_type}
        if include_provenance and isinstance(content, list):
            provenance['count'] = len(content)

        return CompiledSection(
            section_type=section_type,
            content=content_str,
            token_count=token_count,
            resolution_level=resolution_level,
            provenance=provenance,
            importance_score=importance_map.get(section_type, 0.5)
        )

    def compile(
        self,
        goal: str = "",
        memories: List[str] = None,
        observations: List[str] = None,
        artifacts: List[str] = None,
        strategy: str = "",
        system: str = "",
        token_budget: int = 4000,
        include_context_graph: bool = True,
        temporal_decay: bool = True,
        semantic_chunking: bool = True
    ) -> CompiledContext:
        """
        Compile context according to current attention policy.

        Enhanced with:
        - Attention budget allocation per section
        - Temporal decay for observations
        - Context graph for relationships
        - Optimized section ordering

        Args:
            goal: Current task/goal
            memories: Relevant memories from long-term storage
            observations: Recent observations/outputs
            artifacts: Referenced artifact summaries
            strategy: Approach/constraints
            system: System-level instructions
            token_budget: Total token budget
            include_context_graph: Build relationship graph
            temporal_decay: Apply decay to observations
            semantic_chunking: Chunk at logical boundaries

        Returns:
            CompiledContext with optimally allocated sections
        """
        policy = self.attention.get_policy()

        # Allocate token budget per section
        budget = {}
        for category, percentage in policy.allocation.items():
            budget[category] = int(token_budget * percentage)

        # Compile each section
        compiled_sections: List[CompiledSection] = []

        # System (minimal, high importance)
        if system:
            section = self.compile_section('system', system, budget.get('system', 200))
            if section:
                compiled_sections.append(section)

        # Goal (high importance)
        if goal:
            section = self.compile_section('goal', f"Current Goal: {goal}", budget.get('goal', 600))
            if section:
                compiled_sections.append(section)

        # Observations (with temporal decay)
        if observations:
            section = self.compile_section(
                'observations',
                observations,
                budget.get('observations', 600),
                temporal_decay=temporal_decay
            )
            if section:
                compiled_sections.append(section)

        # Memories (with provenance)
        if memories:
            section = self.compile_section(
                'memories',
                memories,
                budget.get('memories', 800),
                include_provenance=True
            )
            if section:
                compiled_sections.append(section)

        # Artifacts (with semantic chunking)
        if artifacts:
            if semantic_chunking:
                chunked_artifacts = []
                for artifact in artifacts:
                    chunks = self.semantic_chunk(artifact, 'code')
                    chunked_artifacts.extend(chunks)
                artifacts = chunked_artifacts

            section = self.compile_section(
                'artifacts',
                artifacts,
                budget.get('artifacts', 600)
            )
            if section:
                compiled_sections.append(section)

        # Strategy
        if strategy:
            section = self.compile_section('strategy', strategy, budget.get('strategy', 400))
            if section:
                compiled_sections.append(section)

        # Build context graph
        context_graph = {}
        if include_context_graph:
            context_graph = self.build_context_graph(compiled_sections)

        # Optimize section ordering (attention sinks)
        ordered_sections = self.optimize_section_order(compiled_sections)

        # Calculate quality score
        quality_score = self.calculate_quality_score(ordered_sections, token_budget)

        # Record compilation
        self.compilation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'strategy': self.attention.current_mode.value,
            'token_budget': token_budget,
            'sections': len(ordered_sections),
            'quality_score': quality_score
        })
        self.quality_metrics.append(quality_score)

        # Keep only last 100 compilations
        if len(self.compilation_history) > 100:
            self.compilation_history = self.compilation_history[-100:]
            self.quality_metrics = self.quality_metrics[-100:]

        # Build sections dict and token usage
        sections = {}
        token_usage = {}
        total_used = 0

        for section in ordered_sections:
            sections[section.section_type] = section.content
            token_usage[section.section_type] = section.token_count
            total_used += section.token_count

        result = CompiledContext(
            mode=self.attention.current_mode,
            policy=policy,
            sections=sections,
            token_usage=token_usage,
            total_tokens=total_used,
            budget=token_budget
        )

        # Add enhanced metadata
        result.compiled_sections = ordered_sections
        result.context_graph = context_graph
        result.quality_score = quality_score

        return result

    def get_compilation_stats(self, last_n: int = 10) -> Dict[str, Any]:
        """
        Get statistics about compilation quality over time.
        (Distilled from mcp_server_context_compiler.py)
        """
        history = self.compilation_history[-last_n:]
        if not history:
            return {'success': False, 'error': 'No compilation history'}

        quality_scores = [c['quality_score'] for c in history]
        strategies_used = defaultdict(int)
        for c in history:
            strategies_used[c['strategy']] += 1

        # Calculate trend
        trend = "insufficient_data"
        if len(quality_scores) >= 2:
            mid = len(quality_scores) // 2
            first_half_avg = sum(quality_scores[:mid]) / mid
            second_half_avg = sum(quality_scores[mid:]) / (len(quality_scores) - mid)
            diff = second_half_avg - first_half_avg
            if diff > 5:
                trend = "improving"
            elif diff < -5:
                trend = "declining"
            else:
                trend = "stable"

        return {
            'success': True,
            'total_compilations': len(history),
            'quality_scores': {
                'mean': sum(quality_scores) / len(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'latest': quality_scores[-1] if quality_scores else 0
            },
            'strategies_used': dict(strategies_used),
            'trend': trend
        }

    def explain_compilation(self) -> Dict[str, Any]:
        """
        Explain the last compilation decision.
        (Distilled from mcp_server_context_compiler.py)
        """
        if not self.compilation_history:
            return {'success': False, 'error': 'No compilation history'}

        last = self.compilation_history[-1]
        policy = self.attention.get_policy()

        explanation = {
            'timestamp': last['timestamp'],
            'strategy_used': last['strategy'],
            'quality_score': last['quality_score'],
            'attention_policy': policy.allocation,
            'decisions': []
        }

        # Add strategy-specific explanations
        if last['strategy'] == 'coding':
            explanation['decisions'].append({
                'decision': 'Allocated 40% of tokens to goal',
                'rationale': 'Task-focused strategy prioritizes goal-relevant content'
            })
        elif last['strategy'] == 'debugging':
            explanation['decisions'].append({
                'decision': 'Allocated 50% of tokens to observations',
                'rationale': 'Debugging needs detailed recent execution history'
            })
        elif last['strategy'] == 'exploring':
            explanation['decisions'].append({
                'decision': 'Balanced allocation across memories and artifacts',
                'rationale': 'Exploration benefits from breadth over depth'
            })

        return {'success': True, 'explanation': explanation}
