"""
SmartContext Context Compression

Provides multiple strategies for compressing context:
- Heuristic: Fast, no API calls, keyword extraction
- LLM: Uses Claude API for intelligent summarization
- Hybrid: Heuristic first, LLM for important content

Based on OpenAI Agents SDK compression patterns.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from session import SessionItem


# ============================================================================
# Compression Strategies
# ============================================================================

class CompressionStrategy(Enum):
    """Available compression strategies."""
    NONE = "none"           # No compression
    HEURISTIC = "heuristic" # Fast, rule-based
    LLM = "llm"             # Claude API summarization
    HYBRID = "hybrid"       # Heuristic + LLM when needed


@dataclass
class CompressionConfig:
    """Configuration for compression."""
    strategy: CompressionStrategy = CompressionStrategy.HYBRID
    target_ratio: float = 0.3           # Target compression ratio (0.3 = 30% of original)
    min_items_to_compress: int = 10     # Don't compress if fewer items
    token_threshold: int = 2000         # Compress when exceeding this
    preserve_recent: int = 5            # Always keep last N items uncompressed
    preserve_system: bool = True        # Always keep system messages
    llm_model: str = "claude-sonnet-4-20250514"  # Model for LLM compression


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_items: int
    original_tokens: int
    compressed_items: int
    compressed_tokens: int
    strategy_used: CompressionStrategy
    summary: str
    preserved_items: List[SessionItem]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original tokens."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens


# ============================================================================
# Heuristic Compression
# ============================================================================

class HeuristicCompressor:
    """
    Fast, rule-based compression without API calls.

    Techniques:
    - Remove filler words and phrases
    - Extract key sentences (first/last of each turn)
    - Preserve code blocks and important markers
    - Deduplicate similar content
    """

    # Filler patterns to remove
    FILLER_PATTERNS = [
        r'\b(um|uh|like|you know|basically|actually|literally|honestly)\b',
        r'\b(I think|I believe|I guess|In my opinion)\b',
        r'\b(kind of|sort of|a bit|a little)\b',
        r'\b(very|really|quite|pretty|fairly)\b',
        r'^\s*(okay|ok|alright|sure|right)\s*[,.]?\s*',
    ]

    # Patterns that indicate important content
    IMPORTANT_MARKERS = [
        r'```[\s\S]*?```',  # Code blocks
        r'error|exception|bug|fix|issue',
        r'important|critical|urgent|must|required',
        r'\d+\.\d+\.\d+',  # Version numbers
        r'https?://\S+',   # URLs
        r'`[^`]+`',        # Inline code
    ]

    def compress(
        self,
        items: List[SessionItem],
        config: CompressionConfig
    ) -> CompressionResult:
        """Compress items using heuristic rules."""
        if len(items) < config.min_items_to_compress:
            return self._no_compression_result(items)

        original_tokens = sum(item.estimate_tokens() for item in items)

        # Separate items to preserve vs compress
        preserve_count = config.preserve_recent
        items_to_compress = items[:-preserve_count] if preserve_count else items
        preserved_items = items[-preserve_count:] if preserve_count else []

        # Also preserve system messages if configured
        if config.preserve_system:
            system_items = [i for i in items_to_compress if i.role == "system"]
            items_to_compress = [i for i in items_to_compress if i.role != "system"]
            preserved_items = system_items + preserved_items

        # Compress the middle items
        compressed_content = self._compress_items(items_to_compress)

        # Create summary item
        summary_item = SessionItem(
            role="system",
            content=f"[Compressed conversation summary]\n{compressed_content}",
            metadata={"compressed": True, "original_count": len(items_to_compress)}
        )

        result_items = [summary_item] + preserved_items
        compressed_tokens = sum(item.estimate_tokens() for item in result_items)

        return CompressionResult(
            original_items=len(items),
            original_tokens=original_tokens,
            compressed_items=len(result_items),
            compressed_tokens=compressed_tokens,
            strategy_used=CompressionStrategy.HEURISTIC,
            summary=compressed_content,
            preserved_items=preserved_items
        )

    def _compress_items(self, items: List[SessionItem]) -> str:
        """Compress a list of items into a summary."""
        if not items:
            return ""

        # Group by role and extract key content
        summaries = []

        for item in items:
            # Skip very short items
            if len(item.content) < 20:
                continue

            # Clean the content
            cleaned = self._clean_content(item.content)

            # Extract key sentences
            key_sentences = self._extract_key_sentences(cleaned)

            if key_sentences:
                summaries.append(f"[{item.role.title()}]: {key_sentences}")

        # Deduplicate similar summaries
        unique_summaries = self._deduplicate(summaries)

        return "\n".join(unique_summaries)

    def _clean_content(self, content: str) -> str:
        """Remove filler words and clean up content."""
        result = content

        for pattern in self.FILLER_PATTERNS:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\n\s*\n', '\n', result)

        return result.strip()

    def _extract_key_sentences(self, content: str, max_sentences: int = 2) -> str:
        """Extract the most important sentences."""
        # Preserve code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return content[:200] if len(content) > 200 else content

        # Score sentences by importance
        scored = []
        for i, sentence in enumerate(sentences):
            score = 0

            # Position bonus (first and last sentences)
            if i == 0:
                score += 3
            if i == len(sentences) - 1:
                score += 2

            # Important markers bonus
            for pattern in self.IMPORTANT_MARKERS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    score += 2

            # Length penalty for very long sentences
            if len(sentence) > 200:
                score -= 1

            scored.append((score, sentence))

        # Sort by score and take top sentences
        scored.sort(key=lambda x: x[0], reverse=True)
        key_sentences = [s for _, s in scored[:max_sentences]]

        # Add back code blocks if any
        result = " ".join(key_sentences)
        if code_blocks:
            result += "\n" + "\n".join(code_blocks[:1])  # Keep first code block

        return result[:500]  # Limit length

    def _deduplicate(self, summaries: List[str], threshold: float = 0.7) -> List[str]:
        """Remove similar summaries."""
        if not summaries:
            return []

        unique = [summaries[0]]

        for summary in summaries[1:]:
            is_duplicate = False
            for existing in unique:
                similarity = self._similarity(summary, existing)
                if similarity > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(summary)

        return unique

    def _similarity(self, a: str, b: str) -> float:
        """Calculate simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    def _no_compression_result(self, items: List[SessionItem]) -> CompressionResult:
        """Return result when no compression needed."""
        total_tokens = sum(item.estimate_tokens() for item in items)
        return CompressionResult(
            original_items=len(items),
            original_tokens=total_tokens,
            compressed_items=len(items),
            compressed_tokens=total_tokens,
            strategy_used=CompressionStrategy.NONE,
            summary="",
            preserved_items=items
        )


# ============================================================================
# LLM Compression
# ============================================================================

class LLMCompressor:
    """
    LLM-based compression using Claude API.

    Uses Claude to intelligently summarize conversation history
    while preserving semantic meaning and important details.
    """

    COMPRESSION_PROMPT = """Summarize this conversation history into a concise summary.

IMPORTANT:
- Preserve key decisions, code snippets, and technical details
- Keep error messages and their resolutions
- Maintain important names, paths, and configurations
- Note any pending tasks or open questions
- Be factual, not interpretive

Conversation:
{conversation}

Provide a concise summary (aim for {target_tokens} tokens):"""

    def __init__(self, api_key: Optional[str] = None):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required for LLM compression")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def compress(
        self,
        items: List[SessionItem],
        config: CompressionConfig
    ) -> CompressionResult:
        """Compress items using Claude API."""
        if len(items) < config.min_items_to_compress:
            return self._no_compression_result(items)

        original_tokens = sum(item.estimate_tokens() for item in items)

        # Separate items to preserve vs compress
        preserve_count = config.preserve_recent
        items_to_compress = items[:-preserve_count] if preserve_count else items
        preserved_items = items[-preserve_count:] if preserve_count else []

        # Preserve system messages
        if config.preserve_system:
            system_items = [i for i in items_to_compress if i.role == "system"]
            items_to_compress = [i for i in items_to_compress if i.role != "system"]
            preserved_items = system_items + preserved_items

        # Format conversation for summarization
        conversation = self._format_conversation(items_to_compress)

        # Calculate target tokens
        target_tokens = int(original_tokens * config.target_ratio)

        # Call Claude API
        summary = self._call_claude(conversation, target_tokens, config.llm_model)

        # Create summary item
        summary_item = SessionItem(
            role="system",
            content=f"[AI-summarized conversation history]\n{summary}",
            metadata={
                "compressed": True,
                "original_count": len(items_to_compress),
                "compression_method": "llm"
            }
        )

        result_items = [summary_item] + preserved_items
        compressed_tokens = sum(item.estimate_tokens() for item in result_items)

        return CompressionResult(
            original_items=len(items),
            original_tokens=original_tokens,
            compressed_items=len(result_items),
            compressed_tokens=compressed_tokens,
            strategy_used=CompressionStrategy.LLM,
            summary=summary,
            preserved_items=preserved_items
        )

    def _format_conversation(self, items: List[SessionItem]) -> str:
        """Format items as readable conversation."""
        lines = []
        for item in items:
            role = item.role.title()
            # Truncate very long content
            content = item.content
            if len(content) > 1000:
                content = content[:1000] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    def _call_claude(self, conversation: str, target_tokens: int, model: str) -> str:
        """Call Claude API to generate summary."""
        prompt = self.COMPRESSION_PROMPT.format(
            conversation=conversation,
            target_tokens=target_tokens
        )

        response = self.client.messages.create(
            model=model,
            max_tokens=target_tokens + 100,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _no_compression_result(self, items: List[SessionItem]) -> CompressionResult:
        """Return result when no compression needed."""
        total_tokens = sum(item.estimate_tokens() for item in items)
        return CompressionResult(
            original_items=len(items),
            original_tokens=total_tokens,
            compressed_items=len(items),
            compressed_tokens=total_tokens,
            strategy_used=CompressionStrategy.NONE,
            summary="",
            preserved_items=items
        )


# ============================================================================
# Hybrid Compression
# ============================================================================

class HybridCompressor:
    """
    Hybrid compression strategy.

    Uses heuristic compression by default, but falls back to LLM
    compression for important or complex content.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.heuristic = HeuristicCompressor()
        self._llm: Optional[LLMCompressor] = None
        self._api_key = api_key

    @property
    def llm(self) -> LLMCompressor:
        """Lazy load LLM compressor."""
        if self._llm is None:
            self._llm = LLMCompressor(self._api_key)
        return self._llm

    def compress(
        self,
        items: List[SessionItem],
        config: CompressionConfig
    ) -> CompressionResult:
        """Compress using hybrid strategy."""
        original_tokens = sum(item.estimate_tokens() for item in items)

        # Use heuristic for smaller contexts
        if original_tokens < config.token_threshold:
            return self.heuristic.compress(items, config)

        # Check if content seems complex (code, errors, technical)
        is_complex = self._is_complex_content(items)

        if is_complex and HAS_ANTHROPIC:
            try:
                return self.llm.compress(items, config)
            except Exception:
                # Fallback to heuristic on LLM failure
                pass

        return self.heuristic.compress(items, config)

    def _is_complex_content(self, items: List[SessionItem]) -> bool:
        """Check if content seems complex enough to warrant LLM compression."""
        combined = " ".join(item.content for item in items)

        # Check for code blocks
        if re.search(r'```[\s\S]{100,}```', combined):
            return True

        # Check for technical patterns
        technical_patterns = [
            r'error|exception|traceback',
            r'function|class|def |const |let |var ',
            r'import |from .+ import',
            r'CREATE TABLE|SELECT |INSERT |UPDATE ',
        ]

        matches = sum(
            1 for pattern in technical_patterns
            if re.search(pattern, combined, re.IGNORECASE)
        )

        return matches >= 2


# ============================================================================
# Context Compressor (Main Interface)
# ============================================================================

class ContextCompressor:
    """
    Main interface for context compression.

    Usage:
        compressor = ContextCompressor()
        result = compressor.compress(items, config)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._compressors = {
            CompressionStrategy.NONE: None,
            CompressionStrategy.HEURISTIC: HeuristicCompressor(),
            CompressionStrategy.LLM: None,  # Lazy loaded
            CompressionStrategy.HYBRID: None,  # Lazy loaded
        }

    def compress(
        self,
        items: List[SessionItem],
        config: Optional[CompressionConfig] = None
    ) -> CompressionResult:
        """Compress items using configured strategy."""
        config = config or CompressionConfig()

        if config.strategy == CompressionStrategy.NONE:
            return self._no_compression_result(items)

        compressor = self._get_compressor(config.strategy)
        return compressor.compress(items, config)

    def should_compress(
        self,
        items: List[SessionItem],
        config: Optional[CompressionConfig] = None
    ) -> bool:
        """Check if compression is recommended."""
        config = config or CompressionConfig()

        if len(items) < config.min_items_to_compress:
            return False

        total_tokens = sum(item.estimate_tokens() for item in items)
        return total_tokens > config.token_threshold

    def _get_compressor(self, strategy: CompressionStrategy):
        """Get or create compressor for strategy."""
        if strategy == CompressionStrategy.LLM:
            if self._compressors[strategy] is None:
                self._compressors[strategy] = LLMCompressor(self._api_key)
            return self._compressors[strategy]

        if strategy == CompressionStrategy.HYBRID:
            if self._compressors[strategy] is None:
                self._compressors[strategy] = HybridCompressor(self._api_key)
            return self._compressors[strategy]

        return self._compressors[strategy]

    def _no_compression_result(self, items: List[SessionItem]) -> CompressionResult:
        """Return result when no compression needed."""
        total_tokens = sum(item.estimate_tokens() for item in items)
        return CompressionResult(
            original_items=len(items),
            original_tokens=total_tokens,
            compressed_items=len(items),
            compressed_tokens=total_tokens,
            strategy_used=CompressionStrategy.NONE,
            summary="",
            preserved_items=items
        )
