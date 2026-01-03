# SmartContext - Nerd Guide

The theory, research, and deep technical details. For those who want to understand *why* it works.

## The Fundamental Problem

LLMs have a **fixed context window**. Everything the model "knows" must fit in this window:

```
[System] [User1] [Asst1] [User2] [Asst2] ... [UserN] [AsstN]
                                              ↑
                                    Old messages get truncated
```

This creates **catastrophic forgetting**.

### Why Not Just Make Windows Bigger?

1. **Quadratic attention complexity**: O(n²) for self-attention
2. **Memory costs**: KV-cache grows linearly with sequence length
3. **"Lost in the middle" phenomenon**: Models struggle with middle content even with long windows

Research: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)

## Theoretical Foundation

SmartContext is built on these research insights:

### 1. Attention Sinks (Xiao et al., 2023)

**Finding**: First few tokens receive disproportionate attention regardless of content.

**Implication**: Put important content at the START of context.

```
[IMPORTANT] ... [less important] ... [IMPORTANT]
      ↑ Attention sink                    ↑ Recency effect
```

SmartContext places system prompt and goal first, strategy last.

### 2. Contextual Retrieval (Anthropic, 2024)

**Finding**: Adding context about *where* information comes from improves retrieval accuracy by 67%.

**Implication**: Don't just include content - include provenance.

Each memory stores:
- Content
- Tags
- Timestamp
- Resolution level (full/summary/reference)
- Importance score

### 3. Resolution Pyramids (Google ADK)

**Finding**: Not all content needs full detail.

```
                    ┌───────────────┐
       Full Detail  │   GOAL        │  ← Current task (100%)
                    ├───────────────┤
       Summary      │  MEMORIES     │  ← Relevant context (50%)
                    ├───────────────┤
       Reference    │  ARTIFACTS    │  ← Background (10%)
                    └───────────────┘
```

SmartContext adjusts detail level based on relevance.

### 4. Temporal Decay (Manus, 2024)

**Finding**: Recent information is more likely to be relevant.

**Implementation**:
```python
def apply_temporal_decay(items, decay_rate=0.15):
    for i, item in enumerate(sorted_by_recency):
        decay_factor = max(0.3, 1.0 - (i * decay_rate))
        # Item 0: 1.0, Item 1: 0.85, Item 2: 0.70, ...
```

### 5. KV-Cache Optimization (Manus, 2024)

**Finding**: Stable prefixes enable KV-cache reuse across requests.

**The math**:
- Request 1: Compute KV for Prefix + Query1 → Cache Prefix
- Request 2: Reuse cached Prefix, compute only Query2
- Savings: O(|Prefix|) compute per request

**Requirements**:
1. Deterministic serialization (same content → same bytes)
2. Stable prefix (system prompt never changes mid-session)
3. Append-only (new content only added at end)

## Memory Architecture

### Why Tiered Memory?

Inspired by human cognition and computer architecture:

| Tier | Human Analog | Computer Analog | Characteristics |
|------|--------------|-----------------|-----------------|
| Working Context | Working memory | CPU Registers | ~7 items, instant |
| Session Memory | Short-term memory | L1 Cache | Current task |
| Long-term Memory | Long-term memory | RAM/SSD | Persistent |
| Artifacts | External notes | Disk | Large, explicit |

### Retrieval vs Accumulation

**Accumulation** (naive):
```
context = context + new_message  # Grows forever, then truncates
```

**Retrieval** (SmartContext):
```
relevant = search(query, memory)  # O(1) context size
context = compile(relevant)       # Always fits budget
```

Key insight from **ACE (Anthropic Context Engineering)**:
> "Context should be a computed view of state, not an accumulation of history."

## Attention Budget Allocation

You have limited tokens. How to spend them?

**Naive**: Equal allocation (20% each)

**SmartContext**: Task-aware allocation

```python
POLICIES = {
    "coding": {
        "goal": 0.40,        # Focus on current task
        "memories": 0.20,
        "observations": 0.15,
        "artifacts": 0.10,
        "strategy": 0.10,
        "system": 0.05
    },
    "debugging": {
        "observations": 0.50,  # Focus on what's happening
        "memories": 0.20,
        "goal": 0.15,
        "artifacts": 0.10,
        "system": 0.05,
        "strategy": 0.00
    }
}
```

## Schema-Driven Compaction

### The Problem with Summarization

Traditional summarization is **lossy**:
```
Original: "Error in auth.py line 47: TypeError"
Summarized: "There was an error"  # Lost critical details!
```

### Schema Preservation

SmartContext preserves structure:
```python
# Original: 3 separate error entries

# Compacted:
{
    "type": "error_summary",
    "count": 3,
    "files_affected": ["auth.py", "user.py"],
    "error_types": ["TypeError", "KeyError", "ValueError"]
}
```

Structure preserved, details compressed.

## Mode Detection

### Keyword-Based Classification

```python
MODE_KEYWORDS = {
    DEBUGGING: ["bug", "error", "broken", "fix", "crash", "fail"],
    PLANNING: ["plan", "architect", "design", "structure", "approach"],
    EXPLORING: ["what is", "how does", "explain", "show me"]
}
```

### Why Not Embeddings?

1. **Latency**: Embedding calls add 100-200ms
2. **Cost**: API calls for every message
3. **Accuracy**: Keywords work surprisingly well
4. **Simplicity**: No external dependencies

## Error Memory and Learning

### Manus Pattern: Error Preservation

> "Maintain failed actions and error traces in context."

Most systems sanitize errors. SmartContext preserves them:

```python
class ErrorMemory:
    def log_error(self, action, error, context):
        entry = ErrorEntry(
            action=action,
            error=error,
            category=self._auto_categorize(error),
            resolved=False
        )
```

### Why Keep Errors?

1. **Avoid repetition**: Don't suggest what already failed
2. **Pattern learning**: Common errors → heuristics
3. **Resolution tracking**: What fixed similar issues?

## Strategy Evolution

Agents develop **evolving playbooks**:

```python
@dataclass
class AgentStrategy:
    domain: str
    heuristics: List[Dict]       # "Always do X before Y"
    learned_patterns: List[Dict]  # "When A happens, try B"
    failure_modes: List[Dict]     # "X doesn't work because Y"
    success_patterns: List[Dict]  # "Z worked for W"
    version: int
```

### Learning Loop

```
Execute action
       ↓
   Success?  ─── Yes ──→ record_success()
       │
      No
       ↓
record_failure()
       ↓
   Retry with new approach
```

## Research Papers & Sources

### Core Research

1. **Attention Sinks** - [StreamingLLM](https://arxiv.org/abs/2309.17453)
2. **Lost in the Middle** - [How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
3. **Contextual Retrieval** - [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval)

### Production Systems

4. **Manus Context Engineering** - [Blog Post](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
5. **OpenAI Agents SDK** - [Session Memory Patterns](https://cookbook.openai.com/examples/agents_sdk/session_memory)

### Theoretical Foundations

6. **ACE** (Anthropic Context Engineering) - Context as computed view
7. **Google ADK** (Agent Development Kit) - Resolution pyramids

## Future Research Directions

- **Embedding-Based Retrieval**: Replace keyword search with semantic search
- **Learned Attention Policies**: Train policies on task performance
- **Cross-Session Learning**: Transfer strategy across sessions
- **Automatic Compaction**: Learn when to compact based on performance
- **Multi-Modal Context**: Extend to images, execution results, file trees

## Code Structure

```
src/
├── attention.py    # Policies, mode detection, compiler
├── memory.py       # Tiered memory, search, decay
├── session.py      # Session protocol, trimming
├── compression.py  # Compaction strategies
├── cache.py        # KV-cache optimization
├── checklist.py    # Task tracking
└── errors.py       # Error preservation
```

Key extension points:
- `POLICIES` dict - add custom attention policies
- `CompressionStrategy` enum - add compression methods
- `StorageBackend` base class - add storage backends
