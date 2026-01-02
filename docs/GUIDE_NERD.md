# SmartContext for Nerds

The theory, research, and deep technical details. For those who want to understand *why* it works.

---

## The Fundamental Problem

LLMs have a **fixed context window**. Everything the model "knows" about your conversation must fit in this window. When it overflows:

```
[System] [User1] [Asst1] [User2] [Asst2] ... [UserN] [AsstN]
                                              ↑
                                    Old messages get truncated
```

This creates **catastrophic forgetting** - the model loses access to early context.

### Why Not Just Make Windows Bigger?

1. **Quadratic attention complexity**: O(n²) for self-attention
2. **Memory costs**: KV-cache grows linearly with sequence length
3. **"Lost in the middle" phenomenon**: Even with long windows, models struggle with middle content

Research: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)

---

## Theoretical Foundation

SmartContext is built on these research insights:

### 1. Attention Sinks (Xiao et al., 2023)

**Finding**: First few tokens receive disproportionate attention regardless of content.

**Implication**: Put important content at the START of context.

```
[IMPORTANT CONTENT] ... [less important] ... [IMPORTANT CONTENT]
      ↑ Attention sink                              ↑ Recency effect
```

SmartContext implementation:
```python
def optimize_section_order(sections):
    priority_order = {
        'system': 0,      # Attention sink #1
        'goal': 1,
        'observations': 2,
        'memories': 3,
        'artifacts': 4,
        'strategy': 5     # Attention sink #2 (recency)
    }
    return sorted(sections, key=lambda s: priority_order[s.section_type])
```

### 2. Contextual Retrieval (Anthropic, 2024)

**Finding**: Adding context about *where* information comes from improves retrieval accuracy by 67%.

**Implication**: Don't just include content - include provenance.

```python
@dataclass
class CompiledSection:
    section_type: str
    content: str
    token_count: int
    resolution_level: str  # "full", "summary", "reference"
    provenance: Dict[str, Any]  # WHERE this came from
    importance_score: float
```

### 3. Resolution Pyramids (Google ADK)

**Finding**: Not all content needs full detail. Critical content → full detail; background → summary.

```
                    ┌───────────────┐
       Full Detail  │   GOAL        │  ← Current task (100% detail)
                    ├───────────────┤
       Summary      │  MEMORIES     │  ← Relevant context (50% detail)
                    ├───────────────┤
       Reference    │  ARTIFACTS    │  ← Background (10% detail)
                    └───────────────┘
```

SmartContext implementation:
```python
if decay_factor > 0.7:
    resolution = 'full'
    text = item_content
elif decay_factor > 0.4:
    resolution = 'summary'
    text = self.summarize(item_content, 50)
else:
    resolution = 'reference'
    text = self.summarize(item_content, 20)
```

### 4. Temporal Decay (Manus, 2024)

**Finding**: Recent information is more likely to be relevant than old information.

**Implication**: Apply exponential decay to older items.

```python
def apply_temporal_decay(items, decay_rate=0.15):
    # Sort by timestamp (most recent first)
    sorted_items = sorted(items, key=lambda x: x['timestamp'], reverse=True)

    weighted = []
    for i, item in enumerate(sorted_items):
        decay_factor = max(0.3, 1.0 - (i * decay_rate))
        # Item 0: 1.0, Item 1: 0.85, Item 2: 0.70, ...
        weighted.append((decay_factor, item))

    return weighted
```

### 5. KV-Cache Optimization (Manus, 2024)

**Finding**: Stable prefixes enable KV-cache reuse across requests.

**The math**: If prefix P is identical between requests:
- Request 1: Compute KV for P + Q1 → Cache P
- Request 2: Reuse cached P, compute only Q2
- Savings: O(|P|) compute per request

**Requirements for cache hits**:
1. **Deterministic serialization**: Same content → same bytes
2. **Stable prefix**: System prompt never changes mid-session
3. **Append-only**: New content only added at end

```python
class CacheOptimizer:
    def serialize_deterministic(self, items):
        # NO timestamps in output - they'd break cache
        # Sort keys for determinism
        return json.dumps(
            [{"role": i.role, "content": i.content} for i in items],
            sort_keys=True
        )
```

---

## Memory Architecture Theory

### Why Tiered Memory?

Inspired by human cognition and computer architecture:

| Tier | Human Analog | Computer Analog | Characteristics |
|------|--------------|-----------------|-----------------|
| Working Context | Working memory | CPU Registers | ~7 items, instant access |
| Session Memory | Short-term memory | L1 Cache | Current task, fast retrieval |
| Long-term Memory | Long-term memory | RAM/SSD | Persistent, slower retrieval |
| Artifacts | External memory (notes) | Disk | Large, explicit retrieval |

### Retrieval vs Accumulation

**Accumulation** (naive approach):
```
context = context + new_message  # Grows forever, then truncates
```

**Retrieval** (SmartContext approach):
```
relevant = search(query, memory)  # O(1) context size
context = compile(relevant)       # Always fits budget
```

This is the key insight from **ACE (Anthropic Context Engineering)**:
> "Context should be a computed view of state, not an accumulation of history."

---

## Attention Budget Allocation

### The Token Economy

You have a limited token budget. How do you spend it?

**Naive**: Equal allocation
```
Goal: 20%, Memories: 20%, Observations: 20%, ...
```

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

### Quality Score Calculation

How do we know if compilation was good?

```python
def calculate_quality_score(sections, max_tokens):
    # Factor 1: Are we using our budget?
    total_tokens = sum(s.token_count for s in sections)
    utilization = total_tokens / max_tokens

    # Factor 2: Do we have diverse content?
    coverage = len(sections) / 6  # 6 possible sections

    # Factor 3: Are we using full resolution?
    full_detail_ratio = len([s for s in sections if s.resolution_level == 'full']) / len(sections)

    # Weighted combination
    score = (utilization * 0.4 + coverage * 0.3 + full_detail_ratio * 0.3) * 100
    return score
```

---

## Schema-Driven Compaction Theory

### The Problem with Summarization

Traditional summarization is **lossy**:
```
Original: "Error in auth.py line 47: TypeError"
Summarized: "There was an error"  # Lost critical details!
```

### Schema Preservation

SmartContext preserves structure:
```python
# Original events
[
    {"type": "error", "file": "auth.py", "line": 47, "msg": "TypeError"},
    {"type": "error", "file": "auth.py", "line": 52, "msg": "KeyError"},
    {"type": "error", "file": "user.py", "line": 12, "msg": "ValueError"},
]

# Schema-preserved compaction
{
    "type": "error_summary",
    "count": 3,
    "files_affected": ["auth.py", "user.py"],
    "error_types": ["TypeError", "KeyError", "ValueError"]
}
```

The **structure** is preserved even though **details** are compressed.

### Reversibility

Key insight: Compaction should be **reversible** when possible.

```python
# Store mapping for potential expansion
compacted = {
    "type": "error_summary",
    "original_ids": ["err1", "err2", "err3"],  # Can retrieve full details
    ...
}
```

---

## Mode Detection Algorithm

### Keyword-Based Classification

```python
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
```

### Why Not Use Embeddings?

1. **Latency**: Embedding calls add 100-200ms
2. **Cost**: API calls for every message
3. **Accuracy**: Keywords work surprisingly well for mode detection
4. **Simplicity**: No external dependencies

For production, you could add embedding-based detection as a fallback:
```python
def detect_mode(message):
    # Fast path: keyword detection
    mode = keyword_detect(message)
    if mode:
        return mode

    # Slow path: embedding similarity (optional)
    embedding = get_embedding(message)
    return embedding_classify(embedding)
```

---

## Context Graph Theory

### Why Build a Graph?

Content has **relationships**:
- Goal → informed by → Memories
- Observations → reference → Artifacts
- Errors → block → Goal

```python
def build_context_graph(sections):
    graph = {'nodes': [], 'edges': []}

    for section in sections:
        graph['nodes'].append({
            'id': section.section_type,
            'importance': section.importance_score,
            'tokens': section.token_count
        })

    # Add semantic relationships
    if 'goal' in section_types and 'memories' in section_types:
        graph['edges'].append({
            'from': 'goal',
            'to': 'memories',
            'relationship': 'informed_by'
        })

    return graph
```

### Future Use: Graph-Aware Retrieval

Currently unused, but enables:
1. **Traversal-based retrieval**: Start at goal, traverse to related content
2. **Importance propagation**: PageRank-style importance scoring
3. **Conflict detection**: Identify contradictory information

---

## Semantic Chunking

### The Problem with Arbitrary Splits

Naive chunking:
```
"function calculateTotal(items) { let total = 0; for"
"(let item of items) { total += item.price; } return"
"total; }"
```

**Lost**: Function boundaries, logical units

### Semantic Boundaries

SmartContext chunks at logical boundaries:

```python
def semantic_chunk(content, content_type):
    if content_type == 'code':
        # Split at function/class definitions
        patterns = [r'\ndef ', r'\nfunction ', r'\nclass ']
        # ... keep functions intact

    elif content_type == 'document':
        # Split at paragraphs
        return content.split('\n\n')
```

This preserves semantic units even when truncating.

---

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
            category=self._auto_categorize(error),  # SYNTAX, RUNTIME, NETWORK, etc.
            context=context,
            resolved=False
        )
        self.errors.append(entry)
```

### Why Keep Errors?

1. **Avoid repetition**: Don't suggest what already failed
2. **Pattern learning**: Common errors → heuristics
3. **Resolution tracking**: What fixed similar issues before?

### Auto-Categorization

```python
def _auto_categorize(self, error):
    error_lower = error.lower()

    if any(kw in error_lower for kw in ['syntax', 'parse', 'unexpected token']):
        return ErrorCategory.SYNTAX
    elif any(kw in error_lower for kw in ['connection', 'timeout', 'refused']):
        return ErrorCategory.NETWORK
    elif any(kw in error_lower for kw in ['permission', 'denied', 'forbidden']):
        return ErrorCategory.PERMISSION
    # ... etc
```

---

## Strategy Evolution

### The Playbook Concept

Agents should develop **evolving playbooks**:

```python
@dataclass
class AgentStrategy:
    domain: str
    heuristics: List[Dict]      # "Always do X before Y"
    learned_patterns: List[Dict] # "When A happens, try B"
    failure_modes: List[Dict]    # "X doesn't work because Y"
    success_patterns: List[Dict] # "Z worked well for W"
    version: int
```

### Learning Loop

```
Execute action
       ↓
   Success?  ─── Yes ──→ record_success("This approach worked")
       │
      No
       ↓
record_failure("This didn't work because...")
       ↓
   Retry with new approach
```

Over time, the strategy accumulates domain knowledge.

---

## Benchmarking Methodology

### What We Measure

| Metric | How | Why |
|--------|-----|-----|
| Token Efficiency | Relevant content / tokens used | Are we wasting context? |
| Recall Accuracy | Relevant items found / total relevant | Are we finding what matters? |
| Mode Detection | Correct mode / total queries | Is auto-detection working? |
| Degradation | Quality at T=100 vs T=0 | Does quality decay over time? |
| Retention | Key info after compaction | Are we losing important data? |

### Baseline Comparison

**Naive baseline**:
- Append everything
- Truncate at limit
- No search/ranking
- No mode awareness

This represents what most chat applications do today.

---

## Research Papers & Sources

### Core Research

1. **Attention Sinks**
   - [StreamingLLM: Efficient Streaming Language Models](https://arxiv.org/abs/2309.17453)
   - First tokens receive disproportionate attention

2. **Lost in the Middle**
   - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
   - Models struggle with middle content in long sequences

3. **Contextual Retrieval**
   - [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
   - Adding context to chunks improves retrieval 67%

### Production Systems

4. **Manus Context Engineering**
   - [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
   - KV-cache optimization, error preservation, task recitation

5. **OpenAI Agents SDK**
   - [Session Memory Patterns](https://cookbook.openai.com/examples/agents_sdk/session_memory)
   - Trimming, compression, branching

### Theoretical Foundations

6. **ACE (Anthropic Context Engineering)**
   - Context as computed view, not accumulation
   - Retrieval over pinning

7. **Google ADK (Agent Development Kit)**
   - Resolution pyramids
   - Multi-level detail

---

## Future Research Directions

### Embedding-Based Retrieval
Replace keyword search with semantic search using embeddings.

### Learned Attention Policies
Train policies on task performance instead of hand-crafting.

### Cross-Session Learning
Transfer strategy knowledge across sessions and projects.

### Automatic Compaction Thresholds
Learn when to compact based on task performance.

### Multi-Modal Context
Extend to images, code execution results, file trees.

---

## Contributing

See something to improve? The codebase is structured for extension:

```
src/
├── attention.py    # Add new policies, improve mode detection
├── memory.py       # New memory tiers, storage backends
├── session.py      # Session protocols, trimming strategies
├── compression.py  # New compression algorithms
├── cache.py        # KV-cache optimizations
├── checklist.py    # Task tracking improvements
└── errors.py       # Error categorization enhancements
```

Key extension points:
- `POLICIES` dict in `attention.py` - add custom attention policies
- `CompressionStrategy` enum - add new compression methods
- `StorageBackend` base class - add new storage backends
