# SmartContext for Experts

Advanced features, customization, and optimization strategies.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SmartContext                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Memory    │  │  Attention  │  │   Session   │              │
│  │   Store     │  │   Manager   │  │   Protocol  │              │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤              │
│  │ Working     │  │ Policies    │  │ Trimming    │              │
│  │ Session     │  │ Compiler    │  │ Compression │              │
│  │ Long-term   │  │ Mode Detect │  │ Branching   │              │
│  │ Artifacts   │  │             │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Cache     │  │  Checklist  │  │   Error     │              │
│  │  Optimizer  │  │  (Manus)    │  │   Memory    │              │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤              │
│  │ Stable Pfx  │  │ Recitation  │  │ Tracking    │              │
│  │ KV-Cache    │  │ Attention   │  │ Learning    │              │
│  │ Breakpoints │  │ Control     │  │ Resolution  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                   │
│  ┌─────────────────────────────────────────────────┐            │
│  │              Storage Backends                    │            │
│  │   JSON (default)  │  SQLite (production)        │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Programmatic Usage

### Direct Python API

```python
from smartcontext import (
    MemoryStore,
    AttentionManager,
    ContextCompiler,
    AttentionMode
)

# Initialize with custom settings
memory = MemoryStore(
    namespace="my-project",
    storage_backend="sqlite",    # Better for production
    max_session_turns=100,       # Before trimming kicks in
    compression_strategy="hybrid"
)

# Custom attention configuration
attention = AttentionManager(default_mode=AttentionMode.CODING)
attention.auto_detect = True  # Enable/disable auto mode switching

# Create compiler
compiler = ContextCompiler(attention)

# Compile context with full control
context = compiler.compile(
    goal="Implement OAuth2 flow",
    memories=["API uses JWT", "Token expiry is 1h"],
    observations=["User clicked login", "Redirect to /auth"],
    artifacts=["#a1: auth_service.py", "#a2: config.json"],
    strategy="Security-first approach",
    system="You are a security expert",
    token_budget=4000,
    include_context_graph=True,
    temporal_decay=True,
    semantic_chunking=True
)

# Access compiled result
print(context.to_prompt())
print(f"Quality score: {context.quality_score}")
print(f"Token usage: {context.total_tokens}/{context.budget}")
```

---

## Custom Attention Policies

### Modifying Built-in Policies

```python
from smartcontext import AttentionPolicy, AttentionMode, POLICIES

# Get existing policy
coding_policy = POLICIES[AttentionMode.CODING]

# Create custom policy
custom_policy = AttentionPolicy(
    name="Security Audit",
    mode=AttentionMode.CODING,
    description="Focus on security concerns",
    allocation={
        "goal": 0.20,
        "memories": 0.25,      # More weight on past vulnerabilities
        "observations": 0.30,  # Recent findings
        "artifacts": 0.15,     # Code being audited
        "strategy": 0.05,
        "system": 0.05
    }
)

# Use custom policy
attention = AttentionManager()
attention._custom_policies = {"security": custom_policy}
```

### Dynamic Policy Adjustment

```python
# Adjust based on context
def adaptive_policy(current_mode, session_length, error_count):
    base = POLICIES[current_mode].allocation.copy()

    # More observation focus if many errors
    if error_count > 5:
        base["observations"] = min(0.6, base["observations"] + 0.2)
        base["goal"] = max(0.1, base["goal"] - 0.1)

    # More memory focus in long sessions
    if session_length > 50:
        base["memories"] = min(0.4, base["memories"] + 0.1)

    return base
```

---

## Strategy Evolution

### Initialize with Domain Knowledge

```python
# Pre-seed with domain expertise
memory.initialize_strategy(domain="security", template="security_audit")

# Or build custom
memory.initialize_strategy(domain="my_domain")
memory.update_strategy("add_heuristic", "Always validate input at API boundary")
memory.update_strategy("add_heuristic", "Use parameterized queries, never string concat")
memory.update_strategy("add_pattern", "Auth errors often come from token expiry")
```

### Learn from Execution

```python
# After successful operations
memory.update_strategy("record_success",
    "Retry with exponential backoff fixed the rate limit issue")

# After failures
memory.update_strategy("record_failure",
    "Using == for None comparison caused subtle bug")

# Check strategy evolution
strategy = memory.get_strategy()
print(f"Strategy v{strategy.version}")
print(f"Heuristics: {len(strategy.heuristics)}")
print(f"Patterns: {len(strategy.learned_patterns)}")
print(f"Failures: {len(strategy.failure_modes)}")
```

---

## Storage Backend Selection

### JSON (Default)
```python
memory = MemoryStore(storage_backend="json")
```
- Human-readable files
- Easy to debug/inspect
- Fine for development
- No concurrent access

### SQLite (Production)
```python
memory = MemoryStore(storage_backend="sqlite")
```
- Better concurrency
- Full-text search built-in
- Query capabilities
- File integrity

### Backend Migration

```python
# Export from JSON
from smartcontext.storage import JSONStorage, SQLiteStorage, StorageConfig

config = StorageConfig(base_dir=Path("~/.smartcontext"), namespace="myproject")
json_store = JSONStorage(config)
data = json_store.export_all()

# Import to SQLite
sqlite_store = SQLiteStorage(config)
sqlite_store.import_all(data)
```

---

## Cache Optimization (KV-Cache)

For production deployments, KV-cache hit rate matters.

### Stable Prefix Strategy

```python
# Set once at session start - NEVER changes
memory.set_stable_prefix("""
System: You are a senior Python developer.
Project: E-commerce platform
Stack: Python 3.11, FastAPI, PostgreSQL, Redis
Constraints:
- Follow PEP 8
- Type hints required
- Async where possible
""")

# This prefix stays at context start, maximizing cache hits
```

### Append-Only Context

```python
# Good: append new info
memory.add_session_item("user", "New requirement: add logging")

# Bad: reordering (breaks cache)
# Avoid reshuffling session items
```

### Explicit Cache Breaks

```python
# When context fundamentally changes
memory.mark_cache_break()

# Example: switching to completely different task
memory.mark_cache_break()
memory.set_goal("Now working on a different feature")
```

---

## Schema-Driven Compaction

### How It Works

```python
# Long session: 200 events
result = memory.compact_session_schema(
    preserve_recent=30,      # Keep last 30 intact
    extract_insights=True    # Mine patterns to long-term
)

# What happens:
# 1. Events 0-169 grouped by type
# 2. Similar events merged: "15 observations" instead of 15 entries
# 3. Key patterns extracted to long-term memory
# 4. Events 170-199 kept intact (most recent)
```

### Compaction vs Truncation

| Approach | What Happens | Information Loss |
|----------|--------------|------------------|
| Truncation | Cut at limit | Loses old OR recent |
| Compaction | Summarize old, keep recent | Minimal - structure preserved |

### Controlling Compaction

```python
# Aggressive (small context, fast)
memory.compact_session_schema(preserve_recent=10)

# Conservative (large context, detailed)
memory.compact_session_schema(preserve_recent=50)

# Extract only, don't compact
insights = memory._extract_insights_from_events(memory.session_log)
```

---

## Multi-Agent / Multi-Session

### Shared Namespace

```python
# Agent A (Research)
agent_a = MemoryStore(namespace="project-x")
agent_a.remember("API uses OAuth2", tags=["research", "auth"])
agent_a.store_artifact("api_docs.md", docs)

# Agent B (Implementation) - same namespace
agent_b = MemoryStore(namespace="project-x")
memories = agent_b.search("authentication")
# Gets Agent A's research!
```

### Isolated Namespaces

```python
# Different projects don't interfere
project_x = MemoryStore(namespace="project-x")
project_y = MemoryStore(namespace="project-y")

project_x.remember("Uses PostgreSQL")
project_y.search("PostgreSQL")  # Returns nothing
```

### Session Branching

```python
from smartcontext import BranchManager

branch_mgr = BranchManager()

# Main development
branch_mgr.session.add_item("user", "Implementing feature A")

# Create branch for experiment
branch_mgr.create_branch("experiment", "Trying alternative approach")
branch_mgr.switch_branch("experiment")

# ... experiment ...

# Compare results
comparison = branch_mgr.compare_branches("main", "experiment")
print(f"Main: {comparison['branch1']['item_count']} items")
print(f"Experiment: {comparison['branch2']['item_count']} items")

# Either merge or discard
branch_mgr.switch_branch("main")
branch_mgr.delete_branch("experiment")
```

---

## Performance Tuning

### Memory Limits

```python
memory = MemoryStore(
    max_session_turns=100,  # Trim after 100 turns
)

# More aggressive trimming
memory.set_max_turns(50)
```

### Compression Strategies

| Strategy | Speed | Quality | When to Use |
|----------|-------|---------|-------------|
| `none` | - | - | Short sessions |
| `heuristic` | Fast | Good | Normal usage |
| `hybrid` | Medium | Excellent | Long sessions, critical context |

```python
memory.set_compression_strategy("heuristic")  # Default
memory.set_compression_strategy("hybrid")      # Better quality
```

### Token Budget Allocation

```python
# Adjust based on model context limit
compiler.compile(
    ...,
    token_budget=4000,   # Small models (GPT-3.5)
    token_budget=16000,  # Medium (GPT-4)
    token_budget=100000, # Large (Claude with 200k)
)
```

---

## Monitoring & Debugging

### Full Transparency Mode

```python
# See everything SmartContext does
ctx.transparency = TransparencyLevel.FULL
```

Output includes:
- Memories loaded
- Mode switches
- Token allocation per section
- Cache hit estimates

### Compilation Statistics

```python
stats = compiler.get_compilation_stats(last_n=20)
print(f"Quality trend: {stats['trend']}")  # improving/declining/stable
print(f"Mean quality: {stats['quality_scores']['mean']}")
```

### Explain Decisions

```python
explanation = compiler.explain_compilation()
print(f"Strategy: {explanation['explanation']['strategy_used']}")
for decision in explanation['explanation']['decisions']:
    print(f"- {decision['decision']}")
    print(f"  Rationale: {decision['rationale']}")
```

---

## Integration Patterns

### With Claude API

```python
import anthropic
from smartcontext import MemoryStore, ContextCompiler, AttentionManager

client = anthropic.Anthropic()
memory = MemoryStore(namespace="my-app")
attention = AttentionManager()
compiler = ContextCompiler(attention)

def chat(user_message: str) -> str:
    # Prepare context
    attention.maybe_switch_mode(user_message)
    relevant = memory.search(user_message, limit=5)

    context = compiler.compile(
        goal=memory.get_goal(),
        memories=[m.content for m in relevant],
        observations=[item.content for item in memory.get_session_items(limit=10)],
        token_budget=4000
    )

    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=context.sections.get("system", ""),
        messages=[
            {"role": "user", "content": context.to_prompt() + "\n\n" + user_message}
        ]
    )

    # Log interaction
    memory.add_session_item("user", user_message)
    memory.add_session_item("assistant", response.content[0].text)

    return response.content[0].text
```

### As MCP Server

```json
// ~/.claude/.mcp.json
{
  "mcpServers": {
    "smartcontext": {
      "command": "python3",
      "args": ["src/smartcontext.py", "--namespace", "myproject", "--backend", "sqlite"],
      "cwd": "/path/to/smartcontext"
    }
  }
}
```

---

## Next Steps

- [Nerd Guide](GUIDE_NERD.md) - The theory and research behind SmartContext
