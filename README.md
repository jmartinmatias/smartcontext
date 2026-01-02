# SmartContext v2.0

**Intelligent context engineering for AI assistants.**

Make Claude remember, focus, and work smarter - invisibly.

```
┌─────────────────────────────────────────────────────────────┐
│  Without SmartContext          │  With SmartContext         │
├────────────────────────────────┼────────────────────────────┤
│  Claude forgets everything     │  Claude remembers forever  │
│  You repeat yourself           │  Context is automatic      │
│  Context fills up fast         │  Artifacts save space      │
│  Same attention for all tasks  │  Smart focus per task      │
│  No learning from errors       │  Errors improve behavior   │
│  Linear conversation only      │  Branch & explore options  │
└─────────────────────────────────────────────────────────────┘
```

## What's New in v2.0

SmartContext v2.0 integrates patterns from:
- **OpenAI Agents SDK**: Session protocol, trimming, compression, branching
- **Manus Production Patterns**: KV-cache optimization, task checklists, error memory

## Features

### Tiered Memory
- **Working Context**: Current focus (~500 tokens, always active)
- **Session Memory**: This conversation's history (with auto-trimming)
- **Long-term Memory**: Persistent, searchable knowledge
- **Artifacts**: Large files stored by reference (10x space savings)

### Attention Management
- **Coding Mode**: 40% focus on goal
- **Debugging Mode**: 50% focus on errors/observations
- **Exploring Mode**: 30% focus on memories/docs
- **Planning Mode**: Balanced goal + strategy
- **Auto-detection**: Switches mode based on your message

### Session Trimming (NEW)
- Automatically prunes old context
- Keeps first N items (context start) + last N items (recent)
- Prevents context explosion in long sessions

### Context Compression (NEW)
- **Heuristic**: Fast, rule-based (no API calls)
- **LLM**: Claude-powered intelligent summarization
- **Hybrid**: Heuristic first, LLM for complex content
- Configurable compression strategies

### Conversation Branching (NEW)
- Fork conversations for "what if" exploration
- Compare different approaches
- Save checkpoints before risky operations

### Cache Optimization (NEW - Manus Pattern)
- Stable prefixes for maximum KV-cache hits
- Append-only context tracking
- Explicit cache breakpoints
- Deterministic serialization

### Task Checklists (NEW - Manus Pattern)
- Keeps current task in recent attention
- Solves "lost in the middle" problem
- Progress tracking and recitation

### Error Memory (NEW - Manus Pattern)
- Preserves failed actions for learning
- Auto-categorization of errors
- Resolution tracking
- Prevents repeating mistakes

### Transparency Modes
- **Off**: Invisible (default)
- **Minimal**: "3 memories loaded"
- **Normal**: Shows what memories were used
- **Full**: Complete debug output

### Storage Backends (NEW)
- **JSON**: File-based, portable, human-readable (default)
- **SQLite**: Better concurrency, querying, full-text search

## Quick Start

### 1. Install

```bash
pip install mcp fastmcp
```

### 2. Run

```bash
cd smartcontext
python src/smartcontext.py
```

With SQLite backend:
```bash
python src/smartcontext.py --backend sqlite
```

### 3. Configure Claude Code

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "smartcontext": {
      "command": "python3",
      "args": ["src/smartcontext.py"],
      "cwd": "/path/to/smartcontext"
    }
  }
}
```

### 4. Use

Just talk to Claude normally. SmartContext works invisibly.

## MCP Tools Reference

### Memory Tools

| Tool | Description |
|------|-------------|
| `remember(content, tags)` | Save to long-term memory |
| `forget(query)` | Remove matching memory |
| `recall(topic, limit)` | Search memories |
| `store_artifact(name, content)` | Store large content |
| `get_artifact(name_or_id)` | Retrieve artifact |
| `list_artifacts()` | List all artifacts |

### Attention Tools

| Tool | Description |
|------|-------------|
| `set_mode(mode)` | Set attention mode |
| `set_goal(goal)` | Set working goal |
| `get_goal()` | Get current goal |
| `set_transparency(level)` | Set visibility level |
| `status()` | Show current state |
| `prepare(message, budget)` | Prepare context (auto-called) |

### Session Tools (NEW)

| Tool | Description |
|------|-------------|
| `set_max_turns(n)` | Set session trimming limit |
| `get_session_stats()` | Show session size, tokens |
| `pop_last()` | Remove last exchange (undo) |
| `end_session(summary)` | End and save session |
| `export_session(format)` | Export session data |

### Compression Tools (NEW)

| Tool | Description |
|------|-------------|
| `compress_context(strategy)` | Trigger compression |
| `set_compression_strategy(s)` | Set strategy (none/heuristic/llm/hybrid) |
| `get_compression_stats()` | Show compression stats |

### Branching Tools (NEW)

| Tool | Description |
|------|-------------|
| `branch_session(name)` | Fork current session |
| `list_branches()` | Show all branches |
| `switch_branch(name)` | Switch to branch |
| `delete_branch(name)` | Delete branch |
| `compare_branches(a, b)` | Compare two branches |

### Cache Optimization Tools (NEW - Manus)

| Tool | Description |
|------|-------------|
| `set_stable_prefix(prefix)` | Set immutable context prefix |
| `mark_cache_break()` | Signal cache should reset |
| `get_cache_stats()` | Show cache hit rate estimate |

### Task Checklist Tools (NEW - Manus)

| Tool | Description |
|------|-------------|
| `add_task(description)` | Add task to checklist |
| `complete_task(id)` | Mark task complete |
| `get_checklist(compact)` | Get checklist for recitation |
| `clear_checklist()` | Clear all tasks |

### Error Memory Tools (NEW - Manus)

| Tool | Description |
|------|-------------|
| `log_error(action, error)` | Log failed action |
| `resolve_error(id, resolution)` | Mark error resolved |
| `get_errors(include_resolved)` | Get errors for context |
| `clear_errors()` | Clear error history |

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR MESSAGE                            │
│              "Fix the authentication bug"                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     SMARTCONTEXT                             │
│                                                              │
│  1. Detect "bug" → Switch to debugging mode                  │
│  2. Search memory for "authentication"                       │
│  3. Load error history (what went wrong before)              │
│  4. Load task checklist (current focus)                      │
│  5. Allocate attention: 50% to observations                  │
│  6. Optimize for KV-cache hits                               │
│  7. Compile optimal context                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        CLAUDE                                │
│                                                              │
│  "Looking at the JWT validation in auth.ts, the issue        │
│   is on line 47. I see this failed before when we tried      │
│   X - let me use a different approach..."                    │
│                                                              │
│  (Claude has full context + learns from past errors)         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
smartcontext/
├── src/
│   ├── __init__.py          # Package exports
│   ├── memory.py            # Tiered memory system
│   ├── attention.py         # Attention policies & compiler
│   ├── session.py           # Session protocol & trimming
│   ├── compression.py       # Compression strategies
│   ├── branching.py         # Conversation branching
│   ├── cache.py             # KV-cache optimization (Manus)
│   ├── checklist.py         # Task recitation (Manus)
│   ├── errors.py            # Error preservation (Manus)
│   ├── smartcontext.py      # Main MCP server
│   └── storage/
│       ├── __init__.py
│       ├── base.py          # Storage interface
│       ├── json_storage.py  # JSON backend
│       └── sqlite_storage.py # SQLite backend
├── tests/
│   └── test_smartcontext.py # Test suite
├── examples/
│   └── demo.py              # Demo script
├── docs/
│   └── ...
├── requirements.txt
└── README.md
```

## Attention Modes

| Mode | Focus | Best For |
|------|-------|----------|
| `coding` | 40% goal | Writing new code |
| `debugging` | 50% observations | Fixing bugs |
| `exploring` | 30% memories | Learning codebase |
| `planning` | 30% goal + 15% strategy | Architecture |
| `balanced` | Even distribution | General work |

Auto-detection keywords:
- **Debugging**: "bug", "error", "fix", "broken", "crash"
- **Planning**: "plan", "architect", "design", "approach"
- **Exploring**: "what is", "how does", "explain", "show me"

## Compression Strategies

| Strategy | Speed | Quality | API Calls |
|----------|-------|---------|-----------|
| `none` | - | - | 0 |
| `heuristic` | Fast | Good | 0 |
| `llm` | Slow | Excellent | 1 per compression |
| `hybrid` | Medium | Excellent | 0-1 |

## Cache Optimization (Manus Pattern)

From Manus: *"KV-cache hit rate is the single most important metric for a production AI agent."*

SmartContext implements:
- **Stable Prefixes**: System instructions never change position
- **Append-Only**: Context only grows, never reorders
- **Deterministic Serialization**: No timestamps in output
- **Explicit Breakpoints**: Mark when cache should reset

## Error Memory (Manus Pattern)

From Manus: *"Maintain failed actions and error traces in context."*

SmartContext:
- Auto-categorizes errors (syntax, runtime, network, etc.)
- Keeps unresolved errors in context for learning
- Tracks resolutions for future reference
- Prevents repeating the same mistakes

## Task Checklist (Manus Pattern)

From Manus: *"Creating and updating a task checklist pushes the global plan into Claude's recent attention window."*

SmartContext:
- Maintains active task list
- Injects checklist into recent context
- Solves "lost in the middle" problem
- Tracks progress automatically

## Advanced Usage

### Namespaces

Use different memory spaces per project:

```bash
python src/smartcontext.py --namespace my-project
```

### SQLite Backend

For better performance and querying:

```bash
python src/smartcontext.py --backend sqlite
```

### Programmatic Usage

```python
from smartcontext import MemoryStore, AttentionManager

memory = MemoryStore(namespace="myproject")
attention = AttentionManager()

# Remember things
memory.remember("API uses rate limiting", tags=["api", "limits"])

# Track tasks
memory.add_task("Implement authentication")
memory.add_task("Write tests")

# Log errors for learning
memory.log_error("Run tests", "ImportError: module not found")

# Search
results = memory.search("rate limits")
```

## Philosophy

1. **Invisible by default** - Just works, no learning curve
2. **Explicit when needed** - Full control available
3. **Space efficient** - Artifacts, not inline content
4. **Task-aware** - Different focus for different work
5. **Persistent** - Never lose important context
6. **Learning** - Errors improve future behavior
7. **Production-ready** - Manus patterns for real-world use

## Sources & Inspiration

### Research & Papers
- [OpenAI Agents SDK - Session Memory](https://cookbook.openai.com/examples/agents_sdk/session_memory)
- [Manus - Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Anthropic ACE: Context as computed view, not accumulation
- Google ADK: Resolution pyramids for multi-level detail
- Anthropic Contextual Retrieval: Chunk context matters
- Meta In-Context Pretraining: Ordering and adjacency matter
- Attention Sinks research: First tokens get disproportionate attention

### Distilled From
Code distilled and enhanced from [mcp-tools-servers](https://github.com/yourusername/mcp-tools-servers):
- `mcp_server_memory_agent.py` - Tiered memory, attention policies, transparency modes
- `mcp_server_context_compiler.py` - Attention budget allocation, semantic chunking, temporal decay, context graphs
- `mcp_server_agentic_memory.py` - WorkingContext, schema-driven compaction, strategy evolution, insight extraction

## License

MIT

---

*SmartContext: Because AI should remember what matters and learn from its mistakes.*
