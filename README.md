# SmartContext

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
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🧠 Tiered Memory
- **Working Context**: Current focus (~500 tokens, always active)
- **Session Memory**: This conversation's history
- **Long-term Memory**: Persistent, searchable knowledge
- **Artifacts**: Large files stored by reference (10x space savings)

### 🎯 Attention Management
- **Coding Mode**: 40% focus on goal
- **Debugging Mode**: 50% focus on errors/observations
- **Exploring Mode**: 30% focus on memories/docs
- **Planning Mode**: Balanced goal + strategy
- **Auto-detection**: Switches mode based on your message

### 👁️ Transparency
- **Off**: Invisible (default)
- **Minimal**: "3 memories loaded"
- **Normal**: Shows what memories were used
- **Full**: Complete debug output

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

**Optional commands** (if you want control):

```
remember("This project uses PostgreSQL")     # Save to long-term memory
forget("PostgreSQL")                          # Remove from memory
recall("database")                            # Search memories
set_mode("debugging")                         # Switch attention mode
set_transparency("normal")                    # See what's happening
status()                                      # Show current state
```

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
│  3. Find: "Auth uses JWT", "endpoint: /api/auth"             │
│  4. Allocate attention: 50% to observations                  │
│  5. Compile optimal context                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        CLAUDE                                │
│                                                              │
│  "Looking at the JWT validation in auth.ts, the issue        │
│   is on line 47 where the token refresh..."                  │
│                                                              │
│  (Claude has full context without you explaining anything)   │
└─────────────────────────────────────────────────────────────┘
```

## Memory System

### Working Context
Current focus. Always included. Stays small (~500 tokens).

```python
set_goal("Build user authentication")  # Sets working goal
get_goal()                              # Retrieve current goal
```

### Long-term Memory
Survives forever. Searched by meaning.

```python
remember("API uses rate limiting of 100 req/min", tags="api,limits")
recall("rate limits")  # Finds it by meaning
forget("rate limits")  # Removes it
```

### Artifacts
Big files stored by reference, not inlined.

```python
store_artifact("schema.sql", "CREATE TABLE users...")  # Store
get_artifact("schema.sql")                              # Retrieve
list_artifacts()                                        # List all
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

## Transparency Modes

```bash
set_transparency("off")      # Invisible (default)
set_transparency("minimal")  # Brief summary
set_transparency("normal")   # Show memories used
set_transparency("full")     # Full debug output
```

Example with `normal`:
```
┌─ SMARTCONTEXT ───────────────────────────────────────┐
│ 📎 3 memories loaded                                  │
│ 🔄 Mode switched to debugging                         │
│ 🎯 Mode: Debugging                                    │
│    • "Auth uses JWT tokens"                           │
│    • "Endpoint is /api/auth"                          │
└───────────────────────────────────────────────────────┘
```

## Project Structure

```
smartcontext/
├── src/
│   ├── __init__.py       # Package exports
│   ├── memory.py         # Tiered memory system
│   ├── attention.py      # Attention policies & compiler
│   └── smartcontext.py   # Main MCP server
├── docs/
│   └── ...
├── examples/
│   └── demo.py
├── requirements.txt
└── README.md
```

## API Reference

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

### Session Tools

| Tool | Description |
|------|-------------|
| `status()` | Show current state |
| `set_transparency(level)` | Set visibility level |
| `prepare(message, budget)` | Prepare context (auto-called) |
| `end_session(summary)` | End and save session |

## Advanced: Namespaces

Use different memory spaces per project:

```bash
python src/smartcontext.py --namespace my-project
```

Or in config:
```json
{
  "mcpServers": {
    "smartcontext-projectA": {
      "command": "python3",
      "args": ["src/smartcontext.py", "--namespace", "projectA"],
      "cwd": "/path/to/smartcontext"
    }
  }
}
```

## Philosophy

1. **Invisible by default** - Just works, no learning curve
2. **Explicit when needed** - Full control available
3. **Space efficient** - Artifacts, not inline content
4. **Task-aware** - Different focus for different work
5. **Persistent** - Never lose important context

## License

MIT

---

*SmartContext: Because AI should remember what matters.*
