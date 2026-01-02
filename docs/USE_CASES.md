# SmartContext Use Cases

Real-world scenarios where context engineering makes a difference.

---

## 1. Long Coding Session (4+ hours)

**Problem**: Claude forgets what you discussed 2 hours ago.

**Without SmartContext**:
```
You: "Remember we decided to use Redis for caching?"
Claude: "I don't have context about previous Redis discussions..."
```

**With SmartContext**:
```python
# Early in session
memory.remember("Architecture decision: Use Redis for caching", tags=["architecture", "redis"])
memory.remember("Redis config: 6379, no auth in dev", tags=["redis", "config"])

# Hours later
memory.search("caching")  # Finds Redis decisions
# Claude: "Yes, we decided on Redis (port 6379, no auth in dev)..."
```

---

## 2. Debugging Complex Bug

**Problem**: Need to track error history, what was tried, what failed.

**Without SmartContext**:
- You manually repeat: "I already tried X, Y, Z"
- Claude suggests things you already tried
- No learning from failures

**With SmartContext**:
```python
# Auto-detects "bug" → switches to debugging mode (50% attention on observations)
memory.log_error("Tried null check", "Still crashes - not the issue")
memory.log_error("Checked DB connection", "Connection is fine")

# Later attempts include error history
# Claude: "I see you tried null check and DB connection. Let me suggest something different..."
```

---

## 3. Large Codebase Navigation

**Problem**: 500+ files, Claude can't hold it all.

**Without SmartContext**:
- Paste entire files into context
- Run out of tokens
- Lose important earlier context

**With SmartContext**:
```python
# Store files as artifacts (referenced, not inlined)
memory.store_artifact("auth_service.py", auth_code)
memory.store_artifact("user_model.py", user_code)
memory.store_artifact("api_routes.py", routes_code)

# Query gets artifact REFERENCES, not full content
# "#a1b2c3: auth_service.py" instead of 500 lines
# Full content retrieved only when needed
```

---

## 4. Multi-Agent Handoff

**Problem**: Agent A does research, Agent B implements. How to transfer context?

**With SmartContext**:
```python
# Agent A (Research)
memory.remember("API uses OAuth2 with PKCE flow", tags=["auth", "research"])
memory.remember("Rate limit: 100/min, backoff required", tags=["api", "research"])
memory.store_artifact("api_docs.md", documentation)

# Agent B (Implementation) - same namespace
relevant = memory.search("authentication implementation")
# Gets Agent A's research findings automatically
```

---

## 5. Learning From Errors Over Time

**Problem**: Same mistakes repeated across sessions.

**Without SmartContext**:
- Each session starts fresh
- No memory of past failures
- Repeat same errors

**With SmartContext**:
```python
# Session 1
memory.log_error("Used == for None check", "Should use 'is None'")
memory.update_strategy("record_failure", "Python None comparison must use 'is'")

# Session 2 (days later)
strategy = memory.get_strategy()
# Strategy includes: "Python None comparison must use 'is'"
# Claude avoids the same mistake
```

---

## 6. Context-Aware Mode Switching

**Problem**: Same attention for all tasks, even though needs differ.

**Example Messages and Auto-Detection**:

| Message | Detected Mode | Focus |
|---------|--------------|-------|
| "There's a bug in login" | `debugging` | 50% on observations/errors |
| "How does the auth work?" | `exploring` | 30% on memories/docs |
| "Let's design the new API" | `planning` | 30% goal + 15% strategy |
| "Implement the feature" | `coding` | 40% on current goal |

```python
# Automatic - no manual switching needed
attention.maybe_switch_mode("There's a bug in the payment flow")
# → Switches to debugging, loads error history, recent observations
```

---

## 7. Task Tracking with Attention Control

**Problem**: Claude loses track of multi-step tasks.

**Without SmartContext**:
- Forgets step 3 of 5
- Needs reminder of overall goal
- "Lost in the middle" of long context

**With SmartContext**:
```python
# Checklist stays in "attention sink" position
memory.add_task("Set up database schema")
memory.add_task("Create API endpoints")
memory.add_task("Write tests")
memory.add_task("Add documentation")

memory.complete_task("1")  # Schema done

# Checklist injected into recent context every turn
# ✓ Set up database schema
# → Create API endpoints  ← CURRENT
# ○ Write tests
# ○ Add documentation
```

---

## 8. Schema-Driven Session Compaction

**Problem**: Long session, can't fit in context, but need history.

**Without SmartContext**:
- Truncate from end (lose recent context!)
- Or truncate from start (lose initial context!)
- Either way, information lost

**With SmartContext**:
```python
# After 100 exchanges
result = memory.compact_session_schema(
    preserve_recent=20,      # Keep last 20 exchanges intact
    extract_insights=True    # Save patterns to long-term memory
)

# Result:
# - 100 exchanges → 35 (compacted older + recent 20)
# - Insights extracted: "Session had 5 errors, all related to auth"
# - Structure preserved, just summarized
```

---

## 9. KV-Cache Optimization (Production)

**Problem**: Slow response times, high API costs.

**Technical Background**:
LLMs cache key-value pairs. If your context prefix stays stable, cache hits = faster + cheaper.

**Without Optimization**:
```
[timestamp: 2024-01-15T10:23:45]  ← Changes every request!
System: You are a helpful assistant
...
```

**With SmartContext**:
```python
# Stable prefix - never changes
memory.set_stable_prefix("""
System: You are a coding assistant.
Project: E-commerce API
Stack: Python, FastAPI, PostgreSQL
""")

# Context is append-only, deterministic serialization
# Result: Higher cache hit rate, faster responses
```

---

## 10. Constraint-Based Working Context

**Problem**: Claude makes suggestions that violate project constraints.

**With SmartContext**:
```python
# Set constraints once
memory.add_constraint("No external API calls in tests")
memory.add_constraint("All dates must be UTC")
memory.add_constraint("Max function length: 50 lines")

# Constraints included in every context compilation
# Claude: "I'll write this without external API calls since that's a constraint..."
```

---

## Quick Reference: When to Use What

| Scenario | Key Tools |
|----------|-----------|
| Long session | `remember()`, `search()`, `end_session()` |
| Debugging | Auto mode detection, `log_error()`, `get_errors()` |
| Large codebase | `store_artifact()`, artifact references |
| Multi-step task | `add_task()`, `complete_task()`, `get_checklist()` |
| Learning | `update_strategy()`, `record_failure()` |
| Production | `set_stable_prefix()`, `compact_session_schema()` |
