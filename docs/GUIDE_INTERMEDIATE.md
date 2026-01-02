# SmartContext for Intermediate Users

You know the basics. Now let's use SmartContext effectively.

---

## Quick Reference

| What You Want | Command |
|--------------|---------|
| Save a fact | `remember("API uses JWT", tags=["auth"])` |
| Find facts | `recall("authentication")` |
| Save big file | `store_artifact("config.py", content)` |
| Get file back | `get_artifact("config.py")` |
| Set current task | `set_goal("Implement login feature")` |
| Track task progress | `add_task("Write tests")` |
| Log an error | `log_error("Build failed", "Missing dependency")` |
| Check status | `status()` |

---

## Feature 1: Tagging Memories

Tags help you organize and find memories faster.

```python
# Without tags - hard to find later
remember("Database is PostgreSQL 14")

# With tags - easy to find
remember("Database is PostgreSQL 14", tags=["database", "infrastructure"])
remember("Use pgvector for embeddings", tags=["database", "ai"])
remember("Connection pool max is 20", tags=["database", "config"])
```

Now you can search by tag:
```
"What do we know about the database?"
# Finds all 3 memories
```

### Good Tagging Practices

| Tag Type | Examples |
|----------|----------|
| Component | `auth`, `payment`, `user`, `api` |
| Type | `config`, `decision`, `bug`, `todo` |
| Priority | `critical`, `important`, `nice-to-have` |

---

## Feature 2: Attention Modes

SmartContext automatically switches modes based on your message:

| Your Message Contains | Mode Activated | Focus |
|----------------------|----------------|-------|
| "bug", "error", "fix", "broken" | Debugging | Recent errors, observations |
| "how does", "explain", "what is" | Exploring | Memories, documentation |
| "design", "architect", "plan" | Planning | Goals, strategy |
| Default | Coding | Current task |

### Manual Mode Switching

Sometimes you want to force a mode:

```
"Switch to debugging mode"
"Set mode to planning"
```

### Why Modes Matter

In **debugging mode**, Claude focuses on:
- Recent error messages (50% of attention)
- What you just tried
- Error history

In **exploring mode**, Claude focuses on:
- Memories and docs (30% of attention)
- Codebase structure
- Past decisions

---

## Feature 3: Task Checklists

Keep Claude focused on multi-step tasks:

```
# Create a checklist
add_task("Set up database schema")
add_task("Create API endpoints")
add_task("Write tests")
add_task("Add documentation")

# As you complete them
complete_task("1")  # Schema done
complete_task("2")  # Endpoints done

# See progress
get_checklist()
```

Output:
```
Task Checklist (2/4 complete):
  [x] Set up database schema
  [x] Create API endpoints
  [ ] Write tests          <- Current
  [ ] Add documentation
```

### Why Use Checklists?

- Keeps the current task in Claude's attention
- Solves the "lost in the middle" problem
- Visible progress tracking

---

## Feature 4: Error Tracking

When something goes wrong, log it:

```
log_error("npm install failed", "EACCES permission denied")
log_error("Tests failing", "Timeout in auth.test.js")
```

Later, Claude can see what went wrong:
```
"What errors have we encountered?"

# Output:
Recent Errors:
  1. npm install failed: EACCES permission denied [UNRESOLVED]
  2. Tests failing: Timeout in auth.test.js [UNRESOLVED]
```

When you fix something:
```
resolve_error("1", "Fixed with sudo chown")
```

### Why Track Errors?

- Claude avoids suggesting things that already failed
- Builds a history of solutions
- Learns from your codebase's quirks

---

## Feature 5: Artifacts for Large Content

Instead of pasting code repeatedly:

```python
# Store once
store_artifact("user_service.py", """
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, id):
        return self.db.query(User).get(id)

    def create_user(self, email, password):
        ...
""")

# Reference many times
"Look at user_service.py and add a delete method"
"What methods does user_service.py have?"
"Compare user_service.py with the payment service"
```

### Artifact Best Practices

| Do | Don't |
|----|-------|
| Store files you reference often | Store one-off pastes |
| Use descriptive names (`auth_middleware.py`) | Use vague names (`code.py`) |
| Update when code changes | Let artifacts go stale |

---

## Feature 6: Session Management

### Set a Goal

```
set_goal("Implement user authentication with JWT")
```

Now every context includes your goal, keeping Claude focused.

### End a Session

When you're done for the day:
```
end_session("Completed auth endpoints, tests passing, need to add docs")
```

This:
- Saves a summary to long-term memory
- Clears working context
- Prepares for next session

---

## Feature 7: Compression

Long sessions fill up context. Compress when needed:

```
# Check if compression is needed
get_compression_stats()

# Compress if recommended
compress_context("heuristic")  # Fast, no API calls
compress_context("hybrid")     # Best quality
```

### When to Compress

- Session has 50+ exchanges
- You see "context limit" warnings
- Claude starts forgetting recent things

---

## Workflow Example: Feature Development

```python
# 1. Start with a goal
set_goal("Add password reset feature")

# 2. Create task checklist
add_task("Add reset token to User model")
add_task("Create /forgot-password endpoint")
add_task("Create /reset-password endpoint")
add_task("Send email with reset link")
add_task("Write tests")

# 3. Store relevant code
store_artifact("user_model.py", existing_user_code)
store_artifact("email_service.py", email_code)

# 4. Remember key decisions
remember("Reset tokens expire in 1 hour", tags=["auth", "security"])
remember("Using SendGrid for emails", tags=["email", "infrastructure"])

# 5. Work through tasks, logging errors
# ... development happens ...
log_error("Email not sending", "SMTP connection refused")
resolve_error("1", "Wrong port, changed to 587")

# 6. Complete tasks as you go
complete_task("1")
complete_task("2")
# ...

# 7. End session with summary
end_session("Password reset complete, all tests passing")
```

---

## Tips for Intermediate Users

### 1. Be Specific with Memories
```
# Bad
remember("Important database thing")

# Good
remember("PostgreSQL 14, max connections 100, on port 5432", tags=["database", "config"])
```

### 2. Update Stale Information
```
forget("Old API endpoint was /v1/users")
remember("API endpoint is /v2/users", tags=["api"])
```

### 3. Use Goals for Focus
```
# Vague
set_goal("Fix stuff")

# Clear
set_goal("Fix the race condition in OrderService.checkout()")
```

### 4. Review Status Regularly
```
status()
```
Shows memories, artifacts, session stats, errors - all at once.

---

## Next Steps

- [Expert Guide](GUIDE_EXPERT.md) - Customization and advanced features
- [Nerd Guide](GUIDE_NERD.md) - How it works under the hood
