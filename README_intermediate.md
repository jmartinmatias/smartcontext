# SmartContext - Intermediate Guide

You know the basics. Now let's use SmartContext effectively.

## Quick Reference

| What You Want | What to Say |
|--------------|-------------|
| Save a fact | "Remember that API uses JWT" |
| Tag a memory | "Remember API uses JWT, tag it as auth" |
| Find facts | "What do you know about authentication?" |
| Save big file | "Store this as config.py: [code]" |
| Get file back | "Show me config.py" |
| Set current task | "Set goal to: Implement login feature" |
| Track task | "Add task: Write tests" |
| Log an error | "Log error: Build failed - missing dependency" |
| Check status | "What's the SmartContext status?" |

## Feature 1: Tagging Memories

Tags help you organize and find memories faster.

```
# Without tags - hard to find later
"Remember that database is PostgreSQL 14"

# With tags - easy to find
"Remember database is PostgreSQL 14, tag as database and infrastructure"
"Remember use pgvector for embeddings, tag as database and ai"
"Remember connection pool max is 20, tag as database and config"
```

Now you can search:
```
"What do we know about the database?"
# Finds all 3 memories
```

### Good Tags to Use

| Type | Examples |
|------|----------|
| Component | auth, payment, user, api |
| Type | config, decision, bug, todo |
| Priority | critical, important, nice-to-have |

## Feature 2: Attention Modes

SmartContext automatically switches modes based on your message:

| Your Message Contains | Mode | Focus |
|----------------------|------|-------|
| "bug", "error", "fix", "broken" | Debugging | Recent errors |
| "how does", "explain", "what is" | Exploring | Memories, docs |
| "design", "architect", "plan" | Planning | Goals, strategy |
| Default | Coding | Current task |

### Manual Mode Switching

```
"Switch to debugging mode"
"Set mode to planning"
```

### Why Modes Matter

**Debugging mode**: Claude focuses 50% on recent errors and observations.

**Exploring mode**: Claude focuses 30% on memories and documentation.

**Planning mode**: Claude balances goals and strategy.

## Feature 3: Task Checklists

Keep Claude focused on multi-step work:

```
"Add task: Set up database schema"
"Add task: Create API endpoints"
"Add task: Write tests"
"Add task: Add documentation"
```

As you work:
```
"Complete task 1"
"Complete task 2"
"Show my tasks"
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

- Keeps current task in Claude's attention
- Visible progress tracking
- Prevents Claude from losing focus

## Feature 4: Error Tracking

When something fails, log it:

```
"Log error: npm install failed - EACCES permission denied"
"Log error: Tests failing - timeout in auth.test.js"
```

Check errors:
```
"What errors have we encountered?"

# Output:
Recent Errors:
  1. npm install failed: EACCES permission denied [UNRESOLVED]
  2. Tests failing: timeout in auth.test.js [UNRESOLVED]
```

When you fix it:
```
"Resolve error 1 with: Fixed with sudo chown"
```

### Why Track Errors?

- Claude avoids suggesting things that already failed
- Builds a history of solutions
- Learns from your codebase's quirks

## Feature 5: Artifacts for Large Content

Instead of pasting code repeatedly:

```
"Store this as user_service.py:
class UserService:
    def __init__(self, db):
        self.db = db
    ..."
```

Then reference it:
```
"Look at user_service.py and add a delete method"
"What methods does user_service.py have?"
```

### Artifact Tips

| Do | Don't |
|----|-------|
| Store files you reference often | Store one-off pastes |
| Use descriptive names | Use vague names like code.py |
| Update when code changes | Let artifacts go stale |

## Feature 6: Session Management

### Set a Goal

```
"Set goal to: Implement user authentication with JWT"
```

Now every response considers your goal.

### End a Session

When done for the day:
```
"End session with summary: Completed auth endpoints, tests passing"
```

This saves to long-term memory and clears working context.

## Feature 7: Compression

Long sessions fill up context. Check and compress:

```
"Show compression stats"
"Compress context"
```

### When to Compress

- Session has 50+ exchanges
- You see "context limit" warnings
- Claude starts forgetting recent things

## Workflow Example

```
# 1. Start with a goal
"Set goal to: Add password reset feature"

# 2. Create checklist
"Add task: Add reset token to User model"
"Add task: Create /forgot-password endpoint"
"Add task: Send email with reset link"
"Add task: Write tests"

# 3. Store relevant code
"Store this as user_model.py: [code]"

# 4. Remember key decisions
"Remember reset tokens expire in 1 hour, tag as auth and security"

# 5. Work through tasks, logging errors
"Log error: Email not sending - SMTP connection refused"
"Resolve error 1: Wrong port, changed to 587"

# 6. Complete tasks
"Complete task 1"
"Complete task 2"

# 7. End session
"End session: Password reset complete, all tests passing"
```

## Tips

### Be Specific with Memories
```
# Bad
"Remember important database thing"

# Good
"Remember PostgreSQL 14, max connections 100, port 5432, tag as database config"
```

### Update Stale Info
```
"Forget the old API endpoint"
"Remember API endpoint is /v2/users, tag as api"
```

### Use Clear Goals
```
# Vague
"Set goal: Fix stuff"

# Clear
"Set goal: Fix the race condition in OrderService.checkout()"
```

## Next Steps

- [README_expert.md](README_expert.md) - Full API and all features
- [README_nerd.md](README_nerd.md) - The theory behind it all
