# SmartContext

**Give Claude a better memory.**

## What Is This?

When you chat with Claude, it forgets everything when you start a new conversation. SmartContext fixes that.

**Without SmartContext:**
```
You (2 hours ago): "Let's use PostgreSQL for the database"
You (now): "What database did we decide on?"
Claude: "I don't have information about previous discussions..."
```

**With SmartContext:**
```
You (now): "What database did we decide on?"
Claude: "We decided on PostgreSQL earlier."
```

## Installation

### Step 1: Get the code

```bash
git clone https://github.com/jmartinmatias/smartcontext.git
cd smartcontext
```

### Step 2: Install requirements

```bash
pip install mcp fastmcp
```

### Step 3: Test it works

```bash
python3 src/smartcontext.py
```

You should see:
```
SmartContext starting...
  Namespace: default
  Storage: ~/.smartcontext/default
```

Press Ctrl+C to stop.

### Step 4: Connect to Claude Code

Add this to `~/.claude/.mcp.json`:

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

Replace `/path/to/smartcontext` with where you downloaded it.

## How to Use It

Just talk to Claude normally. SmartContext works in the background.

### The 3 Things You Need to Know

**1. Remember something**
```
"Remember that we're using PostgreSQL for the database"
"Remember the API key is in the .env file"
"Remember the deadline is March 15th"
```

**2. Find something**
```
"What database are we using?"
"Where's the API key?"
"When's the deadline?"
```

**3. Store big files**
```
"Store this file as 'api-routes.ts':
[paste your code here]"
```

Then reference it later:
```
"Look at api-routes.ts and tell me about the login endpoint"
```

## That's It!

Everything else happens automatically:
- Claude switches focus based on what you're doing
- Old conversations get summarized to save space
- Errors get tracked so Claude learns

## Common Questions

**Where does it store my stuff?**

In `~/.smartcontext/` folder.

**Can I see what it remembers?**

Ask Claude: "Show me everything you remember about this project"

**What if I want it to forget something?**

Say: "Forget the thing about PostgreSQL"

**Do I need to do anything special?**

Nope. Just talk to Claude normally.

## Troubleshooting

**It's not remembering things**
- Make sure SmartContext is running
- Check your `.mcp.json` configuration

**I get an import error**
```bash
pip install mcp fastmcp
```

## Want to Learn More?

- [README_intermediate.md](README_intermediate.md) - Tags, modes, tasks, and error tracking
- [README_expert.md](README_expert.md) - Full API reference and all features
- [README_nerd.md](README_nerd.md) - The theory and research behind it

## License

MIT

---

*SmartContext: Because Claude should remember what matters.*
