# SmartContext

**An MCP server that gives Claude a better memory.**

## What Is This?

SmartContext is an **MCP (Model Context Protocol) server** that runs alongside Claude Code. It provides Claude with persistent memory, intelligent focus modes, and learning capabilities.

### What's an MCP Server?

MCP servers are background processes that extend Claude's capabilities. When you connect SmartContext to Claude Code:
- It runs quietly in the background
- Claude automatically gains access to 45 new tools (remember, recall, store files, etc.)
- You just talk to Claude normally - SmartContext works invisibly

### The Problem It Solves

Without SmartContext:
```
You (2 hours ago): "Let's use PostgreSQL for the database"
You (now): "What database did we decide on?"
Claude: "I don't have information about previous discussions..."
```

With SmartContext:
```
You (now): "What database did we decide on?"
Claude: "We decided on PostgreSQL earlier."
```

## Installation

### Step 1: Clone the repo

```bash
git clone https://github.com/jmartinmatias/smartcontext.git
cd smartcontext
```

### Step 2: Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mcp fastmcp
```

### Step 3: Test the server

```bash
cd src
python3 smartcontext.py
```

The server will start and wait for connections. Press Ctrl+C to stop.

### Step 4: Connect to Claude Code

Add this to your `~/.claude/.mcp.json` file:

```json
{
  "mcpServers": {
    "smartcontext": {
      "command": "/path/to/smartcontext/venv/bin/python3",
      "args": ["smartcontext.py"],
      "cwd": "/path/to/smartcontext/src"
    }
  }
}
```

Replace `/path/to/smartcontext` with your actual path (e.g., `/home/user/smartcontext`).

### Step 5: Restart Claude Code

After updating `.mcp.json`, restart Claude Code. SmartContext will connect automatically.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                     Claude Code                          │
│                                                          │
│   You: "Remember we're using PostgreSQL"                │
│                         │                                │
│                         ▼                                │
│              ┌─────────────────────┐                    │
│              │   SmartContext      │ ◄── MCP Server     │
│              │   (background)      │     running        │
│              │                     │                    │
│              │  • Saves to memory  │                    │
│              │  • Searches context │                    │
│              │  • Tracks errors    │                    │
│              │  • Manages focus    │                    │
│              └─────────────────────┘                    │
│                         │                                │
│                         ▼                                │
│   Claude: "Got it, I'll remember PostgreSQL."           │
└─────────────────────────────────────────────────────────┘
```

## Basic Usage

Once connected, just talk to Claude normally. SmartContext works in the background.

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

## What SmartContext Provides

| Feature | Description |
|---------|-------------|
| **Persistent Memory** | Remembers things across conversations |
| **Artifact Storage** | Stores large files by reference |
| **Focus Modes** | Coding, debugging, exploring, planning |
| **Error Tracking** | Learns from mistakes |
| **Task Checklists** | Keeps Claude focused on multi-step work |
| **Session Management** | Compresses old context to save space |

## Common Questions

**Where does it store my stuff?**

In `~/.smartcontext/` folder as JSON files.

**Can I see what it remembers?**

Ask Claude: "Show me everything you remember about this project"

**What if I want it to forget something?**

Say: "Forget the thing about PostgreSQL"

**How do I know it's connected?**

Ask Claude: "What's the SmartContext status?"

## Troubleshooting

**Claude doesn't seem to remember things**
- Check that the server path in `.mcp.json` is correct
- Make sure you're using the venv python: `/path/to/venv/bin/python3`
- Restart Claude Code after changing `.mcp.json`

**Import errors when running the server**
```bash
source venv/bin/activate
pip install mcp fastmcp
```

**Permission errors**
- Make sure the venv python is executable
- Check that `~/.smartcontext/` is writable

## Learn More

- [README_intermediate.md](README_intermediate.md) - Tags, modes, tasks, error tracking
- [README_expert.md](README_expert.md) - Full API reference and all 45 tools
- [README_nerd.md](README_nerd.md) - Theory, research, and architecture

## License

MIT

---

*SmartContext: An MCP server that gives Claude the memory it deserves.*
