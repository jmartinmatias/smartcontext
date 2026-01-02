# SmartContext for Beginners

**Welcome!** This guide assumes you've never used context engineering before. We'll take it slow.

---

## What Problem Does This Solve?

When you chat with Claude (or any AI), it has a **memory limit**. Imagine talking to someone who can only remember the last 5 minutes of conversation. That's Claude without help.

**Without SmartContext:**
```
You (2 hours ago): "Let's use PostgreSQL for the database"
You (now): "What database did we decide on?"
Claude: "I don't have information about previous database discussions..."
```

**With SmartContext:**
```
You (now): "What database did we decide on?"
Claude: "We decided on PostgreSQL earlier in our session."
```

SmartContext gives Claude a **better memory**.

---

## Installation (5 minutes)

### Step 1: Get the code

```bash
cd ~
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
  Storage: /Users/you/.smartcontext/default
  Memories: 0
  Artifacts: 0
```

Press Ctrl+C to stop it.

---

## Basic Usage (The Only 3 Things You Need)

### 1. Remember something

Tell Claude to remember important things:

```
"Remember that we're using PostgreSQL for the database"
"Remember the API key is in the .env file"
"Remember the deadline is March 15th"
```

Behind the scenes, SmartContext stores this permanently.

### 2. Recall something

Ask Claude about things from earlier:

```
"What database are we using?"
"Where's the API key?"
"When's the deadline?"
```

SmartContext searches its memory and gives Claude the answer.

### 3. Store big files

Instead of pasting huge files into chat:

```
"Store this file as 'api-routes.ts':
[paste your code here]"
```

Now you can reference it later without pasting again:

```
"Look at api-routes.ts and tell me about the login endpoint"
```

---

## That's It!

For beginners, those 3 things are all you need:
- **Remember** - save important facts
- **Recall** - find things later
- **Store** - save big files

Everything else happens automatically:
- Claude switches focus based on what you're doing
- Old conversations get summarized to save space
- Errors get tracked so Claude learns

---

## Common Questions

### "Where does it store my stuff?"

In a folder called `.smartcontext` in your home directory:
```
~/.smartcontext/
  default/
    long_term.json    <- Your memories
    artifacts.json    <- Your stored files
    working_context.json
```

### "Can I see what it remembered?"

Yes! Ask Claude:
```
"Show me everything you remember about this project"
```

Or use the status command:
```
"What's the SmartContext status?"
```

### "What if I want it to forget something?"

Just say:
```
"Forget the thing about PostgreSQL"
```

### "Do I need to do anything special?"

Nope. Just talk to Claude normally. SmartContext works invisibly in the background.

---

## Next Steps

Once you're comfortable:
1. Try the [Intermediate Guide](GUIDE_INTERMEDIATE.md) for more features
2. Look at the [Use Cases](USE_CASES.md) for real examples

---

## Troubleshooting

### "It's not remembering things"

Make sure SmartContext is running:
```bash
python3 src/smartcontext.py
```

### "I get an import error"

Install the requirements:
```bash
pip install mcp fastmcp
```

### "I'm confused"

That's okay! Just use these three phrases:
- "Remember that..."
- "What do you know about..."
- "Store this file as..."

Everything else is optional.
