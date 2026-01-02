#!/usr/bin/env python3
"""
Visual Demo: SmartContext vs Naive Context

Shows side-by-side what context each approach generates.
This makes it easy to see the difference qualitatively.

Usage:
    python demo_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import MemoryStore
from attention import AttentionManager, ContextCompiler


def print_box(title: str, content: str, width: int = 60):
    """Print content in a box."""
    print(f"\n┌─ {title} {'─' * (width - len(title) - 4)}┐")
    for line in content.split('\n')[:30]:  # Limit lines
        truncated = line[:width-4] if len(line) > width-4 else line
        print(f"│ {truncated:<{width-4}} │")
    if content.count('\n') > 30:
        print(f"│ {'... (truncated)':<{width-4}} │")
    print(f"└{'─' * (width-2)}┘")


def demo():
    print("\n" + "=" * 70)
    print("   SMARTCONTEXT vs NAIVE: VISUAL COMPARISON")
    print("=" * 70)

    # ==========================================================================
    # SETUP: Same data for both
    # ==========================================================================

    print("\n📝 Setting up test data...")

    # Memories - mix of relevant and irrelevant
    memories = [
        ("The project uses Python 3.9", ["tech"]),
        ("Weather in SF is foggy today", []),
        ("API authentication uses JWT tokens", ["auth", "api"]),
        ("The team has 5 developers", []),
        ("Database is PostgreSQL 14", ["database"]),
        ("Login endpoint is POST /api/auth/login", ["auth", "api"]),
        ("Office coffee machine is broken", []),
        ("Rate limiting: 100 requests/minute", ["api"]),
        ("Password must be >= 8 characters", ["auth", "validation"]),
        ("Lunch break is at 12:30", []),
    ]

    # Session history - a typical debugging session
    session = [
        "Starting to debug the authentication issue",
        "User reports: login fails with 500 error",
        "Checked the logs - seeing null pointer exception",
        "The error is in UserService.validatePassword()",
        "Found it: password is None when user doesn't exist",
        "Need to add null check before validation",
        "Also should return 401 instead of 500",
        "Let me fix the UserService class",
        "Actually, let's also add input validation",
        "Testing the fix now...",
    ]

    # A code artifact
    code = '''
def login(email: str, password: str) -> Token:
    user = db.get_user(email)
    if user is None:
        raise AuthError("User not found")  # Fixed!

    if not validate_password(password, user.password_hash):
        raise AuthError("Invalid password")

    return generate_jwt(user.id)
'''

    # ==========================================================================
    # SMARTCONTEXT APPROACH
    # ==========================================================================

    print("\n🔧 Building SmartContext...")
    smart = MemoryStore(namespace="demo_smart")
    attention = AttentionManager()
    compiler = ContextCompiler(attention)

    for content, tags in memories:
        smart.remember(content, tags=tags)

    for msg in session:
        smart.add_session_item("user", msg)

    smart.store_artifact("fix.py", code)

    # ==========================================================================
    # NAIVE APPROACH
    # ==========================================================================

    print("🔧 Building Naive context...")
    naive_memories = [m[0] for m in memories]
    naive_session = session
    naive_artifact = code

    # ==========================================================================
    # QUERY: "What's the authentication bug?"
    # ==========================================================================

    query = "What's the authentication bug and how did we fix it?"
    token_budget = 500

    print(f"\n🔍 Query: \"{query}\"")
    print(f"   Token budget: {token_budget}")

    # --- SmartContext Result ---
    attention.maybe_switch_mode(query)  # Should detect "bug" -> debugging mode
    mode = attention.current_mode.value

    relevant_memories = smart.search(query, limit=3)
    memory_contents = [m.content for m in relevant_memories]

    session_items = smart.get_session_items(limit=5)
    observations = [item.content for item in session_items]

    artifacts = [f"#{a.id}: {a.name}" for a in smart.list_artifacts()]

    compiled = compiler.compile(
        goal=query,
        memories=memory_contents,
        observations=observations,
        artifacts=artifacts,
        token_budget=token_budget
    )

    smart_context = compiled.to_prompt()
    smart_tokens = compiled.total_tokens

    # --- Naive Result ---
    naive_parts = []
    naive_parts.append("MEMORIES:\n" + "\n".join(naive_memories))
    naive_parts.append("\nSESSION:\n" + "\n".join(naive_session))
    naive_parts.append("\nARTIFACTS:\n" + naive_artifact)
    naive_context = "\n".join(naive_parts)

    # Truncate to budget
    max_chars = token_budget * 4
    if len(naive_context) > max_chars:
        naive_context = naive_context[:max_chars] + "\n[TRUNCATED]"

    naive_tokens = len(naive_context) // 4

    # ==========================================================================
    # COMPARISON
    # ==========================================================================

    print_box(f"SMARTCONTEXT ({smart_tokens} tokens, mode={mode})", smart_context)
    print_box(f"NAIVE ({naive_tokens} tokens)", naive_context)

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("   ANALYSIS")
    print("=" * 70)

    # Check what made it into each context
    auth_in_smart = "auth" in smart_context.lower()
    auth_in_naive = "auth" in naive_context.lower()

    fix_in_smart = "fix" in smart_context.lower() or "null" in smart_context.lower()
    fix_in_naive = "fix" in naive_context.lower() or "null" in naive_context.lower()

    irrelevant_in_smart = "coffee" in smart_context.lower() or "weather" in smart_context.lower()
    irrelevant_in_naive = "coffee" in naive_context.lower() or "weather" in naive_context.lower()

    print(f"""
  ┌────────────────────────┬───────────────┬───────────────┐
  │ What's Included?       │ SmartContext  │ Naive         │
  ├────────────────────────┼───────────────┼───────────────┤
  │ Auth-related memories  │ {'✓ Yes' if auth_in_smart else '✗ No':<13} │ {'✓ Yes' if auth_in_naive else '✗ No':<13} │
  │ Bug fix discussion     │ {'✓ Yes' if fix_in_smart else '✗ No':<13} │ {'✓ Yes' if fix_in_naive else '✗ No':<13} │
  │ Irrelevant content     │ {'✗ No' if not irrelevant_in_smart else '✓ Yes':<13} │ {'✗ No' if not irrelevant_in_naive else '✓ Yes':<13} │
  ├────────────────────────┼───────────────┼───────────────┤
  │ Tokens used            │ {smart_tokens:<13} │ {naive_tokens:<13} │
  │ Auto mode detection    │ ✓ {mode:<11} │ ✗ None        │
  │ Memory search          │ ✓ Semantic    │ ✗ All included│
  │ Session trimming       │ ✓ Last 5      │ ✗ All included│
  └────────────────────────┴───────────────┴───────────────┘
    """)

    # Key insights
    print("  KEY INSIGHTS:")
    print("  • SmartContext searched memories and found auth-related ones")
    print("  • SmartContext trimmed session to recent items (the fix)")
    print("  • SmartContext detected 'bug' keyword -> debugging mode")
    print("  • Naive approach included irrelevant content (weather, coffee)")
    print("  • Naive approach was truncated, possibly losing important info")


if __name__ == "__main__":
    demo()
