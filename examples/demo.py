#!/usr/bin/env python3
"""
SmartContext Demo

Demonstrates the core features without running as an MCP server.
"""

import sys
sys.path.insert(0, './src')

from memory import MemoryStore
from attention import AttentionManager, AttentionMode, ContextCompiler


def main():
    print("=" * 60)
    print("SMARTCONTEXT DEMO")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing SmartContext...")
    memory = MemoryStore(namespace="demo")
    attention = AttentionManager()
    compiler = ContextCompiler(attention)
    print("   ✓ Memory store created")
    print("   ✓ Attention manager created")
    print("   ✓ Context compiler created")

    # Memory operations
    print("\n2. Memory Operations...")

    # Remember some things
    memory.remember("This project uses React with TypeScript", tags=["tech-stack"])
    memory.remember("Database is PostgreSQL on port 5432", tags=["database", "config"])
    memory.remember("Authentication uses JWT tokens", tags=["auth", "security"])
    print("   ✓ Stored 3 memories")

    # Search
    results = memory.search("database configuration")
    print(f"   ✓ Search 'database configuration': found {len(results)} matches")
    for r in results:
        print(f"      - {r.content[:50]}...")

    # Store an artifact
    memory.store_artifact("schema.sql", """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print("   ✓ Stored artifact: schema.sql")

    # Attention management
    print("\n3. Attention Management...")

    # Show current mode
    print(f"   Current mode: {attention.current_mode.value}")
    print(f"   Policy: {attention.get_policy().name}")

    # Auto-detect from message
    message = "There's a bug in the login system"
    attention.maybe_switch_mode(message)
    print(f"\n   Message: '{message}'")
    print(f"   Auto-detected mode: {attention.current_mode.value}")

    # Show allocation
    print("\n   Attention allocation:")
    print(attention.get_policy().format_allocation())

    # Context compilation
    print("\n4. Context Compilation...")

    compiled = compiler.compile(
        goal="Fix the authentication bug",
        memories=["Auth uses JWT", "Endpoint is /api/auth"],
        observations=["Error: token undefined", "Stack trace: auth.ts:47"],
        token_budget=2000
    )

    print(f"   Mode: {compiled.mode.value}")
    print(f"   Tokens used: {compiled.total_tokens}/{compiled.budget}")
    print("\n   Compiled prompt preview:")
    print("   " + "-" * 40)
    preview = compiled.to_prompt()[:300]
    for line in preview.split("\n"):
        print(f"   {line}")
    print("   ...")

    # Stats
    print("\n5. Memory Stats...")
    stats = memory.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Cleanup demo namespace
    print("\n6. Cleanup...")
    # memory.end_session("Demo completed")
    print("   (Keeping memories for inspection)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nMemories stored in: {memory.storage_dir}")
    print("Run the MCP server with: python src/smartcontext.py")


if __name__ == "__main__":
    main()
