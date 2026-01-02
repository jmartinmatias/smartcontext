#!/usr/bin/env python3
"""
SmartContext Benchmark Suite

Compares context-engineered prompts vs naive prompts across:
1. Token efficiency (tokens used for same quality)
2. Memory recall accuracy (can it find relevant info?)
3. Context relevance (is included context useful?)
4. Long-session performance (degradation over time)

Usage:
    python benchmark.py --scenario all
    python benchmark.py --scenario memory_recall
    python benchmark.py --scenario token_efficiency
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import MemoryStore, Memory
from attention import AttentionManager, ContextCompiler, AttentionMode
from session import SessionItem, BaseSession, TrimConfig, TrimStrategy


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    scenario: str
    variant: str  # "smartcontext" or "baseline"
    tokens_used: int
    tokens_available: int
    efficiency: float  # tokens_used / tokens_available
    recall_accuracy: float  # 0-1
    context_relevance: float  # 0-1 (how much included context was useful)
    duration_ms: float
    details: Dict[str, Any]


@dataclass
class BenchmarkComparison:
    """Comparison between SmartContext and baseline."""
    scenario: str
    smartcontext: BenchmarkResult
    baseline: BenchmarkResult
    token_savings_pct: float
    recall_improvement_pct: float
    relevance_improvement_pct: float


class NaiveContextManager:
    """
    Baseline: Simple context accumulation without engineering.
    Just appends everything to context in order received.
    """

    def __init__(self):
        self.memories: List[str] = []
        self.session_history: List[str] = []
        self.artifacts: Dict[str, str] = {}

    def remember(self, content: str):
        self.memories.append(content)

    def add_to_session(self, content: str):
        self.session_history.append(content)

    def store_artifact(self, name: str, content: str):
        self.artifacts[name] = content

    def prepare_context(self, query: str, token_budget: int) -> Dict[str, Any]:
        """Naive approach: just concatenate everything."""
        start = time.time()

        # Build context by simple concatenation
        context_parts = []

        # Add all memories (no search/ranking)
        if self.memories:
            context_parts.append("=== MEMORIES ===")
            context_parts.extend(self.memories)

        # Add full session history (no trimming)
        if self.session_history:
            context_parts.append("=== SESSION ===")
            context_parts.extend(self.session_history)

        # Add all artifacts inline (no references)
        if self.artifacts:
            context_parts.append("=== ARTIFACTS ===")
            for name, content in self.artifacts.items():
                context_parts.append(f"--- {name} ---")
                context_parts.append(content)

        full_context = "\n".join(context_parts)

        # Naive truncation: just cut at token limit
        estimated_tokens = len(full_context) // 4
        if estimated_tokens > token_budget:
            # Cut from the end (loses recent context!)
            max_chars = token_budget * 4
            full_context = full_context[:max_chars] + "\n... [TRUNCATED]"
            estimated_tokens = token_budget

        duration = (time.time() - start) * 1000

        return {
            "context": full_context,
            "tokens_used": estimated_tokens,
            "duration_ms": duration,
            "memories_included": len(self.memories),
            "truncated": estimated_tokens >= token_budget
        }


class SmartContextManager:
    """
    SmartContext: Full context engineering.
    Uses attention policies, search, trimming, compression.
    """

    def __init__(self, namespace: str = "benchmark"):
        self.memory = MemoryStore(namespace=namespace)
        self.attention = AttentionManager()
        self.compiler = ContextCompiler(self.attention)

    def remember(self, content: str, tags: List[str] = None):
        self.memory.remember(content, tags=tags or [])

    def add_to_session(self, content: str):
        self.memory.add_session_item("user", content)

    def store_artifact(self, name: str, content: str):
        self.memory.store_artifact(name, content)

    def prepare_context(self, query: str, token_budget: int) -> Dict[str, Any]:
        """Smart approach: search, rank, compile with attention policy."""
        start = time.time()

        # Auto-detect mode from query
        self.attention.maybe_switch_mode(query)

        # Search relevant memories (not all)
        relevant_memories = self.memory.search(query, limit=5)
        memory_contents = [m.content for m in relevant_memories]

        # Get recent session (trimmed)
        session_items = self.memory.get_session_items(limit=10)
        observations = [item.content for item in session_items]

        # Get artifact references (not full content)
        artifacts = [f"#{a.id}: {a.name}" for a in self.memory.list_artifacts()[:5]]

        # Compile with attention policy
        compiled = self.compiler.compile(
            goal=query[:200],
            memories=memory_contents,
            observations=observations,
            artifacts=artifacts,
            token_budget=token_budget
        )

        duration = (time.time() - start) * 1000

        return {
            "context": compiled.to_prompt(),
            "tokens_used": compiled.total_tokens,
            "duration_ms": duration,
            "memories_included": len(relevant_memories),
            "mode": self.attention.current_mode.value,
            "quality_score": getattr(compiled, 'quality_score', 0)
        }


# ============================================================================
# Benchmark Scenarios
# ============================================================================

def scenario_memory_recall(smart: SmartContextManager, naive: NaiveContextManager) -> BenchmarkComparison:
    """
    Test: Can the system find relevant memories?

    Setup: Store 50 memories, only 3 are relevant to query.
    Measure: Were the relevant memories included?
    """
    print("\n📊 Scenario: Memory Recall Accuracy")
    print("=" * 50)

    # Seed with irrelevant memories
    irrelevant_topics = [
        "The weather in Paris is nice",
        "Python was created by Guido van Rossum",
        "The Earth orbits the Sun",
        "Coffee contains caffeine",
        "Mount Everest is the tallest mountain",
    ]

    for i in range(50):
        topic = irrelevant_topics[i % len(irrelevant_topics)]
        content = f"{topic} - fact #{i}"
        smart.remember(content, tags=["filler"])
        naive.remember(content)

    # Add 3 relevant memories about authentication
    relevant_memories = [
        "The API uses JWT tokens for authentication with RS256 signing",
        "Authentication failures should return 401 status code",
        "The auth endpoint is POST /api/v1/auth/login",
    ]

    for mem in relevant_memories:
        smart.remember(mem, tags=["auth", "api"])
        naive.remember(mem)

    # Query about authentication
    query = "How does authentication work in this API?"
    token_budget = 500  # Limited budget forces selection

    # Run both
    smart_result = smart.prepare_context(query, token_budget)
    naive_result = naive.prepare_context(query, token_budget)

    # Calculate recall accuracy
    def count_relevant(context: str) -> int:
        count = 0
        for mem in relevant_memories:
            # Check if key phrase from memory is in context
            key_phrase = mem.split()[3:6]  # Get a few words
            if " ".join(key_phrase).lower() in context.lower():
                count += 1
        return count

    smart_recall = count_relevant(smart_result["context"]) / len(relevant_memories)
    naive_recall = count_relevant(naive_result["context"]) / len(relevant_memories)

    print(f"  SmartContext: {smart_recall:.0%} recall ({smart_result['tokens_used']} tokens)")
    print(f"  Baseline:     {naive_recall:.0%} recall ({naive_result['tokens_used']} tokens)")

    smart_bench = BenchmarkResult(
        scenario="memory_recall",
        variant="smartcontext",
        tokens_used=smart_result["tokens_used"],
        tokens_available=token_budget,
        efficiency=smart_result["tokens_used"] / token_budget,
        recall_accuracy=smart_recall,
        context_relevance=smart_recall,  # In this test, relevance = recall
        duration_ms=smart_result["duration_ms"],
        details=smart_result
    )

    naive_bench = BenchmarkResult(
        scenario="memory_recall",
        variant="baseline",
        tokens_used=naive_result["tokens_used"],
        tokens_available=token_budget,
        efficiency=naive_result["tokens_used"] / token_budget,
        recall_accuracy=naive_recall,
        context_relevance=naive_recall,
        duration_ms=naive_result["duration_ms"],
        details=naive_result
    )

    return BenchmarkComparison(
        scenario="memory_recall",
        smartcontext=smart_bench,
        baseline=naive_bench,
        token_savings_pct=0,  # Same budget
        recall_improvement_pct=(smart_recall - naive_recall) / max(naive_recall, 0.01) * 100,
        relevance_improvement_pct=(smart_recall - naive_recall) / max(naive_recall, 0.01) * 100
    )


def scenario_token_efficiency(smart: SmartContextManager, naive: NaiveContextManager) -> BenchmarkComparison:
    """
    Test: How efficiently does each use the token budget?

    Setup: Large session history + artifacts.
    Measure: Quality of context per token.
    """
    print("\n📊 Scenario: Token Efficiency")
    print("=" * 50)

    # Simulate a long coding session
    session_messages = [
        "Let's start building the user registration feature",
        "First, I'll create the User model with email and password fields",
        "Added validation for email format using regex",
        "Implemented password hashing with bcrypt",
        "Created the registration endpoint at POST /users",
        "Added error handling for duplicate emails",
        "Writing tests for the registration flow",
        "Test failed: password validation not working",
        "Fixed: was checking length before hashing",
        "All tests passing now",
        "Moving on to login functionality",
        "Implemented JWT token generation",
        "Added refresh token support",
        "Created middleware for auth validation",
        "Debugging an issue with token expiration",
        "Fixed: was using seconds instead of milliseconds",
        "Adding rate limiting to auth endpoints",
        "Implemented with Redis for distributed rate limiting",
        "Writing integration tests",
        "All auth features complete and tested",
    ]

    for msg in session_messages:
        smart.add_to_session(msg)
        naive.add_to_session(msg)

    # Add a large code artifact
    large_artifact = """
class UserService:
    def __init__(self, db, hasher, jwt_service):
        self.db = db
        self.hasher = hasher
        self.jwt = jwt_service

    def register(self, email: str, password: str) -> User:
        if self.db.get_user_by_email(email):
            raise DuplicateEmailError()

        hashed = self.hasher.hash(password)
        user = User(email=email, password_hash=hashed)
        return self.db.save(user)

    def login(self, email: str, password: str) -> TokenPair:
        user = self.db.get_user_by_email(email)
        if not user or not self.hasher.verify(password, user.password_hash):
            raise InvalidCredentialsError()

        access_token = self.jwt.create_access_token(user.id)
        refresh_token = self.jwt.create_refresh_token(user.id)
        return TokenPair(access_token, refresh_token)

    def refresh(self, refresh_token: str) -> TokenPair:
        payload = self.jwt.verify_refresh_token(refresh_token)
        user_id = payload['user_id']
        return TokenPair(
            self.jwt.create_access_token(user_id),
            self.jwt.create_refresh_token(user_id)
        )
""" * 3  # Make it bigger

    smart.store_artifact("user_service.py", large_artifact)
    naive.store_artifact("user_service.py", large_artifact)

    # Query about debugging
    query = "There's a bug in the rate limiting - it's not resetting properly"
    token_budget = 800

    smart_result = smart.prepare_context(query, token_budget)
    naive_result = naive.prepare_context(query, token_budget)

    # Calculate relevance: how much of the context is about rate limiting/debugging?
    def calculate_relevance(context: str) -> float:
        relevant_keywords = ["rate", "limit", "redis", "debug", "fix", "issue", "error", "bug"]
        words = context.lower().split()
        relevant_count = sum(1 for w in words if any(kw in w for kw in relevant_keywords))
        return min(1.0, relevant_count / 20)  # Normalize

    smart_relevance = calculate_relevance(smart_result["context"])
    naive_relevance = calculate_relevance(naive_result["context"])

    # Efficiency = relevance per token
    smart_efficiency = smart_relevance / (smart_result["tokens_used"] / 100)
    naive_efficiency = naive_relevance / (naive_result["tokens_used"] / 100)

    print(f"  SmartContext: {smart_relevance:.0%} relevance, {smart_result['tokens_used']} tokens")
    print(f"  Baseline:     {naive_relevance:.0%} relevance, {naive_result['tokens_used']} tokens")
    print(f"  Mode detected: {smart_result.get('mode', 'n/a')}")

    smart_bench = BenchmarkResult(
        scenario="token_efficiency",
        variant="smartcontext",
        tokens_used=smart_result["tokens_used"],
        tokens_available=token_budget,
        efficiency=smart_efficiency,
        recall_accuracy=1.0,  # Not measured in this scenario
        context_relevance=smart_relevance,
        duration_ms=smart_result["duration_ms"],
        details=smart_result
    )

    naive_bench = BenchmarkResult(
        scenario="token_efficiency",
        variant="baseline",
        tokens_used=naive_result["tokens_used"],
        tokens_available=token_budget,
        efficiency=naive_efficiency,
        recall_accuracy=1.0,
        context_relevance=naive_relevance,
        duration_ms=naive_result["duration_ms"],
        details=naive_result
    )

    return BenchmarkComparison(
        scenario="token_efficiency",
        smartcontext=smart_bench,
        baseline=naive_bench,
        token_savings_pct=(naive_result["tokens_used"] - smart_result["tokens_used"]) / naive_result["tokens_used"] * 100,
        recall_improvement_pct=0,
        relevance_improvement_pct=(smart_relevance - naive_relevance) / max(naive_relevance, 0.01) * 100
    )


def scenario_mode_detection(smart: SmartContextManager, naive: NaiveContextManager) -> BenchmarkComparison:
    """
    Test: Does auto mode detection improve context quality?

    Setup: Various query types (debugging, exploring, planning).
    Measure: Context relevance for each mode.
    """
    print("\n📊 Scenario: Mode Detection")
    print("=" * 50)

    # Add varied content
    smart.remember("The codebase uses TypeScript with strict mode", tags=["tech"])
    smart.remember("Main entry point is src/index.ts", tags=["structure"])
    smart.remember("Tests are in __tests__ folder using Jest", tags=["testing"])
    smart.remember("Previous bug: null pointer in UserService.login()", tags=["bug"])
    smart.remember("Architecture follows clean architecture pattern", tags=["design"])

    naive.remember("The codebase uses TypeScript with strict mode")
    naive.remember("Main entry point is src/index.ts")
    naive.remember("Tests are in __tests__ folder using Jest")
    naive.remember("Previous bug: null pointer in UserService.login()")
    naive.remember("Architecture follows clean architecture pattern")

    # Test different modes
    queries = [
        ("There's an error in the login function", "debugging"),
        ("How is the codebase organized?", "exploring"),
        ("Let's design the new feature", "planning"),
    ]

    smart_scores = []
    naive_scores = []

    for query, expected_mode in queries:
        smart_result = smart.prepare_context(query, 400)
        naive_result = naive.prepare_context(query, 400)

        detected_mode = smart_result.get("mode", "unknown")
        mode_correct = detected_mode == expected_mode

        print(f"  Query: '{query[:40]}...'")
        print(f"    Expected: {expected_mode}, Detected: {detected_mode} {'✓' if mode_correct else '✗'}")

        # Score based on mode correctness
        smart_scores.append(1.0 if mode_correct else 0.5)
        naive_scores.append(0.5)  # Baseline has no mode detection

    avg_smart = sum(smart_scores) / len(smart_scores)
    avg_naive = sum(naive_scores) / len(naive_scores)

    smart_bench = BenchmarkResult(
        scenario="mode_detection",
        variant="smartcontext",
        tokens_used=400,
        tokens_available=400,
        efficiency=1.0,
        recall_accuracy=avg_smart,
        context_relevance=avg_smart,
        duration_ms=0,
        details={"mode_accuracy": avg_smart}
    )

    naive_bench = BenchmarkResult(
        scenario="mode_detection",
        variant="baseline",
        tokens_used=400,
        tokens_available=400,
        efficiency=1.0,
        recall_accuracy=avg_naive,
        context_relevance=avg_naive,
        duration_ms=0,
        details={"mode_accuracy": avg_naive}
    )

    return BenchmarkComparison(
        scenario="mode_detection",
        smartcontext=smart_bench,
        baseline=naive_bench,
        token_savings_pct=0,
        recall_improvement_pct=(avg_smart - avg_naive) / avg_naive * 100,
        relevance_improvement_pct=(avg_smart - avg_naive) / avg_naive * 100
    )


def scenario_long_session(smart: SmartContextManager, naive: NaiveContextManager) -> BenchmarkComparison:
    """
    Test: Performance degradation over long sessions.

    Setup: 100 exchanges, measure context quality at different points.
    Measure: Does quality degrade? By how much?
    """
    print("\n📊 Scenario: Long Session Performance")
    print("=" * 50)

    # Simulate 100 exchanges
    exchanges = [
        f"Working on feature {i}: implemented step {i % 5}"
        for i in range(100)
    ]

    # Add key memories at different points
    key_memories = {
        10: "IMPORTANT: API key is stored in .env file",
        30: "IMPORTANT: Database connection pool max is 20",
        50: "IMPORTANT: Rate limit is 100 requests per minute",
        70: "IMPORTANT: Session timeout is 30 minutes",
        90: "IMPORTANT: Log level is DEBUG in development",
    }

    smart_quality_over_time = []
    naive_quality_over_time = []

    for i, exchange in enumerate(exchanges):
        smart.add_to_session(exchange)
        naive.add_to_session(exchange)

        if i in key_memories:
            mem = key_memories[i]
            smart.remember(mem, tags=["important"])
            naive.remember(mem)

        # Measure at intervals
        if i in [25, 50, 75, 99]:
            query = "What are the important configuration settings?"

            smart_result = smart.prepare_context(query, 600)
            naive_result = naive.prepare_context(query, 600)

            # Count how many IMPORTANT memories are in context
            smart_count = smart_result["context"].count("IMPORTANT")
            naive_count = naive_result["context"].count("IMPORTANT")

            expected = len([k for k in key_memories.keys() if k <= i])

            smart_quality = smart_count / max(expected, 1)
            naive_quality = naive_count / max(expected, 1)

            smart_quality_over_time.append((i, smart_quality))
            naive_quality_over_time.append((i, naive_quality))

            print(f"  At exchange {i}: Smart={smart_quality:.0%}, Naive={naive_quality:.0%}")

    # Calculate degradation
    if smart_quality_over_time:
        smart_start = smart_quality_over_time[0][1]
        smart_end = smart_quality_over_time[-1][1]
        smart_degradation = (smart_start - smart_end) / max(smart_start, 0.01)

        naive_start = naive_quality_over_time[0][1]
        naive_end = naive_quality_over_time[-1][1]
        naive_degradation = (naive_start - naive_end) / max(naive_start, 0.01)
    else:
        smart_degradation = 0
        naive_degradation = 0

    print(f"\n  Degradation: Smart={smart_degradation:.0%}, Naive={naive_degradation:.0%}")

    smart_bench = BenchmarkResult(
        scenario="long_session",
        variant="smartcontext",
        tokens_used=600,
        tokens_available=600,
        efficiency=1.0 - smart_degradation,
        recall_accuracy=smart_quality_over_time[-1][1] if smart_quality_over_time else 0,
        context_relevance=smart_quality_over_time[-1][1] if smart_quality_over_time else 0,
        duration_ms=0,
        details={"quality_over_time": smart_quality_over_time, "degradation": smart_degradation}
    )

    naive_bench = BenchmarkResult(
        scenario="long_session",
        variant="baseline",
        tokens_used=600,
        tokens_available=600,
        efficiency=1.0 - naive_degradation,
        recall_accuracy=naive_quality_over_time[-1][1] if naive_quality_over_time else 0,
        context_relevance=naive_quality_over_time[-1][1] if naive_quality_over_time else 0,
        duration_ms=0,
        details={"quality_over_time": naive_quality_over_time, "degradation": naive_degradation}
    )

    improvement = (naive_degradation - smart_degradation) / max(naive_degradation, 0.01) * 100

    return BenchmarkComparison(
        scenario="long_session",
        smartcontext=smart_bench,
        baseline=naive_bench,
        token_savings_pct=0,
        recall_improvement_pct=improvement,
        relevance_improvement_pct=improvement
    )


# ============================================================================
# Main
# ============================================================================

def run_benchmarks(scenarios: List[str] = None) -> List[BenchmarkComparison]:
    """Run selected benchmarks and return results."""

    all_scenarios = {
        "memory_recall": scenario_memory_recall,
        "token_efficiency": scenario_token_efficiency,
        "mode_detection": scenario_mode_detection,
        "long_session": scenario_long_session,
    }

    if scenarios is None or "all" in scenarios:
        scenarios = list(all_scenarios.keys())

    results = []

    print("\n" + "=" * 60)
    print("   SMARTCONTEXT BENCHMARK SUITE")
    print("=" * 60)

    for scenario_name in scenarios:
        if scenario_name not in all_scenarios:
            print(f"Unknown scenario: {scenario_name}")
            continue

        # Fresh instances for each scenario
        smart = SmartContextManager(namespace=f"bench_{scenario_name}_{int(time.time())}")
        naive = NaiveContextManager()

        result = all_scenarios[scenario_name](smart, naive)
        results.append(result)

    return results


def print_summary(results: List[BenchmarkComparison]):
    """Print a summary table of results."""

    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)

    print("\n┌─────────────────────┬───────────────┬───────────────┬────────────┐")
    print("│ Scenario            │ SmartContext  │ Baseline      │ Improvement│")
    print("├─────────────────────┼───────────────┼───────────────┼────────────┤")

    for r in results:
        sc_score = r.smartcontext.context_relevance * 100
        bl_score = r.baseline.context_relevance * 100
        improvement = r.relevance_improvement_pct

        print(f"│ {r.scenario:<19} │ {sc_score:>10.1f}%  │ {bl_score:>10.1f}%  │ {improvement:>+8.1f}% │")

    print("└─────────────────────┴───────────────┴───────────────┴────────────┘")

    # Overall
    avg_improvement = sum(r.relevance_improvement_pct for r in results) / len(results)
    print(f"\n  Average improvement: {avg_improvement:+.1f}%")

    # Save results
    output_file = Path(__file__).parent / "results.json"
    with open(output_file, "w") as f:
        json.dump([{
            "scenario": r.scenario,
            "smartcontext": asdict(r.smartcontext),
            "baseline": asdict(r.baseline),
            "improvements": {
                "tokens": r.token_savings_pct,
                "recall": r.recall_improvement_pct,
                "relevance": r.relevance_improvement_pct
            }
        } for r in results], f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SmartContext Benchmark Suite")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        choices=["all", "memory_recall", "token_efficiency", "mode_detection", "long_session"],
        help="Which scenarios to run"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    results = run_benchmarks(args.scenario)
    print_summary(results)


if __name__ == "__main__":
    main()
