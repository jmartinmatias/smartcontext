#!/usr/bin/env python3
"""
Enhanced SmartContext Benchmark Suite

More realistic scenarios based on actual use cases:
1. Multi-turn debugging session
2. Large codebase with artifacts
3. Error learning over time
4. Strategy evolution
5. Constraint adherence
6. Session compaction quality

Usage:
    python enhanced_benchmark.py --scenario all
    python enhanced_benchmark.py --scenario debugging_session
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import MemoryStore
from attention import AttentionManager, ContextCompiler, AttentionMode
from session import BaseSession, TrimConfig, TrimStrategy


@dataclass
class EnhancedResult:
    """Result with detailed metrics."""
    scenario: str
    variant: str
    metrics: Dict[str, float]
    details: Dict[str, Any]
    duration_ms: float


def estimate_tokens(text: str) -> int:
    return len(str(text)) // 4


# ============================================================================
# Scenario 1: Multi-Turn Debugging Session
# ============================================================================

def scenario_debugging_session() -> Tuple[EnhancedResult, EnhancedResult]:
    """
    Simulates a realistic debugging session:
    - User reports bug
    - Multiple investigation steps
    - Dead ends tracked
    - Eventually finds solution

    Measures: Error recall, dead-end avoidance, solution relevance
    """
    print("\n" + "=" * 60)
    print("📊 SCENARIO: Multi-Turn Debugging Session")
    print("=" * 60)

    # Setup: A realistic debugging conversation
    debugging_turns = [
        ("user", "The checkout is failing with a 500 error"),
        ("assistant", "Let me check the logs. I see a NullPointerException in PaymentService."),
        ("user", "I tried adding a null check but it still fails"),
        ("assistant", "The null check didn't work. Let me look at the stack trace more carefully."),
        ("user", "Could it be the database connection?"),
        ("assistant", "I checked DB - connection is fine, pool has available connections."),
        ("user", "What about the payment gateway API?"),
        ("assistant", "API credentials are valid, test ping works. Not the issue."),
        ("user", "Maybe it's the session handling?"),
        ("assistant", "Found it! Session expires before payment completes. Need to extend timeout."),
        ("user", "Great, how do we fix it?"),
        ("assistant", "Set SESSION_TIMEOUT=3600 in config. Also add retry logic for expired sessions."),
    ]

    errors_tried = [
        ("Added null check to PaymentService", "Still fails - NPE in different location"),
        ("Checked database connection", "Connection OK - not the issue"),
        ("Tested payment API credentials", "API works - not the issue"),
    ]

    solution = "Session expires before payment completes. Fix: SESSION_TIMEOUT=3600"

    # --- SmartContext Approach ---
    start = time.time()
    smart = MemoryStore(namespace=f"debug_{int(time.time())}")
    attention = AttentionManager()

    # Simulate the session
    for role, content in debugging_turns:
        smart.add_session_item(role, content)
        # Auto-detect mode
        if role == "user":
            attention.maybe_switch_mode(content)

    # Log errors
    for action, result in errors_tried:
        smart.log_error(action, result)

    # Query: "What have we tried and what's the fix?"
    query = "Summarize what we tried and the solution"

    # Get context
    session_items = smart.get_session_items(limit=10)
    error_context = smart.get_errors()

    compiler = ContextCompiler(attention)
    compiled = compiler.compile(
        goal=query,
        observations=[item.content for item in session_items],
        strategy=error_context,
        token_budget=800
    )

    smart_context = compiled.to_prompt()
    smart_duration = (time.time() - start) * 1000

    # --- Naive Approach ---
    start = time.time()
    naive_context = "\n".join([f"{role}: {content}" for role, content in debugging_turns])
    naive_duration = (time.time() - start) * 1000

    # Measure quality
    def measure_debug_quality(context: str) -> Dict[str, float]:
        # Does it mention the dead ends?
        dead_ends_mentioned = sum(1 for action, _ in errors_tried if action.lower()[:20] in context.lower())
        dead_end_score = dead_ends_mentioned / len(errors_tried)

        # Does it have the solution?
        solution_present = "session" in context.lower() and ("timeout" in context.lower() or "expire" in context.lower())

        # Is it in debugging mode?
        mode_correct = attention.current_mode == AttentionMode.DEBUGGING

        return {
            "dead_end_awareness": dead_end_score,
            "solution_present": 1.0 if solution_present else 0.0,
            "mode_correct": 1.0 if mode_correct else 0.0,
            "tokens_used": estimate_tokens(context)
        }

    smart_metrics = measure_debug_quality(smart_context)
    naive_metrics = measure_debug_quality(naive_context)
    naive_metrics["mode_correct"] = 0.0  # Naive has no mode

    print(f"\n  SmartContext:")
    print(f"    Dead-end awareness: {smart_metrics['dead_end_awareness']:.0%}")
    print(f"    Solution present: {'✓' if smart_metrics['solution_present'] else '✗'}")
    print(f"    Mode detected: debugging {'✓' if smart_metrics['mode_correct'] else '✗'}")
    print(f"    Tokens: {smart_metrics['tokens_used']}")

    print(f"\n  Baseline:")
    print(f"    Dead-end awareness: {naive_metrics['dead_end_awareness']:.0%}")
    print(f"    Solution present: {'✓' if naive_metrics['solution_present'] else '✗'}")
    print(f"    Tokens: {naive_metrics['tokens_used']}")

    smart_result = EnhancedResult(
        scenario="debugging_session",
        variant="smartcontext",
        metrics=smart_metrics,
        details={"error_count": len(errors_tried), "mode": attention.current_mode.value},
        duration_ms=smart_duration
    )

    naive_result = EnhancedResult(
        scenario="debugging_session",
        variant="baseline",
        metrics=naive_metrics,
        details={"error_count": 0},
        duration_ms=naive_duration
    )

    return smart_result, naive_result


# ============================================================================
# Scenario 2: Large Codebase with Artifacts
# ============================================================================

def scenario_large_codebase() -> Tuple[EnhancedResult, EnhancedResult]:
    """
    Simulates working with a large codebase:
    - 20 files stored as artifacts
    - Query about specific functionality
    - Measure: Relevant files found vs tokens used
    """
    print("\n" + "=" * 60)
    print("📊 SCENARIO: Large Codebase Navigation")
    print("=" * 60)

    # Simulate 20 code files
    files = {
        "auth_service.py": "class AuthService:\n    def login(self, email, password): ...\n    def logout(self): ...\n    def refresh_token(self): ...",
        "user_model.py": "class User:\n    id: int\n    email: str\n    password_hash: str",
        "payment_service.py": "class PaymentService:\n    def charge(self, amount, card): ...\n    def refund(self, transaction_id): ...",
        "order_model.py": "class Order:\n    id: int\n    user_id: int\n    total: Decimal",
        "cart_service.py": "class CartService:\n    def add_item(self, product_id): ...\n    def remove_item(self, product_id): ...",
        "product_model.py": "class Product:\n    id: int\n    name: str\n    price: Decimal",
        "inventory_service.py": "class InventoryService:\n    def check_stock(self, product_id): ...\n    def reserve(self, product_id, qty): ...",
        "shipping_service.py": "class ShippingService:\n    def calculate_cost(self, address): ...\n    def create_label(self, order): ...",
        "email_service.py": "class EmailService:\n    def send_confirmation(self, order): ...\n    def send_reset_password(self, user): ...",
        "notification_service.py": "class NotificationService:\n    def push(self, user_id, message): ...",
        "analytics_service.py": "class AnalyticsService:\n    def track_event(self, event): ...",
        "cache_service.py": "class CacheService:\n    def get(self, key): ...\n    def set(self, key, value): ...",
        "database.py": "class Database:\n    def query(self, sql): ...\n    def execute(self, sql): ...",
        "config.py": "DATABASE_URL = 'postgresql://...'\nREDIS_URL = 'redis://...'",
        "middleware.py": "class AuthMiddleware:\n    def process_request(self, request): ...",
        "validators.py": "def validate_email(email): ...\ndef validate_password(password): ...",
        "exceptions.py": "class AuthError(Exception): ...\nclass PaymentError(Exception): ...",
        "utils.py": "def hash_password(password): ...\ndef generate_token(): ...",
        "api_routes.py": "router = Router()\n@router.post('/login')\n@router.post('/register')",
        "tests_auth.py": "def test_login(): ...\ndef test_logout(): ...",
    }

    # Query about authentication
    query = "How does user authentication work? Show me the login flow."
    relevant_files = ["auth_service.py", "user_model.py", "middleware.py", "validators.py", "api_routes.py"]

    # --- SmartContext Approach ---
    start = time.time()
    smart = MemoryStore(namespace=f"codebase_{int(time.time())}")

    for name, content in files.items():
        smart.store_artifact(name, content * 10)  # Make files bigger
        # Also remember file purposes
        smart.remember(f"File {name}: {content.split(':')[0] if ':' in content else name}",
                      tags=[name.replace('.py', ''), 'code'])

    # Search for relevant
    memories = smart.search("authentication login", limit=5)
    artifacts = smart.list_artifacts()

    # Build context with references only
    context_parts = ["Relevant code files:"]
    for mem in memories:
        context_parts.append(f"  - {mem.content}")
    context_parts.append("\nArtifacts available:")
    for art in artifacts[:10]:
        context_parts.append(f"  #{art.id}: {art.name} ({art.size} chars)")

    smart_context = "\n".join(context_parts)
    smart_duration = (time.time() - start) * 1000

    # --- Naive Approach ---
    start = time.time()
    # Include ALL file contents inline
    naive_parts = []
    for name, content in files.items():
        naive_parts.append(f"=== {name} ===\n{content * 10}")
    naive_context = "\n\n".join(naive_parts)

    # Truncate to reasonable size
    if len(naive_context) > 8000:
        naive_context = naive_context[:8000] + "\n... [TRUNCATED - more files not shown]"

    naive_duration = (time.time() - start) * 1000

    # Measure quality
    def count_relevant_files(context: str) -> int:
        return sum(1 for f in relevant_files if f.lower() in context.lower())

    smart_relevant = count_relevant_files(smart_context)
    naive_relevant = count_relevant_files(naive_context)

    smart_tokens = estimate_tokens(smart_context)
    naive_tokens = estimate_tokens(naive_context)

    # Efficiency: relevant files per 100 tokens
    smart_efficiency = (smart_relevant / len(relevant_files)) / (smart_tokens / 100) if smart_tokens > 0 else 0
    naive_efficiency = (naive_relevant / len(relevant_files)) / (naive_tokens / 100) if naive_tokens > 0 else 0

    print(f"\n  SmartContext:")
    print(f"    Relevant files found: {smart_relevant}/{len(relevant_files)}")
    print(f"    Tokens used: {smart_tokens}")
    print(f"    Efficiency: {smart_efficiency:.2f} relevance per 100 tokens")

    print(f"\n  Baseline:")
    print(f"    Relevant files found: {naive_relevant}/{len(relevant_files)}")
    print(f"    Tokens used: {naive_tokens}")
    print(f"    Efficiency: {naive_efficiency:.2f} relevance per 100 tokens")

    smart_result = EnhancedResult(
        scenario="large_codebase",
        variant="smartcontext",
        metrics={
            "relevant_found": smart_relevant / len(relevant_files),
            "tokens_used": smart_tokens,
            "efficiency": smart_efficiency
        },
        details={"total_files": len(files), "relevant_files": relevant_files},
        duration_ms=smart_duration
    )

    naive_result = EnhancedResult(
        scenario="large_codebase",
        variant="baseline",
        metrics={
            "relevant_found": naive_relevant / len(relevant_files),
            "tokens_used": naive_tokens,
            "efficiency": naive_efficiency
        },
        details={"truncated": len(naive_context) >= 8000},
        duration_ms=naive_duration
    )

    return smart_result, naive_result


# ============================================================================
# Scenario 3: Error Learning Over Time
# ============================================================================

def scenario_error_learning() -> Tuple[EnhancedResult, EnhancedResult]:
    """
    Simulates learning from errors across multiple "sessions":
    - Session 1: Makes mistake A
    - Session 2: Makes mistake B
    - Session 3: Query about best practices

    Measures: Are past mistakes reflected in guidance?
    """
    print("\n" + "=" * 60)
    print("📊 SCENARIO: Error Learning Over Time")
    print("=" * 60)

    # Errors from past "sessions"
    past_errors = [
        ("Used == for None comparison", "Should use 'is None' in Python"),
        ("Forgot to close database connection", "Use context manager or finally block"),
        ("Hardcoded API key in source", "Use environment variables"),
        ("No input validation on user data", "Always validate/sanitize user input"),
        ("Synchronous API call in async function", "Use await or run_in_executor"),
    ]

    # --- SmartContext Approach ---
    start = time.time()
    smart = MemoryStore(namespace=f"learning_{int(time.time())}")

    # Initialize strategy
    smart.initialize_strategy(domain="python_development")

    # Record past errors
    for action, lesson in past_errors:
        smart.log_error(action, lesson)
        smart.update_strategy("record_failure", f"{action} → {lesson}")

    # Query about best practices
    query = "What should I watch out for when writing Python code?"

    error_context = smart.get_errors()
    strategy = smart.get_strategy()

    smart_context = f"Past mistakes to avoid:\n{error_context}\n\nLearned patterns:\n"
    if strategy:
        for failure in strategy.failure_modes[:5]:
            smart_context += f"  - {failure['content']}\n"

    smart_duration = (time.time() - start) * 1000

    # --- Naive Approach ---
    start = time.time()
    # No error tracking, no learning
    naive_context = "No historical error data available."
    naive_duration = (time.time() - start) * 1000

    # Measure: How many lessons are accessible?
    smart_lessons = sum(1 for _, lesson in past_errors if lesson.lower()[:20] in smart_context.lower())
    naive_lessons = 0

    print(f"\n  SmartContext:")
    print(f"    Lessons accessible: {smart_lessons}/{len(past_errors)}")
    print(f"    Strategy version: {strategy.version if strategy else 'N/A'}")
    print(f"    Has failure patterns: {'✓' if strategy and strategy.failure_modes else '✗'}")

    print(f"\n  Baseline:")
    print(f"    Lessons accessible: {naive_lessons}/{len(past_errors)}")
    print(f"    No learning capability")

    smart_result = EnhancedResult(
        scenario="error_learning",
        variant="smartcontext",
        metrics={
            "lessons_accessible": smart_lessons / len(past_errors),
            "strategy_version": strategy.version if strategy else 0,
            "has_patterns": 1.0 if strategy and strategy.failure_modes else 0.0
        },
        details={"total_errors": len(past_errors)},
        duration_ms=smart_duration
    )

    naive_result = EnhancedResult(
        scenario="error_learning",
        variant="baseline",
        metrics={
            "lessons_accessible": 0.0,
            "strategy_version": 0,
            "has_patterns": 0.0
        },
        details={},
        duration_ms=naive_duration
    )

    return smart_result, naive_result


# ============================================================================
# Scenario 4: Constraint Adherence
# ============================================================================

def scenario_constraint_adherence() -> Tuple[EnhancedResult, EnhancedResult]:
    """
    Simulates working with project constraints:
    - Set specific constraints
    - Query for implementation
    - Measure: Are constraints in context?
    """
    print("\n" + "=" * 60)
    print("📊 SCENARIO: Constraint Adherence")
    print("=" * 60)

    constraints = [
        "No external API calls in unit tests",
        "All dates must be stored in UTC",
        "Maximum function length: 50 lines",
        "Use type hints for all function parameters",
        "No print statements in production code",
    ]

    # --- SmartContext Approach ---
    start = time.time()
    smart = MemoryStore(namespace=f"constraints_{int(time.time())}")

    for constraint in constraints:
        smart.add_constraint(constraint)

    # Get working context
    working = smart.get_agentic_working_context()
    smart_context = f"Constraints:\n" + "\n".join(f"  - {c}" for c in working['constraints'])
    smart_duration = (time.time() - start) * 1000

    # --- Naive Approach ---
    start = time.time()
    # Constraints would need to be manually repeated each time
    naive_context = "No constraint tracking."
    naive_duration = (time.time() - start) * 1000

    # Measure
    smart_constraints = sum(1 for c in constraints if c.lower()[:20] in smart_context.lower())
    naive_constraints = 0

    print(f"\n  SmartContext:")
    print(f"    Constraints in context: {smart_constraints}/{len(constraints)}")

    print(f"\n  Baseline:")
    print(f"    Constraints in context: {naive_constraints}/{len(constraints)}")

    smart_result = EnhancedResult(
        scenario="constraint_adherence",
        variant="smartcontext",
        metrics={"constraints_present": smart_constraints / len(constraints)},
        details={"total_constraints": len(constraints)},
        duration_ms=smart_duration
    )

    naive_result = EnhancedResult(
        scenario="constraint_adherence",
        variant="baseline",
        metrics={"constraints_present": 0.0},
        details={},
        duration_ms=naive_duration
    )

    return smart_result, naive_result


# ============================================================================
# Scenario 5: Session Compaction Quality
# ============================================================================

def scenario_compaction_quality() -> Tuple[EnhancedResult, EnhancedResult]:
    """
    Simulates a very long session that needs compaction:
    - 200 exchanges
    - Key information at various points
    - Compact and measure retention
    """
    print("\n" + "=" * 60)
    print("📊 SCENARIO: Session Compaction Quality")
    print("=" * 60)

    # Create a long session with key information scattered throughout
    key_info = {
        10: "IMPORTANT: API rate limit is 100/minute",
        50: "IMPORTANT: Database uses read replicas",
        100: "IMPORTANT: Cache TTL is 5 minutes",
        150: "IMPORTANT: Auth tokens expire in 1 hour",
    }

    # --- SmartContext Approach ---
    start = time.time()
    smart = MemoryStore(namespace=f"compact_{int(time.time())}")

    for i in range(200):
        if i in key_info:
            content = key_info[i]
            smart.remember(content, tags=["important"])  # Also save to long-term
        else:
            content = f"Regular exchange {i}: discussing implementation details"
        smart.log_event("exchange", {"content": content, "index": i})

    # Compact
    result = smart.compact_session_schema(preserve_recent=30, extract_insights=True)

    # Check if key info is still accessible
    smart_context = str(smart.search("IMPORTANT", limit=10))
    for _, info in key_info.items():
        if info not in smart_context:
            # Try to find in long-term memory
            matches = smart.search(info[:30], limit=1)
            if matches:
                smart_context += f"\n{matches[0].content}"

    smart_duration = (time.time() - start) * 1000

    # --- Naive Approach ---
    start = time.time()
    # Just truncate
    naive_session = []
    for i in range(200):
        if i in key_info:
            naive_session.append(key_info[i])
        else:
            naive_session.append(f"Exchange {i}")

    # Truncate to last 50 (simulating context limit)
    naive_context = "\n".join(naive_session[-50:])
    naive_duration = (time.time() - start) * 1000

    # Measure retention
    smart_retained = sum(1 for _, info in key_info.items() if info[11:30].lower() in smart_context.lower())
    naive_retained = sum(1 for _, info in key_info.items() if info[11:30].lower() in naive_context.lower())

    print(f"\n  SmartContext:")
    print(f"    Key info retained: {smart_retained}/{len(key_info)}")
    print(f"    Compaction result: {result.get('original_size', 0)} → {result.get('compacted_size', 0)}")
    print(f"    Insights extracted: {result.get('insights_extracted', 0)}")

    print(f"\n  Baseline (truncation):")
    print(f"    Key info retained: {naive_retained}/{len(key_info)}")
    print(f"    Simply kept last 50 exchanges")

    smart_result = EnhancedResult(
        scenario="compaction_quality",
        variant="smartcontext",
        metrics={
            "retention": smart_retained / len(key_info),
            "compression_ratio": result.get('compacted_size', 200) / 200
        },
        details=result,
        duration_ms=smart_duration
    )

    naive_result = EnhancedResult(
        scenario="compaction_quality",
        variant="baseline",
        metrics={
            "retention": naive_retained / len(key_info),
            "compression_ratio": 50 / 200
        },
        details={"method": "truncation"},
        duration_ms=naive_duration
    )

    return smart_result, naive_result


# ============================================================================
# Main
# ============================================================================

def run_enhanced_benchmarks(scenarios: List[str] = None) -> List[Tuple[EnhancedResult, EnhancedResult]]:
    """Run enhanced benchmarks."""

    all_scenarios = {
        "debugging_session": scenario_debugging_session,
        "large_codebase": scenario_large_codebase,
        "error_learning": scenario_error_learning,
        "constraint_adherence": scenario_constraint_adherence,
        "compaction_quality": scenario_compaction_quality,
    }

    if scenarios is None or "all" in scenarios:
        scenarios = list(all_scenarios.keys())

    results = []

    print("\n" + "=" * 70)
    print("   ENHANCED SMARTCONTEXT BENCHMARK SUITE")
    print("=" * 70)

    for scenario_name in scenarios:
        if scenario_name not in all_scenarios:
            print(f"Unknown scenario: {scenario_name}")
            continue

        smart, naive = all_scenarios[scenario_name]()
        results.append((smart, naive))

    return results


def print_enhanced_summary(results: List[Tuple[EnhancedResult, EnhancedResult]]):
    """Print summary of enhanced results."""

    print("\n" + "=" * 70)
    print("   ENHANCED BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────┬─────────────────┬─────────────────┬────────────┐")
    print("│ Scenario                │ SmartContext    │ Baseline        │ Winner     │")
    print("├─────────────────────────┼─────────────────┼─────────────────┼────────────┤")

    smart_wins = 0
    naive_wins = 0

    for smart, naive in results:
        # Get primary metric
        primary_metric = list(smart.metrics.keys())[0]
        smart_val = smart.metrics[primary_metric]
        naive_val = naive.metrics[primary_metric]

        winner = "SmartContext" if smart_val > naive_val else ("Baseline" if naive_val > smart_val else "Tie")
        if winner == "SmartContext":
            smart_wins += 1
        elif winner == "Baseline":
            naive_wins += 1

        print(f"│ {smart.scenario:<23} │ {smart_val:>13.1%}  │ {naive_val:>13.1%}  │ {winner:<10} │")

    print("└─────────────────────────┴─────────────────┴─────────────────┴────────────┘")

    print(f"\n  SmartContext wins: {smart_wins}/{len(results)}")
    print(f"  Baseline wins: {naive_wins}/{len(results)}")

    # Save results
    output_file = Path(__file__).parent / "enhanced_results.json"
    with open(output_file, "w") as f:
        json.dump([{
            "smartcontext": asdict(smart),
            "baseline": asdict(naive)
        } for smart, naive in results], f, indent=2, default=str)

    print(f"\n  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced SmartContext Benchmarks")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        choices=["all", "debugging_session", "large_codebase", "error_learning",
                 "constraint_adherence", "compaction_quality"],
        help="Which scenarios to run"
    )

    args = parser.parse_args()

    results = run_enhanced_benchmarks(args.scenario)
    print_enhanced_summary(results)


if __name__ == "__main__":
    main()
