"""Red Teaming CLI runner.

Automatically attacks the Kavach gateway with mutated adversarial prompts
and generates a report of bypasses and detection metrics.

Usage:
    python -m kavach.redteam.runner --category jailbreak --iterations 500
    python -m kavach.redteam.runner --all --output reports/redteam.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from kavach.ml.dataset import TRAINING_DATA, BENIGN
from kavach.redteam.mutator import mutate_batch

logger = logging.getLogger(__name__)


def run_redteam(
    category: str | None = None,
    max_iterations: int = 500,
    strategies: list[str] | None = None,
    gateway: Any | None = None,
) -> dict[str, Any]:
    """Run a red team attack campaign against the gateway.
    
    Args:
        category: Attack category to test (or None for all).
        max_iterations: Max number of mutated prompts to generate.
        strategies: Mutation strategies to apply.
        gateway: KavachGateway instance (creates default if None).
    """
    if gateway is None:
        from kavach.core.gateway import KavachGateway
        gateway = KavachGateway()

    # Collect seed attack prompts
    seeds = []
    for s in TRAINING_DATA:
        if s.label == BENIGN:
            continue
        if category and s.label != category:
            continue
        seeds.append((s.text, s.label))

    if not seeds:
        return {"error": f"No seeds found for category: {category}"}

    # Generate mutations
    seed_texts = [s[0] for s in seeds]
    mutations = mutate_batch(seed_texts, strategies=strategies, max_per_seed=max_iterations // len(seeds) + 1)
    mutations = mutations[:max_iterations]

    logger.info("Running red team: %d mutations from %d seeds (category=%s)", len(mutations), len(seeds), category or "all")

    # Attack
    bypasses: list[dict[str, Any]] = []
    blocked = 0
    total = 0
    start = time.monotonic()

    for mutation in mutations:
        total += 1
        try:
            result = gateway.analyze(prompt=mutation, user_id="redteam_bot", role="guest")
            action = result.get("decision", "allow")
            risk_score = result.get("risk_score", 0.0)

            if action in ("block", "sanitize"):
                blocked += 1
            else:
                bypasses.append({
                    "prompt": mutation[:200],
                    "decision": action,
                    "risk_score": round(risk_score, 4),
                })
        except Exception as e:
            logger.warning("Red team probe error: %s", e)

    elapsed = time.monotonic() - start

    report = {
        "category": category or "all",
        "total_probes": total,
        "blocked": blocked,
        "bypasses": len(bypasses),
        "detection_rate": round(blocked / total, 4) if total > 0 else 0.0,
        "bypass_rate": round(len(bypasses) / total, 4) if total > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "avg_latency_ms": round((elapsed / total) * 1000, 2) if total > 0 else 0.0,
        "bypass_samples": bypasses[:20],  # Cap for report readability
    }

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Kavach Red Teaming Engine.")
    parser.add_argument("--category", type=str, default=None,
                        help="Attack category (injection, jailbreak, exfiltration, etc.)")
    parser.add_argument("--all", action="store_true", help="Run against all categories.")
    parser.add_argument("--iterations", type=int, default=500, help="Max mutations to test.")
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON file.")
    args = parser.parse_args()

    categories = [None] if args.all else [args.category]
    if args.all:
        categories = ["injection", "jailbreak", "exfiltration", "obfuscation", "social_engineering"]

    all_reports = []
    for cat in categories:
        print(f"\n{'='*50}")
        print(f"Red Team: {cat or 'ALL'}")
        print(f"{'='*50}")

        report = run_redteam(category=cat, max_iterations=args.iterations)
        all_reports.append(report)

        print(f"  Probes:         {report['total_probes']}")
        print(f"  Blocked:        {report['blocked']}")
        print(f"  Bypasses:       {report['bypasses']}")
        print(f"  Detection Rate: {report['detection_rate']:.1%}")
        print(f"  Avg Latency:    {report['avg_latency_ms']:.1f}ms")

        if report["bypass_samples"]:
            print(f"\n  Sample Bypasses:")
            for bp in report["bypass_samples"][:5]:
                print(f"    [{bp['risk_score']:.2f}] {bp['prompt'][:80]}...")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
