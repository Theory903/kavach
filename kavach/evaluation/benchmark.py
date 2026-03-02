"""CLI benchmark runner for Kavach evaluation.

Usage:
    python -m kavach.evaluation.benchmark
    python -m kavach.evaluation.benchmark --output reports/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kavach.evaluation import KavachEvaluator


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run Kavach evaluation benchmark.")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON report.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Block threshold.")
    args = parser.parse_args()

    evaluator = KavachEvaluator(block_threshold=args.threshold)

    print("Running evaluation on bundled dataset...")
    result = evaluator.evaluate_bundled()
    report = result.to_report()

    print("\n" + "=" * 60)
    print("KAVACH EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total Samples:          {report['total_samples']}")
    print(f"  Accuracy:               {report['accuracy']:.2%}")
    print(f"  Precision:              {report['precision']:.2%}")
    print(f"  Recall (ADR):           {report['recall']:.2%}")
    print(f"  F1 Score:               {report['f1_score']:.2%}")
    print(f"  False Positive Rate:    {report['false_positive_rate']:.2%}")
    print(f"  False Negative Rate:    {report['false_negative_rate']:.2%}")
    print()
    print("Confusion Matrix:")
    cm = report["confusion_matrix"]
    print(f"  TP={cm['TP']}  FP={cm['FP']}")
    print(f"  FN={cm['FN']}  TN={cm['TN']}")

    if "per_category" in report:
        print("\nPer-Category Detection Rates:")
        for cat, stats in report["per_category"].items():
            print(f"  {cat:24s}  {stats['detected']}/{stats['total']}  ({stats['detection_rate']:.1%})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
