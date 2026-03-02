"""Evaluation framework for Kavach detection performance.

Runs labeled datasets through the full ensemble pipeline and computes
standard classification metrics: precision, recall, F1, FPR, FNR,
attack detection rate, and per-category breakdowns.

Usage:
    python -m kavach.evaluation.benchmark --output reports/
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Holds evaluation metrics for a single run."""
    total_samples: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def false_negative_rate(self) -> float:
        denom = self.false_negatives + self.true_positives
        return self.false_negatives / denom if denom > 0 else 0.0

    @property
    def attack_detection_rate(self) -> float:
        return self.recall  # Same metric, different name

    @property
    def accuracy(self) -> float:
        total = self.total_samples
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def confusion_matrix(self) -> dict[str, int]:
        return {
            "TP": self.true_positives,
            "TN": self.true_negatives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
        }

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "attack_detection_rate": round(self.attack_detection_rate, 4),
            "confusion_matrix": self.confusion_matrix(),
        }
        if self.per_category:
            cat_report = {}
            for cat, counts in self.per_category.items():
                tp = counts.get("tp", 0)
                total = counts.get("total", 0)
                cat_report[cat] = {
                    "total": total,
                    "detected": tp,
                    "missed": counts.get("fn", 0),
                    "detection_rate": round(tp / total, 4) if total > 0 else 0.0,
                }
            report["per_category"] = cat_report
        return report


class KavachEvaluator:
    """Runs evaluation benchmarks against the Kavach detection pipeline.
    
    Usage:
        evaluator = KavachEvaluator()
        result = evaluator.evaluate_dataset(samples)
        print(result.to_report())
    """

    def __init__(self, block_threshold: float = 0.5) -> None:
        self.block_threshold = block_threshold

    def evaluate_dataset(
        self,
        samples: list[tuple[str, str]],
        gateway: Any | None = None,
    ) -> EvalResult:
        """Evaluate a list of (text, label) pairs.
        
        Args:
            samples: List of (prompt_text, ground_truth_label) where
                     label is 'benign' or an attack category name.
            gateway: Optional KavachGateway instance. If None, creates default.
        """
        if gateway is None:
            from kavach.core.gateway import KavachGateway
            gateway = KavachGateway()

        result = EvalResult()
        category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "total": 0})

        for text, label in samples:
            is_attack = label != "benign"
            decision = gateway.analyze(prompt=text, user_id="eval_user", role="default")

            risk_score = decision.get("risk_score", 0.0)
            action = decision.get("decision", "allow")
            was_blocked = action in ("block", "sanitize")

            result.total_samples += 1

            if is_attack and was_blocked:
                result.true_positives += 1
                category_counts[label]["tp"] += 1
            elif is_attack and not was_blocked:
                result.false_negatives += 1
                category_counts[label]["fn"] += 1
            elif not is_attack and was_blocked:
                result.false_positives += 1
            else:
                result.true_negatives += 1

            if is_attack:
                category_counts[label]["total"] += 1

            result.predictions.append({
                "text": text[:80],
                "label": label,
                "predicted": action,
                "risk_score": round(risk_score, 4),
                "correct": (is_attack == was_blocked),
            })

        result.per_category = dict(category_counts)
        return result

    def evaluate_bundled(self, gateway: Any | None = None) -> EvalResult:
        """Evaluate against the bundled curated dataset."""
        from kavach.ml.dataset import TRAINING_DATA
        samples = [(s.text, s.label) for s in TRAINING_DATA]
        return self.evaluate_dataset(samples, gateway=gateway)
