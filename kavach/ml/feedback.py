"""Production feedback store for continuous learning.

Records decision outcomes from human reviewers or automated verification,
then exports them as training data for the next retraining cycle.

Storage: JSONL file — append-only, one record per line.

Usage:
    store = FeedbackStore("data/feedback.jsonl")
    store.record(trace_id="abc123", prompt="...", decision="block",
                 actual_label="benign", was_correct=False)
    
    # Export for retraining
    pairs = store.export_training_data()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Append-only JSONL store for production decision feedback."""

    def __init__(self, path: str = "data/feedback.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        trace_id: str,
        prompt: str,
        decision: str,
        actual_label: str,
        was_correct: bool,
        reviewer: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single feedback entry."""
        entry = {
            "timestamp": time.time(),
            "trace_id": trace_id,
            "prompt": prompt,
            "decision": decision,
            "actual_label": actual_label,
            "was_correct": was_correct,
            "reviewer": reviewer,
            "metadata": metadata or {},
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def export_training_data(self) -> list[tuple[str, str]]:
        """Export feedback as (text, label) pairs for retraining."""
        if not self._path.exists():
            return []

        pairs: list[tuple[str, str]] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    pairs.append((entry["prompt"], entry["actual_label"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        return pairs

    def get_false_positives(self) -> list[dict[str, Any]]:
        """Return all recorded false positives (safe content that was blocked)."""
        return self._filter(lambda e: not e["was_correct"] and e["actual_label"] == "benign")

    def get_false_negatives(self) -> list[dict[str, Any]]:
        """Return all recorded false negatives (attacks that were missed)."""
        return self._filter(lambda e: not e["was_correct"] and e["actual_label"] != "benign")

    def stats(self) -> dict[str, Any]:
        """Return summary statistics of recorded feedback."""
        if not self._path.exists():
            return {"total": 0}

        total = 0
        correct = 0
        fp = 0
        fn = 0
        with open(self._path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    total += 1
                    if entry.get("was_correct"):
                        correct += 1
                    elif entry.get("actual_label") == "benign":
                        fp += 1
                    else:
                        fn += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return {
            "total": total,
            "correct": correct,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        }

    def _filter(self, fn: Any) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        results = []
        with open(self._path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if fn(entry):
                        results.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue
        return results
