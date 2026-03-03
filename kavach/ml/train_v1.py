#!/usr/bin/env python3
"""Train all Kavach V1 models on the full downloaded corpus.

Usage:
    python -m kavach.ml.train_v1
    python -m kavach.ml.train_v1 --skip-rl
    python -m kavach.ml.train_v1 --dataset data/datasets/all_datasets.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def train_ml_ensemble(
    texts: list[str],
    labels: list[int],
    save_dir: Path,
) -> dict[str, Any]:
    """Train and persist the ML ensemble on the full corpus."""
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from kavach.ml.features import extract_features_batch

    log.info("Extracting features from %d samples...", len(texts))
    X = extract_features_batch(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log.info("Training GBM (500 estimators, depth 6)...")
    gbm = GradientBoostingClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_train_s, y_train)

    log.info("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    log.info("Training Isolation Forest...")
    X_benign = X_train_s[y_train == 0]
    iforest = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    iforest.fit(X_benign if len(X_benign) > 50 else X_train_s)

    # Report
    y_pred = gbm.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=["benign", "attack"])
    log.info("GBM Evaluation:\n%s", report)

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("gbm", gbm), ("lr", lr), ("iforest", iforest), ("scaler", scaler)]:
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    accuracy = float((y_pred == y_test).mean())
    from sklearn.metrics import precision_score, recall_score, f1_score

    meta = {
        "version": "v1.0",
        "n_total": len(texts),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "attack_ratio": float(sum(labels) / len(labels)),
        "gbm_test_accuracy": accuracy,
        "gbm_test_precision": float(precision_score(y_test, y_pred)),
        "gbm_test_recall": float(recall_score(y_test, y_pred)),
        "gbm_test_f1": float(f1_score(y_test, y_pred)),
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def train_rl_advisor(epochs: int = 30) -> dict[str, Any]:
    """Train the RL Q-table via simulated gateway episodes."""
    from kavach.core.gateway import KavachGateway
    from kavach.ml.dataset import TRAINING_DATA, BENIGN

    gateway = KavachGateway()
    advisor = gateway._rl_advisor

    total_steps = 0
    start = time.monotonic()

    for epoch in range(epochs):
        data = list(TRAINING_DATA)
        random.shuffle(data)

        for sample in data:
            is_attack = sample.label != BENIGN

            result = gateway.analyze(
                prompt=sample.text,
                user_id="rl_trainer",
                role=random.choice(["guest", "default", "admin"]),
            )

            rl_sugg = result.get("rl_suggestion", {})
            state_index = rl_sugg.get("state_index")
            action_index = rl_sugg.get("action_index")

            if state_index is not None and action_index is not None:
                action_taken = result.get("decision", "allow")
                was_blocked = action_taken in ("block", "sanitize")
                reward = advisor.compute_reward(
                    action_taken=action_taken,
                    actual_is_attack=is_attack,
                    was_blocked=was_blocked,
                )
                advisor.update(
                    state_index=state_index,
                    action_index=action_index,
                    reward=reward,
                )
                total_steps += 1

    advisor.save()
    elapsed = time.monotonic() - start
    stats = advisor.get_stats()
    stats["elapsed_secs"] = round(elapsed, 1)

    log.info(
        "RL training complete: %d updates, %.1f%% coverage, %.1fs",
        stats["total_updates"], stats["coverage_pct"], elapsed,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kavach V1 models.")
    parser.add_argument("--dataset", type=str, default="data/datasets/all_datasets.json")
    parser.add_argument("--save-path", type=str, default="data/trained_models")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--rl-epochs", type=int, default=30)
    args = parser.parse_args()

    save_dir = Path(args.save_path)

    # ── Load dataset ──
    print("=" * 60)
    print("KAVACH V1 — FULL MODEL TRAINING")
    print("=" * 60)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run: python -m kavach.ml.download_datasets")
        return

    with open(dataset_path) as f:
        raw = json.load(f)

    # Also add bundled dataset
    from kavach.ml.dataset import TRAINING_DATA, BENIGN

    all_texts: list[str] = []
    all_labels: list[int] = []

    # Bundled curated
    for sample in TRAINING_DATA:
        all_texts.append(sample.text)
        all_labels.append(0 if sample.label == BENIGN else 1)

    # Downloaded datasets
    for entry in raw:
        text = entry["text"] if isinstance(entry, dict) else entry[0]
        label = entry["label"] if isinstance(entry, dict) else entry[1]
        all_texts.append(text)
        all_labels.append(0 if label == "benign" else 1)

    benign = all_labels.count(0)
    attack = all_labels.count(1)
    print(f"\n📦 Corpus: {len(all_texts):,} samples ({benign:,} benign, {attack:,} attack)")
    print(f"   Attack ratio: {attack/len(all_texts):.1%}")

    # ── Train ML ──
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING ML ENSEMBLE")
    print("=" * 60)
    meta = train_ml_ensemble(all_texts, all_labels, save_dir)
    if meta:
        print(f"\n  📊 Results:")
        print(f"     Accuracy:   {meta['gbm_test_accuracy']:.1%}")
        print(f"     Precision:  {meta['gbm_test_precision']:.1%}")
        print(f"     Recall:     {meta['gbm_test_recall']:.1%}")
        print(f"     F1:         {meta['gbm_test_f1']:.1%}")
        print(f"     Features:   {meta['n_features']}")
        print(f"     Train/Test: {meta['n_train']:,}/{meta['n_test']:,}")

    # ── Train RL ──
    if not args.skip_rl:
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING RL ADVISOR")
        print("=" * 60)
        rl_stats = train_rl_advisor(epochs=args.rl_epochs)
        print(f"\n  🧠 Results:")
        print(f"     Updates:  {rl_stats['total_updates']:,}")
        print(f"     Coverage: {rl_stats['coverage_pct']}%")
        print(f"     Time:     {rl_stats['elapsed_secs']}s")

    print("\n" + "=" * 60)
    print("✅ KAVACH V1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Models saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
