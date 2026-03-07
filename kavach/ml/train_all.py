"""Comprehensive dataset downloader and full-stack model trainer.

Downloads security-focused datasets from HuggingFace, extracts
attack/benign pairs, then trains:
  1. ML Ensemble (GBM, LR, IsolationForest)
  2. RL Q-table (via simulated gateway episodes)

Datasets:
  - Anthropic/hh-rlhf (harmless-base split — 42K+ chosen/rejected RLHF pairs)
  - rubend18/ChatGPT-Jailbreak-Prompts (79 curated jailbreak prompts)
  - deepset/prompt-injections (train split — injection/benign classification)
  - jackhhao/jailbreak-classification (jailbreak/normal classification)
  - Bundled kavach.ml.dataset (196 curated samples)

Usage:
    python -m kavach.ml.train_all
    python -m kavach.ml.train_all --skip-rl
    python -m kavach.ml.train_all --skip-download  # use cached data only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATTACK_KEYWORDS = [
    "ignore previous", "ignore all", "disregard", "bypass", "jailbreak",
    "DAN", "developer mode", "pretend you", "act as", "you are now",
    "forget your", "new persona", "system prompt", "reveal your",
    "override", "hack", "exploit", "injection", "execute code",
    "sudo", "admin mode", "do anything now", "no restrictions",
    "ignore safety", "ignore rules", "ignore guidelines", "harmful",
    "illegal", "unethical", "violence", "weapon", "bomb", "kill",
    "steal", "rob", "poison", "drug", "ransomware", "malware",
    "how to hack", "break into", "fake id", "counterfeit",
]


def _classify_text(text: str) -> str:
    """Heuristic classifier for unlabeled text into attack categories."""
    lower = text.lower()

    injection_patterns = [
        "ignore previous", "ignore all", "disregard", "system prompt",
        "reveal your", "override", "injection", "execute code",
    ]
    jailbreak_patterns = [
        "jailbreak", "DAN", "developer mode", "pretend you",
        "act as", "you are now", "forget your", "new persona",
        "do anything now", "no restrictions", "bypass",
    ]
    exfiltration_patterns = [
        "reveal", "leak", "extract", "send to", "email me",
        "http_post", "exfiltrate", "social security", "credit card",
        "password", "api key", "secret",
    ]
    social_eng_patterns = [
        "as a doctor", "as a lawyer", "authority", "pretend",
        "impersonate", "role play", "act like",
    ]

    for p in injection_patterns:
        if p in lower:
            return "injection"
    for p in jailbreak_patterns:
        if p in lower:
            return "jailbreak"
    for p in exfiltration_patterns:
        if p in lower:
            return "exfiltration"
    for p in social_eng_patterns:
        if p in lower:
            return "social_engineering"

    # Count attack keyword hits
    hits = sum(1 for kw in ATTACK_KEYWORDS if kw in lower)
    if hits >= 2:
        return "injection"
    elif hits == 1:
        return "obfuscation"
    return "benign"


def _extract_rlhf_attacks(chosen: str, rejected: str) -> list[tuple[str, str]]:
    """Extract attack samples from Anthropic HH-RLHF conversation pairs.
    
    The 'rejected' response often represents the model complying with harmful
    requests, while 'chosen' is the better/safer response. We extract the
    *human prompts* from harmful conversations as attack training data.
    """
    pairs = []

    # Parse human turns from the rejected conversation
    human_turns = re.findall(r"Human:\s*(.+?)(?=\s*(?:Assistant:|$))", rejected, re.DOTALL)

    for turn in human_turns:
        turn = turn.strip()
        if len(turn) < 10:
            continue

        # Classify the human prompt
        category = _classify_text(turn)
        if category != "benign":
            pairs.append((turn, category))

    return pairs


# ---------------------------------------------------------------------------
# Dataset Downloaders
# ---------------------------------------------------------------------------

def download_anthropic_rlhf() -> list[tuple[str, str]]:
    """Download Anthropic HH-RLHF harmless-base and extract harmful prompts."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not installed")
        return []

    pairs: list[tuple[str, str]] = []

    for data_dir in ["harmless-base", "red-team-attempts"]:
        try:
            logger.info("Downloading Anthropic/hh-rlhf (%s)...", data_dir)
            ds = load_dataset(
                "Anthropic/hh-rlhf",
                data_dir=data_dir,
                split="train",
                verification_mode="no_checks",
            )

            if data_dir == "harmless-base":
                count = 0
                for row in ds:
                    extracted = _extract_rlhf_attacks(
                        row.get("chosen", ""),
                        row.get("rejected", ""),
                    )
                    pairs.extend(extracted)
                    count += len(extracted)
                    if count >= 5000:  # Cap to avoid overwhelming training
                        break
                logger.info("  -> %d attack prompts from hh-rlhf/%s", count, data_dir)

            elif data_dir == "red-team-attempts":
                count = 0
                for row in ds:
                    transcript = row.get("transcript", "")
                    score = row.get("min_harmlessness_score_transcript", 0)
                    # Extract harmful conversations (low harmlessness score)
                    if score is not None and score < -1.0:
                        human_turns = re.findall(
                            r"Human:\s*(.+?)(?=\s*(?:Assistant:|$))",
                            transcript, re.DOTALL,
                        )
                        for turn in human_turns:
                            turn = turn.strip()
                            if len(turn) > 10:
                                category = _classify_text(turn)
                                if category != "benign":
                                    pairs.append((turn, category))
                                    count += 1
                                    if count >= 3000:
                                        break
                    if count >= 3000:
                        break
                logger.info("  -> %d attack prompts from hh-rlhf/%s", count, data_dir)

        except Exception as e:
            logger.warning("Failed on Anthropic/hh-rlhf/%s: %s", data_dir, e)

    return pairs


def download_jailbreak_prompts() -> list[tuple[str, str]]:
    """Download rubend18/ChatGPT-Jailbreak-Prompts."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs: list[tuple[str, str]] = []
    try:
        logger.info("Downloading rubend18/ChatGPT-Jailbreak-Prompts...")
        ds = load_dataset(
            "rubend18/ChatGPT-Jailbreak-Prompts",
            split="train",
            verification_mode="no_checks",
        )
        for row in ds:
            prompt = str(row.get("Prompt", "")).strip()
            if len(prompt) > 20:
                pairs.append((prompt, "jailbreak"))
        logger.info("  -> %d jailbreak prompts", len(pairs))
    except Exception as e:
        logger.warning("Failed on ChatGPT-Jailbreak-Prompts: %s", e)

    return pairs


def download_deepset_injections() -> list[tuple[str, str]]:
    """Download deepset/prompt-injections."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs: list[tuple[str, str]] = []
    try:
        logger.info("Downloading deepset/prompt-injections...")
        ds = load_dataset(
            "deepset/prompt-injections",
            split="train",
            verification_mode="no_checks",
        )
        for row in ds:
            text = str(row.get("text", "")).strip()
            label = row.get("label", 0)
            if len(text) > 5:
                pairs.append((text, "injection" if label == 1 else "benign"))
        logger.info("  -> %d samples from deepset", len(pairs))
    except Exception as e:
        logger.warning("Failed on deepset/prompt-injections: %s", e)

    return pairs


def download_all_datasets() -> list[tuple[str, str]]:
    """Download all registered datasets."""
    all_pairs: list[tuple[str, str]] = []

    # 1. Anthropic HH-RLHF (biggest dataset)
    all_pairs.extend(download_anthropic_rlhf())

    # 2. ChatGPT Jailbreak Prompts
    all_pairs.extend(download_jailbreak_prompts())

    # 3. Deepset prompt injections
    all_pairs.extend(download_deepset_injections())

    return all_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ml_ensemble(
    texts: list[str],
    labels: list[int],
    save_dir: Path,
) -> dict[str, Any]:
    """Train and persist the ML ensemble."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn not installed")
        return {}

    from kavach.ml.features import extract_features_batch

    logger.info("Extracting features from %d samples...", len(texts))
    X = extract_features_batch(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # GBM
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
    gbm.fit(X_train_s, y_train)

    # LR
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)

    # IsolationForest
    X_benign = X_train_s[y_train == 0]
    iforest = IsolationForest(contamination=0.05, random_state=42)
    iforest.fit(X_benign if len(X_benign) > 10 else X_train_s)

    # Report
    y_pred = gbm.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=["benign", "attack"])
    logger.info("GBM Evaluation:\n%s", report)

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("gbm", gbm), ("lr", lr), ("iforest", iforest), ("scaler", scaler)]:
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    accuracy = float((y_pred == y_test).mean())
    meta = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "attack_ratio": float(sum(labels) / len(labels)),
        "gbm_test_accuracy": accuracy,
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("ML ensemble saved to %s (accuracy: %.2f%%)", save_dir, accuracy * 100)
    return meta


def train_rl_advisor(epochs: int = 50) -> dict[str, Any]:
    """Train the RL Q-table via simulated gateway episodes."""
    from kavach.core.gateway import KavachGateway
    from kavach.ml.dataset import TRAINING_DATA, BENIGN

    gateway = KavachGateway()
    advisor = gateway._rl_advisor

    logger.info("RL Bootstrap: %d samples × %d epochs", len(TRAINING_DATA), epochs)
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

    logger.info(
        "RL training complete: %d updates, %.1f%% coverage, %.1fs",
        stats["total_updates"], stats["coverage_pct"], elapsed,
    )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Full Kavach training pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training")
    parser.add_argument("--save-path", type=str, default="data/trained_models")
    parser.add_argument("--rl-epochs", type=int, default=50)
    args = parser.parse_args()

    save_dir = Path(args.save_path)

    # ── Step 1: Download datasets ──
    hf_pairs: list[tuple[str, str]] = []
    if not args.skip_download:
        print("=" * 60)
        print("STEP 1: DOWNLOADING DATASETS")
        print("=" * 60)
        hf_pairs = download_all_datasets()
        print(f"  Downloaded: {len(hf_pairs)} samples from HuggingFace")

        # Save raw pairs for future use
        raw_path = save_dir / "raw_dataset.json"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(hf_pairs[:10000], f)  # Cap at 10K for disk
        print(f"  Raw data cached to: {raw_path}")
    else:
        # Try to load cached data
        raw_path = save_dir / "raw_dataset.json"
        if raw_path.exists():
            with open(raw_path) as f:
                hf_pairs = json.load(f)
            print(f"  Loaded {len(hf_pairs)} cached samples")

    # ── Step 2: Build combined corpus ──
    from kavach.ml.dataset import TRAINING_DATA, BENIGN

    all_texts: list[str] = []
    all_labels: list[int] = []

    # Bundled dataset
    for sample in TRAINING_DATA:
        all_texts.append(sample.text)
        all_labels.append(0 if sample.label == BENIGN else 1)

    # HF datasets
    for text, label in hf_pairs:
        all_texts.append(text)
        all_labels.append(0 if label == "benign" else 1)

    benign = all_labels.count(0)
    attack = all_labels.count(1)
    print(f"\n  Combined corpus: {len(all_texts)} samples ({benign} benign, {attack} attack)")

    # ── Step 3: Train ML Ensemble ──
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING ML ENSEMBLE")
    print("=" * 60)
    ml_meta = train_ml_ensemble(all_texts, all_labels, save_dir)
    if ml_meta:
        print(f"  Accuracy:  {ml_meta['gbm_test_accuracy']:.2%}")
        print(f"  Features:  {ml_meta['n_features']}")
        print(f"  Train/Test: {ml_meta['n_train']}/{ml_meta['n_test']}")

    # ── Step 3.5: Train OOD Detector ──
    print("\n" + "=" * 60)
    print("STEP 3.5: TRAINING OOD DETECTOR")
    print("=" * 60)
    try:
        from kavach.ml.embeddings import EmbeddingRiskScorer
        from kavach.ml.ood_detector import OODDetector
        
        scorer = EmbeddingRiskScorer()
        scorer.load_and_encode_corpus()
        benign_texts = [t for t, l in zip(all_texts, all_labels) if l == 0]
        
        # OOD sampling ceiling
        if len(benign_texts) > 5000:
            benign_texts = random.sample(benign_texts, 5000)
            
        print(f"  Extracting ONNX embeddings for {len(benign_texts)} benign samples...")
        benign_embeddings = []
        for text in benign_texts:
            emb = scorer.encode(text)
            if emb is not None:
                benign_embeddings.append(emb)
                
        if benign_embeddings:
            ood = OODDetector(model_dir=str(save_dir))
            ood.fit(np.array(benign_embeddings))
            ood.save()
            print(f"  OOD Detector fitted on {len(benign_embeddings)} semantic vectors.")
    except Exception as e:
        print(f"  OOD Detector training skipped/failed: {e}")

    # ── Step 4: Train RL Advisor ──
    if not args.skip_rl:
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING RL ADVISOR")
        print("=" * 60)
        rl_stats = train_rl_advisor(epochs=args.rl_epochs)
        print(f"  Updates:     {rl_stats['total_updates']}")
        print(f"  Coverage:    {rl_stats['coverage_pct']}%")
        print(f"  Non-zero:    {rl_stats['non_zero_cells']}/{rl_stats['q_table_cells']}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Models saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
