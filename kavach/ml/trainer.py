"""External dataset downloader and ensemble trainer for Kavach.

Downloads real-world prompt injection, jailbreak, and security datasets
from HuggingFace and other public sources, then retrains the ML ensemble
with the combined corpus.

Usage:
    python -m kavach.ml.trainer              # Download all + train
    python -m kavach.ml.trainer --dry-run    # List datasets without downloading
    python -m kavach.ml.trainer --save-path data/training  # Specify output dir

Datasets fetched (HuggingFace, open access):
  - qualifire/prompt-injection-benchmark  (~5K labeled)
  - xxz224/prompt-injection-attack-dataset (~3.7K)
  - yanismiraoui/prompt_injections (~1K, multilingual)
  - deepset/prompt-injections (Deepset curated)
  - rubend18/ChatGPT-Jailbreak-Prompts (jailbreak collection)

Combined with bundled curated dataset from kavach.ml.dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset
    _HAS_HF_DATASETS = True
except ImportError:
    _HAS_HF_DATASETS = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Dataset Registry — HuggingFace repos and their column mappings
# ---------------------------------------------------------------------------

HF_DATASETS: list[dict[str, Any]] = [
    {
        "name": "deepset/prompt-injections",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "label_map": {1: "injection", 0: "benign"},
        "trust_remote_code": False,
    },
    {
        "name": "Skywork/Jailbreak_Dataset_v2",
        "split": "train",
        "text_col": "prompt",
        "label_col": None,
        "label_map": None,
        "forced_label": "jailbreak",
        "trust_remote_code": False,
    },
    {
        "name": "jackhhao/jailbreak-classification",
        "split": "train",
        "text_col": "prompt",
        "label_col": "type",
        "label_map": {"jailbreak": "jailbreak", "normal": "benign"},
        "trust_remote_code": False,
    },
    {
        "name": "hackaprompt/hackaprompt-dataset",
        "split": "train",
        "text_col": "user_input",
        "label_col": None,
        "forced_label": "injection",
        "trust_remote_code": False,
    },
]


def _download_hf_dataset(spec: dict[str, Any]) -> list[tuple[str, str]]:
    """Download a single HuggingFace dataset and return (text, label) pairs."""
    name = spec["name"]
    split = spec.get("split", "train")
    text_col = spec["text_col"]
    label_col = spec.get("label_col")
    label_map = spec.get("label_map")
    forced_label = spec.get("forced_label")

    logger.info("Downloading %s ...", name)
    try:
        # Use verification_mode='no_checks' to avoid torch issues
        ds = load_dataset(
            name,
            split=split,
            trust_remote_code=spec.get("trust_remote_code", False),
            verification_mode="no_checks",
            download_mode="reuse_cache_if_exists",
        )
    except Exception as e:
        logger.warning("Failed to load %s: %s — skipping.", name, e)
        return []

    pairs: list[tuple[str, str]] = []
    for row in ds:
        text = str(row.get(text_col, "")).strip()
        if not text or len(text) < 5:
            continue

        if forced_label:
            label = forced_label
        elif label_col and label_map:
            raw = row.get(label_col)
            if raw is None:
                continue
            # Handle both int keys and string keys in label_map
            label = label_map.get(raw) or label_map.get(int(raw) if str(raw).lstrip('-').isdigit() else raw)
            if label is None:
                continue
        else:
            continue

        pairs.append((text, label))

    logger.info("  -> %d samples from %s", len(pairs), name)
    return pairs


def download_all_datasets() -> list[tuple[str, str]]:
    """Download all registered HuggingFace datasets and return combined corpus."""
    if not _HAS_HF_DATASETS:
        logger.error(
            "HuggingFace 'datasets' library not installed. "
            "Install with: pip install datasets"
        )
        return []

    all_pairs: list[tuple[str, str]] = []
    for spec in HF_DATASETS:
        pairs = _download_hf_dataset(spec)
        all_pairs.extend(pairs)

    return all_pairs


def build_combined_corpus(
    hf_pairs: list[tuple[str, str]],
    include_bundled: bool = True,
) -> tuple[list[str], list[int]]:
    """Merge HuggingFace data with the curated bundled dataset.
    
    Returns:
        (texts, binary_labels) where 1=malicious, 0=benign
    """
    from kavach.ml.dataset import TRAINING_DATA, BENIGN

    all_texts: list[str] = []
    all_labels: list[int] = []

    if include_bundled:
        for sample in TRAINING_DATA:
            all_texts.append(sample.text)
            all_labels.append(0 if sample.label == BENIGN else 1)

    for text, label in hf_pairs:
        all_texts.append(text)
        all_labels.append(0 if label == "benign" else 1)

    benign_count = all_labels.count(0)
    attack_count = all_labels.count(1)
    logger.info(
        "Combined corpus: %d total (%d benign, %d attack)",
        len(all_texts), benign_count, attack_count
    )
    return all_texts, all_labels


def train_and_save(
    texts: list[str],
    labels: list[int],
    save_dir: Path,
) -> None:
    """Train the GBM/LR ensemble on the combined corpus and persist to disk."""
    if not _HAS_SKLEARN:
        logger.error("scikit-learn not installed. Cannot train.")
        return

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

    # GBM — primary classifier
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    gbm.fit(X_train_s, y_train)

    # LR — calibrated baseline
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)

    # IsolationForest on benign only
    X_benign = X_train_s[y_train == 0]
    iforest = IsolationForest(contamination=0.05, random_state=42)
    iforest.fit(X_benign if len(X_benign) > 0 else X_train_s)

    # Evaluation report
    y_pred = gbm.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=["benign", "attack"])
    logger.info("GBM Evaluation:\n%s", report)

    # Persist artifacts
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    with open(save_dir / "lr.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(save_dir / "iforest.pkl", "wb") as f:
        pickle.dump(iforest, f)
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "attack_ratio": float(labels.count(1) / len(labels)),  # type: ignore
        "gbm_test_accuracy": float((y_pred == y_test).mean()),
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model artifacts saved to %s", save_dir)
    logger.info("Training summary: %s", meta)


def load_pretrained(save_dir: Path) -> dict[str, Any] | None:
    """Load a previously trained model set from disk.
    
    Returns a dict with 'gbm', 'lr', 'iforest', 'scaler', 'meta' keys,
    or None if artifacts don't exist.
    """
    required = ["gbm.pkl", "lr.pkl", "iforest.pkl", "scaler.pkl"]
    if not all((save_dir / f).exists() for f in required):
        return None

    artifacts: dict[str, Any] = {}
    for name in required:
        with open(save_dir / name, "rb") as f:
            artifacts[name.replace(".pkl", "")] = pickle.load(f)

    meta_path = save_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            artifacts["meta"] = json.load(f)

    return artifacts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download datasets and train the Kavach ML ensemble.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List available datasets without downloading.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/trained_models",
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--no-bundled",
        action="store_true",
        help="Skip the bundled curated dataset and use only downloaded data.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("\nAvailable dataset sources:")
        for i, ds in enumerate(HF_DATASETS, 1):
            print(f"  [{i}] {ds['name']}")
        return

    print("Downloading datasets from HuggingFace...")
    hf_pairs = download_all_datasets()

    if not hf_pairs and not args.no_bundled:
        logger.warning("No HuggingFace data downloaded. Training on bundled dataset only.")

    texts, labels = build_combined_corpus(
        hf_pairs,
        include_bundled=not args.no_bundled,
    )

    if len(texts) < 10:
        logger.error("Corpus too small (%d samples). Aborting.", len(texts))
        return

    save_dir = Path(args.save_path)
    train_and_save(texts, labels, save_dir)
    print(f"\nDone. Model saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
