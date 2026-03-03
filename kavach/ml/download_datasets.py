#!/usr/bin/env python3
"""Download ALL security datasets listed in the Kavach V1 master prompt.

Sources:
  1. TrustAIRLab/in-the-wild-jailbreak-prompts (1,405 jailbreaks + 13,735 benign)
  2. deepset/prompt-injections (546 injection/benign pairs)
  3. Mindgard/evaded-prompt-injection-and-jailbreak-samples (evasion attacks)
  4. xTRam1/safe-guard-prompt-injection (8,240 train + 2,056 test)
  5. S-Labs/prompt-injection-dataset (train/val/test splits)
  6. J1N2/mix-prompt-injection-dataset (1.15M mixed — capped)
  7. Anthropic/hh-rlhf (42K harmless-base JSONL)
  8. rubend18/ChatGPT-Jailbreak-Prompts (79 curated jailbreaks)
  9. ai-safety-institute/AgentHarm (agent-level harmful prompts)
  10. qualifire/prompt-injections-benchmark (test benchmark)
  11. HumanCompatibleAI/tensor-trust-data (GitHub clone)

Usage:
    python -m kavach.ml.download_datasets
    python -m kavach.ml.download_datasets --output data/datasets
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace Parquet download helper
# ---------------------------------------------------------------------------

HF_PARQUET_BASE = "https://huggingface.co/datasets"


def _hf_parquet_url(repo: str, config: str = "default", split: str = "train", shard: int = 0) -> str:
    return f"{HF_PARQUET_BASE}/{repo}/resolve/refs%2Fconvert%2Fparquet/{config}/{split}/{shard:04d}.parquet"


def _download_parquet(url: str, name: str) -> pd.DataFrame:
    log.info("  Downloading %s ...", name)
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    except Exception as e:
        log.warning("  FAILED %s: %s", name, e)
        return pd.DataFrame()


def _download_csv(url: str, name: str) -> pd.DataFrame:
    log.info("  Downloading %s ...", name)
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        log.warning("  FAILED %s: %s", name, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def download_trustairlab_jailbreaks() -> list[tuple[str, str, str]]:
    """TrustAIRLab/in-the-wild-jailbreak-prompts — 1,405 jailbreaks + regular prompts."""
    pairs: list[tuple[str, str, str]] = []

    for config in ["jailbreak_2023_12_25", "regular_2023_12_25"]:
        is_jailbreak = "jailbreak" in config
        label = "jailbreak" if is_jailbreak else "benign"
        url = _hf_parquet_url("TrustAIRLab/in-the-wild-jailbreak-prompts", config)
        df = _download_parquet(url, f"TrustAIRLab/{config}")
        if df.empty:
            continue

        col = "prompt" if "prompt" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            text = str(row.get(col, "")).strip()
            if 10 < len(text) < 5000:
                pairs.append((text[:2000], label, "trustairlab"))
                if len(pairs) >= 3000:
                    break

    log.info("  TrustAIRLab: %d samples", len(pairs))
    return pairs


def download_deepset_injections() -> list[tuple[str, str, str]]:
    """deepset/prompt-injections — labeled injection/benign."""
    pairs: list[tuple[str, str, str]] = []
    url = _hf_parquet_url("deepset/prompt-injections")
    df = _download_parquet(url, "deepset/prompt-injections")
    if df.empty:
        return pairs

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        label = row.get("label", 0)
        if len(text) > 5:
            pairs.append((text, "injection" if label == 1 else "benign", "deepset"))

    log.info("  deepset: %d samples", len(pairs))
    return pairs


def download_mindgard_evasions() -> list[tuple[str, str, str]]:
    """Mindgard/evaded-prompt-injection-and-jailbreak-samples."""
    pairs: list[tuple[str, str, str]] = []
    url = _hf_parquet_url("Mindgard/evaded-prompt-injection-and-jailbreak-samples")
    df = _download_parquet(url, "Mindgard evasion samples")
    if df.empty:
        return pairs

    for col in df.columns:
        if "prompt" in col.lower() or "text" in col.lower():
            for _, row in df.iterrows():
                text = str(row.get(col, "")).strip()
                if len(text) > 10:
                    pairs.append((text[:2000], "injection", "mindgard"))
            break

    log.info("  Mindgard: %d samples", len(pairs))
    return pairs


def download_safeguard() -> list[tuple[str, str, str]]:
    """xTRam1/safe-guard-prompt-injection — 8.24K train."""
    pairs: list[tuple[str, str, str]] = []
    url = _hf_parquet_url("xTRam1/safe-guard-prompt-injection")
    df = _download_parquet(url, "xTRam1/safe-guard-prompt-injection")
    if df.empty:
        return pairs

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        label = row.get("label", 0)
        if len(text) > 5:
            pairs.append((text[:2000], "injection" if label == 1 else "benign", "safeguard"))

    log.info("  SafeGuard: %d samples", len(pairs))
    return pairs


def download_slabs() -> list[tuple[str, str, str]]:
    """S-Labs/prompt-injection-dataset — train CSV."""
    pairs: list[tuple[str, str, str]] = []
    url = "https://huggingface.co/datasets/S-Labs/prompt-injection-dataset/resolve/main/data/train.csv"
    df = _download_csv(url, "S-Labs/prompt-injection-dataset")
    if df.empty:
        return pairs

    text_col = None
    label_col = None
    for c in df.columns:
        if "text" in c.lower() or "prompt" in c.lower():
            text_col = c
        if "label" in c.lower():
            label_col = c

    if text_col and label_col:
        for _, row in df.iterrows():
            text = str(row.get(text_col, "")).strip()
            label = row.get(label_col, 0)
            if len(text) > 5:
                is_attack = label == 1 if isinstance(label, (int, float)) else str(label).lower() in ("1", "true", "injection")
                pairs.append((text[:2000], "injection" if is_attack else "benign", "slabs"))

    log.info("  S-Labs: %d samples", len(pairs))
    return pairs


def download_j1n2_mix(max_samples: int = 20000) -> list[tuple[str, str, str]]:
    """J1N2/mix-prompt-injection-dataset — 1.15M (capped for sanity)."""
    pairs: list[tuple[str, str, str]] = []

    # This is a huge dataset, try first parquet shard only
    url = _hf_parquet_url("J1N2/mix-prompt-injection-dataset")
    df = _download_parquet(url, "J1N2/mix (shard 0)")
    if df.empty:
        return pairs

    count = 0
    for _, row in df.iterrows():
        text = str(row.get("prompt", "")).strip()
        label = row.get("label", False)
        source = str(row.get("source", "j1n2"))
        if len(text) > 5:
            is_attack = label is True or label == 1 or str(label).lower() == "true"
            pairs.append((text[:2000], "injection" if is_attack else "benign", f"j1n2/{source}"))
            count += 1
            if count >= max_samples:
                break

    log.info("  J1N2: %d samples (capped at %d)", len(pairs), max_samples)
    return pairs


def download_anthropic_rlhf() -> list[tuple[str, str, str]]:
    """Anthropic/hh-rlhf — harmless-base (42K JSONL.GZ)."""
    ATTACK_KW = [
        "ignore previous", "ignore all", "system prompt", "reveal your",
        "kill", "steal", "poison", "weapon", "bomb", "hack", "exploit",
        "jailbreak", "DAN", "developer mode", "pretend you",
    ]

    def classify(t: str) -> str:
        lo = t.lower()
        for k in ["ignore previous", "system prompt", "injection"]:
            if k in lo:
                return "injection"
        for k in ["jailbreak", "DAN", "developer mode", "pretend you"]:
            if k in lo:
                return "jailbreak"
        hits = sum(1 for kw in ATTACK_KW if kw in lo)
        if hits >= 2:
            return "social_engineering"
        if hits == 1:
            return "obfuscation"
        return "benign"

    pairs: list[tuple[str, str, str]] = []
    url = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl.gz"
    log.info("  Downloading Anthropic/hh-rlhf harmless-base...")
    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        data = gzip.decompress(r.content)
        lines = data.decode("utf-8", errors="ignore").strip().split("\n")
        log.info("  -> %d lines", len(lines))

        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            for key in ["rejected", "chosen"]:
                conv = str(row.get(key, ""))
                human_turns = re.findall(r"Human:\s*(.+?)(?=\s*(?:Assistant:|$))", conv, re.DOTALL)
                for turn in human_turns:
                    turn = turn.strip()
                    if 15 < len(turn) < 500:
                        cat = classify(turn)
                        pairs.append((turn, cat, "anthropic_rlhf"))

            if len(pairs) >= 10000:
                break

    except Exception as e:
        log.warning("  FAILED Anthropic RLHF: %s", e)

    log.info("  Anthropic RLHF: %d samples", len(pairs))
    return pairs


def download_chatgpt_jailbreaks() -> list[tuple[str, str, str]]:
    """rubend18/ChatGPT-Jailbreak-Prompts — 79 curated."""
    pairs: list[tuple[str, str, str]] = []
    url = _hf_parquet_url("rubend18/ChatGPT-Jailbreak-Prompts")
    df = _download_parquet(url, "rubend18/ChatGPT-Jailbreak-Prompts")
    if df.empty:
        return pairs

    col = "Prompt" if "Prompt" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        text = str(row.get(col, "")).strip()
        if len(text) > 20:
            pairs.append((text[:3000], "jailbreak", "chatgpt_jailbreaks"))

    log.info("  ChatGPT Jailbreaks: %d samples", len(pairs))
    return pairs


def download_agentharm() -> list[tuple[str, str, str]]:
    """ai-safety-institute/AgentHarm — agent-level harmful behaviors."""
    pairs: list[tuple[str, str, str]] = []
    for subset in ["chat"]:
        url = _hf_parquet_url("ai-safety-institute/AgentHarm", subset, "test_public")
        df = _download_parquet(url, f"AgentHarm/{subset}")
        if df.empty:
            continue
        for _, row in df.iterrows():
            text = str(row.get("prompt", "")).strip()
            cat = str(row.get("category", "")).lower()
            if len(text) > 10:
                category = "social_engineering" if "harm" in cat or "fraud" in cat else "injection"
                pairs.append((text[:2000], category, "agentharm"))

    log.info("  AgentHarm: %d samples", len(pairs))
    return pairs


def download_qualifire_benchmark() -> list[tuple[str, str, str]]:
    """qualifire/prompt-injections-benchmark."""
    pairs: list[tuple[str, str, str]] = []
    url = "https://huggingface.co/datasets/qualifire/prompt-injections-benchmark/resolve/main/test.csv"
    df = _download_csv(url, "qualifire/prompt-injections-benchmark")
    if df.empty:
        return pairs

    for _, row in df.iterrows():
        text = str(row.get("text", row.get(df.columns[0], ""))).strip()
        label = row.get("label", row.get(df.columns[-1], 0))
        if len(text) > 5:
            is_attack = str(label) in ("1", "True", "true", "injection")
            pairs.append((text[:2000], "injection" if is_attack else "benign", "qualifire"))

    log.info("  Qualifire: %d samples", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Master download function
# ---------------------------------------------------------------------------

ALL_DOWNLOADERS = [
    ("TrustAIRLab Jailbreaks", download_trustairlab_jailbreaks),
    ("Deepset Injections", download_deepset_injections),
    ("Mindgard Evasions", download_mindgard_evasions),
    ("SafeGuard (xTRam1)", download_safeguard),
    ("S-Labs Injections", download_slabs),
    ("J1N2 Mix (capped)", download_j1n2_mix),
    ("Anthropic HH-RLHF", download_anthropic_rlhf),
    ("ChatGPT Jailbreaks", download_chatgpt_jailbreaks),
    ("AgentHarm (AISI)", download_agentharm),
    ("Qualifire Benchmark", download_qualifire_benchmark),
]


def download_all(output_dir: Path) -> Path:
    """Download all datasets, deduplicate, save to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs: list[tuple[str, str, str]] = []

    print("=" * 60)
    print("KAVACH V1 — DOWNLOADING ALL SECURITY DATASETS")
    print("=" * 60)

    for name, downloader in ALL_DOWNLOADERS:
        print(f"\n🔽 {name}")
        try:
            pairs = downloader()
            all_pairs.extend(pairs)
            print(f"   ✅ {len(pairs)} samples")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Deduplicate by first 100 chars of text
    seen: set[str] = set()
    deduped: list[dict] = []
    for text, label, source in all_pairs:
        key = text[:100].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append({"text": text, "label": label, "source": source})

    # Save
    out_path = output_dir / "all_datasets.json"
    with open(out_path, "w") as f:
        json.dump(deduped, f)

    # Stats
    cats = Counter(d["label"] for d in deduped)
    sources = Counter(d["source"] for d in deduped)

    print("\n" + "=" * 60)
    print(f"📊 FINAL CORPUS: {len(deduped):,} unique samples")
    print("=" * 60)
    print("\nBy category:")
    for cat, count in cats.most_common():
        pct = count / len(deduped) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:25s} {count:>6,} ({pct:5.1f}%) {bar}")
    print(f"\nBy source ({len(sources)} sources):")
    for src, count in sources.most_common(15):
        print(f"  {src:30s} {count:>6,}")
    print(f"\nSaved to: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download all Kavach V1 security datasets.")
    parser.add_argument("--output", type=str, default="data/datasets", help="Output directory")
    args = parser.parse_args()
    download_all(Path(args.output))


if __name__ == "__main__":
    main()
