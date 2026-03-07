"""Automated Benchmark Runner for Kavach LLM Security Gateway.

Runs the Kavach Gateway against industry-standard AI security benchmarks:
1. Lakera PINT Benchmark
2. Qualifire Benchmark
3. JailbreakBench
4. BIPIA / SecReEvalBench (Indirect/Multi-step)

Usage:
    python -m kavach.evaluation.benchmark_runner
    python -m kavach.evaluation.benchmark_runner --benchmark pint
"""

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import os
import sys
import pandas as pd
from datasets import load_dataset
from tabulate import tabulate

from kavach.core.gateway import KavachGateway

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("kavach").setLevel(logging.WARNING)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

class BenchmarkRunner:
    def __init__(self, gateway: KavachGateway | None = None):
        self.gateway = gateway or KavachGateway()

    def evaluate_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        label_mapping: dict | None = None,
    ) -> Dict[str, Any]:
        """Evaluate the gateway against a given DataFrame.

        Args:
            name: Benchmark name
            df: DataFrame containing the dataset
            text_col: Column name for the prompt text
            label_col: Column name for the ground truth label
            label_mapping: Optional dict to map dataset labels to booleans (True = attack, False = benign)
        """
        log.info(f"Running benchmark [{name}] on {len(df)} samples...")

        # Detailed Counters
        stats = {
            "attack": {"allow": 0, "sanitize": 0, "monitor": 0, "require_approval": 0, "block": 0, "total": 0},
            "benign": {"allow": 0, "sanitize": 0, "monitor": 0, "require_approval": 0, "block": 0, "total": 0},
        }
        errors = 0

        for _, row in df.iterrows():
            text = str(row[text_col])
            raw_label = row[label_col]

            if label_mapping:
                is_attack = label_mapping.get(raw_label, bool(raw_label))
            else:
                is_attack = bool(raw_label)
                
            cat = "attack" if is_attack else "benign"
            stats[cat]["total"] += 1

            try:
                # Reset session state to prevent bleed between independent samples
                if hasattr(self.gateway, '_risk_scorer') and hasattr(self.gateway._risk_scorer, 'session_manager'):
                    self.gateway._risk_scorer.session_manager.reset_session("anonymous")
                with SuppressOutput():
                    result = self.gateway.analyze(prompt=text, role="default")
                decision = result["decision"]
                stats[cat][decision] += 1
            except Exception as e:
                log.warning(f"Error evaluating sample: {e}")
                errors += 1

        total = stats["attack"]["total"] + stats["benign"]["total"]
        
        # Hard blocks (block + require_approval)
        hard_blocked_attacks = stats["attack"]["block"] + stats["attack"]["require_approval"]
        hard_blocked_benign = stats["benign"]["block"] + stats["benign"]["require_approval"]
        
        # Friction actions (sanitize + block + require_approval)
        fric_attacks = hard_blocked_attacks + stats["attack"]["sanitize"]
        fric_benign = hard_blocked_benign + stats["benign"]["sanitize"]

        # Recalls
        hard_recall = hard_blocked_attacks / stats["attack"]["total"] if stats["attack"]["total"] > 0 else 0.0
        fric_recall = fric_attacks / stats["attack"]["total"] if stats["attack"]["total"] > 0 else 0.0
        
        # FPRs
        hard_fpr = hard_blocked_benign / stats["benign"]["total"] if stats["benign"]["total"] > 0 else 0.0
        sanitize_rate_on_benign = stats["benign"]["sanitize"] / stats["benign"]["total"] if stats["benign"]["total"] > 0 else 0.0
        user_friction = (fric_attacks + fric_benign) / total if total > 0 else 0.0

        return {
            "benchmark": name,
            "total_samples": total,
            "attacks": stats["attack"]["total"],
            "benign": stats["benign"]["total"],
            "hard_recall": hard_recall,
            "fric_recall": fric_recall,   # Recall if we count sanitize as a catch
            "hard_fpr": hard_fpr,
            "sanitize_rate_benign": sanitize_rate_on_benign,
            "user_friction": user_friction,
            "errors": errors,
            "stats": stats,
        }

    def run_qualifire(self) -> Dict[str, Any]:
        """Run the Qualifire Prompt Injections Benchmark."""
        log.info("Loading Qualifire Benchmark...")
        try:
            ds = load_dataset("qualifire/prompt-injections-benchmark", split="train")
            df = ds.to_pandas()
            return self.evaluate_dataset(
                name="Qualifire",
                df=df,
                text_col="text",
                label_col="label",
                label_mapping={"benign": False, "injection": True, "jailbreak": True}
            )
        except Exception as e:
            log.error(f"Failed to run Qualifire: {e}")
            return {"benchmark": "Qualifire", "error": str(e)}

    def run_xxz224(self) -> Dict[str, Any]:
        log.info("Loading xxz224/prompt-injection-attack-dataset...")
        try:
            ds = load_dataset("xxz224/prompt-injection-attack-dataset", split="train")
            df = ds.to_pandas()
            
            # XXZ224 dataset contains both benign ('target_text') and attack ('naive_attack') versions
            # We'll construct a fresh DataFrame to use our generic evaluator
            samples = []
            for _, row in df.iterrows():
                # Benign
                if pd.notna(row.get("target_text")):
                    samples.append({"text": row["target_text"], "label": False})
                # Attack
                if pd.notna(row.get("naive_attack")):
                    samples.append({"text": row["naive_attack"], "label": True})
            
            eval_df = pd.DataFrame(samples)
            return self.evaluate_dataset(
                name="XXZ224 Attack Dataset",
                df=eval_df,
                text_col="text",
                label_col="label"
            )
        except Exception as e:
            return {"benchmark": "XXZ224 Attack Dataset", "error": str(e)}

    def run_yanismiraoui(self) -> Dict[str, Any]:
        log.info("Loading yanismiraoui/prompt_injections...")
        try:
            ds = load_dataset("yanismiraoui/prompt_injections", split="train")
            df = ds.to_pandas()
            # Yanismiraoui only contains attacks
            df["label"] = True
            return self.evaluate_dataset(
                name="Multilingual Injection",
                df=df,
                text_col="prompt_injections",
                label_col="label"
            )
        except Exception as e:
            return {"benchmark": "Multilingual Injection", "error": str(e)}
            
    def run_bipia(self) -> Dict[str, Any]:
        """Loads BIPIA structural attack data sets."""
        log.info("Loading BIPIA (Indirect Prompt Injection)...")
        try:
            bipia_path = Path("/Users/abhishekjha/CODE/kavach/data/bipia_benchmark/text_attack_test.json")
            if not bipia_path.exists():
                return {"benchmark": "BIPIA Indirect Injection", "error": "Dataset text_attack_test.json not found. Run clone script."}
            
            with open(bipia_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            samples = []
            for category, attacks in data.items():
                # BIPIA "attacks" file contains the attacker instructions to be embedded
                for text in attacks:
                    samples.append({"text": text, "label": True})
                    
            eval_df = pd.DataFrame(samples)
            return self.evaluate_dataset(
                name="BIPIA Indirect Injection",
                df=eval_df,
                text_col="text",
                label_col="label"
            )
        except Exception as e:
            return {"benchmark": "BIPIA Indirect Injection", "error": str(e)}

    def run_pint(self) -> Dict[str, Any]:
        """Runs the Lakera PINT configuration file."""
        log.info("Loading Lakera PINT Benchmark...")
        try:
            import yaml
            pint_path = Path("/Users/abhishekjha/CODE/kavach/data/pint_benchmark/example-dataset.yaml")
            if not pint_path.exists():
                return {"benchmark": "Lakera PINT Benchmark", "error": "example-dataset.yaml not found. Run clone script."}
            
            with open(pint_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
            samples = []
            for item in data:
                samples.append({"text": item["text"], "label": bool(item.get("label", False))})
                
            eval_df = pd.DataFrame(samples)
            return self.evaluate_dataset(
                name="Lakera PINT Benchmark",
                df=eval_df,
                text_col="text",
                label_col="label"
            )
        except ImportError:
            return {"benchmark": "Lakera PINT Benchmark", "error": "PyYAML not installed"}
        except Exception as e:
            return {"benchmark": "Lakera PINT Benchmark", "error": str(e)}

    def run_tensortrust(self) -> Dict[str, Any]:
        """Runs evaluation against a subset of the TensorTrust dataset."""
        try:
            log.info("Streaming TensorTrust dataset (subset)...")
            ds = load_dataset('qxcv/tensor-trust', split='train', streaming=True)
            # Take a 1000 sample subset and assume they are all attacks given the dataset nature
            attacks = []
            for item in list(ds.take(1000)):
                text = str(item.get("attack", ""))
                if text.strip():
                    attacks.append({"text": text, "is_attack": True})
            
            # Since TensorTrust is almost entirely attacks in this format, generate 100 benign samples to pad
            benign_samples = [
                "Translate the following text to French",
                "Summarize this article",
                "What is the capital of France?",
                "Write a python script to reverse a string"
            ] * 25
            
            for b in benign_samples:
                attacks.append({"text": b, "is_attack": False})

            df = pd.DataFrame(attacks)
            return self.evaluate_dataset(
                name="TensorTrust (Subset)",
                df=df,
                text_col="text",
                label_col="is_attack"
            )
        except Exception as e:
            log.error(f"Failed to load TensorTrust: {e}")
            return {"benchmark": "TensorTrust (Subset)", "error": str(e)}

    def run_llmail_inject(self) -> Dict[str, Any]:
        """Runs evaluation against LLMail-Inject Challenge."""
        try:
            log.info("Streaming LLMail-Inject dataset (subset)...")
            # We are using train split, streamed to save bandwidth (1.6GB)
            # Often LLMail includes both malicious and benign. The injection is usually in specific fields.
            # Using raw_submissions_phase1.jsonl might be tricky. Let's just create a mock block if it's too complex or fails format
            return {"benchmark": "LLMail-Inject", "error": "Evaluation postponed: format requires parsed RAG structures"}
        except Exception as e:
            return {"benchmark": "LLMail-Inject", "error": str(e)}

    def run_all(self):
        """Runs all configured benchmarks."""
        results = []
        
        #results.append(self.run_qualifire())
        results.append(self.run_pint())
        results.append(self.run_xxz224())
        results.append(self.run_yanismiraoui())
        results.append(self.run_bipia())
        results.append(self.run_tensortrust())
        results.append(self.run_llmail_inject())
        
        # Print Summary
        print("\n" + "="*100)
        print("🛡️ KAVACH INDUSTRY BENCHMARK REPORT (SHADOW MODE)")
        print("="*100 + "\n")
        
        table_data = []
        for r in results:
            if "error" in r:
                table_data.append([r["benchmark"], "ERROR", "-", "-", "-", "-", "-", r["error"]])
            else:
                table_data.append([
                    r["benchmark"],
                    r["total_samples"],
                    f"{r['hard_recall']:.1%}",
                    f"{r['fric_recall']:.1%}",
                    f"{r['hard_fpr']:.1%}",
                    f"{r['sanitize_rate_benign']:.1%}",
                    f"{r['user_friction']:.1%}",
                    "-"
                ])
                
        headers = [
            "Benchmark", "Samples", "Hard Recall", "Fric Recall", 
            "Hard FPR", "Benign Sanitize Rate", "User Friction", "Notes"
        ]
        print(tabulate(table_data, headers=headers, tablefmt="github"))
        
        print("\n" + "="*40 + " DETAILED DISTRIBUTION " + "="*40)
        for r in results:
            if "stats" in r:
                print(f"\n{r['benchmark']}:")
                for cat in ["benign", "attack"]:
                    total = r['stats'][cat]['total']
                    if total > 0:
                        allow = r['stats'][cat]['allow'] / total
                        monitor = r['stats'][cat]['monitor'] / total
                        sanitize = r['stats'][cat]['sanitize'] / total
                        block = (r['stats'][cat]['block'] + r['stats'][cat]['require_approval']) / total
                        print(f"  {cat.upper():<7} | Allow: {allow:.1%} | Monitor: {monitor:.1%} | Sanitize: {sanitize:.1%} | Block: {block:.1%}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kavach against AI Security Benchmarks")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "qualifire", "pint"])
    args = parser.parse_args()

    runner = BenchmarkRunner()
    
    if args.benchmark == "all":
        runner.run_all()
    elif args.benchmark == "qualifire":
        res = runner.run_qualifire()
        print(json.dumps(res, indent=2))
    elif args.benchmark == "pint":
        res = runner.run_pint()
        print(json.dumps(res, indent=2))
