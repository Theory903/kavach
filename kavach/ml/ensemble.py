"""Meta-model aggregation layer.

Coordinates the rule engine, ML classifiers, embedding scorer,
and behavioral tracker to produce one unified risk score and decision.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any
import numpy as np

from kavach.core.identity import Identity
from kavach.ml.behavioral import BehavioralTracker
from kavach.ml.classifiers import MLEnsembleClassifier, _HAS_SKLEARN
from kavach.ml.embeddings import EmbeddingRiskScorer, _HAS_ONNX
from kavach.ml.features import extract_features
from kavach.ml.intent import IntentClassifier
from kavach.classifier.attack_classifier import AttackClassifier
from kavach.ml.ood_detector import OODDetector
from kavach.ml.redis_behavioral import RedisBehavioralTracker, _HAS_REDIS
from kavach.memory.session_manager import SessionManager
import kavach.observability.prometheus as prom
from kavach.ml.sfm.model import SecurityFoundationModel
import torch

logger = logging.getLogger(__name__)

# Global thread pool for ML timeout isolation (prevents thread init overhead)
_ML_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)

class EnsembleRiskScorer:
    """The master risk engine.
    
    Combines:
    1. Rule-based signals (from standard detectors)
    2. ML Classifier ensemble risk
    3. Embedding similarity risk
    4. Behavioral historical multiplier
    """
    def __init__(
        self, 
        enable_ml: bool = True, 
        enable_embeddings: bool = True,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379/0",
        ml_timeout_seconds: float = 0.05
    ) -> None:
        """Initialize the ensemble."""
        self.ml_timeout_seconds = ml_timeout_seconds
        
        # Determine if we should instantiate the SFM (Model 1)
        # SFM is the preferred intelligence engine and replaces the old ensemble
        self.sfm: SecurityFoundationModel | None = None
        self.sfm_tokenizer = None
        
        # Load SFM if enabled and PyTorch is available
        try:
            self.sfm = SecurityFoundationModel()
            self.sfm.eval() # ensure inference mode
            # Standard fast tokenizer matching the default base model
            from transformers import AutoTokenizer
            self.sfm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Security Foundation Model (SFM) initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize SFM (falling back to legacy ML if any): {e}")

        if use_redis and _HAS_REDIS:
            self._behavioral = RedisBehavioralTracker(redis_url=redis_url)
        else:
            if use_redis:
                logger.warning("Redis requested but redis-py not installed. Falling back to in-memory tracker.")
            self._behavioral = BehavioralTracker()

        self.ml_classifier: MLEnsembleClassifier | None = None
        self.embedding_scorer: EmbeddingRiskScorer | None = None
        self.intent_classifier = IntentClassifier()
        self.attack_classifier = AttackClassifier()

        self.is_ml_active = False
        
        if enable_ml and _HAS_SKLEARN:
            self.ml_classifier = MLEnsembleClassifier()
            # Train sync on init (fast since dataset is small)
            self.ml_classifier.train_on_bundled_dataset()
            self.is_ml_active = True
            
        if enable_embeddings and _HAS_ONNX:
            self.embedding_scorer = EmbeddingRiskScorer()
            self.embedding_scorer.load_and_encode_corpus()
            self.is_ml_active = True

        # Phase 17: OOD Detector + Session Memory
        self.ood_detector = OODDetector()
        self.ood_detector.load()
        self.session_manager = SessionManager(risk_decay_factor=0.8)

    def _blend_scores(
        self,
        rule_score: float,
        ml_score: float,
        emb_score: float,
        intent_category: str | None,
        intent_score: float,
        multiplier: float
    ) -> float:
        """Calibrated score blending requiring multi-signal corroboration.
        
        Key design principles:
        1. No single signal can force a high risk score alone
        2. At least 2 signals must agree above threshold for escalation
        3. Benign text with low rule scores stays in the ALLOW zone (< 0.2)
        """
        # Count how many signals indicate high risk (above 0.5)
        HIGH_THRESHOLD = 0.5
        high_signals = sum([
            1 if rule_score > HIGH_THRESHOLD else 0,
            1 if ml_score > HIGH_THRESHOLD else 0,
            1 if emb_score > HIGH_THRESHOLD else 0,
            1 if intent_score > HIGH_THRESHOLD else 0,
        ])

        # Hard override ONLY when rules AND at least one ML signal both agree
        if rule_score > 0.8 and high_signals >= 2:
            return min(max(rule_score, ml_score, intent_score) * multiplier, 1.0)

        # Weighted blend with calibrated weights
        total_weight = 0.0
        sum_weighted = 0.0

        # Rule weight: 0.40 (rules are deterministic, trust them most)
        sum_weighted += rule_score * 0.40
        total_weight += 0.40

        # ML classifier weight: 0.20
        if self.ml_classifier is not None and self.ml_classifier.is_trained:
            sum_weighted += ml_score * 0.20
            total_weight += 0.20

        # Embedding similarity weight: 0.15
        if self.embedding_scorer is not None and self.embedding_scorer.is_loaded:
            sum_weighted += emb_score * 0.15
            total_weight += 0.15

        # SLM intent weight: 0.25
        if self.intent_classifier.is_loaded:
            sum_weighted += intent_score * 0.25
            total_weight += 0.25

        base_score = sum_weighted / total_weight if total_weight > 0 else rule_score

        # If rule and intention strongly agree, boost slightly
        if rule_score > 0.7 and intent_score > 0.7:
            base_score = min(base_score * 1.2, 1.0)

        # Apply multiplier but cap appropriately
        final_score = base_score * multiplier
        
        # Soft cap to allow continuous ML signal distribution
        # Keep it continuous, avoiding hard jumps to 1.0 unless really malicious
        return min(max(final_score, 0.0), 1.0)
        
    def _run_sfm_inference(self, prompt: str) -> dict[str, Any]:
        """Run the prompt through the Security Foundation Model (SFM)."""
        if not self.sfm or not self.sfm_tokenizer:
            return {}
            
        try:
            with torch.no_grad():
                encoded = self.sfm_tokenizer(
                    prompt, 
                    padding="max_length", 
                    truncation=True, 
                    max_length=256, 
                    return_tensors="pt"
                )
                
                outputs = self.sfm(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                
                # Attack Logits to Probability (Softmax)
                attack_probs = torch.softmax(outputs["attack_logits"], dim=-1)[0]
                attack_prob = attack_probs[1].item() # probability of class 1 (attack)
                
                # Intent Logits to Probabilities
                intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)[0]
                intent_idx = torch.argmax(intent_probs).item()
                intent_score = intent_probs[intent_idx].item()
                
                # Map indices to taxonomy
                taxonomy = ["benign", "prompt_injection", "jailbreak", "data_exfiltration", "tool_abuse"]
                intent_category = taxonomy[intent_idx] if intent_idx < len(taxonomy) else "unknown"
                
                return {
                    "attack_probability": attack_prob,
                    "intent_category": intent_category,
                    "intent_score": intent_score,
                    "risk_score": outputs["risk_score"].item(),
                    "ood_embeddings": outputs["ood_embeddings"].numpy()
                }
        except Exception as e:
            logger.error(f"SFM Inference failed: {e}")
            return {}

    def analyze(
        self,
        prompt: str,
        rule_signals: dict[str, float],
        identity: Identity
    ) -> dict[str, Any]:
        """Run the full ensemble scoring pipeline.
        
        Args:
            prompt: The raw user prompt.
            rule_signals: Dictionary of rule-based scores (e.g. {"injection": 0.9})
            identity: The user identity to apply behavioral history.
            
        Returns:
            Dict containing final_score and a full breakdown of all components.
        """
        rule_score = max(rule_signals.values()) if rule_signals else 0.0

        ml_score = 0.0
        emb_score = 0.0
        intent_score = 0.0
        ml_breakdown: dict[str, Any] = {}
        intent_result: dict[str, Any] = {}
        prompt_embedding: np.ndarray | None = None
        
        # Fast semantic attack classification
        attack_label = self.attack_classifier.classify(prompt)

        with prom.latency_timer(prom.KAVACH_ML_INFERENCE_TIME):
            def _ml_task() -> tuple[float, float, dict[str, Any], dict[str, Any], np.ndarray | None]:
                m_score = 0.0
                e_score = 0.0
                m_break: dict[str, Any] = {}
                i_result: dict[str, Any] = {}
                p_emb: np.ndarray | None = None

                if self.ml_classifier is not None and self.ml_classifier.is_trained:
                    features = extract_features(prompt)
                    m_break = self.ml_classifier.predict_risk(features)
                    m_score = m_break.get("ensemble_risk", 0.0)

                if self.embedding_scorer is not None and self.embedding_scorer.is_loaded:
                    # Get raw embedding for OOD + Session reuse
                    p_emb = self.embedding_scorer.encode(prompt)
                    e_score = self.embedding_scorer.predict_risk(prompt)

                if self.intent_classifier.is_loaded:
                    i_result = self.intent_classifier.classify(prompt)
                    
                # Run the unified SFM to harvest all 4 major signals in one pass
                sfm_signals = self._run_sfm_inference(prompt)
                if sfm_signals:
                    # If SFM is active, it heavily influences the fallback scores
                    m_score = max(m_score, sfm_signals.get("risk_score", 0.0))
                    i_result["predicted_category"] = sfm_signals.get("intent_category")
                    i_result["risk_score"] = sfm_signals.get("intent_score")
                    i_result["slm_active"] = True
                    # Set the prompt embedding to the SFM OOD representation 
                    # so the session manager/drift calculation uses the highly distilled SFM space
                    p_emb = sfm_signals.get("ood_embeddings")

                return m_score, e_score, m_break, i_result, p_emb

            try:
                future = _ML_EXECUTOR.submit(_ml_task)
                ml_score, emb_score, ml_breakdown, intent_result, prompt_embedding = future.result(
                    timeout=self.ml_timeout_seconds
                )
                intent_score = intent_result.get("risk_score", 0.0)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "ML Inference exceeded %.3fs timeout. Using rule-based scoring only.",
                    self.ml_timeout_seconds
                )
            except Exception as e:
                logger.error("ML Inference pipeline error: %s. Falling back to rules.", e)

        multiplier = self._behavioral.get_behavioral_multiplier(identity.user_id)
        
        intent_cat = intent_result.get("predicted_category")
        if attack_label.category != "benign":
            intent_cat = attack_label.category

        final_score = self._blend_scores(
            rule_score=rule_score,
            ml_score=ml_score,
            emb_score=emb_score,
            intent_category=intent_cat,
            intent_score=intent_score,
            multiplier=multiplier
        )

        # ── Phase 17: OOD + Session Memory ──
        ood_score = 0.0
        session_state = {"session_risk": 0.0, "semantic_drift": 0.0, "turn_count": 0}

        if prompt_embedding is not None:
            # OOD: how far is this embedding from the benign manifold?
            ood_score = self.ood_detector.evaluate(prompt_embedding)

            # Session: accumulate risk and track drift
            session_state = self.session_manager.record_prompt(
                session_id=identity.user_id,
                current_risk=final_score,
                current_embedding=prompt_embedding
            )

        # Boost final score if OOD or session signals are alarming
        if ood_score > self.ood_detector.ood_threshold:
            # Unknown territory — bump score into at least soft-sanitize
            ood_boost = ood_score * 0.3
            final_score = min(final_score + ood_boost, 1.0)

        if session_state["session_risk"] > 0.7:
            # Cumulative session risk breached — escalate
            session_boost = min(session_state["session_risk"] * 0.15, 0.3)
            final_score = min(final_score + session_boost, 1.0)

        if session_state["semantic_drift"] > 0.4:
            # Big semantic jump detected inside the session
            drift_boost = session_state["semantic_drift"] * 0.2
            final_score = min(final_score + drift_boost, 1.0)

        breakdown = {
            "final_score": float(final_score),
            "components": {
                "rule_score": float(rule_score),
                "ml_classifier_score": float(ml_score),
                "embedding_sim_score": float(emb_score),
                "intent_score": float(intent_score),
                "behavioral_multiplier": float(multiplier),
                "ood_score": float(ood_score),
                "session_risk": float(session_state["session_risk"]),
                "semantic_drift": float(session_state["semantic_drift"]),
            },
            "rule_signals": rule_signals,
        }

        if ml_breakdown:
            breakdown["components"]["ml_details"] = ml_breakdown

        if intent_result.get("slm_active"):
            breakdown["components"]["intent_analysis"] = {
                "predicted_category": intent_result.get("predicted_category"),
                "confidence": intent_result.get("confidence"),
                "all_scores": intent_result.get("all_scores", {}),
            }

        breakdown["components"]["attack_classification"] = attack_label.to_dict()

        return breakdown

    def update_behavior(self, user_id: str, risk_score: float, action: str) -> None:
        """Update the behavioral tracker after an action is taken."""
        self._behavioral.record_interaction(user_id, risk_score, action)
