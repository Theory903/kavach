"""Meta-model aggregation layer.

Coordinates the rule engine, ML classifiers, embedding scorer,
and behavioral tracker to produce one unified risk score and decision.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from kavach.core.identity import Identity
from kavach.ml.behavioral import BehavioralTracker
from kavach.ml.classifiers import MLEnsembleClassifier, _HAS_SKLEARN
from kavach.ml.embeddings import EmbeddingRiskScorer, _HAS_ONNX
from kavach.ml.features import extract_features
from kavach.ml.intent import IntentClassifier
from kavach.ml.redis_behavioral import RedisBehavioralTracker, _HAS_REDIS
import kavach.observability.prometheus as prom

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

        if use_redis and _HAS_REDIS:
            self._behavioral = RedisBehavioralTracker(redis_url=redis_url)
        else:
            if use_redis:
                logger.warning("Redis requested but redis-py not installed. Falling back to in-memory tracker.")
            self._behavioral = BehavioralTracker()

        self.ml_classifier: MLEnsembleClassifier | None = None
        self.embedding_scorer: EmbeddingRiskScorer | None = None
        self.intent_classifier = IntentClassifier()

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

    def _blend_scores(
        self,
        rule_score: float,
        ml_score: float,
        emb_score: float,
        intent_score: float,
        multiplier: float
    ) -> float:
        """Weights and aggregates all signals into a final risk score.
        
        Rule score is deterministic and takes precedence when high-confidence.
        """
        if rule_score > 0.8:
            base_score = max(rule_score, ml_score, intent_score)
        else:
            total_weight = 0.0
            sum_weighted = 0.0

            # Rule weight: 0.35
            sum_weighted += rule_score * 0.35
            total_weight += 0.35

            # ML classifier weight: 0.30
            if self.ml_classifier is not None and self.ml_classifier.is_trained:
                sum_weighted += ml_score * 0.30
                total_weight += 0.30

            # Embedding similarity weight: 0.20
            if self.embedding_scorer is not None and self.embedding_scorer.is_loaded:
                sum_weighted += emb_score * 0.20
                total_weight += 0.20

            # SLM intent weight: 0.15
            if self.intent_classifier.is_loaded:
                sum_weighted += intent_score * 0.15
                total_weight += 0.15

            base_score = sum_weighted / total_weight if total_weight > 0 else rule_score

        return min(max(base_score * multiplier, 0.0), 1.0)

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

        with prom.latency_timer(prom.KAVACH_ML_INFERENCE_TIME):
            def _ml_task() -> tuple[float, float, dict[str, Any], dict[str, Any]]:
                m_score = 0.0
                e_score = 0.0
                m_break: dict[str, Any] = {}
                i_result: dict[str, Any] = {}

                if self.ml_classifier is not None and self.ml_classifier.is_trained:
                    features = extract_features(prompt)
                    m_break = self.ml_classifier.predict_risk(features)
                    m_score = m_break.get("ensemble_risk", 0.0)

                if self.embedding_scorer is not None and self.embedding_scorer.is_loaded:
                    e_score = self.embedding_scorer.predict_risk(prompt)

                # SLM intent classification runs concurrently with embeddings
                if self.intent_classifier.is_loaded:
                    i_result = self.intent_classifier.classify(prompt)

                return m_score, e_score, m_break, i_result

            try:
                future = _ML_EXECUTOR.submit(_ml_task)
                ml_score, emb_score, ml_breakdown, intent_result = future.result(
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
        final_score = self._blend_scores(rule_score, ml_score, emb_score, intent_score, multiplier)

        breakdown = {
            "final_score": float(final_score),
            "components": {
                "rule_score": float(rule_score),
                "ml_classifier_score": float(ml_score),
                "embedding_sim_score": float(emb_score),
                "intent_score": float(intent_score),
                "behavioral_multiplier": float(multiplier),
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

        return breakdown

    def update_behavior(self, user_id: str, risk_score: float, action: str) -> None:
        """Update the behavioral tracker after an action is taken."""
        self._behavioral.record_interaction(user_id, risk_score, action)
