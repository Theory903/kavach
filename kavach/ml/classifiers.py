"""ML Classifiers — Gradient Boosting and Logistic Regression.

This module provides the tabular ML classifiers that run on top
of the feature extraction pipeline. It uses scikit-learn to train
a fast, accurate ensemble.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    # Need internal scaler to handle bounds
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from kavach.ml.dataset import get_binary_labels, get_training_texts
from kavach.ml.features import extract_features_batch


logger = logging.getLogger(__name__)


class MLEnsembleClassifier:
    """Production classifier ensemble using extracted text features.

    Combines:
    1. GradientBoostingClassifier (strong nonlinear signal, like XGBoost)
    2. LogisticRegression (stable, calibrated baseline)
    3. IsolationForest (unsupervised anomaly detection)
    """

    def __init__(self, pretrained_dir: str | None = None) -> None:
        self.is_trained = False

        if not _HAS_SKLEARN:
            logger.warning("scikit-learn not installed. MLEnsembleClassifier disabled.")
            return

        # Keep estimators lightweight for fast <10ms inference
        self._gbm = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        self._lr = LogisticRegression(
            class_weight="balanced", max_iter=200, random_state=42
        )
        self._iforest = IsolationForest(
            contamination=0.1, random_state=42
        )
        self._scaler = StandardScaler()

        # Try to load persisted artifacts from a previous training run
        if pretrained_dir and self.try_load_pretrained(Path(pretrained_dir)):
            logger.info("Loaded pretrained models from %s", pretrained_dir)
        elif pretrained_dir:
            logger.warning("Pretrained artifacts not found at %s. Will train on bundled dataset.", pretrained_dir)

    def try_load_pretrained(self, save_dir: Path) -> bool:
        """Load persisted model artifacts from a previous `trainer.py` run.
        
        Returns True if artifacts were found and loaded successfully.
        """
        if not _HAS_SKLEARN:
            return False

        required = ["gbm.pkl", "lr.pkl", "iforest.pkl", "scaler.pkl"]
        if not all((save_dir / f).exists() for f in required):
            return False

        try:
            with open(save_dir / "gbm.pkl", "rb") as f:
                self._gbm = pickle.load(f)
            with open(save_dir / "lr.pkl", "rb") as f:
                self._lr = pickle.load(f)
            with open(save_dir / "iforest.pkl", "rb") as f:
                self._iforest = pickle.load(f)
            with open(save_dir / "scaler.pkl", "rb") as f:
                self._scaler = pickle.load(f)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error("Failed to load pretrained artifacts: %s", e)
            return False

    def train_on_bundled_dataset(self) -> None:
        """Train models on the bundled dataset in kavach.ml.dataset."""
        if not _HAS_SKLEARN:
            return

        texts = get_training_texts()
        labels = np.array(get_binary_labels())

        logger.info("Extracting features for %d training samples...", len(texts))
        X = extract_features_batch(texts)

        self.fit(X, labels)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit models on feature matrix X and labels y.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels, 1=malicious, 0=benign
        """
        if not _HAS_SKLEARN:
            return

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train models
        self._gbm.fit(X_scaled, y)
        self._lr.fit(X_scaled, y)
        
        # Only fit IsolationForest on benign data to learn "normal" distribution
        X_benign = X_scaled[y == 0]
        if len(X_benign) > 0:
            self._iforest.fit(X_benign)
        else:
            self._iforest.fit(X_scaled) # fallback

        self.is_trained = True
        logger.info("ML Ensemble training complete.")

    def predict_risk(self, features: np.ndarray) -> dict[str, float]:
        """Predict risk score for a single feature vector.

        Args:
            features: 1D numpy array of shape (30,)

        Returns:
            Dict with component scores and a blended 'ensemble_risk'.
        """
        if not self.is_trained or not _HAS_SKLEARN:
            return {"ensemble_risk": 0.0}

        # Reshape to 2D for scikit-learn
        X_single = features.reshape(1, -1)
        X_scaled = self._scaler.transform(X_single)

        # 1. GBM prediction (probability of malicious class)
        gbm_prob = self._gbm.predict_proba(X_scaled)[0, 1]

        # 2. LR prediction
        lr_prob = self._lr.predict_proba(X_scaled)[0, 1]

        # 3. Anomaly score (Isolation Forest)
        # Returns -1 for outlier, 1 for inlier. Convert to 0-1 risk score.
        iso_pred = self._iforest.decision_function(X_scaled)[0]
        # Invert and scale: more negative = higher risk. Max cap around 1.0.
        anomaly_risk = min(max(0.0, -iso_pred * 2.0), 1.0) 

        # Blend scores — GBM gets highest weight, LR acts as stabilizer,
        # Anomaly detection catches weird lengths/character patterns not in training
        ensemble_risk = (0.5 * gbm_prob) + (0.3 * lr_prob) + (0.2 * anomaly_risk)

        return {
            "gbm_risk": float(gbm_prob),
            "lr_risk": float(lr_prob),
            "anomaly_risk": float(anomaly_risk),
            "ensemble_risk": float(ensemble_risk),
        }
