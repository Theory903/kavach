"""Out-Of-Distribution (OOD) semantic detector.

Uses z-score normalized centroid distances and IsolationForest
to identify prompts outside the benign training manifold.

Key insight: in high-dimensional embedding space (384d), cosine distances
cluster tightly (0.6-0.7 for all inputs). Raw distances don't discriminate.
We must normalize against the TRAINING distribution's statistics.
"""
import os
import logging
import pickle
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class OODDetector:
    """Detects out-of-distribution inputs using relative embedding geometry."""

    def __init__(self, model_dir: str = "data/trained_models"):
        self.model_dir = model_dir
        self.is_loaded = False
        self._iforest = None
        self._centroid = None
        # Distribution statistics from training (key calibration data)
        self._dist_mean = 0.0
        self._dist_std = 1.0
        self._iso_mean = 0.0
        self._iso_std = 1.0
        self.ood_threshold = 0.5  # z-score based: 0.5 = moderately unusual

    def fit(self, benign_embeddings: np.ndarray) -> None:
        """Fit centroid, IsolationForest, AND distance statistics on benign data."""
        if not _HAS_SKLEARN:
            logger.warning("Scikit-learn not installed. Cannot fit OOD detector.")
            return
        if len(benign_embeddings) == 0:
            logger.warning("No benign embeddings provided for OOD fitting.")
            return

        # 1. Centroid
        self._centroid = np.mean(benign_embeddings, axis=0)

        # 2. IsolationForest trained on benign embeddings
        self._iforest = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=42
        )
        self._iforest.fit(benign_embeddings)

        # 3. Compute distance distribution statistics for z-score normalization
        # Calculate cosine distance from centroid for every training sample
        centroid_norm = np.linalg.norm(self._centroid)
        distances = []
        iso_scores = []
        for emb in benign_embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0 and centroid_norm > 0:
                cos_sim = np.dot(emb, self._centroid) / (emb_norm * centroid_norm)
                distances.append(1.0 - cos_sim)

            iso_dec = self._iforest.decision_function(emb.reshape(1, -1))[0]
            iso_scores.append(iso_dec)

        distances = np.array(distances)
        iso_scores = np.array(iso_scores)

        self._dist_mean = float(np.mean(distances))
        self._dist_std = float(np.std(distances)) if np.std(distances) > 1e-8 else 0.01
        self._iso_mean = float(np.mean(iso_scores))
        self._iso_std = float(np.std(iso_scores)) if np.std(iso_scores) > 1e-8 else 0.01

        self.is_loaded = True
        logger.info(
            f"OOD Detector fitted on {len(benign_embeddings)} samples. "
            f"dist_mean={self._dist_mean:.4f}, dist_std={self._dist_std:.4f}, "
            f"iso_mean={self._iso_mean:.4f}, iso_std={self._iso_std:.4f}"
        )

    def evaluate(self, target_embedding: np.ndarray) -> float:
        """
        Score how far outside the benign manifold an input is.
        Uses z-score normalization so only truly unusual inputs score high.
        Returns 0.0 (normal) to 1.0 (extreme outlier).
        """
        if not self.is_loaded or self._centroid is None:
            return 0.0

        if len(target_embedding.shape) == 1:
            target = target_embedding.reshape(1, -1)
        else:
            target = target_embedding

        # 1. Centroid distance z-score
        emb = target[0]
        norm_a = np.linalg.norm(emb)
        norm_b = np.linalg.norm(self._centroid)
        cosine_sim = np.dot(emb, self._centroid) / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
        cosine_dist = 1.0 - cosine_sim

        # How many standard deviations away from the training mean?
        dist_zscore = (cosine_dist - self._dist_mean) / self._dist_std

        # 2. IsolationForest z-score
        iso_decision = self._iforest.decision_function(target)[0]
        # Lower decision = more anomalous, so we negate
        iso_zscore = -(iso_decision - self._iso_mean) / self._iso_std

        # 3. Combine: both z-scores contribute. Only flag when BOTH indicate outlier.
        # Sigmoid to compress into 0-1 range
        combined_z = (dist_zscore * 0.5) + (iso_zscore * 0.5)
        ood_score = 1.0 / (1.0 + np.exp(-combined_z + 1.5))  # shifted sigmoid: ~0 for z<1, ramps up above

        return float(np.clip(ood_score, 0.0, 1.0))

    def save(self) -> None:
        """Serialize detector state including distribution statistics."""
        if not self.is_loaded:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        state = {
            "centroid": self._centroid,
            "iforest": self._iforest,
            "dist_mean": self._dist_mean,
            "dist_std": self._dist_std,
            "iso_mean": self._iso_mean,
            "iso_std": self._iso_std,
        }
        path = os.path.join(self.model_dir, "ood_detector.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved OOD Detector to {path}")

    def load(self) -> bool:
        """Load detector state."""
        path = os.path.join(self.model_dir, "ood_detector.pkl")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._centroid = state.get("centroid")
            self._iforest = state.get("iforest")
            self._dist_mean = state.get("dist_mean", 0.0)
            self._dist_std = state.get("dist_std", 1.0)
            self._iso_mean = state.get("iso_mean", 0.0)
            self._iso_std = state.get("iso_std", 1.0)
            if self._centroid is not None and self._iforest is not None:
                self.is_loaded = True
                return True
        except Exception as e:
            logger.warning(f"Failed to load OOD Detector: {e}")
        return False
