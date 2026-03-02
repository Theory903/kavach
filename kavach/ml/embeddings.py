"""Embedding similarity scorer.

Uses a lightweight, local embedding model via sentence-transformers
to compare incoming prompts against a known corpus of attacks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

from kavach.ml.dataset import get_attack_texts


logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    
    if a_norm == 0:
        return np.zeros(len(b))
        
    dot_products = np.dot(b, a.T).flatten()
    
    # Avoid zero division
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)
    
    return dot_products / (a_norm * b_norm)


class EmbeddingRiskScorer:
    """Computes risk by embedding similarity to known attacks.
    
    Loads a tiny ONNX/transformer model (like all-MiniLM-L6-v2) 
    that runs locally on CPU with <50ms latency.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.is_loaded = False
        self._model = None
        self._attack_embeddings: np.ndarray | None = None
        
        if not _HAS_SBERT:
            logger.warning("sentence-transformers not installed. Embedding scorer disabled.")
            return

        try:
            # We load the model lazily/explicitly, to avoid hanging on import
            self._model_name = model_name
        except Exception as e:
            logger.error("Failed to initialize embedding scorer: %s", e)

    def load_and_encode_corpus(self) -> None:
        """Load the model and compute embeddings for the attack corpus."""
        if not _HAS_SBERT:
            return

        logger.info("Loading embedding model %s (this may take a moment)...", self._model_name)
        try:
            self._model = SentenceTransformer(self._model_name)
            
            attack_texts = get_attack_texts()
            logger.info("Encoding %d attack patterns...", len(attack_texts))
            self._attack_embeddings = self._model.encode(attack_texts, convert_to_numpy=True)
            self.is_loaded = True
            logger.info("Embedding scorer ready.")
        except Exception as e:
            logger.error("Error setting up embedding scorer: %s", e)
            self.is_loaded = False

    def predict_risk(self, prompt: str) -> float:
        """Score a prompt based on maximum similarity to any known attack.
        
        Args:
            prompt: The text prompt.
            
        Returns:
            Float representing the maximum cosine similarity (0.0 to 1.0).
        """
        if not self.is_loaded or self._model is None or self._attack_embeddings is None:
            return 0.0
            
        if not prompt.strip():
            return 0.0

        # Encode single prompt
        prompt_emb = self._model.encode([prompt], convert_to_numpy=True)[0]
        
        # Compute similarities against all attack patterns
        similarities = cosine_similarity(prompt_emb, self._attack_embeddings)
        
        # Get maximum similarity
        max_sim = float(np.max(similarities))
        
        # Negative similarities are not risk (opposite meaning)
        return max(0.0, max_sim)
