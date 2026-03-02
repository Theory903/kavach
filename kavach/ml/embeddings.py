"""Embedding similarity scorer using ONNX Runtime.

Uses a lightweight ONNX model (Xenova/all-MiniLM-L6-v2) to extract
embeddings without blocking the GIL or requiring 1.5GB of PyTorch.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

from kavach.ml.dataset import get_attack_texts


logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b, axis=1)
    
    if a_norm == 0:
        return np.zeros(len(b))
        
    dot_products = np.dot(b, a.T).flatten()
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)
    return dot_products / (a_norm * b_norm)


class EmbeddingRiskScorer:
    """Computes risk by embedding similarity to known attacks.
    
    Loads an ONNX model (all-MiniLM-L6-v2) locally on CPU with <5ms latency.
    """

    def __init__(self, model_id: str = "Xenova/all-MiniLM-L6-v2") -> None:
        self.is_loaded = False
        self._session = None
        self._tokenizer = None
        self._attack_embeddings: np.ndarray | None = None
        self._model_id = model_id
        
        if not _HAS_ONNX:
            logger.warning("onnxruntime or tokenizers not installed. Embedding scorer disabled.")

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        sum_mask = np.maximum(sum_mask, 1e-9)
        return sum_embeddings / sum_mask

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings using ONNX."""
        if not self._tokenizer or not self._session:
            return np.zeros((len(texts), 384))

        # Tokenize (padding=True, truncation=True)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=512)
        
        encodings = self._tokenizer.encode_batch(texts)
        
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        # ONNX inference
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        model_output = self._session.run(None, ort_inputs)
        # model_output[0] is last_hidden_state
        last_hidden_state = model_output[0]
        
        # Pooling
        embeddings = self._mean_pooling(last_hidden_state, attention_mask)
        
        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def load_and_encode_corpus(self) -> None:
        """Load the ONNX model and compute embeddings for the attack corpus."""
        if not _HAS_ONNX:
            return

        logger.info("Downloading/Loading ONNX model %s ...", self._model_id)
        try:
            # Download files from HuggingFace
            model_path = hf_hub_download(repo_id=self._model_id, filename="onnx/model_quantized.onnx")
            vocab_path = hf_hub_download(repo_id=self._model_id, filename="tokenizer.json")
            
            self._tokenizer = Tokenizer.from_file(vocab_path)
            
            # Setup ONNX session for ultra-fast CPU inference
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1 # Keep to 1 for high-concurrency web servers
            self._session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
            
            # Encode attack patterns
            attack_texts = get_attack_texts()
            logger.info("Encoding %d attack patterns with ONNX...", len(attack_texts))
            self._attack_embeddings = self._encode_batch(attack_texts)
            self.is_loaded = True
            logger.info("ONNX Embedding scorer ready.")
        except Exception as e:
            logger.error("Error setting up ONNX embedding scorer: %s", e)
            self.is_loaded = False

    def predict_risk(self, prompt: str) -> float:
        """Score a prompt based on maximum similarity to any known attack."""
        if not self.is_loaded or self._session is None or self._attack_embeddings is None:
            return 0.0
            
        if not prompt.strip():
            return 0.0

        # Encode single prompt
        prompt_emb = self._encode_batch([prompt])[0]
        
        # Compute similarities against all attack patterns
        similarities = cosine_similarity(prompt_emb, self._attack_embeddings)
        
        # Get maximum similarity
        max_sim = float(np.max(similarities))
        
        return max(0.0, max_sim)
