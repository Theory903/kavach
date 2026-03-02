"""Kavach ML module — production-grade ensemble risk engine.

Provides feature engineering, ML classifiers, embedding similarity,
behavioral scoring, and meta-model aggregation for production-grade
risk analysis of LLM interactions.
"""

from kavach.ml.ensemble import EnsembleRiskScorer

__all__ = ["EnsembleRiskScorer"]