"""Kavach vector attack store — semantic attack signature matching.

Uses FAISS (local) or Qdrant (server) to find semantically similar
attack patterns via nearest-neighbor search on embeddings.
"""

from kavach.vectors.attack_store import AttackVectorStore, AttackMatch

__all__ = ["AttackVectorStore", "AttackMatch"]
