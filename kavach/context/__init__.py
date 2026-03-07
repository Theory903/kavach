"""Kavach context security — RAG / retrieval pipeline protection.

Sanitizes retrieved documents to prevent indirect prompt injection
before they are fed into the LLM context window.
"""

from kavach.context.rag_sanitizer import RAGDocumentSanitizer, SanitizedContext

__all__ = ["RAGDocumentSanitizer", "SanitizedContext"]
