"""Structured audit logger for Kavach decisions.

Logs every security decision as structured JSON. Supports
raw, hashed, and redacted prompt logging modes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from kavach.crypto.kms_provider import EnvKMSProvider, KMSProvider
from kavach.policies.validator import PromptLogMode


def _setup_logger(name: str = "kavach") -> logging.Logger:
    """Set up a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class KavachLogger:
    """Structured audit logger.

    All security-relevant events are logged as JSON objects
    with consistent structure for easy parsing by SIEM/log tools.

    Usage:
        logger = KavachLogger(log_prompts=PromptLogMode.HASHED)
        logger.log_decision(decision, prompt="user input...", identity=identity)
    """

    def __init__(
        self,
        log_prompts: PromptLogMode = PromptLogMode.HASHED,
        logger_name: str = "kavach",
        kms_provider: KMSProvider | None = None,
    ) -> None:
        self._log_prompt_mode = log_prompts
        self._logger = _setup_logger(logger_name)
        self._kms = kms_provider or EnvKMSProvider()
        # Seed for hash chain. In production this would be loaded from secure storage.
        self._prev_hash = "0" * 64

    def _compute_hmac(self, log_dict: dict[str, Any]) -> str:
        """Compute HMAC-SHA256 of the log entry linked to the previous log."""
        secret = self._kms.get_hmac_secret()
        
        # Serialize deterministically for hashing
        canonical_json = json.dumps(log_dict, sort_keys=True)
        payload = f"{self._prev_hash}|{canonical_json}".encode("utf-8")
        
        current_hash = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        self._prev_hash = current_hash
        return current_hash

    def _prepare_prompt(self, prompt: str | None) -> str | None:
        """Prepare prompt for logging based on configured mode."""
        if prompt is None:
            return None

        if self._log_prompt_mode == PromptLogMode.RAW:
            return prompt
        elif self._log_prompt_mode == PromptLogMode.HASHED:
            return f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        else:  # REDACTED
            return f"[REDACTED:{len(prompt)}chars]"

    def log_decision(
        self,
        decision: dict[str, Any],
        prompt: str | None = None,
        identity: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a security decision.

        Args:
            decision: The Decision.to_dict() output.
            prompt: The original prompt (will be hashed/redacted per policy).
            identity: The Identity.to_dict() output.
            extra: Additional context to include.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "kavach.decision",
            "decision": decision.get("decision", "unknown"),
            "risk_score": decision.get("risk_score", 0.0),
            "reasons": decision.get("reasons", []),
            "matched_rules": decision.get("matched_rules", []),
            "latency_ms": decision.get("latency_ms", 0.0),
            "session_id": decision.get("session_id", ""),
        }

        if prompt is not None:
            entry["prompt"] = self._prepare_prompt(prompt)

        if identity:
            entry["identity"] = identity

        if extra:
            entry["extra"] = extra

        entry["_chain_hash"] = self._compute_hmac(entry)
        self._logger.info(json.dumps(entry, default=str))

    def log_tool_call(
        self,
        tool_name: str,
        decision: str,
        role: str,
        risk_score: float = 0.0,
        reasons: list[str] | None = None,
    ) -> None:
        """Log a tool guard decision."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "kavach.tool_call",
            "tool_name": tool_name,
            "decision": decision,
            "role": role,
            "risk_score": risk_score,
            "reasons": reasons or [],
        }
        
        entry["_chain_hash"] = self._compute_hmac(entry)
        self._logger.info(json.dumps(entry, default=str))

    def log_error(self, error: str, context: dict[str, Any] | None = None) -> None:
        """Log an error event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "kavach.error",
            "error": error,
            "context": context or {},
        }
        
        entry["_chain_hash"] = self._compute_hmac(entry)
        self._logger.error(json.dumps(entry, default=str))
