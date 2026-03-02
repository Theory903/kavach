"""KMS Provider interfaces for RBI compliance.

Financial regulations require that log keys and API tokens are never
stored in plaintext. They must be managed by an external Key Management Service.
"""

from __future__ import annotations

import os


class KMSProvider:
    """Base interface for fetching encryption/HMAC keys."""
    
    def get_hmac_secret(self) -> bytes:
        """Get the secret key used for WORM audit log chaining."""
        raise NotImplementedError()


class EnvKMSProvider(KMSProvider):
    """Fallback provider that reads from environment variables."""
    
    def get_hmac_secret(self) -> bytes:
        """Get the secret from KAVACH_HMAC_SECRET or generate a default one."""
        secret = os.environ.get("KAVACH_HMAC_SECRET", "default-insecure-hmac-secret-key")
        return secret.encode("utf-8")
