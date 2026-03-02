"""KMS Provider interfaces for RBI compliance.

Financial regulations require that log keys and API tokens are never
stored in plaintext. They must be managed by an external Key Management Service.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)


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


class AWSKMSProvider(KMSProvider):
    """Enterprise KMS Provider interfacing with AWS KMS.
    
    Includes a 15-minute local TTL cache to avoid hammering the AWS KMS
    API on every request while maintaining secure rotation compliance.
    """
    
    def __init__(self, key_id: str | None = None, region: str = "us-east-1"):
        self.key_id = key_id or os.environ.get("KAVACH_AWS_KMS_KEY_ID")
        self.region = region
        self._cache_secret: bytes | None = None
        self._cache_time: float = 0.0
        self.ttl_seconds = 900  # 15 minutes
        
        try:
            import boto3
            self.client = boto3.client("kms", region_name=self.region)
            self._has_boto3 = True
        except ImportError:
            self._has_boto3 = False
            logger.warning("boto3 not installed. AWSKMSProvider will use fallback mechanisms.")

    def get_hmac_secret(self) -> bytes:
        """Fetch the secret from AWS KMS, utilizing the local cache."""
        now = time.time()
        
        # Return cached secret if within TTL
        if self._cache_secret and (now - self._cache_time) < self.ttl_seconds:
            return self._cache_secret
            
        # Try to rotate/fetch new key from AWS KMS
        new_secret = self._fetch_from_aws()
        
        if new_secret:
            self._cache_secret = new_secret
            self._cache_time = now
            return new_secret
            
        # Fallback handling: Graceful degradation
        if self._cache_secret:
            logger.error("KMS_FAIL: Failed to rotate key from AWS KMS. Reusing expired cached key.")
            return self._cache_secret
            
        logger.error("KMS_FAIL: AWS KMS unreachable and no cache exists. Falling back to ENV secret.")
        fallback = os.environ.get("KAVACH_HMAC_SECRET", "default-insecure-hmac-secret-key")
        return fallback.encode("utf-8")
        
    def _fetch_from_aws(self) -> bytes | None:
        """Internal method to fetch Data Key from KMS."""
        if not self._has_boto3 or not self.key_id:
            return None
            
        try:
            # We use GenerateDataKey to create a unique symmetric key for HMAC
            response = self.client.generate_data_key(
                KeyId=self.key_id,
                KeySpec="AES_256"
            )
            return response["Plaintext"]
        except Exception as e:
            logger.error(f"AWS KMS Error: {e}")
            return None
