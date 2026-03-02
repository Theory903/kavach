"""Distributed behavioral risk tracking using Redis.

Extends the in-memory BehavioralTracker to work across a distributed
Kubernetes deployment. Stores rolling windows of user scores.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


logger = logging.getLogger(__name__)


class RedisBehavioralTracker:
    """Tracks user history and computes risk multipliers over Redis.
    
    Uses Redis Hashes and Lists to store session data with TTLs.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", history_size: int = 10, ttl_seconds: int = 86400) -> None:
        self._history_size = history_size
        self._ttl = ttl_seconds
        
        self.is_connected = False
        self._redis = None
        
        if not _HAS_REDIS:
            logger.warning("redis-py not installed. Redis tracking disabled.")
            return
            
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self.is_connected = True
            
            # Preload Lua script for atomic sliding window modifications
            self._record_script = """
            local state_key = KEYS[1]
            local list_key = KEYS[2]
            
            local risk_score = tonumber(ARGV[1])
            local is_block = tonumber(ARGV[2])
            local timestamp = ARGV[3]
            local history_size = tonumber(ARGV[4])
            local ttl = tonumber(ARGV[5])
            
            redis.call('HINCRBY', state_key, 'request_count', 1)
            redis.call('HINCRBYFLOAT', state_key, 'total_risk', risk_score)
            
            if is_block == 1 then
                redis.call('HINCRBY', state_key, 'violation_count', 1)
            end
            
            redis.call('HSET', state_key, 'last_seen_ts', timestamp)
            
            redis.call('RPUSH', list_key, ARGV[1])
            redis.call('LTRIM', list_key, -history_size, -1)
            
            redis.call('EXPIRE', state_key, ttl)
            redis.call('EXPIRE', list_key, ttl)
            
            return 1
            """
            self._record_sha = self._redis.script_load(self._record_script)
            
        except Exception as e:
            logger.error("Failed to connect to Redis for behavioral tracking: %s", e)

    def _get_user_key(self, user_id: str) -> str:
        return f"kavach:user_state:{user_id}"
        
    def _get_list_key(self, user_id: str) -> str:
        return f"kavach:user_history:{user_id}"

    def record_interaction(self, user_id: str, risk_score: float, action_taken: str) -> None:
        """Update state after an interaction atomically via loaded Lua script."""
        if user_id == "anonymous" or not self.is_connected or not self._redis:
            return

        state_key = self._get_user_key(user_id)
        list_key = self._get_list_key(user_id)
        is_block = 1 if action_taken == "block" else 0
        
        try:
            self._redis.evalsha(
                self._record_sha, 
                2, 
                state_key, 
                list_key, 
                str(risk_score), 
                is_block, 
                time.time(), 
                self._history_size, 
                self._ttl
            )
        except redis.exceptions.NoScriptError:
            # Re-compile if Redis wiped script cache locally
            self._record_sha = self._redis.script_load(self._record_script)
            self._redis.evalsha(
                self._record_sha, 
                2, 
                state_key, 
                list_key, 
                str(risk_score), 
                is_block, 
                time.time(), 
                self._history_size, 
                self._ttl
            )
        except Exception as e:
            logger.error("Redis Lua record script failed: %s", e)

    def get_behavioral_multiplier(self, user_id: str) -> float:
        """Calculate a risk multiplier based on decentralized history."""
        if user_id == "anonymous" or not self.is_connected or not self._redis:
            return 1.0

        state_key = self._get_user_key(user_id)
        list_key = self._get_list_key(user_id)
        
        try:
            state = self._redis.hgetall(state_key)
            if not state:
                return 1.0 # Brand new user
                
            request_count = int(state.get("request_count", 0))
            violation_count = int(state.get("violation_count", 0))
            
            # Immediate penalty for blocks
            if violation_count > 0:
                return min(1.5, 1.0 + (0.1 * violation_count))
                
            if request_count < 3:
                return 1.0
                
            # Average recent risk
            history_strs = self._redis.lrange(list_key, 0, -1)
            
            if history_strs:
                scores = [float(x) for x in history_strs]
                avg_recent_risk = sum(scores) / len(scores)
                
                # If they keep generating borderline risk (e.g. 0.4)
                if avg_recent_risk > 0.3:
                    return 1.2
                
                # Very clean history
                if avg_recent_risk < 0.1 and request_count >= 5:
                    return 0.85
                    
        except Exception as e:
            logger.error("Redis fetch failed: %s", e)
            
        return 1.0
