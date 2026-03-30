import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
import structlog
from pydantic import BaseModel

from ..core.exceptions import CacheError, CacheConnectionError

logger = structlog.get_logger(__name__)


class CacheKey:
    """Redis cache key constants and generators."""
    
    PREDICTION_PREFIX = "pred"
    DRIFT_PREFIX = "drift"
    METRICS_PREFIX = "metrics"
    MODEL_STATE_PREFIX = "model_state"
    PERFORMANCE_PREFIX = "perf"
    
    @staticmethod
    def prediction(model_id: str, request_hash: str) -> str:
        """Generate cache key for prediction results."""
        return f"{CacheKey.PREDICTION_PREFIX}:{model_id}:{request_hash}"
    
    @staticmethod
    def drift_score(model_id: str, timestamp: datetime) -> str:
        """Generate cache key for drift scores."""
        ts = int(timestamp.timestamp())
        return f"{CacheKey.DRIFT_PREFIX}:{model_id}:{ts}"
    
    @staticmethod
    def metrics_window(model_id: str, window_minutes: int) -> str:
        """Generate cache key for metrics time windows."""
        return f"{CacheKey.METRICS_PREFIX}:{model_id}:{window_minutes}m"
    
    @staticmethod
    def model_state(model_id: str) -> str:
        """Generate cache key for model state."""
        return f"{CacheKey.MODEL_STATE_PREFIX}:{model_id}"
    
    @staticmethod
    def performance_metrics(model_id: str, metric_type: str) -> str:
        """Generate cache key for performance metrics."""
        return f"{CacheKey.PERFORMANCE_PREFIX}:{model_id}:{metric_type}"


class CacheConfig(BaseModel):
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    max_connections: int = 20
    decode_responses: bool = True


class RedisClient:
    """Production-grade Redis client for model performance prediction caching."""
    
    def __init__(self, config: CacheConfig) -> None:
        """Initialize Redis client with configuration.
        
        Args:
            config: Redis configuration parameters
        """
        self.config = config
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connected = False
        
    async def connect(self) -> None:
        """Establish Redis connection pool."""
        try:
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
                password=self.config.password,
                ssl=self.config.ssl,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            self._connected = True
            
            logger.info(
                "Redis connection established",
                host=self.config.host,
                port=self.config.port,
                db=self.config.database,
            )
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise CacheConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.info("Redis connection closed")
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            if not self._client:
                return False
            await self._client.ping()
            return True
        except Exception as e:
            logger.warning("Redis health check failed", error=str(e))
            return False
    
    def _ensure_connected(self) -> None:
        """Ensure Redis client is connected."""
        if not self._connected or not self._client:
            raise CacheError("Redis client not connected. Call connect() first.")
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from Redis storage."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None,
        if_not_exists: bool = False,
    ) -> bool:
        """Set cache value with optional expiration.
        
        Args:
            key: Cache key
            value: Value to cache
            expire_seconds: Expiration time in seconds
            if_not_exists: Only set if key doesn't exist
            
        Returns:
            True if value was set, False otherwise
        """
        self._ensure_connected()
        
        try:
            serialized_value = self._serialize_value(value)
            
            if if_not_exists:
                result = await self._client.set(
                    key, serialized_value, ex=expire_seconds, nx=True
                )
            else:
                result = await self._client.set(
                    key, serialized_value, ex=expire_seconds
                )
            
            success = bool(result)
            if success:
                logger.debug("Cache set successful", key=key, expires=expire_seconds)
            
            return success
            
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            raise CacheError(f"Failed to set cache key {key}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        self._ensure_connected()
        
        try:
            value = await self._client.get(key)
            if value is None:
                return None
            
            logger.debug("Cache hit", key=key)
            return self._deserialize_value(value)
            
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            raise CacheError(f"Failed to get cache key {key}: {e}")
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """Get multiple cached values.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values
        """
        self._ensure_connected()
        
        try:
            if not keys:
                return {}
            
            values = await self._client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize_value(value)
                else:
                    result[key] = None
            
            hit_count = sum(1 for v in result.values() if v is not None)
            logger.debug("Cache multi-get", keys_requested=len(keys), hits=hit_count)
            
            return result
            
        except Exception as e:
            logger.error("Cache multi-get failed", keys=keys, error=str(e))
            raise CacheError(f"Failed to get multiple cache keys: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete cached value.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        self._ensure_connected()
        
        try:
            result = await self._client.delete(key)
            deleted = bool(result)
            
            if deleted:
                logger.debug("Cache delete successful", key=key)
            
            return deleted
            
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            raise CacheError(f"Failed to delete cache key {key}: {e}")
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.
        
        Args:
            pattern: Pattern to match (supports Redis glob patterns)
            
        Returns:
            Number of keys deleted
        """
        self._ensure_connected()
        
        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._client.delete(*keys)
                logger.info("Pattern delete completed", pattern=pattern, deleted=deleted)
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error("Pattern delete failed", pattern=pattern, error=str(e))
            raise CacheError(f"Failed to delete pattern {pattern}: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        self._ensure_connected()
        
        try:
            result = await self._client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error("Cache exists check failed", key=key, error=str(e))
            raise CacheError(f"Failed to check key existence {key}: {e}")
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for key.
        
        Args:
            key: Cache key
            seconds: Expiration time in seconds
            
        Returns:
            True if expiration was set, False if key doesn't exist
        """
        self._ensure_connected()
        
        try:
            result = await self._client.expire(key, seconds)
            success = bool(result)
            
            if success:
                logger.debug("Expiration set", key=key, seconds=seconds)
            
            return success
            
        except Exception as e:
            logger.error("Set expiration failed", key=key, error=str(e))
            raise CacheError(f"Failed to set expiration for key {key}: {e}")
    
    async def ttl(self, key: str) -> int:
        """Get time-to-live for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        self._ensure_connected()
        
        try:
            ttl_seconds = await self._client.ttl(key)
            return ttl_seconds
            
        except Exception as e:
            logger.error("TTL check failed", key=key, error=str(e))
            raise CacheError(f"Failed to get TTL for key {key}: {e}")
    
    async def zadd_with_score(
        self,
        key: str,
        score: float,
        member: str,
        expire_seconds: Optional[int] = None,
    ) -> bool:
        """Add member to sorted set with score.
        
        Args:
            key: Sorted set key
            score: Score for the member
            member: Member to add
            expire_seconds: Optional expiration time
            
        Returns:
            True if member was added (new), False if updated
        """
        self._ensure_connected()
        
        try:
            result = await self._client.zadd(key, {member: score})
            
            if expire_seconds:
                await self._client.expire(key, expire_seconds)
            
            added = bool(result)
            logger.debug("Sorted set add", key=key, member=member, score=score, new=added)
            
            return added
            
        except Exception as e:
            logger.error("Sorted set add failed", key=key, error=str(e))
            raise CacheError(f"Failed to add to sorted set {key}: {e}")
    
    async def zrange_with_scores(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        desc: bool = False,
    ) -> List[tuple[str, float]]:
        """Get sorted set members with scores.
        
        Args:
            key: Sorted set key
            start: Start index
            end: End index (-1 for all)
            desc: Sort in descending order
            
        Returns:
            List of (member, score) tuples
        """
        self._ensure_connected()
        
        try:
            result = await self._client.zrange(
                key, start, end, desc=desc, withscores=True
            )
            
            # Convert to list of tuples with proper types
            members_scores = [(str(member), float(score)) for member, score in result]
            
            logger.debug("Sorted set range", key=key, count=len(members_scores))
            return members_scores
            
        except Exception as e:
            logger.error("Sorted set range failed", key=key, error=str(e))
            raise CacheError(f"Failed to get sorted set range {key}: {e}")
    
    async def pipeline(self) -> redis.Pipeline:
        """Create Redis pipeline for batch operations."""
        self._ensure_connected()
        return self._client.pipeline()
    
    async def info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        self._ensure_connected()
        
        try:
            info_data = await self._client.info()
            return info_data
            
        except Exception as e:
            logger.error("Redis info failed", error=str(e))
            raise CacheError(f"Failed to get Redis info: {e}")


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get or create global Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        config = CacheConfig()
        _redis_client = RedisClient(config)
        await _redis_client.connect()
    
    return _redis_client


async def close_redis_client() -> None:
    """Close global Redis client instance."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None