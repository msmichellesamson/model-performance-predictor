import redis
import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 max_connections: int = 10, retry_attempts: int = 3):
        self.pool = redis.ConnectionPool(
            host=host, port=port, 
            max_connections=max_connections,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self.retry_attempts = retry_attempts
        self._client = None
    
    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.pool)
        return self._client
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry Redis operations with exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Redis operation failed after {self.retry_attempts} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Redis operation failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def get_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cached model metrics with retry logic"""
        key = f"metrics:{model_id}"
        try:
            data = self._retry_operation(self.client.get, key)
            return eval(data) if data else None
        except Exception as e:
            logger.error(f"Failed to get metrics for {model_id}: {e}")
            return None
    
    def cache_prediction(self, model_id: str, features_hash: str, 
                        prediction: Dict[str, Any], ttl: int = 300) -> bool:
        """Cache prediction with automatic expiration and retry"""
        key = f"prediction:{model_id}:{features_hash}"
        try:
            self._retry_operation(
                self.client.setex, key, ttl, str(prediction)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache prediction: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            self._retry_operation(self.client.ping)
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Clean up connection pool"""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")