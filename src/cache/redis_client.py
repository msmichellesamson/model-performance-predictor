import redis
import logging
import time
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

def retry_redis_operation(max_retries: int = 3, delay: float = 0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Redis operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"Redis operation failed after {max_retries} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator

class RedisClient:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 max_connections: int = 10, socket_timeout: int = 5):
        self.pool = redis.ConnectionPool(
            host=host, 
            port=port, 
            db=db,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            health_check_interval=30
        )
        self.client = redis.Redis(connection_pool=self.pool, decode_responses=True)
        
    @retry_redis_operation()
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis with retry logic"""
        return self.client.get(key)
    
    @retry_redis_operation()
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value in Redis with retry logic"""
        return self.client.set(key, value, ex=ex)
    
    @retry_redis_operation()
    def delete(self, key: str) -> int:
        """Delete key from Redis with retry logic"""
        return self.client.delete(key)
    
    @retry_redis_operation()
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis with retry logic"""
        return bool(self.client.exists(key))
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Close connection pool"""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")
