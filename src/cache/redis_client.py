import redis
import logging
import time
from typing import Optional, Any
from redis.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 max_connections: int = 10, socket_timeout: int = 5):
        self.pool = ConnectionPool(
            host=host, port=port, db=db,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            retry_on_timeout=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        
    def _retry_operation(self, operation, *args, max_retries: int = 3, **kwargs) -> Any:
        """Retry Redis operations with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Redis operation failed after {max_retries} attempts: {e}")
                    raise
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Redis operation failed (attempt {attempt + 1}), retrying in {wait_time}s")
                time.sleep(wait_time)
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value with retry logic"""
        return self._retry_operation(self.client.get, key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value with retry logic"""
        return self._retry_operation(self.client.set, key, value, ex=ex)
    
    def delete(self, key: str) -> int:
        """Delete key with retry logic"""
        return self._retry_operation(self.client.delete, key)
    
    def ping(self) -> bool:
        """Health check with retry logic"""
        try:
            return self._retry_operation(self.client.ping)
        except Exception:
            return False
    
    def close(self):
        """Close connection pool"""
        self.pool.disconnect()