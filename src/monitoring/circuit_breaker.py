from typing import Optional, Callable, Any
from datetime import datetime, timedelta
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern for external service calls with exponential backoff."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        max_backoff: int = 300
    ):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.max_backoff = max_backoff
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.backoff_multiplier = 1
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN, next attempt in {self._get_backoff_delay()}s")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self._get_backoff_delay()
    
    def _get_backoff_delay(self) -> int:
        """Calculate exponential backoff delay."""
        delay = min(self.timeout_duration * (2 ** (self.backoff_multiplier - 1)), self.max_backoff)
        return delay
    
    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.backoff_multiplier = 1
        self.state = CircuitState.CLOSED
        logger.debug("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.backoff_multiplier += 1
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures, "
                f"backoff: {self._get_backoff_delay()}s"
            )
