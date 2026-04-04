import pytest
from unittest.mock import patch, MagicMock
from src.monitoring.circuit_breaker import CircuitBreaker, CircuitBreakerState
import time


class TestCircuitBreaker:
    def test_circuit_breaker_initialization(self):
        cb = CircuitBreaker(threshold=3, timeout=60)
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.threshold == 3
        assert cb.timeout == 60

    def test_successful_call_in_closed_state(self):
        cb = CircuitBreaker(threshold=3, timeout=60)
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_failure_increments_count(self):
        cb = CircuitBreaker(threshold=3, timeout=60)
        
        def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.failure_count == 1
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=2, timeout=60)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2

    def test_open_circuit_blocks_calls(self):
        cb = CircuitBreaker(threshold=1, timeout=60)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit open
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Should block subsequent calls
        def any_func():
            return "should not execute"
        
        with pytest.raises(Exception, match="Circuit breaker is open"):
            cb.call(any_func)

    @patch('time.time')
    def test_circuit_moves_to_half_open_after_timeout(self, mock_time):
        cb = CircuitBreaker(threshold=1, timeout=60)
        
        # Set initial time
        mock_time.return_value = 1000
        
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit open
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Move time forward past timeout
        mock_time.return_value = 1070
        
        # Next call should move to half-open
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @patch('time.time')
    def test_half_open_success_closes_circuit(self, mock_time):
        cb = CircuitBreaker(threshold=1, timeout=60)
        mock_time.return_value = 1000
        
        # Open circuit
        with pytest.raises(Exception):
            cb.call(lambda: exec('raise Exception("fail")'))
        
        # Move to half-open
        mock_time.return_value = 1070
        
        def success_func():
            return "recovery"
        
        result = cb.call(success_func)
        assert result == "recovery"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    def test_metrics_collection(self):
        cb = CircuitBreaker(threshold=2, timeout=60)
        
        metrics = cb.get_metrics()
        expected_keys = {'state', 'failure_count', 'total_requests', 'failed_requests'}
        assert set(metrics.keys()) == expected_keys
        assert metrics['state'] == 'CLOSED'
        assert metrics['failure_count'] == 0
        assert metrics['total_requests'] == 0
        assert metrics['failed_requests'] == 0