import pytest
import time
from unittest.mock import Mock, patch
from src.monitoring.latency_monitor import LatencyMonitor


class TestLatencyMonitor:
    
    def setup_method(self):
        self.mock_redis = Mock()
        self.monitor = LatencyMonitor(self.mock_redis)
    
    def test_record_latency(self):
        """Test recording latency metrics"""
        self.monitor.record_latency('model_v1', 150.5)
        
        # Verify Redis calls
        self.mock_redis.lpush.assert_called_once()
        self.mock_redis.ltrim.assert_called_once()
        self.mock_redis.expire.assert_called_once()
    
    def test_get_p99_latency(self):
        """Test P99 latency calculation"""
        # Mock latencies: 10 values from 100-1000ms
        mock_latencies = [str(100 + i * 100) for i in range(10)]
        self.mock_redis.lrange.return_value = mock_latencies
        
        p99 = self.monitor.get_p99_latency('model_v1')
        assert p99 == 1000.0  # 99th percentile of our test data
    
    def test_get_avg_latency(self):
        """Test average latency calculation"""
        mock_latencies = ['100', '200', '300', '400', '500']
        self.mock_redis.lrange.return_value = mock_latencies
        
        avg = self.monitor.get_avg_latency('model_v1')
        assert avg == 300.0
    
    def test_empty_latency_data(self):
        """Test handling of empty latency data"""
        self.mock_redis.lrange.return_value = []
        
        p99 = self.monitor.get_p99_latency('model_v1')
        avg = self.monitor.get_avg_latency('model_v1')
        
        assert p99 == 0.0
        assert avg == 0.0
    
    def test_check_latency_threshold(self):
        """Test latency threshold checking"""
        mock_latencies = ['50', '60', '1500', '70', '80']  # One spike
        self.mock_redis.lrange.return_value = mock_latencies
        
        # Should trigger alert for P99 > 1000ms
        is_degraded = self.monitor.check_latency_threshold('model_v1', p99_threshold=1000.0)
        assert is_degraded is True
        
        # Should not trigger alert for higher threshold
        is_degraded = self.monitor.check_latency_threshold('model_v1', p99_threshold=2000.0)
        assert is_degraded is False
    
    def test_get_latency_trend(self):
        """Test latency trend calculation"""
        # Simulate increasing latency trend
        mock_latencies = [str(100 + i * 10) for i in range(20)]  # 100, 110, 120, ..., 290
        self.mock_redis.lrange.return_value = mock_latencies
        
        trend = self.monitor.get_latency_trend('model_v1')
        assert trend > 0  # Positive trend (increasing)
    
    @patch('time.time')
    def test_redis_key_expiry(self, mock_time):
        """Test Redis key expiration is set correctly"""
        mock_time.return_value = 1000000
        self.monitor.record_latency('model_v1', 100.0)
        
        # Verify expire is called with 1 hour (3600 seconds)
        self.mock_redis.expire.assert_called_with('latency:model_v1', 3600)