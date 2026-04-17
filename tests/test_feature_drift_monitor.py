import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.monitoring.feature_drift_monitor import FeatureDriftMonitor
from src.cache.redis_client import RedisClient


class TestFeatureDriftMonitor:
    @pytest.fixture
    def redis_client(self):
        mock_redis = Mock(spec=RedisClient)
        mock_redis.get_baseline_stats.return_value = {
            'feature_1': {'mean': 0.5, 'std': 0.2},
            'feature_2': {'mean': 1.0, 'std': 0.3}
        }
        return mock_redis
    
    @pytest.fixture
    def monitor(self, redis_client):
        return FeatureDriftMonitor(redis_client, drift_threshold=0.1)
    
    def test_calculate_drift_no_drift(self, monitor):
        current_data = pd.DataFrame({
            'feature_1': [0.48, 0.52, 0.51],
            'feature_2': [0.98, 1.02, 1.01]
        })
        
        drift_score = monitor.calculate_drift(current_data)
        assert drift_score < 0.1
    
    def test_calculate_drift_high_drift(self, monitor):
        current_data = pd.DataFrame({
            'feature_1': [0.8, 0.9, 0.85],  # Mean shift from 0.5
            'feature_2': [1.5, 1.6, 1.55]  # Mean shift from 1.0
        })
        
        drift_score = monitor.calculate_drift(current_data)
        assert drift_score > 0.1
    
    def test_detect_drift_triggers_alert(self, monitor):
        with patch.object(monitor, 'calculate_drift', return_value=0.15):
            with patch.object(monitor, '_send_drift_alert') as mock_alert:
                current_data = pd.DataFrame({'feature_1': [0.8, 0.9]})
                
                result = monitor.detect_drift(current_data)
                
                assert result is True
                mock_alert.assert_called_once_with(0.15)
    
    def test_detect_drift_no_alert(self, monitor):
        with patch.object(monitor, 'calculate_drift', return_value=0.05):
            with patch.object(monitor, '_send_drift_alert') as mock_alert:
                current_data = pd.DataFrame({'feature_1': [0.48, 0.52]})
                
                result = monitor.detect_drift(current_data)
                
                assert result is False
                mock_alert.assert_not_called()
    
    def test_missing_baseline_stats(self, redis_client):
        redis_client.get_baseline_stats.return_value = None
        monitor = FeatureDriftMonitor(redis_client, drift_threshold=0.1)
        
        current_data = pd.DataFrame({'feature_1': [0.5, 0.6]})
        drift_score = monitor.calculate_drift(current_data)
        
        assert drift_score == 0.0  # No drift when no baseline
    
    def test_empty_current_data(self, monitor):
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Current data cannot be empty"):
            monitor.calculate_drift(empty_data)
    
    @pytest.mark.parametrize("threshold,expected_alert", [
        (0.05, True),   # Low threshold, should alert
        (0.20, False),  # High threshold, should not alert
    ])
    def test_drift_threshold_sensitivity(self, redis_client, threshold, expected_alert):
        monitor = FeatureDriftMonitor(redis_client, drift_threshold=threshold)
        
        with patch.object(monitor, 'calculate_drift', return_value=0.1):
            with patch.object(monitor, '_send_drift_alert') as mock_alert:
                current_data = pd.DataFrame({'feature_1': [0.7, 0.8]})
                
                monitor.detect_drift(current_data)
                
                if expected_alert:
                    mock_alert.assert_called_once()
                else:
                    mock_alert.assert_not_called()