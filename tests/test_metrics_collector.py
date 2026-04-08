import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

from src.core.metrics_collector import MetricsCollector
from src.db.models import ModelMetrics


class TestMetricsCollector:
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        return mock_db
    
    @pytest.fixture
    def collector(self, mock_db):
        """Create MetricsCollector with mocked dependencies"""
        return MetricsCollector(db=mock_db)
    
    def test_collect_latency_metrics(self, collector, mock_db):
        """Test latency metrics collection from database"""
        # Mock database response
        mock_metrics = [
            Mock(response_time=0.15, created_at=datetime.utcnow()),
            Mock(response_time=0.23, created_at=datetime.utcnow()),
            Mock(response_time=0.18, created_at=datetime.utcnow())
        ]
        mock_db.query.return_value.filter.return_value.all.return_value = mock_metrics
        
        metrics = collector.collect_latency_metrics(minutes=5)
        
        assert len(metrics) == 3
        assert metrics[0]['latency'] == 0.15
        assert 'timestamp' in metrics[0]
        mock_db.query.assert_called_once_with(ModelMetrics)
    
    def test_collect_accuracy_metrics(self, collector, mock_db):
        """Test accuracy metrics collection from database"""
        mock_metrics = [
            Mock(accuracy=0.92, model_version='v1.2', created_at=datetime.utcnow()),
            Mock(accuracy=0.89, model_version='v1.2', created_at=datetime.utcnow())
        ]
        mock_db.query.return_value.filter.return_value.all.return_value = mock_metrics
        
        metrics = collector.collect_accuracy_metrics(hours=1)
        
        assert len(metrics) == 2
        assert metrics[0]['accuracy'] == 0.92
        assert metrics[0]['model_version'] == 'v1.2'
    
    def test_get_prediction_volume(self, collector, mock_db):
        """Test prediction volume calculation"""
        mock_db.query.return_value.filter.return_value.count.return_value = 1543
        
        volume = collector.get_prediction_volume(hours=24)
        
        assert volume == 1543
        mock_db.query.return_value.filter.return_value.count.assert_called_once()
    
    def test_calculate_throughput(self, collector):
        """Test throughput calculation from metrics"""
        now = datetime.utcnow()
        metrics = [
            {'timestamp': now - timedelta(minutes=4)},
            {'timestamp': now - timedelta(minutes=3)},
            {'timestamp': now - timedelta(minutes=1)}
        ]
        
        throughput = collector.calculate_throughput(metrics, window_minutes=5)
        
        # 3 predictions in 5 minutes = 0.6 predictions/minute
        assert throughput == 0.6
    
    def test_get_error_rate(self, collector, mock_db):
        """Test error rate calculation"""
        mock_db.query.return_value.filter.return_value.count.side_effect = [150, 10]  # total, errors
        
        error_rate = collector.get_error_rate(minutes=30)
        
        assert error_rate == 10 / 150
        assert mock_db.query.call_count == 2
    
    def test_empty_metrics_handling(self, collector, mock_db):
        """Test handling of empty metrics gracefully"""
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        metrics = collector.collect_latency_metrics(minutes=5)
        throughput = collector.calculate_throughput(metrics, window_minutes=5)
        
        assert metrics == []
        assert throughput == 0.0