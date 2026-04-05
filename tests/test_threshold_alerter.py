import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime, timedelta
import json

from src.alerts.threshold_alerter import ThresholdAlerter, AlertSeverity
from src.core.metrics_collector import ModelMetrics


class TestThresholdAlerter:
    @pytest.fixture
    def mock_redis(self):
        return Mock()
    
    @pytest.fixture
    def alerter(self, mock_redis):
        return ThresholdAlerter(
            redis_client=mock_redis,
            accuracy_threshold=0.85,
            latency_threshold=500.0,
            drift_threshold=0.3
        )
    
    def test_check_accuracy_threshold_critical(self, alerter, mock_redis):
        mock_redis.get.return_value = None  # No recent alert
        
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.75,  # Below threshold
            latency=100.0,
            drift_score=0.1,
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert "accuracy" in alerts[0].message.lower()
        assert alerts[0].metric_value == 0.75
    
    def test_check_latency_threshold_warning(self, alerter, mock_redis):
        mock_redis.get.return_value = None
        
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.90,
            latency=600.0,  # Above threshold
            drift_score=0.1,
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "latency" in alerts[0].message.lower()
    
    def test_check_drift_threshold_warning(self, alerter, mock_redis):
        mock_redis.get.return_value = None
        
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.90,
            latency=100.0,
            drift_score=0.4,  # Above threshold
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "drift" in alerts[0].message.lower()
    
    def test_no_alerts_when_within_thresholds(self, alerter, mock_redis):
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.90,
            latency=100.0,
            drift_score=0.1,
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        assert len(alerts) == 0
    
    def test_alert_suppression_within_cooldown(self, alerter, mock_redis):
        # Simulate recent alert
        recent_alert_time = datetime.utcnow() - timedelta(minutes=5)
        mock_redis.get.return_value = recent_alert_time.isoformat()
        
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.70,  # Below threshold
            latency=100.0,
            drift_score=0.1,
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        assert len(alerts) == 0  # Suppressed due to cooldown
    
    @patch('requests.post')
    def test_send_webhook_alert(self, mock_post, alerter):
        mock_post.return_value.status_code = 200
        
        from src.alerts.threshold_alerter import Alert
        alert = Alert(
            model_id="test-model",
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            metric_name="accuracy",
            metric_value=0.70,
            threshold=0.85,
            timestamp=datetime.utcnow()
        )
        
        webhook_url = "http://test-webhook.com/alert"
        result = alerter.send_webhook_alert(alert, webhook_url)
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        assert payload['model_id'] == 'test-model'
        assert payload['severity'] == 'CRITICAL'
    
    def test_multiple_threshold_violations(self, alerter, mock_redis):
        mock_redis.get.return_value = None
        
        metrics = ModelMetrics(
            model_id="test-model",
            accuracy=0.70,  # Below threshold
            latency=600.0,  # Above threshold
            drift_score=0.4,  # Above threshold
            timestamp=datetime.utcnow()
        )
        
        alerts = alerter.check_thresholds(metrics)
        
        assert len(alerts) == 3
        severities = [alert.severity for alert in alerts]
        assert AlertSeverity.CRITICAL in severities
        assert AlertSeverity.WARNING in severities