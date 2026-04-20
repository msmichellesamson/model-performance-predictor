"""Integration test for end-to-end performance monitoring pipeline."""
import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch

from src.core.metrics_collector import MetricsCollector
from src.core.predictor import PerformancePredictor
from src.monitoring.latency_monitor import LatencyMonitor
from src.monitoring.accuracy_monitor import AccuracyMonitor
from src.cache.redis_client import RedisClient
from src.alerts.threshold_alerter import ThresholdAlerter


class TestPerformanceMonitoringPipeline:
    """Integration tests for the complete performance monitoring pipeline."""
    
    @pytest.fixture
    async def setup_pipeline(self):
        """Setup complete monitoring pipeline for testing."""
        redis_client = AsyncMock(spec=RedisClient)
        metrics_collector = MetricsCollector(redis_client=redis_client)
        predictor = PerformancePredictor()
        latency_monitor = LatencyMonitor(threshold_ms=200)
        accuracy_monitor = AccuracyMonitor(threshold=0.8)
        alerter = ThresholdAlerter()
        
        return {
            'redis': redis_client,
            'collector': metrics_collector,
            'predictor': predictor,
            'latency_monitor': latency_monitor,
            'accuracy_monitor': accuracy_monitor,
            'alerter': alerter
        }
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection_flow(self, setup_pipeline):
        """Test complete flow from metrics collection to alert generation."""
        pipeline = await setup_pipeline
        
        # Simulate degrading model performance over time
        model_id = "test-model-v1"
        
        # Initial good performance
        await pipeline['collector'].record_prediction_latency(model_id, 50)
        await pipeline['collector'].record_accuracy(model_id, 0.95)
        
        # Gradual degradation
        degradation_steps = [
            {'latency': 150, 'accuracy': 0.85},
            {'latency': 250, 'accuracy': 0.75},  # Should trigger alerts
            {'latency': 350, 'accuracy': 0.65}   # Critical degradation
        ]
        
        alerts_triggered = []
        
        for step in degradation_steps:
            await pipeline['collector'].record_prediction_latency(model_id, step['latency'])
            await pipeline['collector'].record_accuracy(model_id, step['accuracy'])
            
            # Check latency monitoring
            latency_alert = pipeline['latency_monitor'].check_threshold(step['latency'])
            if latency_alert:
                alerts_triggered.append(f"latency_{step['latency']}ms")
            
            # Check accuracy monitoring  
            accuracy_alert = pipeline['accuracy_monitor'].check_threshold(step['accuracy'])
            if accuracy_alert:
                alerts_triggered.append(f"accuracy_{step['accuracy']}")
        
        # Verify alerts were triggered for degraded performance
        assert len(alerts_triggered) >= 2
        assert any('latency_250' in alert for alert in alerts_triggered)
        assert any('accuracy_0.75' in alert for alert in alerts_triggered)
    
    @pytest.mark.asyncio
    async def test_prediction_accuracy_with_real_metrics(self, setup_pipeline):
        """Test predictor accuracy using realistic metric patterns."""
        pipeline = await setup_pipeline
        model_id = "prod-model-v2"
        
        # Generate realistic degradation pattern
        base_latency = 80
        base_accuracy = 0.92
        
        historical_metrics = []
        for i in range(20):
            # Simulate gradual degradation with noise
            latency_drift = i * 5 + (i % 3) * 10  # Trending up with noise
            accuracy_drift = -i * 0.01 + (i % 2) * 0.005  # Trending down
            
            metrics = {
                'latency': base_latency + latency_drift,
                'accuracy': base_accuracy + accuracy_drift,
                'timestamp': time.time() - (20-i) * 3600  # Hourly data
            }
            historical_metrics.append(metrics)
        
        # Use predictor to forecast next performance
        with patch.object(pipeline['predictor'], 'get_historical_metrics') as mock_history:
            mock_history.return_value = historical_metrics
            
            prediction = await pipeline['predictor'].predict_performance_degradation(
                model_id, horizon_hours=6
            )
        
        # Verify prediction indicates degradation
        assert prediction['degradation_risk'] > 0.7  # High risk expected
        assert prediction['predicted_latency'] > base_latency + 100
        assert prediction['predicted_accuracy'] < 0.85
        
        # Verify confidence in prediction
        assert 0.6 <= prediction['confidence'] <= 1.0
