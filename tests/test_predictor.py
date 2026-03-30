import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import asyncio

from src.core.predictor import ModelPerformancePredictor, PredictionResult
from src.core.metrics_collector import MetricsCollector, InferenceMetrics
from src.core.drift_detector import DriftDetector, DriftResult
from src.db.models import ModelMetrics, DriftAlert
from src.cache.redis_client import RedisClient
from src.monitoring.prometheus import PrometheusMetrics


class TestModelPerformancePredictor:
    """Test suite for ModelPerformancePredictor."""
    
    @pytest.fixture
    async def predictor(self) -> ModelPerformancePredictor:
        """Create predictor instance with mocked dependencies."""
        mock_db = AsyncMock()
        mock_redis = Mock(spec=RedisClient)
        mock_prometheus = Mock(spec=PrometheusMetrics)
        
        predictor = ModelPerformancePredictor(
            db_session=mock_db,
            redis_client=mock_redis,
            prometheus_metrics=mock_prometheus
        )
        return predictor
    
    @pytest.fixture
    def sample_metrics(self) -> InferenceMetrics:
        """Create sample inference metrics."""
        return InferenceMetrics(
            model_name="test_model",
            version="v1.0.0",
            latency_p50=45.2,
            latency_p95=120.5,
            latency_p99=200.1,
            throughput=1250.0,
            error_rate=0.02,
            memory_usage=512.0,
            cpu_usage=35.5,
            timestamp=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def sample_prediction_features(self) -> pd.DataFrame:
        """Create sample features for prediction."""
        return pd.DataFrame({
            'latency_trend': [0.15, 0.22, 0.18],
            'error_rate_trend': [0.001, 0.003, 0.002],
            'throughput_trend': [-0.05, -0.08, -0.03],
            'memory_trend': [0.12, 0.18, 0.15],
            'cpu_trend': [0.08, 0.12, 0.10],
            'drift_score': [0.3, 0.7, 0.5],
            'time_since_deployment': [72, 120, 96]
        })
    
    async def test_predict_performance_degradation_no_drift(self, predictor: ModelPerformancePredictor, sample_metrics: InferenceMetrics):
        """Test performance prediction when no drift is detected."""
        # Mock dependencies
        predictor.drift_detector = Mock(spec=DriftDetector)
        predictor.drift_detector.detect_drift.return_value = DriftResult(
            has_drift=False,
            drift_score=0.2,
            affected_features=[],
            confidence=0.95
        )
        
        predictor._get_historical_metrics = AsyncMock(return_value=[])
        predictor._extract_trend_features = Mock(return_value=pd.DataFrame({
            'latency_trend': [0.05],
            'error_rate_trend': [0.001],
            'throughput_trend': [-0.02],
            'memory_trend': [0.03],
            'cpu_trend': [0.02],
            'drift_score': [0.2],
            'time_since_deployment': [48]
        }))
        predictor._predict_degradation_probability = Mock(return_value=0.15)
        
        result = await predictor.predict_performance_degradation(sample_metrics)
        
        assert isinstance(result, PredictionResult)
        assert result.model_name == "test_model"
        assert result.degradation_probability == 0.15
        assert not result.requires_attention
        assert result.confidence > 0.8
        assert "No significant drift detected" in result.explanation
    
    async def test_predict_performance_degradation_with_drift(self, predictor: ModelPerformancePredictor, sample_metrics: InferenceMetrics):
        """Test performance prediction when drift is detected."""
        # Mock drift detection
        predictor.drift_detector = Mock(spec=DriftDetector)
        predictor.drift_detector.detect_drift.return_value = DriftResult(
            has_drift=True,
            drift_score=0.8,
            affected_features=['feature1', 'feature2'],
            confidence=0.92
        )
        
        predictor._get_historical_metrics = AsyncMock(return_value=[])
        predictor._extract_trend_features = Mock(return_value=pd.DataFrame({
            'latency_trend': [0.25],
            'error_rate_trend': [0.008],
            'throughput_trend': [-0.15],
            'memory_trend': [0.18],
            'cpu_trend': [0.12],
            'drift_score': [0.8],
            'time_since_deployment': [168]
        }))
        predictor._predict_degradation_probability = Mock(return_value=0.85)
        
        result = await predictor.predict_performance_degradation(sample_metrics)
        
        assert result.degradation_probability == 0.85
        assert result.requires_attention
        assert "Data drift detected" in result.explanation
        assert len(result.affected_features) == 2
    
    async def test_get_historical_metrics_with_cache_hit(self, predictor: ModelPerformancePredictor):
        """Test historical metrics retrieval with Redis cache hit."""
        cached_data = [
            {"latency_p95": 100.0, "error_rate": 0.01, "timestamp": "2023-01-01T00:00:00Z"},
            {"latency_p95": 110.0, "error_rate": 0.02, "timestamp": "2023-01-01T01:00:00Z"}
        ]
        
        predictor.redis_client.get.return_value = cached_data
        
        result = await predictor._get_historical_metrics("test_model", hours=24)
        
        assert len(result) == 2
        assert result[0]["latency_p95"] == 100.0
        predictor.redis_client.get.assert_called_once_with("metrics:test_model:24h")
    
    async def test_get_historical_metrics_with_cache_miss(self, predictor: ModelPerformancePredictor):
        """Test historical metrics retrieval with Redis cache miss."""
        # Mock cache miss
        predictor.redis_client.get.return_value = None
        
        # Mock database query
        mock_metrics = [
            Mock(
                latency_p50=45.0, latency_p95=100.0, latency_p99=150.0,
                throughput=1000.0, error_rate=0.01, memory_usage=400.0,
                cpu_usage=30.0, timestamp=datetime.now(timezone.utc)
            )
        ]
        
        predictor.db_session.execute = AsyncMock()
        predictor.db_session.execute.return_value.scalars.return_value.all.return_value = mock_metrics
        
        result = await predictor._get_historical_metrics("test_model", hours=24)
        
        assert len(result) == 1
        assert result[0]["latency_p95"] == 100.0
        assert result[0]["error_rate"] == 0.01
        
        # Verify cache was set
        predictor.redis_client.set.assert_called_once()
    
    def test_extract_trend_features(self, predictor: ModelPerformancePredictor):
        """Test trend feature extraction from historical data."""
        historical_data = [
            {
                "latency_p95": 100.0, "error_rate": 0.01, "throughput": 1000.0,
                "memory_usage": 400.0, "cpu_usage": 30.0,
                "timestamp": "2023-01-01T00:00:00Z"
            },
            {
                "latency_p95": 110.0, "error_rate": 0.015, "throughput": 950.0,
                "memory_usage": 420.0, "cpu_usage": 32.0,
                "timestamp": "2023-01-01T01:00:00Z"
            },
            {
                "latency_p95": 120.0, "error_rate": 0.02, "throughput": 900.0,
                "memory_usage": 450.0, "cpu_usage": 35.0,
                "timestamp": "2023-01-01T02:00:00Z"
            }
        ]
        
        current_metrics = InferenceMetrics(
            model_name="test_model",
            version="v1.0.0",
            latency_p50=50.0,
            latency_p95=125.0,
            latency_p99=180.0,
            throughput=850.0,
            error_rate=0.025,
            memory_usage=480.0,
            cpu_usage=38.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        with patch('pandas.Timestamp.now') as mock_now:
            mock_now.return_value = pd.Timestamp('2023-01-01T03:00:00Z')
            
            features = predictor._extract_trend_features(historical_data, current_metrics, 0.6)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        
        # Check that trends are calculated correctly
        assert features['latency_trend'].iloc[0] > 0  # Increasing trend
        assert features['error_rate_trend'].iloc[0] > 0  # Increasing trend
        assert features['throughput_trend'].iloc[0] < 0  # Decreasing trend
        assert features['drift_score'].iloc[0] == 0.6
        assert features['time_since_deployment'].iloc[0] >= 0
    
    def test_predict_degradation_probability(self, predictor: ModelPerformancePredictor, sample_prediction_features: pd.DataFrame):
        """Test degradation probability prediction logic."""
        # Test low risk scenario
        low_risk_features = pd.DataFrame({
            'latency_trend': [0.05],
            'error_rate_trend': [0.001],
            'throughput_trend': [-0.02],
            'memory_trend': [0.03],
            'cpu_trend': [0.02],
            'drift_score': [0.2],
            'time_since_deployment': [24]
        })
        
        prob_low = predictor._predict_degradation_probability(low_risk_features)
        assert 0.0 <= prob_low <= 1.0
        assert prob_low < 0.5  # Should be low risk
        
        # Test high risk scenario
        high_risk_features = pd.DataFrame({
            'latency_trend': [0.3],
            'error_rate_trend': [0.02],
            'throughput_trend': [-0.2],
            'memory_trend': [0.25],
            'cpu_trend': [0.18],
            'drift_score': [0.9],
            'time_since_deployment': [168]
        })
        
        prob_high = predictor._predict_degradation_probability(high_risk_features)
        assert 0.0 <= prob_high <= 1.0
        assert prob_high > 0.7  # Should be high risk
    
    async def test_store_prediction_result(self, predictor: ModelPerformancePredictor):
        """Test storing prediction results to database and cache."""
        prediction_result = PredictionResult(
            model_name="test_model",
            degradation_probability=0.75,
            confidence=0.88,
            requires_attention=True,
            explanation="High latency trend detected",
            affected_features=["latency", "throughput"],
            timestamp=datetime.now(timezone.utc)
        )
        
        await predictor._store_prediction_result(prediction_result)
        
        # Verify database storage
        predictor.db_session.add.assert_called_once()
        predictor.db_session.commit.assert_called_once()
        
        # Verify Redis caching
        predictor.redis_client.set.assert_called()
        
        # Verify Prometheus metrics
        predictor.prometheus_metrics.record_prediction.assert_called_once_with(
            model_name="test_model",
            probability=0.75,
            requires_attention=True
        )
    
    async def test_error_handling_database_failure(self, predictor: ModelPerformancePredictor, sample_metrics: InferenceMetrics):
        """Test error handling when database operations fail."""
        predictor.drift_detector = Mock(spec=DriftDetector)
        predictor.drift_detector.detect_drift.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            await predictor.predict_performance_degradation(sample_metrics)
        
        assert "Database connection failed" in str(exc_info.value)
    
    async def test_concurrent_predictions(self, predictor: ModelPerformancePredictor):
        """Test handling multiple concurrent prediction requests."""
        metrics_list = [
            InferenceMetrics(
                model_name=f"model_{i}",
                version="v1.0.0",
                latency_p50=40.0 + i,
                latency_p95=100.0 + i * 10,
                latency_p99=180.0 + i * 15,
                throughput=1000.0 - i * 50,
                error_rate=0.01 + i * 0.005,
                memory_usage=400.0 + i * 20,
                cpu_usage=30.0 + i * 5,
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]
        
        # Mock all dependencies
        predictor.drift_detector = Mock(spec=DriftDetector)
        predictor.drift_detector.detect_drift.return_value = DriftResult(
            has_drift=False,
            drift_score=0.3,
            affected_features=[],
            confidence=0.9
        )
        
        predictor._get_historical_metrics = AsyncMock(return_value=[])
        predictor._extract_trend_features = Mock(return_value=pd.DataFrame({
            'latency_trend': [0.1],
            'error_rate_trend': [0.002],
            'throughput_trend': [-0.05],
            'memory_trend': [0.08],
            'cpu_trend': [0.05],
            'drift_score': [0.3],
            'time_since_deployment': [72]
        }))
        predictor._predict_degradation_probability = Mock(return_value=0.35)
        predictor._store_prediction_result = AsyncMock()
        
        # Execute concurrent predictions
        tasks = [
            predictor.predict_performance_degradation(metrics)
            for metrics in metrics_list
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(isinstance(result, PredictionResult) for result in results)
        assert all(result.model_name == f"model_{i}" for i, result in enumerate(results))
    
    def test_calculate_confidence_score(self, predictor: ModelPerformancePredictor):
        """Test confidence score calculation."""
        # Test with sufficient historical data and no drift
        confidence_high = predictor._calculate_confidence_score(
            historical_data_points=100,
            drift_detected=False,
            drift_confidence=0.95
        )
        assert confidence_high > 0.85
        
        # Test with limited historical data and drift
        confidence_low = predictor._calculate_confidence_score(
            historical_data_points=10,
            drift_detected=True,
            drift_confidence=0.7
        )
        assert confidence_low < 0.7
        
        # Test edge case with no historical data
        confidence_minimal = predictor._calculate_confidence_score(
            historical_data_points=0,
            drift_detected=False,
            drift_confidence=0.9
        )
        assert confidence_minimal < 0.5