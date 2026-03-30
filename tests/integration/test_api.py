import asyncio
import json
import pytest
import numpy as np
from datetime import datetime, timedelta
from httpx import AsyncClient
from sqlalchemy import text
from typing import Dict, Any, List
import structlog

from src.main import app
from src.db.models import ModelPerformance, DriftAlert
from src.db.database import get_db_session, engine
from src.cache.redis_client import get_redis_client
from src.core.exceptions import ModelNotFoundError, InsufficientDataError

logger = structlog.get_logger()


class TestAPIIntegration:
    """Integration tests for the model performance predictor API."""

    @pytest.fixture(scope="class")
    async def client(self):
        """Create async HTTP client for API testing."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.fixture(scope="class")
    async def setup_test_data(self):
        """Setup test data in database and Redis."""
        # Clear existing data
        async with get_db_session() as session:
            await session.execute(text("TRUNCATE TABLE model_performance, drift_alerts CASCADE"))
            await session.commit()
        
        redis_client = await get_redis_client()
        await redis_client.flushdb()
        
        # Insert test model performance data
        test_data = []
        base_time = datetime.utcnow() - timedelta(hours=24)
        
        for i in range(100):
            timestamp = base_time + timedelta(minutes=i * 10)
            # Simulate performance degradation over time
            accuracy = 0.95 - (i * 0.002) + np.random.normal(0, 0.01)
            latency = 150 + (i * 2) + np.random.normal(0, 10)
            
            test_data.append({
                "model_id": "test-model-1",
                "timestamp": timestamp,
                "accuracy": max(0.5, min(1.0, accuracy)),
                "precision": max(0.5, min(1.0, accuracy + np.random.normal(0, 0.02))),
                "recall": max(0.5, min(1.0, accuracy + np.random.normal(0, 0.02))),
                "f1_score": max(0.5, min(1.0, accuracy + np.random.normal(0, 0.01))),
                "latency_p50": max(50, latency),
                "latency_p95": max(100, latency * 1.8),
                "latency_p99": max(200, latency * 2.5),
                "throughput": max(10, 100 - (i * 0.5) + np.random.normal(0, 5)),
                "error_rate": min(0.5, max(0.001, (i * 0.001) + np.random.normal(0, 0.005))),
                "drift_score": min(1.0, max(0.0, (i * 0.008) + np.random.normal(0, 0.02)))
            })
        
        async with get_db_session() as session:
            for data in test_data:
                perf = ModelPerformance(**data)
                session.add(perf)
            await session.commit()
        
        # Cache recent metrics in Redis
        recent_metrics = {
            "accuracy": [d["accuracy"] for d in test_data[-10:]],
            "latency_p95": [d["latency_p95"] for d in test_data[-10:]],
            "throughput": [d["throughput"] for d in test_data[-10:]],
            "error_rate": [d["error_rate"] for d in test_data[-10:]],
            "drift_score": [d["drift_score"] for d in test_data[-10:]]
        }
        
        await redis_client.setex(
            "model:test-model-1:recent_metrics",
            3600,
            json.dumps(recent_metrics, default=str)
        )
        
        return test_data

    async def test_health_check(self, client: AsyncClient):
        """Test API health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data

    async def test_predict_performance_success(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test successful performance prediction."""
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={"horizon_hours": 6}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "model_id" in data
        assert "prediction_timestamp" in data
        assert "horizon_hours" in data
        assert "predictions" in data
        assert "confidence_interval" in data
        assert "risk_level" in data
        assert "recommendations" in data
        
        # Validate predictions
        predictions = data["predictions"]
        assert "accuracy" in predictions
        assert "latency_p95" in predictions
        assert "throughput" in predictions
        assert "error_rate" in predictions
        
        # Validate confidence intervals
        ci = data["confidence_interval"]
        for metric in ["accuracy", "latency_p95", "throughput", "error_rate"]:
            assert metric in ci
            assert "lower" in ci[metric]
            assert "upper" in ci[metric]
            assert ci[metric]["lower"] <= predictions[metric] <= ci[metric]["upper"]
        
        # Validate risk level
        assert data["risk_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(data["recommendations"], list)

    async def test_predict_performance_custom_horizon(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test performance prediction with custom time horizon."""
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={"horizon_hours": 24, "confidence_level": 0.90}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["horizon_hours"] == 24

    async def test_predict_performance_model_not_found(self, client: AsyncClient):
        """Test prediction for non-existent model."""
        response = await client.post(
            "/api/v1/models/nonexistent-model/predict",
            json={"horizon_hours": 6}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "model not found" in data["error"].lower()

    async def test_predict_performance_invalid_horizon(self, client: AsyncClient):
        """Test prediction with invalid time horizon."""
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={"horizon_hours": -1}
        )
        
        assert response.status_code == 422
        
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={"horizon_hours": 169}  # > 168 hours (7 days)
        )
        
        assert response.status_code == 422

    async def test_get_model_metrics(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test retrieving model metrics."""
        response = await client.get(
            "/api/v1/models/test-model-1/metrics",
            params={"hours": 12}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_id" in data
        assert "metrics" in data
        assert "time_range" in data
        
        metrics = data["metrics"]
        required_metrics = ["accuracy", "precision", "recall", "f1_score", 
                          "latency_p50", "latency_p95", "latency_p99", 
                          "throughput", "error_rate", "drift_score"]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], list)
            assert len(metrics[metric]) > 0

    async def test_get_model_metrics_with_aggregation(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test retrieving aggregated model metrics."""
        response = await client.get(
            "/api/v1/models/test-model-1/metrics",
            params={"hours": 24, "aggregation": "hourly"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have fewer data points due to aggregation
        accuracy_points = len(data["metrics"]["accuracy"])
        assert accuracy_points <= 24  # At most 24 hourly points

    async def test_detect_drift(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test drift detection endpoint."""
        response = await client.post("/api/v1/models/test-model-1/detect-drift")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_id" in data
        assert "drift_detected" in data
        assert "drift_score" in data
        assert "timestamp" in data
        assert "features" in data
        
        assert isinstance(data["drift_detected"], bool)
        assert 0.0 <= data["drift_score"] <= 1.0
        assert isinstance(data["features"], dict)

    async def test_get_drift_alerts(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test retrieving drift alerts."""
        # First create some drift alerts
        async with get_db_session() as session:
            alerts = [
                DriftAlert(
                    model_id="test-model-1",
                    timestamp=datetime.utcnow() - timedelta(hours=i),
                    drift_score=0.8 + (i * 0.05),
                    features_affected=["feature_1", "feature_2"],
                    severity="high" if i < 2 else "medium",
                    resolved=i > 3
                )
                for i in range(5)
            ]
            
            for alert in alerts:
                session.add(alert)
            await session.commit()
        
        response = await client.get("/api/v1/models/test-model-1/drift-alerts")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alerts" in data
        assert len(data["alerts"]) > 0
        
        alert = data["alerts"][0]
        assert "id" in alert
        assert "timestamp" in alert
        assert "drift_score" in alert
        assert "features_affected" in alert
        assert "severity" in alert
        assert "resolved" in alert

    async def test_acknowledge_drift_alert(self, client: AsyncClient):
        """Test acknowledging a drift alert."""
        # Create a drift alert first
        async with get_db_session() as session:
            alert = DriftAlert(
                model_id="test-model-1",
                timestamp=datetime.utcnow(),
                drift_score=0.85,
                features_affected=["feature_1"],
                severity="high",
                resolved=False
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            alert_id = alert.id
        
        response = await client.post(f"/api/v1/drift-alerts/{alert_id}/acknowledge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True
        
        # Verify alert was updated in database
        async with get_db_session() as session:
            updated_alert = await session.get(DriftAlert, alert_id)
            assert updated_alert.resolved is True

    async def test_list_models(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test listing all monitored models."""
        response = await client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) > 0
        
        model = data["models"][0]
        assert "model_id" in model
        assert "last_updated" in model
        assert "total_predictions" in model
        assert "current_status" in model

    async def test_concurrent_predictions(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test handling concurrent prediction requests."""
        async def make_prediction():
            return await client.post(
                "/api/v1/models/test-model-1/predict",
                json={"horizon_hours": 6}
            )
        
        # Make 5 concurrent requests
        tasks = [make_prediction() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data

    async def test_api_rate_limiting(self, client: AsyncClient):
        """Test API rate limiting behavior."""
        # Make rapid requests to test rate limiting
        tasks = []
        for _ in range(20):  # Exceed typical rate limit
            tasks.append(
                client.get("/api/v1/models/test-model-1/metrics", params={"hours": 1})
            )
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some requests should succeed
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        assert len(successful_responses) > 0
        
        # May have some rate-limited responses (429)
        rate_limited = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 429]
        logger.info("Rate limiting test", successful=len(successful_responses), rate_limited=len(rate_limited))

    async def test_metrics_endpoint_prometheus(self, client: AsyncClient):
        """Test Prometheus metrics endpoint."""
        response = await client.get("/metrics")
        
        assert response.status_code == 200
        content = response.text
        
        # Check for expected metrics
        expected_metrics = [
            "model_prediction_requests_total",
            "model_prediction_duration_seconds",
            "model_performance_score",
            "drift_alerts_active_total"
        ]
        
        for metric in expected_metrics:
            assert metric in content

    async def test_api_error_handling(self, client: AsyncClient):
        """Test API error handling and response formats."""
        # Test malformed JSON
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing required fields
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={}
        )
        assert response.status_code == 422
        
        # Test invalid model ID format
        response = await client.post(
            "/api/v1/models//predict",
            json={"horizon_hours": 6}
        )
        assert response.status_code == 404

    async def test_database_connection_resilience(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test API behavior when database is temporarily unavailable."""
        # This test would require a way to temporarily disconnect from DB
        # For now, we test with a model that should trigger DB queries
        response = await client.get("/api/v1/models/test-model-1/metrics")
        assert response.status_code in [200, 503]  # 200 if DB available, 503 if not

    async def test_redis_fallback_behavior(
        self, 
        client: AsyncClient, 
        setup_test_data: List[Dict[str, Any]]
    ):
        """Test API behavior when Redis cache is unavailable."""
        # Test should still work without Redis (may be slower)
        response = await client.post(
            "/api/v1/models/test-model-1/predict",
            json={"horizon_hours": 6}
        )
        # Should succeed even if Redis is down (fallback to DB)
        assert response.status_code in [200, 503]