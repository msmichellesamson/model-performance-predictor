import pytest
import asyncio
import json
from httpx import AsyncClient
from src.main import app
from src.cache.redis_client import RedisClient
from src.core.drift_detector import DriftDetector
import numpy as np


@pytest.fixture
async def drift_detector():
    detector = DriftDetector()
    yield detector
    # Cleanup
    redis_client = RedisClient()
    await redis_client.delete_pattern("drift:*")


@pytest.mark.asyncio
async def test_drift_detection_workflow(drift_detector):
    """Test complete drift detection workflow with API integration."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Submit baseline data
        baseline_data = {
            "model_id": "test-model-v1",
            "features": np.random.normal(0, 1, 100).tolist(),
            "timestamp": "2024-01-01T10:00:00Z"
        }
        
        response = await client.post("/drift/baseline", json=baseline_data)
        assert response.status_code == 200
        
        # 2. Submit current data with drift
        drifted_data = {
            "model_id": "test-model-v1", 
            "features": np.random.normal(2, 1.5, 100).tolist(),  # Significant drift
            "timestamp": "2024-01-01T11:00:00Z"
        }
        
        response = await client.post("/drift/detect", json=drifted_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["has_drift"] is True
        assert result["drift_score"] > 0.5
        assert "population_stability_index" in result
        
        # 3. Verify drift status endpoint
        response = await client.get(f"/drift/status/{baseline_data['model_id']}")
        assert response.status_code == 200
        
        status = response.json()
        assert status["drift_detected"] is True
        assert len(status["recent_scores"]) > 0


@pytest.mark.asyncio 
async def test_drift_alert_integration(drift_detector):
    """Test that drift detection triggers alerts properly."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Submit high drift data
        high_drift_data = {
            "model_id": "alert-test-model",
            "features": np.random.normal(5, 3, 50).tolist(),
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        await client.post("/drift/baseline", json=high_drift_data)
        
        # Submit significantly different data
        different_data = {
            "model_id": "alert-test-model",
            "features": np.random.normal(-5, 0.5, 50).tolist(), 
            "timestamp": "2024-01-01T13:00:00Z"
        }
        
        response = await client.post("/drift/detect", json=different_data)
        assert response.status_code == 200
        
        # Check alerts endpoint
        response = await client.get("/alerts/active")
        assert response.status_code == 200
        
        alerts = response.json()["alerts"]
        drift_alerts = [a for a in alerts if a["alert_type"] == "drift"]
        assert len(drift_alerts) > 0
        assert drift_alerts[0]["model_id"] == "alert-test-model"