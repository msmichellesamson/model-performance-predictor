"""Risk assessment API endpoint."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

from ..core.predictor import PerformancePredictor
from ..core.metrics_collector import MetricsCollector
from ..cache.redis_client import RedisClient

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/risk/{model_id}")
async def get_degradation_risk(
    model_id: str,
    predictor: PerformancePredictor = Depends(lambda: PerformancePredictor()),
    metrics_collector: MetricsCollector = Depends(lambda: MetricsCollector()),
    redis_client: RedisClient = Depends(lambda: RedisClient())
) -> Dict[str, Any]:
    """Get current degradation risk level for a model."""
    try:
        # Get cached risk if available
        cache_key = f"risk:{model_id}"
        cached_risk = await redis_client.get_json(cache_key)
        if cached_risk:
            logger.info(f"Returning cached risk for model {model_id}")
            return cached_risk

        # Collect recent metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        metrics = await metrics_collector.get_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for model {model_id}")

        # Calculate risk score
        risk_factors = {
            "drift_score": metrics.get("feature_drift_score", 0.0),
            "accuracy_drop": metrics.get("accuracy_drop_percent", 0.0),
            "latency_spike": metrics.get("latency_p99_spike", 0.0),
            "confidence_drop": metrics.get("confidence_drop", 0.0)
        }
        
        # Simple weighted risk calculation
        risk_score = (
            risk_factors["drift_score"] * 0.3 +
            risk_factors["accuracy_drop"] * 0.4 +
            risk_factors["latency_spike"] * 0.2 +
            risk_factors["confidence_drop"] * 0.1
        )
        
        risk_level = "low"
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
            
        result = {
            "model_id": model_id,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl_seconds": 300
        }
        
        # Cache for 5 minutes
        await redis_client.set_json(cache_key, result, ttl=300)
        
        logger.info(f"Calculated risk score {risk_score} for model {model_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating risk for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
