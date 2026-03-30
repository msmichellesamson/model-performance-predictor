from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from ..core.predictor import PerformancePredictor
from ..db.models import ModelMetrics

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)

@router.get("/model/{model_id}")
async def get_model_health(model_id: str) -> Dict[str, Any]:
    """Get health status for a specific model."""
    try:
        predictor = PerformancePredictor()
        
        # Get recent metrics (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = ModelMetrics.get_recent(model_id, cutoff)
        
        if not recent_metrics:
            raise HTTPException(status_code=404, detail="No recent metrics found")
        
        # Calculate health score
        health_score = predictor.calculate_health_score(recent_metrics)
        
        # Determine status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.6:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "model_id": model_id,
            "status": status,
            "health_score": health_score,
            "last_updated": datetime.utcnow().isoformat(),
            "metrics_count": len(recent_metrics)
        }
        
    except Exception as e:
        logger.error(f"Health check failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/models")
async def get_all_models_health() -> Dict[str, Any]:
    """Get health status for all monitored models."""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        active_models = ModelMetrics.get_active_models(cutoff)
        
        health_statuses = []
        for model_id in active_models:
            try:
                health_data = await get_model_health(model_id)
                health_statuses.append(health_data)
            except HTTPException:
                continue
        
        healthy_count = sum(1 for h in health_statuses if h["status"] == "healthy")
        
        return {
            "total_models": len(health_statuses),
            "healthy_models": healthy_count,
            "unhealthy_models": len(health_statuses) - healthy_count,
            "models": health_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get all models health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health status")