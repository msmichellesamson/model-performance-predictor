from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ..core.predictor import ModelPerformancePredictor
from ..core.metrics_collector import MetricsCollector
from ..cache.redis_client import RedisClient

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    model_id: str
    feature_names: list[str]
    prediction_confidence: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    input_size: Optional[int] = None

class PredictionResponse(BaseModel):
    model_id: str
    predicted_performance_drop: float
    confidence_score: float
    risk_level: str  # "low", "medium", "high"
    contributing_factors: list[str]
    timestamp: datetime

def get_predictor() -> ModelPerformancePredictor:
    return ModelPerformancePredictor()

def get_metrics_collector() -> MetricsCollector:
    return MetricsCollector()

def get_redis_client() -> RedisClient:
    return RedisClient()

@router.post("/predict", response_model=PredictionResponse)
async def predict_performance(
    request: PredictionRequest,
    predictor: ModelPerformancePredictor = Depends(get_predictor),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Predict model performance degradation based on current metrics."""
    try:
        # Get historical metrics for the model
        historical_metrics = await metrics_collector.get_model_metrics(
            request.model_id, hours=24
        )
        
        if not historical_metrics:
            raise HTTPException(
                status_code=404, 
                detail=f"No historical metrics found for model {request.model_id}"
            )
        
        # Build current inference context
        current_context = {
            "feature_names": request.feature_names,
            "prediction_confidence": request.prediction_confidence,
            "inference_latency_ms": request.inference_latency_ms,
            "input_size": request.input_size,
            "timestamp": datetime.utcnow()
        }
        
        # Generate prediction
        prediction_result = await predictor.predict_performance_drop(
            model_id=request.model_id,
            historical_metrics=historical_metrics,
            current_context=current_context
        )
        
        # Determine risk level
        risk_level = "low"
        if prediction_result["performance_drop"] > 0.3:
            risk_level = "high"
        elif prediction_result["performance_drop"] > 0.15:
            risk_level = "medium"
        
        # Cache result for 5 minutes
        cache_key = f"prediction:{request.model_id}:{int(datetime.utcnow().timestamp() // 300)}"
        await redis_client.setex(
            cache_key, 
            300, 
            prediction_result["performance_drop"]
        )
        
        logger.info(
            f"Performance prediction for model {request.model_id}: "
            f"{prediction_result['performance_drop']:.3f} ({risk_level} risk)"
        )
        
        return PredictionResponse(
            model_id=request.model_id,
            predicted_performance_drop=prediction_result["performance_drop"],
            confidence_score=prediction_result["confidence"],
            risk_level=risk_level,
            contributing_factors=prediction_result["factors"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error predicting performance for model {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/risk")
async def get_model_risk_level(
    model_id: str,
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Get current risk level for a model from cache."""
    try:
        cache_key = f"prediction:{model_id}:*"
        cached_prediction = await redis_client.get(cache_key)
        
        if not cached_prediction:
            raise HTTPException(
                status_code=404, 
                detail=f"No recent prediction found for model {model_id}"
            )
        
        performance_drop = float(cached_prediction)
        risk_level = "low"
        if performance_drop > 0.3:
            risk_level = "high"
        elif performance_drop > 0.15:
            risk_level = "medium"
            
        return {"model_id": model_id, "risk_level": risk_level}
        
    except Exception as e:
        logger.error(f"Error getting risk level for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))