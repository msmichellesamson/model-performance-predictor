from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging

from ..core.drift_detector import DriftDetector
from ..cache.redis_client import RedisClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/drift", tags=["drift"])

class DriftRequest(BaseModel):
    feature_values: List[float]
    model_version: str
    timestamp: Optional[int] = None

class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    threshold: float
    features_affected: List[int]

@router.post("/detect", response_model=DriftResponse)
async def detect_drift(
    request: DriftRequest,
    detector: DriftDetector = Depends(lambda: DriftDetector()),
    cache: RedisClient = Depends(lambda: RedisClient())
):
    """Detect feature drift in real-time inference data."""
    try:
        # Check cache for recent drift analysis
        cache_key = f"drift:{request.model_version}:{hash(tuple(request.feature_values))}"
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached drift result for model {request.model_version}")
            return DriftResponse(**cached_result)
        
        # Perform drift detection
        drift_result = detector.detect_drift(
            feature_values=request.feature_values,
            model_version=request.model_version,
            timestamp=request.timestamp
        )
        
        response = DriftResponse(
            drift_detected=drift_result['drift_detected'],
            drift_score=drift_result['drift_score'],
            threshold=drift_result['threshold'],
            features_affected=drift_result['features_affected']
        )
        
        # Cache result for 5 minutes
        await cache.setex(cache_key, 300, response.dict())
        
        logger.info(f"Drift detection completed: score={drift_result['drift_score']:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@router.get("/status/{model_version}")
async def get_drift_status(
    model_version: str,
    detector: DriftDetector = Depends(lambda: DriftDetector())
):
    """Get current drift status for a model version."""
    try:
        status = detector.get_model_drift_status(model_version)
        return {
            "model_version": model_version,
            "drift_status": status["status"],
            "last_check": status["last_check"],
            "drift_score_trend": status["trend"]
        }
    except Exception as e:
        logger.error(f"Failed to get drift status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drift status")
