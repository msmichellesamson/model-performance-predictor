"""Alerts API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict
import logging

from ..alerts.threshold_alerter import ThresholdAlerter, Alert
from ..core.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

alerter = ThresholdAlerter()
metrics_collector = MetricsCollector()


@router.get("/models/{model_id}", response_model=List[Dict])
async def get_model_alerts(model_id: str):
    """Get all active alerts for a specific model."""
    try:
        alerts = alerter.get_active_alerts(model_id)
        return [
            {
                "model_id": alert.model_id,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp,
                "description": alert.description
            }
            for alert in alerts
        ]
    except Exception as e:
        logger.error(f"Failed to get alerts for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/models/{model_id}/check")
async def check_model_thresholds(model_id: str):
    """Check current metrics against thresholds and return any new alerts."""
    try:
        # Get latest metrics for the model
        metrics = await metrics_collector.get_latest_metrics(model_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics found for model")
        
        # Check thresholds
        alerts = alerter.check_thresholds(model_id, metrics)
        
        return {
            "model_id": model_id,
            "alerts_count": len(alerts),
            "alerts": [
                {
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "severity": alert.severity.value,
                    "description": alert.description
                }
                for alert in alerts
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check thresholds for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check thresholds")
