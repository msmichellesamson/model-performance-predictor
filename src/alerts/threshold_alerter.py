"""Performance threshold alerting system."""
import logging
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    model_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    timestamp: str
    description: str


class ThresholdAlerter:
    """Monitors performance metrics and triggers alerts when thresholds are exceeded."""
    
    def __init__(self):
        self.thresholds = {
            "accuracy_drop": {"warning": 0.05, "critical": 0.15},
            "latency_increase": {"warning": 100, "critical": 500},  # ms
            "error_rate": {"warning": 0.02, "critical": 0.10},
            "drift_score": {"warning": 0.3, "critical": 0.7}
        }
        self.active_alerts: Dict[str, List[Alert]] = {}
    
    def check_thresholds(self, model_id: str, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
                
            alert = self._evaluate_threshold(model_id, metric_name, value)
            if alert:
                alerts.append(alert)
                
        return alerts
    
    def _evaluate_threshold(self, model_id: str, metric_name: str, value: float) -> Optional[Alert]:
        """Evaluate a single metric against its threshold."""
        thresholds = self.thresholds[metric_name]
        
        severity = None
        threshold_value = None
        
        if value >= thresholds["critical"]:
            severity = AlertSeverity.CRITICAL
            threshold_value = thresholds["critical"]
        elif value >= thresholds["warning"]:
            severity = AlertSeverity.MEDIUM
            threshold_value = thresholds["warning"]
        
        if severity:
            return Alert(
                model_id=model_id,
                metric_name=metric_name,
                current_value=value,
                threshold_value=threshold_value,
                severity=severity,
                timestamp="2024-01-01T00:00:00Z",  # Would use datetime.utcnow()
                description=f"{metric_name} exceeded {severity.value} threshold: {value:.3f} > {threshold_value}"
            )
        
        return None
    
    def get_active_alerts(self, model_id: str) -> List[Alert]:
        """Get all active alerts for a model."""
        return self.active_alerts.get(model_id, [])
