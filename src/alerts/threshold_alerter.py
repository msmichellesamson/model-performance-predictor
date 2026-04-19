import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    timestamp: float
    model_id: str
    message: str

class ThresholdAlerter:
    def __init__(self, alert_thresholds: Dict[str, Dict[str, float]]):
        self.alert_thresholds = alert_thresholds
        self.logger = logging.getLogger(__name__)
        self.alert_history: List[Alert] = []
        self.cooldown_period = 300  # 5 minutes
        self.last_alert_time: Dict[str, float] = {}
    
    def check_thresholds(self, metrics: Dict[str, float], model_id: str) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        try:
            for metric_name, value in metrics.items():
                if metric_name not in self.alert_thresholds:
                    continue
                
                alert = self._evaluate_metric(metric_name, value, model_id)
                if alert and self._should_alert(alert):
                    alerts.append(alert)
                    self._log_alert(alert)
                    
        except Exception as e:
            self.logger.error(
                "Error checking thresholds",
                extra={
                    "error": str(e),
                    "model_id": model_id,
                    "metrics_count": len(metrics)
                }
            )
            
        return alerts
    
    def _evaluate_metric(self, metric: str, value: float, model_id: str) -> Optional[Alert]:
        """Evaluate single metric against thresholds"""
        thresholds = self.alert_thresholds[metric]
        
        severity = None
        threshold_value = None
        
        if value >= thresholds.get('critical', float('inf')):
            severity = AlertSeverity.CRITICAL
            threshold_value = thresholds['critical']
        elif value >= thresholds.get('high', float('inf')):
            severity = AlertSeverity.HIGH
            threshold_value = thresholds['high']
        elif value >= thresholds.get('medium', float('inf')):
            severity = AlertSeverity.MEDIUM
            threshold_value = thresholds['medium']
        elif value >= thresholds.get('low', float('inf')):
            severity = AlertSeverity.LOW
            threshold_value = thresholds['low']
        
        if severity:
            return Alert(
                metric=metric,
                value=value,
                threshold=threshold_value,
                severity=severity,
                timestamp=time.time(),
                model_id=model_id,
                message=f"{metric} ({value:.3f}) exceeded {severity.value} threshold ({threshold_value})"
            )
        
        return None
    
    def _should_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on cooldown"""
        alert_key = f"{alert.model_id}:{alert.metric}:{alert.severity.value}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.cooldown_period:
                return False
        
        self.last_alert_time[alert_key] = current_time
        return True
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert with structured data"""
        self.logger.warning(
            "Threshold alert triggered",
            extra={
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "severity": alert.severity.value,
                "model_id": alert.model_id,
                "message": alert.message
            }
        )
        self.alert_history.append(alert)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]