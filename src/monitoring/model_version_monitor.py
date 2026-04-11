"""Model version drift monitoring for A/B testing scenarios."""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Prometheus metrics
model_version_requests = Counter(
    'model_version_requests_total',
    'Total requests per model version',
    ['version', 'endpoint']
)

version_performance_drift = Gauge(
    'model_version_performance_drift',
    'Performance drift between model versions',
    ['baseline_version', 'comparison_version', 'metric']
)

version_traffic_split = Gauge(
    'model_version_traffic_split_ratio',
    'Traffic split ratio between model versions',
    ['version']
)

@dataclass
class VersionMetrics:
    """Metrics for a specific model version."""
    version: str
    request_count: int
    avg_latency: float
    avg_confidence: float
    error_rate: float
    timestamp: datetime

class ModelVersionMonitor:
    """Monitor performance drift between different model versions."""
    
    def __init__(self, drift_threshold: float = 0.1, window_minutes: int = 60):
        self.drift_threshold = drift_threshold
        self.window_minutes = window_minutes
        self.version_metrics: Dict[str, List[VersionMetrics]] = defaultdict(list)
        self.baseline_version: Optional[str] = None
        
    def record_version_metrics(
        self,
        version: str,
        latency: float,
        confidence: float,
        error_occurred: bool,
        endpoint: str = "predict"
    ) -> None:
        """Record metrics for a specific model version."""
        model_version_requests.labels(version=version, endpoint=endpoint).inc()
        
        now = datetime.utcnow()
        
        # Clean old metrics outside window
        cutoff = now - timedelta(minutes=self.window_minutes)
        self.version_metrics[version] = [
            m for m in self.version_metrics[version] 
            if m.timestamp > cutoff
        ]
        
        # Calculate current metrics
        recent_metrics = self.version_metrics[version]
        request_count = len(recent_metrics) + 1
        
        if recent_metrics:
            avg_latency = (sum(m.avg_latency for m in recent_metrics) + latency) / request_count
            avg_confidence = (sum(m.avg_confidence for m in recent_metrics) + confidence) / request_count
            error_count = sum(1 for m in recent_metrics if m.error_rate > 0) + (1 if error_occurred else 0)
            error_rate = error_count / request_count
        else:
            avg_latency = latency
            avg_confidence = confidence
            error_rate = 1.0 if error_occurred else 0.0
            
        metrics = VersionMetrics(
            version=version,
            request_count=request_count,
            avg_latency=avg_latency,
            avg_confidence=avg_confidence,
            error_rate=error_rate,
            timestamp=now
        )
        
        self.version_metrics[version].append(metrics)
        
    def set_baseline_version(self, version: str) -> None:
        """Set the baseline version for drift comparison."""
        self.baseline_version = version
        logger.info(f"Set baseline version: {version}")
        
    def detect_version_drift(self) -> Dict[str, Dict[str, float]]:
        """Detect performance drift between versions."""
        if not self.baseline_version or self.baseline_version not in self.version_metrics:
            return {}
            
        baseline_metrics = self._get_current_metrics(self.baseline_version)
        if not baseline_metrics:
            return {}
            
        drift_results = {}
        
        for version, metrics_list in self.version_metrics.items():
            if version == self.baseline_version or not metrics_list:
                continue
                
            current_metrics = self._get_current_metrics(version)
            if not current_metrics:
                continue
                
            # Calculate drift for each metric
            latency_drift = abs(current_metrics.avg_latency - baseline_metrics.avg_latency) / baseline_metrics.avg_latency
            confidence_drift = abs(current_metrics.avg_confidence - baseline_metrics.avg_confidence) / max(baseline_metrics.avg_confidence, 0.01)
            error_drift = abs(current_metrics.error_rate - baseline_metrics.error_rate)
            
            drift_results[version] = {
                'latency_drift': latency_drift,
                'confidence_drift': confidence_drift,
                'error_drift': error_drift
            }
            
            # Update Prometheus metrics
            version_performance_drift.labels(
                baseline_version=self.baseline_version,
                comparison_version=version,
                metric='latency'
            ).set(latency_drift)
            
            version_performance_drift.labels(
                baseline_version=self.baseline_version,
                comparison_version=version,
                metric='confidence'
            ).set(confidence_drift)
            
            version_performance_drift.labels(
                baseline_version=self.baseline_version,
                comparison_version=version,
                metric='error_rate'
            ).set(error_drift)
            
        return drift_results
        
    def update_traffic_split(self, traffic_splits: Dict[str, float]) -> None:
        """Update traffic split ratios for monitoring."""
        for version, ratio in traffic_splits.items():
            version_traffic_split.labels(version=version).set(ratio)
            
    def get_active_versions(self) -> Set[str]:
        """Get currently active model versions."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
        active_versions = set()
        
        for version, metrics_list in self.version_metrics.items():
            if any(m.timestamp > cutoff for m in metrics_list):
                active_versions.add(version)
                
        return active_versions
        
    def _get_current_metrics(self, version: str) -> Optional[VersionMetrics]:
        """Get the most recent metrics for a version."""
        if version not in self.version_metrics or not self.version_metrics[version]:
            return None
            
        return self.version_metrics[version][-1]