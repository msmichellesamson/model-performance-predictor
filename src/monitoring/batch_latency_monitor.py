"""Batch prediction latency monitoring for performance degradation detection."""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
from threading import Lock

import prometheus_client


@dataclass
class BatchMetrics:
    """Batch prediction metrics."""
    batch_id: str
    batch_size: int
    start_time: float
    end_time: float
    prediction_count: int
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        return self.prediction_count / self.duration if self.duration > 0 else 0.0


class BatchLatencyMonitor:
    """Monitor batch prediction latency and detect performance degradation."""
    
    def __init__(self, window_size: int = 100, degradation_threshold: float = 2.0):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self._metrics = deque(maxlen=window_size)
        self._lock = Lock()
        
        # Prometheus metrics
        self.batch_duration_histogram = prometheus_client.Histogram(
            'batch_prediction_duration_seconds',
            'Time spent processing prediction batches',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.batch_throughput_gauge = prometheus_client.Gauge(
            'batch_prediction_throughput',
            'Predictions processed per second in batch'
        )
        
        self.degradation_alert = prometheus_client.Counter(
            'batch_latency_degradation_alerts_total',
            'Number of batch latency degradation alerts'
        )
    
    def record_batch(self, metrics: BatchMetrics) -> None:
        """Record batch metrics and check for degradation."""
        with self._lock:
            self._metrics.append(metrics)
            
        # Update Prometheus metrics
        self.batch_duration_histogram.observe(metrics.duration)
        self.batch_throughput_gauge.set(metrics.throughput)
        
        # Check for degradation
        if self._is_degraded():
            self.degradation_alert.inc()
    
    def _is_degraded(self) -> bool:
        """Check if current performance is degraded compared to baseline."""
        if len(self._metrics) < 10:  # Need minimum samples
            return False
            
        recent_metrics = list(self._metrics)[-5:]  # Last 5 batches
        baseline_metrics = list(self._metrics)[:-5]  # All but last 5
        
        if not baseline_metrics:
            return False
            
        recent_avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
        baseline_avg_duration = sum(m.duration for m in baseline_metrics) / len(baseline_metrics)
        
        return recent_avg_duration > baseline_avg_duration * self.degradation_threshold
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        with self._lock:
            if not self._metrics:
                return {}
                
            durations = [m.duration for m in self._metrics]
            throughputs = [m.throughput for m in self._metrics]
            
            return {
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'avg_throughput': sum(throughputs) / len(throughputs),
                'total_batches': len(self._metrics)
            }