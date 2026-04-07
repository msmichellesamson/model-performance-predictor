from typing import Dict, List
import time
import numpy as np
from prometheus_client import Histogram, Counter
from dataclasses import dataclass


@dataclass
class LatencyMetrics:
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    count: int


class LatencyMonitor:
    """Monitor prediction latency with percentile tracking."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: List[float] = []
        
        # Prometheus metrics
        self.latency_histogram = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        self.high_latency_counter = Counter(
            'high_latency_predictions_total',
            'Total high latency predictions (>1s)'
        )
    
    def record_latency(self, latency: float) -> None:
        """Record a prediction latency measurement."""
        self._latencies.append(latency)
        
        # Keep sliding window
        if len(self._latencies) > self.window_size:
            self._latencies.pop(0)
        
        # Update Prometheus metrics
        self.latency_histogram.observe(latency)
        
        if latency > 1.0:  # 1 second threshold
            self.high_latency_counter.inc()
    
    def get_metrics(self) -> LatencyMetrics:
        """Get current latency percentile metrics."""
        if not self._latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0)
        
        latencies = np.array(self._latencies)
        
        return LatencyMetrics(
            p50=float(np.percentile(latencies, 50)),
            p90=float(np.percentile(latencies, 90)),
            p95=float(np.percentile(latencies, 95)),
            p99=float(np.percentile(latencies, 99)),
            mean=float(np.mean(latencies)),
            count=len(latencies)
        )
    
    def is_degrading(self, threshold_p95: float = 0.5) -> bool:
        """Check if latency is degrading based on P95."""
        metrics = self.get_metrics()
        return metrics.count > 10 and metrics.p95 > threshold_p95
