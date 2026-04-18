"""Batch processing metrics monitor for tracking throughput and queue depths."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
from prometheus_client import Histogram, Counter, Gauge


@dataclass
class BatchMetrics:
    """Batch processing metrics."""
    batch_size: int
    processing_time: float
    queue_depth: int
    throughput: float
    error_count: int


class BatchProcessingMonitor:
    """Monitor batch processing performance and queue metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._batch_times = deque(maxlen=window_size)
        self._batch_sizes = deque(maxlen=window_size)
        
        # Prometheus metrics
        self.batch_duration = Histogram(
            'batch_processing_duration_seconds',
            'Time spent processing batches',
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        self.batch_size_metric = Histogram(
            'batch_size_total',
            'Number of items in batch',
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        )
        self.queue_depth = Gauge(
            'processing_queue_depth_total',
            'Current queue depth'
        )
        self.throughput_metric = Gauge(
            'batch_throughput_items_per_second',
            'Current batch processing throughput'
        )
        self.batch_errors = Counter(
            'batch_processing_errors_total',
            'Total batch processing errors'
        )
    
    def record_batch(self, batch_size: int, processing_time: float, 
                    queue_depth: int, error_count: int = 0) -> None:
        """Record batch processing metrics."""
        self._batch_times.append(processing_time)
        self._batch_sizes.append(batch_size)
        
        # Update Prometheus metrics
        self.batch_duration.observe(processing_time)
        self.batch_size_metric.observe(batch_size)
        self.queue_depth.set(queue_depth)
        
        if error_count > 0:
            self.batch_errors.inc(error_count)
        
        # Calculate throughput
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.throughput_metric.set(throughput)
    
    def get_metrics(self) -> Optional[BatchMetrics]:
        """Get current batch processing metrics."""
        if not self._batch_times:
            return None
        
        recent_time = self._batch_times[-1]
        recent_size = self._batch_sizes[-1]
        
        # Calculate average throughput over window
        total_items = sum(self._batch_sizes)
        total_time = sum(self._batch_times)
        avg_throughput = total_items / total_time if total_time > 0 else 0
        
        return BatchMetrics(
            batch_size=recent_size,
            processing_time=recent_time,
            queue_depth=int(self.queue_depth._value.get()),
            throughput=avg_throughput,
            error_count=int(self.batch_errors._value.get())
        )