import logging
import time
from typing import Dict, List, Optional
import psutil
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    timestamp: float
    used_mb: float
    available_mb: float
    percent: float
    process_rss_mb: float

class MemoryMonitor:
    """Monitors memory usage with batch processing for efficiency."""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 30.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.metrics_batch: List[MemoryMetrics] = []
        self.last_flush = time.time()
        self._lock = Lock()
    
    def collect_metrics(self) -> MemoryMetrics:
        """Collect current memory metrics."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics = MemoryMetrics(
                timestamp=time.time(),
                used_mb=memory.used / 1024 / 1024,
                available_mb=memory.available / 1024 / 1024,
                percent=memory.percent,
                process_rss_mb=process_memory.rss / 1024 / 1024
            )
            
            self._add_to_batch(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect memory metrics: {e}")
            raise
    
    def _add_to_batch(self, metrics: MemoryMetrics) -> None:
        """Add metrics to batch and flush if needed."""
        with self._lock:
            self.metrics_batch.append(metrics)
            
            should_flush = (
                len(self.metrics_batch) >= self.batch_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush_batch()
    
    def _flush_batch(self) -> None:
        """Process and clear the current batch."""
        if not self.metrics_batch:
            return
            
        try:
            # Calculate batch statistics
            avg_memory_percent = sum(m.percent for m in self.metrics_batch) / len(self.metrics_batch)
            max_process_rss = max(m.process_rss_mb for m in self.metrics_batch)
            
            logger.info(f"Memory batch processed: {len(self.metrics_batch)} samples, "
                       f"avg_memory_percent={avg_memory_percent:.2f}, "
                       f"max_process_rss={max_process_rss:.2f}MB")
            
            self.metrics_batch.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Failed to flush memory metrics batch: {e}")
            self.metrics_batch.clear()
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage summary."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'memory_percent': memory.percent,
                'process_rss_mb': process.memory_info().rss / 1024 / 1024,
                'available_gb': memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Failed to get current memory usage: {e}")
            return {}
    
    def force_flush(self) -> None:
        """Force flush current batch."""
        with self._lock:
            self._flush_batch()
