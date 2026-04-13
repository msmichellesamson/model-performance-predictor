import logging
import psutil
from typing import Dict, Optional
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Prometheus metrics
memory_usage_gauge = Gauge('model_memory_usage_bytes', 'Memory usage during inference', ['model_id', 'version'])
memory_threshold_exceeded = Counter('memory_threshold_exceeded_total', 'Memory threshold violations', ['model_id', 'version'])

class MemoryMonitor:
    """Monitors memory usage during model inference operations."""
    
    def __init__(self, threshold_mb: float = 1024.0):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.baseline_memory: Optional[float] = None
        logger.info(f"Memory monitor initialized with threshold: {threshold_mb}MB")
    
    def start_monitoring(self, model_id: str, version: str) -> Dict[str, float]:
        """Start memory monitoring for inference session."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            current_memory = memory_info.rss
            if self.baseline_memory is None:
                self.baseline_memory = current_memory
            
            # Calculate memory increase from baseline
            memory_delta = current_memory - self.baseline_memory
            
            # Update Prometheus metrics
            memory_usage_gauge.labels(model_id=model_id, version=version).set(current_memory)
            
            # Check threshold violation
            if memory_delta > self.threshold_bytes:
                memory_threshold_exceeded.labels(model_id=model_id, version=version).inc()
                logger.warning(
                    f"Memory threshold exceeded: {memory_delta / 1024 / 1024:.2f}MB "
                    f"(threshold: {self.threshold_bytes / 1024 / 1024:.2f}MB)"
                )
            
            return {
                'current_memory_mb': current_memory / 1024 / 1024,
                'memory_delta_mb': memory_delta / 1024 / 1024,
                'threshold_exceeded': memory_delta > self.threshold_bytes
            }
            
        except psutil.Error as e:
            logger.error(f"Failed to monitor memory: {e}")
            return {'error': str(e)}
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current system memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_percent': memory.percent
            }
        except psutil.Error as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    def reset_baseline(self):
        """Reset memory baseline for new monitoring session."""
        self.baseline_memory = None
        logger.info("Memory baseline reset")