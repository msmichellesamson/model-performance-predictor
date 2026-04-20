from typing import Dict, List, Optional
import psutil
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass
class ResourceMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]
    disk_io_read: int
    disk_io_write: int
    network_bytes_sent: int
    network_bytes_recv: int

class ResourceMonitor:
    """Monitor system resource utilization during ML inference."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self._baseline_metrics: Optional[ResourceMetrics] = None
        self._metrics_history: List[ResourceMetrics] = []
        
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        try:
            # Get CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            
            # Try to get GPU utilization (nvidia-ml-py)
            gpu_percent = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = gpu_util.gpu
            except (ImportError, Exception):
                pass  # GPU monitoring optional
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_percent=gpu_percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_bytes_sent=net_io.bytes_sent if net_io else 0,
                network_bytes_recv=net_io.bytes_recv if net_io else 0
            )
            
            self._metrics_history.append(metrics)
            # Keep last 1000 metrics
            if len(self._metrics_history) > 1000:
                self._metrics_history.pop(0)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            raise
    
    def detect_resource_spike(self, current_metrics: ResourceMetrics, 
                            cpu_threshold: float = 80.0,
                            memory_threshold: float = 85.0) -> Dict[str, bool]:
        """Detect if resource usage has spiked beyond thresholds."""
        return {
            'cpu_spike': current_metrics.cpu_percent > cpu_threshold,
            'memory_spike': current_metrics.memory_percent > memory_threshold,
            'gpu_spike': (current_metrics.gpu_percent or 0) > 90.0
        }
    
    def get_resource_trend(self, window_minutes: int = 5) -> Dict[str, float]:
        """Calculate resource usage trend over time window."""
        if len(self._metrics_history) < 2:
            return {'cpu_trend': 0.0, 'memory_trend': 0.0}
        
        cutoff_time = datetime.now(timezone.utc)
        recent_metrics = [
            m for m in self._metrics_history[-100:] 
            if (cutoff_time - m.timestamp).seconds <= window_minutes * 60
        ]
        
        if len(recent_metrics) < 2:
            return {'cpu_trend': 0.0, 'memory_trend': 0.0}
        
        # Simple linear trend calculation
        cpu_trend = recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent
        memory_trend = recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent
        
        return {
            'cpu_trend': cpu_trend / len(recent_metrics),
            'memory_trend': memory_trend / len(recent_metrics)
        }
