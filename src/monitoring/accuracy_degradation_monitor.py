import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
from datetime import datetime, timedelta

@dataclass
class AccuracyWindow:
    timestamp: datetime
    accuracy: float
    sample_count: int

class AccuracyDegradationMonitor:
    """Monitors ML model accuracy trends and detects degradation patterns."""
    
    def __init__(self, 
                 window_size: int = 24,  # hours
                 degradation_threshold: float = 0.05,  # 5% drop
                 min_samples_per_window: int = 100):
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.min_samples_per_window = min_samples_per_window
        self.accuracy_windows: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
    
    def add_accuracy_sample(self, accuracy: float, sample_count: int) -> None:
        """Add new accuracy measurement."""
        window = AccuracyWindow(
            timestamp=datetime.utcnow(),
            accuracy=accuracy,
            sample_count=sample_count
        )
        self.accuracy_windows.append(window)
        self.logger.debug(f"Added accuracy sample: {accuracy:.3f} ({sample_count} samples)")
    
    def detect_degradation(self) -> Optional[Dict]:
        """Detect if model accuracy is degrading over time."""
        if len(self.accuracy_windows) < 3:
            return None
        
        # Filter windows with sufficient samples
        valid_windows = [
            w for w in self.accuracy_windows 
            if w.sample_count >= self.min_samples_per_window
        ]
        
        if len(valid_windows) < 3:
            return None
        
        # Calculate trend using linear regression
        accuracies = [w.accuracy for w in valid_windows]
        trend_slope = self._calculate_trend(accuracies)
        
        # Check for significant degradation
        recent_avg = np.mean(accuracies[-3:])
        baseline_avg = np.mean(accuracies[:3])
        degradation_pct = (baseline_avg - recent_avg) / baseline_avg
        
        if degradation_pct > self.degradation_threshold:
            return {
                'type': 'accuracy_degradation',
                'severity': 'high' if degradation_pct > 0.1 else 'medium',
                'degradation_percentage': degradation_pct,
                'trend_slope': trend_slope,
                'recent_accuracy': recent_avg,
                'baseline_accuracy': baseline_avg,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope."""
        x = np.arange(len(values))
        y = np.array(values)
        return np.polyfit(x, y, 1)[0]  # slope coefficient
    
    def get_metrics(self) -> Dict:
        """Get current monitoring metrics."""
        if not self.accuracy_windows:
            return {'status': 'no_data'}
        
        recent_windows = list(self.accuracy_windows)[-6:]  # last 6 hours
        valid_windows = [
            w for w in recent_windows 
            if w.sample_count >= self.min_samples_per_window
        ]
        
        if not valid_windows:
            return {'status': 'insufficient_samples'}
        
        accuracies = [w.accuracy for w in valid_windows]
        return {
            'current_accuracy': accuracies[-1],
            'accuracy_trend': self._calculate_trend(accuracies),
            'windows_monitored': len(valid_windows),
            'total_samples': sum(w.sample_count for w in valid_windows)
        }