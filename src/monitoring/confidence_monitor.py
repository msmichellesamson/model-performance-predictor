"""Model confidence monitoring for prediction quality tracking."""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Confidence metrics for model predictions."""
    timestamp: float
    confidence_score: float
    prediction_id: str
    model_version: str
    low_confidence_count: int = 0

class ConfidenceMonitor:
    """Monitor model prediction confidence scores and detect quality degradation."""
    
    def __init__(self, 
                 window_size: int = 1000,
                 low_confidence_threshold: float = 0.6,
                 alert_threshold: float = 0.3):
        self.window_size = window_size
        self.low_confidence_threshold = low_confidence_threshold
        self.alert_threshold = alert_threshold  # Alert if >30% predictions are low confidence
        self.confidence_history: deque = deque(maxlen=window_size)
        self._metrics_cache: Optional[Dict] = None
        self._cache_timestamp = 0
        
    def record_prediction(self, prediction_id: str, confidence_score: float, 
                         model_version: str) -> None:
        """Record a prediction confidence score."""
        try:
            metrics = ConfidenceMetrics(
                timestamp=time.time(),
                confidence_score=confidence_score,
                prediction_id=prediction_id,
                model_version=model_version
            )
            
            self.confidence_history.append(metrics)
            self._invalidate_cache()
            
            logger.debug(f"Recorded confidence {confidence_score:.3f} for {prediction_id}")
            
        except Exception as e:
            logger.error(f"Failed to record confidence: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current confidence metrics with caching."""
        current_time = time.time()
        
        # Use cache if valid (within 10 seconds)
        if self._metrics_cache and (current_time - self._cache_timestamp) < 10:
            return self._metrics_cache
            
        if not self.confidence_history:
            return {
                'mean_confidence': 0.0,
                'low_confidence_ratio': 0.0,
                'total_predictions': 0,
                'quality_alert': False
            }
        
        confidences = [m.confidence_score for m in self.confidence_history]
        low_confidence_count = sum(1 for c in confidences if c < self.low_confidence_threshold)
        
        metrics = {
            'mean_confidence': statistics.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'std_confidence': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            'low_confidence_ratio': low_confidence_count / len(confidences),
            'total_predictions': len(confidences),
            'quality_alert': (low_confidence_count / len(confidences)) > self.alert_threshold
        }
        
        # Cache results
        self._metrics_cache = metrics
        self._cache_timestamp = current_time
        
        return metrics
    
    def _invalidate_cache(self) -> None:
        """Invalidate metrics cache."""
        self._metrics_cache = None
        self._cache_timestamp = 0