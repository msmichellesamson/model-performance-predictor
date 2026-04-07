"""Feature importance drift detection for ML models."""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureDriftAlert:
    feature_name: str
    drift_score: float
    threshold: float
    timestamp: datetime
    baseline_importance: float
    current_importance: float

class FeatureDriftMonitor:
    """Monitor feature importance drift in ML models."""
    
    def __init__(self, 
                 drift_threshold: float = 0.15,
                 min_samples: int = 100,
                 baseline_window: int = 1000):
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.baseline_window = baseline_window
        self.baseline_importance: Dict[str, float] = {}
        self.current_samples: List[Dict[str, float]] = []
        
    def update_baseline(self, feature_importance: Dict[str, float]) -> None:
        """Update baseline feature importance."""
        self.baseline_importance = feature_importance.copy()
        logger.info(f"Updated baseline with {len(feature_importance)} features")
        
    def add_sample(self, feature_importance: Dict[str, float]) -> None:
        """Add a new feature importance sample."""
        self.current_samples.append(feature_importance)
        
        # Keep sliding window
        if len(self.current_samples) > self.baseline_window:
            self.current_samples.pop(0)
            
    def calculate_drift_scores(self) -> Dict[str, float]:
        """Calculate drift scores for all features."""
        if len(self.current_samples) < self.min_samples:
            return {}
            
        if not self.baseline_importance:
            logger.warning("No baseline set, cannot calculate drift")
            return {}
            
        # Calculate current average importance
        current_avg = {}
        for feature in self.baseline_importance.keys():
            values = [s.get(feature, 0.0) for s in self.current_samples]
            current_avg[feature] = np.mean(values)
            
        # Calculate drift scores using Jensen-Shannon divergence approximation
        drift_scores = {}
        for feature in self.baseline_importance.keys():
            baseline_val = self.baseline_importance[feature]
            current_val = current_avg[feature]
            
            # Simple drift score: normalized absolute difference
            if baseline_val > 0:
                drift_scores[feature] = abs(current_val - baseline_val) / baseline_val
            else:
                drift_scores[feature] = abs(current_val)
                
        return drift_scores
        
    def detect_drift(self) -> List[FeatureDriftAlert]:
        """Detect feature importance drift."""
        drift_scores = self.calculate_drift_scores()
        alerts = []
        
        for feature, score in drift_scores.items():
            if score > self.drift_threshold:
                alert = FeatureDriftAlert(
                    feature_name=feature,
                    drift_score=score,
                    threshold=self.drift_threshold,
                    timestamp=datetime.utcnow(),
                    baseline_importance=self.baseline_importance[feature],
                    current_importance=np.mean([s.get(feature, 0.0) 
                                              for s in self.current_samples])
                )
                alerts.append(alert)
                logger.warning(f"Feature drift detected: {feature} (score: {score:.3f})")
                
        return alerts
        
    def get_monitoring_metrics(self) -> Dict[str, float]:
        """Get metrics for monitoring dashboard."""
        drift_scores = self.calculate_drift_scores()
        
        return {
            'max_drift_score': max(drift_scores.values()) if drift_scores else 0.0,
            'avg_drift_score': np.mean(list(drift_scores.values())) if drift_scores else 0.0,
            'features_drifting': sum(1 for s in drift_scores.values() if s > self.drift_threshold),
            'total_features': len(drift_scores),
            'sample_count': len(self.current_samples)
        }