import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class QualityMetrics:
    """Data quality metrics for model inputs."""
    null_rate: float
    outlier_rate: float
    schema_violations: int
    feature_completeness: Dict[str, float]
    timestamp: datetime

class DataQualityMonitor:
    """Monitors data quality degradation in model inputs."""
    
    def __init__(self, outlier_threshold: float = 3.0, null_threshold: float = 0.1):
        self.logger = logging.getLogger(__name__)
        self.outlier_threshold = outlier_threshold
        self.null_threshold = null_threshold
        self.baseline_stats: Dict[str, Tuple[float, float]] = {}  # mean, std
        
    def set_baseline(self, features: Dict[str, np.ndarray]) -> None:
        """Set baseline statistics for quality monitoring."""
        self.baseline_stats = {
            name: (np.mean(values), np.std(values))
            for name, values in features.items()
            if np.issubdtype(values.dtype, np.number)
        }
        self.logger.info(f"Baseline set for {len(self.baseline_stats)} features")
    
    def calculate_quality_metrics(self, features: Dict[str, np.ndarray]) -> QualityMetrics:
        """Calculate current data quality metrics."""
        total_features = len(features)
        total_samples = len(next(iter(features.values())))
        
        null_count = sum(
            np.isnan(values).sum() if np.issubdtype(values.dtype, np.number)
            else (values == None).sum() if hasattr(values, '__len__')
            else 0
            for values in features.values()
        )
        null_rate = null_count / (total_features * total_samples)
        
        outlier_count = 0
        completeness = {}
        
        for name, values in features.items():
            if np.issubdtype(values.dtype, np.number):
                # Calculate completeness
                valid_count = np.sum(~np.isnan(values))
                completeness[name] = valid_count / len(values)
                
                # Detect outliers using baseline if available
                if name in self.baseline_stats:
                    mean, std = self.baseline_stats[name]
                    z_scores = np.abs((values - mean) / (std + 1e-8))
                    outlier_count += np.sum(z_scores > self.outlier_threshold)
            else:
                completeness[name] = 1.0  # Assume categorical features are complete
        
        outlier_rate = outlier_count / (total_samples * len(self.baseline_stats))
        
        return QualityMetrics(
            null_rate=null_rate,
            outlier_rate=outlier_rate,
            schema_violations=0,  # TODO: implement schema validation
            feature_completeness=completeness,
            timestamp=datetime.utcnow()
        )
    
    def detect_quality_degradation(self, current_metrics: QualityMetrics) -> Dict[str, bool]:
        """Detect if data quality has degraded significantly."""
        alerts = {
            'high_null_rate': current_metrics.null_rate > self.null_threshold,
            'high_outlier_rate': current_metrics.outlier_rate > 0.05,  # 5% outliers
            'low_completeness': any(
                completeness < 0.95 for completeness in current_metrics.feature_completeness.values()
            )
        }
        
        if any(alerts.values()):
            self.logger.warning(f"Data quality degradation detected: {alerts}")
        
        return alerts
