#!/usr/bin/env python3
"""Model accuracy degradation monitoring."""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass
class AccuracyMetric:
    """Single accuracy measurement."""
    timestamp: datetime
    accuracy: float
    sample_count: int
    model_version: str


class AccuracyMonitor:
    """Monitors model accuracy degradation over time."""
    
    def __init__(
        self,
        baseline_accuracy: float,
        degradation_threshold: float = 0.05,
        window_hours: int = 24,
        min_samples: int = 100
    ):
        self.baseline_accuracy = baseline_accuracy
        self.degradation_threshold = degradation_threshold
        self.window_hours = window_hours
        self.min_samples = min_samples
        self.metrics: List[AccuracyMetric] = []
        self.logger = logging.getLogger(__name__)
    
    def record_accuracy(
        self,
        accuracy: float,
        sample_count: int,
        model_version: str
    ) -> None:
        """Record new accuracy measurement."""
        if not 0 <= accuracy <= 1:
            raise ValueError(f"Accuracy must be between 0 and 1, got {accuracy}")
        
        if sample_count < 1:
            raise ValueError(f"Sample count must be positive, got {sample_count}")
        
        metric = AccuracyMetric(
            timestamp=datetime.utcnow(),
            accuracy=accuracy,
            sample_count=sample_count,
            model_version=model_version
        )
        
        self.metrics.append(metric)
        self._cleanup_old_metrics()
        
        self.logger.info(
            f"Recorded accuracy {accuracy:.4f} for model {model_version} "
            f"with {sample_count} samples"
        )
    
    def check_degradation(self) -> Tuple[bool, Optional[float]]:
        """Check if model accuracy has degraded significantly.
        
        Returns:
            Tuple of (is_degraded, current_accuracy)
        """
        recent_metrics = self._get_recent_metrics()
        
        if not recent_metrics:
            return False, None
        
        total_samples = sum(m.sample_count for m in recent_metrics)
        if total_samples < self.min_samples:
            self.logger.debug(
                f"Insufficient samples ({total_samples} < {self.min_samples})"
            )
            return False, None
        
        # Weighted average by sample count
        weighted_accuracy = sum(
            m.accuracy * m.sample_count for m in recent_metrics
        ) / total_samples
        
        degradation = self.baseline_accuracy - weighted_accuracy
        is_degraded = degradation > self.degradation_threshold
        
        if is_degraded:
            self.logger.warning(
                f"Model degradation detected: {degradation:.4f} > {self.degradation_threshold:.4f}"
            )
        
        return is_degraded, weighted_accuracy
    
    def _get_recent_metrics(self) -> List[AccuracyMetric]:
        """Get metrics within the monitoring window."""
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        return [m for m in self.metrics if m.timestamp > cutoff]
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than the monitoring window."""
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours * 2)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]