"""
Statistical drift detection algorithms for ML model performance monitoring.

This module implements various statistical tests to detect distribution drift
in model inputs, outputs, and feature distributions.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog
from prometheus_client import Counter, Histogram, Gauge
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..exceptions import DriftDetectionError, InsufficientDataError


logger = structlog.get_logger(__name__)

# Prometheus metrics
DRIFT_DETECTIONS = Counter(
    "drift_detections_total",
    "Total number of drift detections",
    ["detector_type", "feature", "severity"]
)

DRIFT_DETECTION_DURATION = Histogram(
    "drift_detection_duration_seconds",
    "Time spent detecting drift",
    ["detector_type"]
)

DRIFT_SCORE = Gauge(
    "drift_score",
    "Current drift score",
    ["detector_type", "feature"]
)

REFERENCE_WINDOW_SIZE = Gauge(
    "reference_window_size",
    "Size of reference window",
    ["detector_type", "feature"]
)


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift that can be detected."""
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONCEPT = "concept"
    COVARIATE = "covariate"


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    detected: bool
    severity: DriftSeverity
    drift_type: DriftType
    feature_name: str
    score: float
    threshold: float
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    reference_size: int = 0
    current_size: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors."""

    def __init__(
        self,
        reference_window_size: int = 1000,
        detection_window_size: int = 100,
        min_samples: int = 50,
        name: str = "base_detector"
    ) -> None:
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.min_samples = min_samples
        self.name = name
        self.reference_data: Dict[str, List[Union[float, int]]] = {}
        self.is_fitted = False
        self.logger = logger.bind(detector=name)

    @abstractmethod
    async def detect_drift(
        self,
        current_data: Dict[str, List[Union[float, int]]],
        feature_name: str
    ) -> DriftResult:
        """Detect drift in current data compared to reference."""
        pass

    async def fit(self, reference_data: Dict[str, List[Union[float, int]]]) -> None:
        """Fit the detector on reference data."""
        self.logger.info("Fitting drift detector", features=list(reference_data.keys()))
        
        for feature, data in reference_data.items():
            if len(data) < self.min_samples:
                raise InsufficientDataError(
                    f"Feature {feature} has only {len(data)} samples, "
                    f"minimum {self.min_samples} required"
                )
            
            # Keep only the most recent reference_window_size samples
            self.reference_data[feature] = list(data[-self.reference_window_size:])
            
            REFERENCE_WINDOW_SIZE.labels(
                detector_type=self.name,
                feature=feature
            ).set(len(self.reference_data[feature]))
        
        self.is_fitted = True
        self.logger.info("Drift detector fitted successfully")

    def _get_severity(self, score: float, thresholds: Dict[str, float]) -> DriftSeverity:
        """Determine severity based on score and thresholds."""
        if score >= thresholds.get("critical", 0.9):
            return DriftSeverity.CRITICAL
        elif score >= thresholds.get("high", 0.7):
            return DriftSeverity.HIGH
        elif score >= thresholds.get("medium", 0.5):
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW


class KSDriftDetector(BaseDriftDetector):
    """Kolmogorov-Smirnov test-based drift detector."""

    def __init__(
        self,
        alpha: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__(name="ks_detector", **kwargs)
        self.alpha = alpha
        self.thresholds = {
            "critical": 0.001,
            "high": 0.01,
            "medium": 0.05,
            "low": 0.1
        }

    async def detect_drift(
        self,
        current_data: Dict[str, List[Union[float, int]]],
        feature_name: str
    ) -> DriftResult:
        """Detect drift using Kolmogorov-Smirnov test."""
        if not self.is_fitted:
            raise DriftDetectionError("Detector must be fitted before detecting drift")
        
        if feature_name not in self.reference_data:
            raise DriftDetectionError(f"Feature {feature_name} not found in reference data")
        
        if feature_name not in current_data:
            raise DriftDetectionError(f"Feature {feature_name} not found in current data")

        start_time = time.time()
        
        try:
            reference = np.array(self.reference_data[feature_name])
            current = np.array(current_data[feature_name][-self.detection_window_size:])
            
            if len(current) < self.min_samples:
                raise InsufficientDataError(
                    f"Current data has only {len(current)} samples, "
                    f"minimum {self.min_samples} required"
                )
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(reference, current)
            
            detected = p_value < self.alpha
            severity = self._get_severity(1 - p_value, self.thresholds)
            
            result = DriftResult(
                detected=detected,
                severity=severity,
                drift_type=DriftType.FEATURE,
                feature_name=feature_name,
                score=1 - p_value,
                threshold=1 - self.alpha,
                p_value=p_value,
                test_statistic=statistic,
                reference_size=len(reference),
                current_size=len(current),
                metadata={
                    "test_name": "kolmogorov_smirnov",
                    "alpha": self.alpha
                }
            )
            
            # Update metrics
            DRIFT_SCORE.labels(
                detector_type=self.name,
                feature=feature_name
            ).set(result.score)
            
            if detected:
                DRIFT_DETECTIONS.labels(
                    detector_type=self.name,
                    feature=feature_name,
                    severity=severity.value
                ).inc()
            
            self.logger.info(
                "KS drift detection completed",
                feature=feature_name,
                detected=detected,
                p_value=p_value,
                statistic=statistic,
                severity=severity.value
            )
            
            return result
            
        except Exception as e:
            self.logger.error("KS drift detection failed", feature=feature_name, error=str(e))
            raise DriftDetectionError(f"KS drift detection failed: {str(e)}") from e
        
        finally:
            duration = time.time() - start_time
            DRIFT_DETECTION_DURATION.labels(detector_type=self.name).observe(duration)


class PSIDriftDetector(BaseDriftDetector):
    """Population Stability Index (PSI) drift detector."""

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        **kwargs
    ) -> None:
        super().__init__(name="psi_detector", **kwargs)
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.thresholds = {
            "critical": 0.25,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05
        }
        self.bin_edges: Dict[str, np.ndarray] = {}

    async def detect_drift(
        self,
        current_data: Dict[str, List[Union[float, int]]],
        feature_name: str
    ) -> DriftResult:
        """Detect drift using Population Stability Index."""
        if not self.is_fitted:
            raise DriftDetectionError("Detector must be fitted before detecting drift")
        
        if feature_name not in self.reference_data:
            raise DriftDetectionError(f"Feature {feature_name} not found in reference data")
        
        if feature_name not in current_data:
            raise DriftDetectionError(f"Feature {feature_name} not found in current data")

        start_time = time.time()
        
        try:
            reference = np.array(self.reference_data[feature_name])
            current = np.array(current_data[feature_name][-self.detection_window_size:])
            
            if len(current) < self.min_samples:
                raise InsufficientDataError(
                    f"Current data has only {len(current)} samples, "
                    f"minimum {self.min_samples} required"
                )
            
            # Use pre-computed bin edges or compute them
            if feature_name not in self.bin_edges:
                self.bin_edges[feature_name] = np.histogram_bin_edges(
                    reference, bins=self.n_bins
                )
            
            bin_edges = self.bin_edges[feature_name]
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to get proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            ref_props = np.where(ref_props == 0, epsilon, ref_props)
            cur_props = np.where(cur_props == 0, epsilon, cur_props)
            
            # Calculate PSI
            psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
            psi_score = np.sum(psi_values)
            
            detected = psi_score > self.psi_threshold
            severity = self._get_severity(psi_score, self.thresholds)
            
            result = DriftResult(
                detected=detected,
                severity=severity,
                drift_type=DriftType.FEATURE,
                feature_name=feature_name,
                score=psi_score,
                threshold=self.psi_threshold,
                reference_size=len(reference),
                current_size=len(current),
                metadata={
                    "test_name": "population_stability_index",
                    "n_bins": self.n_bins,
                    "bin_edges": bin_edges.tolist()
                }
            )
            
            # Update metrics
            DRIFT_SCORE.labels(
                detector_type=self.name,
                feature=feature_name
            ).set(result.score)
            
            if detected:
                DRIFT_DETECTIONS.labels(
                    detector_type=self.name,
                    feature=feature_name,
                    severity=severity.value
                ).inc()
            
            self.logger.info(
                "PSI drift detection completed",
                feature=feature_name,
                detected=detected,
                psi_score=psi_score,
                severity=severity.value
            )
            
            return result
            
        except Exception as e:
            self.logger.error("PSI drift detection failed", feature=feature_name, error=str(e))
            raise DriftDetectionError(f"PSI drift detection failed: {str(e)}") from e
        
        finally:
            duration = time.time() - start_time
            DRIFT_DETECTION_DURATION.labels(detector_type=self.name).observe(duration)

    async def fit(self, reference_data: Dict[str, List[Union[float, int]]]) -> None:
        """Fit the PSI detector and compute bin edges."""
        await super().fit(reference_data)
        
        # Pre-compute bin edges for each feature
        for feature, data in self.reference_data.items():
            self.bin_edges[feature] = np.histogram_bin_edges(
                np.array(data), bins=self.n_bins
            )


class IsolationForestDriftDetector(BaseDriftDetector):
    """Isolation Forest-based drift detector for multivariate drift."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        anomaly_threshold: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(name="isolation_forest_detector", **kwargs)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.anomaly_threshold = anomaly_threshold
        self.thresholds = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.15,
            "low": 0.1
        }
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    async def detect_drift(
        self,
        current_data: Dict[str, List[Union[float, int]]],
        feature_name: str
    ) -> DriftResult:
        """Detect drift using Isolation Forest anomaly detection."""
        if not self.is_fitted:
            raise DriftDetectionError("Detector must be fitted before detecting drift")
        
        if feature_name not in self.models:
            raise DriftDetectionError(f"Feature {feature_name} not found in fitted models")
        
        if feature_name not in current_data:
            raise DriftDetectionError(f"Feature {feature_name} not found in current data")

        start_time = time.time()
        
        try:
            current = np.array(current_data[feature_name][-self.detection_window_size:])
            
            if len(current) < self.min_samples:
                raise InsufficientDataError(
                    f"Current data has only {len(current)} samples, "
                    f"minimum {self.min_samples} required"
                )
            
            # Reshape for univariate data
            current = current.reshape(-1, 1)
            
            # Scale current data
            current_scaled = self.scalers[feature_name].transform(current)
            
            # Get anomaly scores
            anomaly_scores = self.models[feature_name].decision_function(current_scaled)
            outliers = self.models[feature_name].predict(current_scaled)
            
            # Calculate drift score as proportion of anomalies
            anomaly_rate = np.mean(outliers == -1)
            
            detected = anomaly_rate > self.anomaly_threshold
            severity = self._get_severity(anomaly_rate, self.thresholds)
            
            result = DriftResult(
                detected=detected,
                severity=severity,
                drift_type=DriftType.FEATURE,
                feature_name=feature_name,
                score=anomaly_rate,
                threshold=self.anomaly_threshold,
                reference_size=len(self.reference_data[feature_name]),
                current_size=len(current),
                metadata={
                    "test_name": "isolation_forest",
                    "contamination": self.contamination,
                    "n_estimators": self.n_estimators,
                    "mean_anomaly_score": float(np.mean(anomaly_scores)),
                    "std_anomaly_score": float(np.std(anomaly_scores))
                }
            )
            
            # Update metrics
            DRIFT_SCORE.labels(
                detector_type=self.name,
                feature=feature_name
            ).set(result.score)
            
            if detected:
                DRIFT_DETECTIONS.labels(
                    detector_type=self.name,
                    feature=feature_name,
                    severity=severity.value
                ).inc()
            
            self.logger.info(
                "Isolation Forest drift detection completed",
                feature=feature_name,
                detected=detected,
                anomaly_rate=anomaly_rate,
                severity=severity.value
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Isolation Forest drift detection failed",
                feature=feature_name,
                error=str(e)
            )
            raise DriftDetectionError(
                f"Isolation Forest drift detection failed: {str(e)}"
            ) from e
        
        finally:
            duration = time.time() - start_time
            DRIFT_DETECTION_DURATION.labels(detector_type=self.name).observe(duration)

    async def fit(self, reference_data: Dict[str, List[Union[float, int]]]) -> None:
        """Fit Isolation Forest models for each feature."""
        await super().fit(reference_data)
        
        for feature, data in self.reference_data.items():
            # Prepare data
            X = np.array(data).reshape(-1, 1)
            
            # Fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            self.models[feature] = model
            self.scalers[feature] = scaler
            
            self.logger.info(
                "Fitted Isolation Forest model",
                feature=feature,
                n_samples=len(data)
            )


class EnsembleDriftDetector:
    """Ensemble of multiple drift detectors for robust detection."""

    def __init__(
        self,
        detectors: List[BaseDriftDetector],
        voting_strategy: str = "majority",
        min_detectors: int = 2
    ) -> None:
        self.detectors = detectors
        self.voting_strategy = voting_strategy
        self.min_detectors = min_detectors
        self.is_fitted = False
        self.logger = logger.bind(detector="ensemble")

    async def fit(self, reference_data: Dict[str, List[Union[float, int]]]) -> None:
        """Fit all detectors in the ensemble."""
        self.logger.info("Fitting ensemble drift detector", n_detectors=len(self.detectors))
        
        fit_tasks = [detector.fit(reference_data) for detector in self.detectors]
        await asyncio.gather(*fit_tasks)
        
        self.is_fitted = True
        self.logger.info("Ensemble drift detector fitted successfully")

    async def detect_drift(
        self,
        current_data: Dict[str, List[Union[float, int]]],
        feature_name: str
    ) -> DriftResult:
        """Detect drift using ensemble of detectors."""
        if not self.is_fitted:
            raise DriftDetectionError("Ensemble must be fitted before detecting drift")

        # Run all detectors in parallel
        detection_tasks = [
            detector.detect_drift(current_data, feature_name)
            for detector in self.detectors
        ]
        
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Filter out exceptions and get valid results
        valid_results = [r for r in results if isinstance(r, DriftResult)]
        
        if len(valid_results) < self.min_detectors:
            raise DriftDetectionError(
                f"Only {len(valid_results)} detectors succeeded, "
                f"minimum {self.min_detectors} required"
            )

        # Apply voting strategy
        if self.voting_strategy == "majority":
            detected = sum(r.detected for r in valid_results) > len(valid_results) / 2
        elif self.voting_strategy == "unanimous":
            detected = all(r.detected for r in valid_results)
        elif self.voting_strategy == "any":
            detected = any(r.detected for r in valid_results)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        # Aggregate scores and determine severity
        avg_score = np.mean([r.score for r in valid_results])
        max_severity = max([r.severity for r in valid_results], key=lambda x: x.value)
        
        ensemble_result = DriftResult(
            detected=detected,
            severity=max_severity if detected else DriftSeverity.LOW,
            drift_type=DriftType.FEATURE,
            feature_name=feature_name,
            score=avg_score,
            threshold=0.5,  # Ensemble threshold
            reference_size=valid_results[0].reference_size,
            current_size=valid_results[0].current_size,
            metadata={
                "voting_strategy": self.voting_strategy,
                "n_detectors": len(valid_results),
                "individual_results": [
                    {
                        "detector": type(detector).__name__,
                        "detected": result.detected,
                        "score": result.score,
                        "severity": result.severity.value
                    }
                    for detector, result in zip(self.detectors, valid_results)
                ]
            }
        )

        self.logger.info(
            "Ensemble drift detection completed",
            feature=feature_name,
            detected=detected,
            avg_score=avg_score,
            n_detectors=len(valid_results),
            voting_strategy=self.voting_strategy
        )

        return ensemble_result


# Factory function for creating detector ensembles
def create_default_ensemble(
    reference_window_size: int = 1000,
    detection_window_size: int = 100
) -> EnsembleDriftDetector:
    """Create a default ensemble of drift detectors."""
    detectors = [
        KSDriftDetector(
            reference_window_size=reference_window_size,
            detection_window_size=detection_window_size
        ),
        PSIDriftDetector(
            reference_window_size=reference_window_size,
            detection_window_size=detection_window_size
        ),
        IsolationForestDriftDetector(
            reference_window_size=reference_window_size,
            detection_window_size=detection_window_size
        )
    ]
    
    return EnsembleDriftDetector(
        detectors=detectors,
        voting_strategy="majority",
        min_detectors=2
    )