"""
Performance degradation ML predictor for real-time model monitoring.

This module implements the core ML predictor that analyzes model inference
metrics and data drift to predict when model performance will degrade.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import structlog
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import redis.asyncio as redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge
import joblib
from pathlib import Path


logger = structlog.get_logger(__name__)

# Metrics
PREDICTION_COUNTER = Counter('degradation_predictions_total', ['model_id', 'prediction'])
PREDICTION_LATENCY = Histogram('degradation_prediction_duration_seconds', 'Time spent on degradation prediction')
DRIFT_SCORE = Gauge('model_drift_score', 'Current drift score for model', ['model_id'])
CONFIDENCE_SCORE = Gauge('prediction_confidence', 'Confidence in degradation prediction', ['model_id'])


class DegradationLevel(Enum):
    """Model performance degradation levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    FAILED = "failed"


class PredictorError(Exception):
    """Base exception for predictor errors."""
    pass


class ModelNotFoundError(PredictorError):
    """Raised when model is not found."""
    pass


class InsufficientDataError(PredictorError):
    """Raised when insufficient data for prediction."""
    pass


class DriftDetectionError(PredictorError):
    """Raised when drift detection fails."""
    pass


@dataclass
class ModelMetrics:
    """Container for model inference metrics."""
    model_id: str
    timestamp: datetime
    latency_p95: float
    latency_p99: float
    error_rate: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    prediction_confidence: float
    input_drift_score: float
    output_drift_score: float


@dataclass
class PredictionResult:
    """Container for degradation prediction results."""
    model_id: str
    prediction_time: datetime
    degradation_level: DegradationLevel
    confidence: float
    risk_factors: List[str]
    recommended_actions: List[str]
    time_to_degradation: Optional[timedelta]
    feature_importance: Dict[str, float]


class FeatureExtractor:
    """Extracts features from model metrics for ML prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_features(self, metrics_window: List[ModelMetrics]) -> np.ndarray:
        """Extract ML features from metrics window."""
        if len(metrics_window) < 2:
            raise InsufficientDataError("Need at least 2 metrics points for feature extraction")
        
        features = []
        
        # Time-based features
        timestamps = [m.timestamp for m in metrics_window]
        time_deltas = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                      for i in range(1, len(timestamps))]
        features.extend([
            np.mean(time_deltas),
            np.std(time_deltas) if len(time_deltas) > 1 else 0.0
        ])
        
        # Statistical features for each metric
        for metric_name in ['latency_p95', 'latency_p99', 'error_rate', 'throughput',
                           'memory_usage', 'cpu_usage', 'prediction_confidence',
                           'input_drift_score', 'output_drift_score']:
            values = [getattr(m, metric_name) for m in metrics_window]
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0.0,
                np.max(values),
                np.min(values),
                values[-1] - values[0] if len(values) > 1 else 0.0,  # trend
            ])
        
        # Rolling window features
        if len(metrics_window) >= 5:
            recent_window = metrics_window[-5:]
            older_window = metrics_window[:-5] if len(metrics_window) > 5 else metrics_window[:1]
            
            for metric_name in ['latency_p95', 'error_rate', 'input_drift_score']:
                recent_avg = np.mean([getattr(m, metric_name) for m in recent_window])
                older_avg = np.mean([getattr(m, metric_name) for m in older_window])
                features.append(recent_avg - older_avg)  # change detection
        else:
            features.extend([0.0, 0.0, 0.0])  # padding
        
        return np.array(features).reshape(1, -1)
    
    def fit_scaler(self, feature_matrix: np.ndarray) -> None:
        """Fit the feature scaler."""
        self.scaler.fit(feature_matrix)
        self._is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self._is_fitted:
            logger.warning("Scaler not fitted, returning unscaled features")
            return features
        return self.scaler.transform(features)


class DriftDetector:
    """Detects data drift using statistical methods."""
    
    def __init__(self, sensitivity: float = 0.05):
        self.sensitivity = sensitivity
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
    
    def set_baseline(self, baseline_metrics: List[ModelMetrics]) -> None:
        """Set baseline statistics for drift detection."""
        if len(baseline_metrics) < 10:
            raise InsufficientDataError("Need at least 10 baseline metrics")
        
        # Calculate baseline statistics
        for metric_name in ['input_drift_score', 'output_drift_score', 'prediction_confidence']:
            values = [getattr(m, metric_name) for m in baseline_metrics]
            self._baseline_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'q95': np.percentile(values, 95),
                'q5': np.percentile(values, 5)
            }
        
        # Train isolation forest on baseline
        feature_matrix = np.array([[m.input_drift_score, m.output_drift_score, m.prediction_confidence] 
                                  for m in baseline_metrics])
        self.isolation_forest.fit(feature_matrix)
        
        logger.info("Drift detector baseline set", 
                   baseline_size=len(baseline_metrics),
                   baseline_stats=self._baseline_stats)
    
    def detect_drift(self, current_metrics: ModelMetrics) -> Tuple[bool, float, Dict[str, float]]:
        """Detect if current metrics show drift from baseline."""
        if not self._baseline_stats:
            raise DriftDetectionError("Baseline not set")
        
        drift_scores = {}
        overall_drift = False
        
        # Statistical drift detection
        for metric_name in ['input_drift_score', 'output_drift_score']:
            current_value = getattr(current_metrics, metric_name)
            baseline = self._baseline_stats[metric_name]
            
            # Z-score based drift
            z_score = abs(current_value - baseline['mean']) / (baseline['std'] + 1e-8)
            drift_scores[f"{metric_name}_zscore"] = z_score
            
            # Percentile based drift
            if current_value > baseline['q95'] or current_value < baseline['q5']:
                drift_scores[f"{metric_name}_percentile_drift"] = 1.0
                overall_drift = True
            else:
                drift_scores[f"{metric_name}_percentile_drift"] = 0.0
        
        # Anomaly detection
        feature_vector = np.array([[current_metrics.input_drift_score, 
                                   current_metrics.output_drift_score,
                                   current_metrics.prediction_confidence]])
        anomaly_score = self.isolation_forest.decision_function(feature_vector)[0]
        is_anomaly = self.isolation_forest.predict(feature_vector)[0] == -1
        
        drift_scores['anomaly_score'] = anomaly_score
        drift_scores['is_anomaly'] = float(is_anomaly)
        
        if is_anomaly:
            overall_drift = True
        
        # Combined drift score
        combined_score = np.mean([
            drift_scores['input_drift_score_zscore'],
            drift_scores['output_drift_score_zscore'],
            abs(anomaly_score)
        ])
        
        return overall_drift, combined_score, drift_scores


class PerformancePredictor:
    """Main class for predicting model performance degradation."""
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 db_pool: asyncpg.Pool,
                 model_storage_path: str = "/app/models"):
        self.redis = redis_client
        self.db_pool = db_pool
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.drift_detector = DriftDetector()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self._models: Dict[str, bool] = {}  # model_id -> is_trained
    
    async def load_historical_metrics(self, 
                                    model_id: str, 
                                    hours_back: int = 168) -> List[ModelMetrics]:
        """Load historical metrics from database."""
        query = """
        SELECT model_id, timestamp, latency_p95, latency_p99, error_rate, 
               throughput, memory_usage, cpu_usage, prediction_confidence,
               input_drift_score, output_drift_score
        FROM model_metrics 
        WHERE model_id = $1 AND timestamp > $2
        ORDER BY timestamp ASC
        """
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, model_id, cutoff_time)
        
        if not rows:
            raise ModelNotFoundError(f"No metrics found for model {model_id}")
        
        metrics = []
        for row in rows:
            metrics.append(ModelMetrics(
                model_id=row['model_id'],
                timestamp=row['timestamp'],
                latency_p95=row['latency_p95'],
                latency_p99=row['latency_p99'], 
                error_rate=row['error_rate'],
                throughput=row['throughput'],
                memory_usage=row['memory_usage'],
                cpu_usage=row['cpu_usage'],
                prediction_confidence=row['prediction_confidence'],
                input_drift_score=row['input_drift_score'],
                output_drift_score=row['output_drift_score']
            ))
        
        return metrics
    
    def _create_training_labels(self, metrics: List[ModelMetrics]) -> np.ndarray:
        """Create training labels based on degradation patterns."""
        labels = []
        
        for i, metric in enumerate(metrics):
            # Look ahead window for labeling
            future_window = metrics[i:i+24]  # next 24 points
            
            if len(future_window) < 5:
                labels.append(0)  # healthy
                continue
            
            # Check for degradation patterns
            error_rates = [m.error_rate for m in future_window]
            latencies = [m.latency_p95 for m in future_window]
            drift_scores = [max(m.input_drift_score, m.output_drift_score) for m in future_window]
            
            # Critical degradation
            if (max(error_rates) > 0.1 or 
                max(latencies) > metric.latency_p95 * 3 or
                max(drift_scores) > 0.8):
                labels.append(3)  # critical
            # Warning degradation  
            elif (max(error_rates) > 0.05 or 
                  max(latencies) > metric.latency_p95 * 2 or
                  max(drift_scores) > 0.5):
                labels.append(2)  # warning
            # Healthy
            else:
                labels.append(0)  # healthy
        
        return np.array(labels)
    
    async def train_model(self, model_id: str) -> None:
        """Train degradation prediction model."""
        logger.info("Starting model training", model_id=model_id)
        
        # Load historical data
        try:
            historical_metrics = await self.load_historical_metrics(model_id, hours_back=336)  # 2 weeks
        except ModelNotFoundError:
            logger.error("Insufficient historical data for training", model_id=model_id)
            raise InsufficientDataError(f"No historical data for model {model_id}")
        
        if len(historical_metrics) < 100:
            raise InsufficientDataError(f"Need at least 100 historical metrics, got {len(historical_metrics)}")
        
        # Set baseline for drift detection
        baseline_metrics = historical_metrics[:len(historical_metrics)//3]  # First third as baseline
        self.drift_detector.set_baseline(baseline_metrics)
        
        # Extract features and labels
        features_list = []
        labels_list = []
        
        # Use sliding window approach
        window_size = 10
        for i in range(window_size, len(historical_metrics) - 24):  # Leave 24 for future labeling
            try:
                window = historical_metrics[i-window_size:i]
                features = self.feature_extractor.extract_features(window)
                features_list.append(features[0])
                
                # Create labels based on future degradation
                future_metrics = historical_metrics[i:i+24]
                label = self._determine_degradation_label(future_metrics)
                labels_list.append(label)
                
            except Exception as e:
                logger.warning("Failed to extract features for window", error=str(e), index=i)
                continue
        
        if len(features_list) < 50:
            raise InsufficientDataError(f"Insufficient feature vectors: {len(features_list)}")
        
        # Prepare training data
        X = np.vstack(features_list)
        y = np.array(labels_list)
        
        # Fit scaler
        self.feature_extractor.fit_scaler(X)
        X_scaled = self.feature_extractor.transform_features(X)
        
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        model_path = self.model_storage_path / f"{model_id}_predictor.pkl"
        joblib.dump({
            'classifier': self.classifier,
            'feature_extractor': self.feature_extractor,
            'drift_detector': self.drift_detector,
            'training_report': report
        }, model_path)
        
        self._models[model_id] = True
        
        logger.info("Model training completed", 
                   model_id=model_id,
                   training_samples=len(X_train),
                   test_accuracy=report['accuracy'],
                   model_path=str(model_path))
    
    def _determine_degradation_label(self, future_metrics: List[ModelMetrics]) -> int:
        """Determine degradation label based on future metrics."""
        error_rates = [m.error_rate for m in future_metrics]
        latencies = [m.latency_p95 for m in future_metrics]
        drift_scores = [max(m.input_drift_score, m.output_drift_score) for m in future_metrics]
        
        # Critical conditions
        if (max(error_rates) > 0.15 or 
            np.mean(error_rates) > 0.1 or
            max(latencies) > np.mean(latencies[:5]) * 4 or
            max(drift_scores) > 0.9):
            return 3  # critical
        
        # Warning conditions
        elif (max(error_rates) > 0.08 or
              np.mean(error_rates) > 0.05 or  
              max(latencies) > np.mean(latencies[:5]) * 2.5 or
              max(drift_scores) > 0.6):
            return 2  # warning
        
        # Healthy
        return 0
    
    async def load_model(self, model_id: str) -> bool:
        """Load trained model from disk."""
        model_path = self.model_storage_path / f"{model_id}_predictor.pkl"
        
        if not model_path.exists():
            return False
        
        try:
            model_data = joblib.load(model_path)
            self.classifier = model_data['classifier']
            self.feature_extractor = model_data['feature_extractor'] 
            self.drift_detector = model_data['drift_detector']
            
            self._models[model_id] = True
            logger.info("Model loaded successfully", model_id=model_id, path=str(model_path))
            return True
            
        except Exception as e:
            logger.error("Failed to load model", model_id=model_id, error=str(e))
            return False
    
    @PREDICTION_LATENCY.time()
    async def predict_degradation(self, model_id: str) -> PredictionResult:
        """Predict model performance degradation."""
        logger.info("Starting degradation prediction", model_id=model_id)
        
        # Ensure model is loaded
        if model_id not in self._models or not self._models[model_id]:
            if not await self.load_model(model_id):
                raise ModelNotFoundError(f"No trained model found for {model_id}")
        
        # Get recent metrics
        try:
            recent_metrics = await self.load_historical_metrics(model_id, hours_back=24)
        except ModelNotFoundError:
            raise InsufficientDataError(f"No recent metrics for model {model_id}")
        
        if len(recent_metrics) < 10:
            raise InsufficientDataError(f"Need at least 10 recent metrics, got {len(recent_metrics)}")
        
        # Extract features
        features = self.feature_extractor.extract_features(recent_metrics[-10:])
        features_scaled = self.feature_extractor.transform_features(features)
        
        # Make prediction
        prediction = self.classifier.predict(features_scaled)[0]
        prediction_proba = self.classifier.predict_proba(features_scaled)[0]
        confidence = max(prediction_proba)
        
        # Detect drift
        current_metrics = recent_metrics[-1]
        has_drift, drift_score, drift_details = self.drift_detector.detect_drift(current_metrics)
        
        # Determine degradation level
        degradation_level = DegradationLevel.HEALTHY
        if prediction == 3:
            degradation_level = DegradationLevel.CRITICAL
        elif prediction == 2:
            degradation_level = DegradationLevel.WARNING
        elif has_drift and drift_score > 0.7:
            degradation_level = DegradationLevel.WARNING
        
        # Identify risk factors
        risk_factors = []
        if current_metrics.error_rate > 0.05:
            risk_factors.append(f"High error rate: {current_metrics.error_rate:.3f}")
        if current_metrics.latency_p95 > 1.0:
            risk_factors.append(f"High latency: {current_metrics.latency_p95:.2f}s")
        if has_drift:
            risk_factors.append(f"Data drift detected (score: {drift_score:.3f})")
        if current_metrics.prediction_confidence < 0.8:
            risk_factors.append(f"Low prediction confidence: {current_metrics.prediction_confidence:.3f}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(degradation_level, risk_factors, current_metrics)
        
        # Estimate time to degradation
        time_to_degradation = None
        if degradation_level != DegradationLevel.HEALTHY:
            # Simple heuristic based on trend
            recent_error_rates = [m.error_rate for m in recent_metrics[-5:]]
            if len(recent_error_rates) > 1:
                error_trend = (recent_error_rates[-1] - recent_error_rates[0]) / len(recent_error_rates)
                if error_trend > 0:
                    # Rough estimate: hours until critical threshold
                    hours_to_critical = max(1, int((0.15 - current_metrics.error_rate) / (error_trend + 1e-8)))
                    time_to_degradation = timedelta(hours=min(hours_to_critical, 72))
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.classifier, 'feature_importances_'):
            feature_names = [f"feature_{i}" for i in range(len(self.classifier.feature_importances_))]
            feature_importance = dict(zip(feature_names, self.classifier.feature_importances_.tolist()))
        
        result = PredictionResult(
            model_id=model_id,
            prediction_time=datetime.utcnow(),
            degradation_level=degradation_level,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_actions=recommendations,
            time_to_degradation=time_to_degradation,
            feature_importance=feature_importance
        )
        
        # Update metrics
        PREDICTION_COUNTER.labels(model_id=model_id, prediction=degradation_level.value).inc()
        DRIFT_SCORE.labels(model_id=model_id).set(drift_score)
        CONFIDENCE_SCORE.labels(model_id=model_id).set(confidence)
        
        # Cache result
        await self.redis.setex(
            f"degradation_prediction:{model_id}",
            300,  # 5 minutes
            result.__dict__
        )
        
        logger.info("Degradation prediction completed",
                   model_id=model_id,
                   degradation_level=degradation_level.value,
                   confidence=confidence,
                   drift_score=drift_score)
        
        return result
    
    def _generate_recommendations(self, 
                                degradation_level: DegradationLevel,
                                risk_factors: List[str], 
                                current_metrics: ModelMetrics) -> List[str]:
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        if degradation_level == DegradationLevel.CRITICAL:
            recommendations.extend([
                "URGENT: Consider immediate model rollback or circuit breaker activation",
                "Scale up infrastructure resources immediately",
                "Investigate recent model or data changes"
            ])
        
        elif degradation_level == DegradationLevel.WARNING:
            recommendations.extend([
                "Monitor model closely for further degradation",
                "Prepare rollback plan",
                "Review recent deployments and data changes"
            ])
        
        # Specific recommendations based on metrics
        if current_metrics.error_rate > 0.05:
            recommendations.append("Investigate root cause of increased error rate")
        
        if current_metrics.latency_p95 > 1.0:
            recommendations.extend([
                "Consider horizontal scaling",
                "Review model complexity and optimize inference"
            ])
        
        if current_metrics.input_drift_score > 0.5 or current_metrics.output_drift_score > 0.5:
            recommendations.extend([
                "Analyze input data distribution changes", 
                "Consider model retraining with recent data"
            ])
        
        if current_metrics.prediction_confidence < 0.8:
            recommendations.append("Review model confidence thresholds and fallback mechanisms")
        
        return recommendations
    
    async def health_check(self) -> Dict[str, Union[str, int, bool]]:
        """Health check for the predictor service."""
        try:
            # Check database connection
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_healthy = True
        except Exception:
            db_healthy = False
        
        # Check Redis connection
        try:
            await self.redis.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False
        
        return {
            "status": "healthy" if db_healthy and redis_healthy else "unhealthy",
            "database_healthy": db_healthy,
            "redis_healthy": redis_healthy,
            "loaded_models": len(self._models),
            "timestamp": datetime.utcnow().isoformat()
        }