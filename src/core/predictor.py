import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result with confidence and metadata."""
    prediction: float
    confidence: float
    feature_drift: bool
    data_quality_score: float
    timestamp: datetime
    model_version: str

class PerformancePredictor:
    """Predicts ML model performance degradation using metrics and drift."""
    
    def __init__(self, model_id: str, metrics_collector=None, drift_detector=None):
        self.model_id = model_id
        self.metrics_collector = metrics_collector
        self.drift_detector = drift_detector
        self._connection_pool = None
        self._is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize predictor resources."""
        try:
            if self.metrics_collector:
                await self.metrics_collector.start()
            if self.drift_detector:
                await self.drift_detector.initialize()
            self._is_initialized = True
            logger.info(f"Predictor initialized for model {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> None:
        """Clean up predictor resources."""
        try:
            if self.metrics_collector:
                await self.metrics_collector.stop()
            if self.drift_detector:
                await self.drift_detector.cleanup()
            self._is_initialized = False
            logger.info(f"Predictor cleaned up for model {self.model_id}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @asynccontextmanager
    async def managed_session(self) -> AsyncGenerator['PerformancePredictor', None]:
        """Async context manager for predictor lifecycle."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def predict_performance(self, features: Dict) -> PredictionResult:
        """Predict performance degradation for given features."""
        if not self._is_initialized:
            raise RuntimeError("Predictor not initialized. Use managed_session() or call initialize()")
        
        try:
            # Collect current metrics
            metrics = await self.metrics_collector.get_latest_metrics()
            
            # Detect drift
            drift_result = await self.drift_detector.detect_drift(features)
            
            # Simple performance prediction based on metrics + drift
            performance_score = self._calculate_performance_score(metrics, drift_result)
            
            return PredictionResult(
                prediction=performance_score,
                confidence=drift_result.get('confidence', 0.95),
                feature_drift=drift_result.get('has_drift', False),
                data_quality_score=metrics.get('data_quality', 1.0),
                timestamp=datetime.utcnow(),
                model_version=metrics.get('model_version', 'unknown')
            )
        except Exception as e:
            logger.error(f"Prediction failed for model {self.model_id}: {e}")
            raise
    
    def _calculate_performance_score(self, metrics: Dict, drift_result: Dict) -> float:
        """Calculate performance score based on metrics and drift."""
        base_score = metrics.get('accuracy', 0.95)
        drift_penalty = 0.1 if drift_result.get('has_drift', False) else 0.0
        latency_penalty = max(0, (metrics.get('avg_latency', 0) - 100) / 1000)
        
        return max(0.0, base_score - drift_penalty - latency_penalty)