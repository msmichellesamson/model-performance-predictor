from typing import Dict, List, Optional
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway
import logging
import time
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class PrometheusClient:
    """Enhanced Prometheus client with circuit breaker and better error handling."""
    
    def __init__(self, gateway_url: str, job_name: str = "model-performance"):
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_duration=30,
            max_backoff=300
        )
        
        # Initialize metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total ML model predictions',
            ['model_name', 'status'],
            registry=self.registry
        )
        
        self.drift_gauge = Gauge(
            'ml_drift_score',
            'Model drift detection score',
            ['model_name', 'feature'],
            registry=self.registry
        )
        
        self.performance_histogram = Histogram(
            'ml_model_performance_score',
            'Model performance metrics',
            ['model_name', 'metric_type'],
            registry=self.registry
        )
    
    def record_prediction(self, model_name: str, status: str = "success"):
        """Record a model prediction with error handling."""
        try:
            self.prediction_counter.labels(model_name=model_name, status=status).inc()
            logger.debug(f"Recorded prediction for model {model_name} with status {status}")
        except Exception as e:
            logger.error(f"Failed to record prediction metric: {e}")
    
    def record_drift_score(self, model_name: str, feature: str, score: float):
        """Record drift detection score."""
        try:
            self.drift_gauge.labels(model_name=model_name, feature=feature).set(score)
            logger.debug(f"Recorded drift score {score} for {model_name}/{feature}")
        except Exception as e:
            logger.error(f"Failed to record drift metric: {e}")
    
    def record_performance(self, model_name: str, metric_type: str, score: float):
        """Record model performance score."""
        try:
            self.performance_histogram.labels(
                model_name=model_name, 
                metric_type=metric_type
            ).observe(score)
            logger.debug(f"Recorded performance {metric_type}={score} for {model_name}")
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
    
    def push_metrics(self, grouping_key: Optional[Dict[str, str]] = None) -> bool:
        """Push metrics to Prometheus gateway with circuit breaker protection."""
        try:
            def _push():
                push_to_gateway(
                    gateway=self.gateway_url,
                    job=self.job_name,
                    registry=self.registry,
                    grouping_key=grouping_key or {}
                )
            
            self.circuit_breaker.call(_push)
            logger.info("Successfully pushed metrics to Prometheus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
            return False
