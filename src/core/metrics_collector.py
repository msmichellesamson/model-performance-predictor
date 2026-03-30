import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import json

import structlog
import redis.asyncio as redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import numpy as np
from pydantic import BaseModel, validator


logger = structlog.get_logger(__name__)


class MetricsCollectorError(Exception):
    """Base exception for metrics collector operations"""
    pass


class DatabaseConnectionError(MetricsCollectorError):
    """Database connection related errors"""
    pass


class RedisConnectionError(MetricsCollectorError):
    """Redis connection related errors"""
    pass


class InvalidMetricError(MetricsCollectorError):
    """Invalid metric data errors"""
    pass


@dataclass
class InferenceMetric:
    """Single inference metric record"""
    model_id: str
    version: str
    timestamp: datetime
    prediction_latency: float
    confidence_score: float
    input_features: Dict[str, Any]
    prediction: Any
    ground_truth: Optional[Any] = None
    feature_drift_score: Optional[float] = None
    concept_drift_score: Optional[float] = None
    request_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with timestamp serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricBatch(BaseModel):
    """Batch of metrics for processing"""
    model_id: str
    version: str
    metrics: List[Dict[str, Any]]
    batch_timestamp: datetime
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if len(v) == 0:
            raise ValueError("Metrics batch cannot be empty")
        if len(v) > 10000:
            raise ValueError("Batch size too large (max 10000)")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PrometheusMetrics:
    """Prometheus metrics definitions"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        self.inference_counter = Counter(
            'model_inferences_total',
            'Total number of model inferences',
            ['model_id', 'version', 'status'],
            registry=self.registry
        )
        
        self.latency_histogram = Histogram(
            'model_inference_duration_seconds',
            'Model inference latency in seconds',
            ['model_id', 'version'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.confidence_histogram = Histogram(
            'model_confidence_score',
            'Model confidence score distribution',
            ['model_id', 'version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.drift_gauge = Gauge(
            'model_drift_score',
            'Current drift score for model',
            ['model_id', 'version', 'drift_type'],
            registry=self.registry
        )
        
        self.performance_gauge = Gauge(
            'model_performance_prediction',
            'Predicted performance degradation score',
            ['model_id', 'version'],
            registry=self.registry
        )
    
    def record_inference(self, metric: InferenceMetric, status: str = "success"):
        """Record inference metrics in Prometheus"""
        labels = [metric.model_id, metric.version]
        
        self.inference_counter.labels(*labels, status).inc()
        self.latency_histogram.labels(*labels).observe(metric.prediction_latency)
        self.confidence_histogram.labels(*labels).observe(metric.confidence_score)
        
        if metric.feature_drift_score is not None:
            self.drift_gauge.labels(*labels, "feature").set(metric.feature_drift_score)
        
        if metric.concept_drift_score is not None:
            self.drift_gauge.labels(*labels, "concept").set(metric.concept_drift_score)


class MetricsCollector:
    """Production-grade metrics collector for ML model performance monitoring"""
    
    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        batch_size: int = 100,
        flush_interval: int = 30,
        max_memory_buffer: int = 10000
    ):
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_memory_buffer = max_memory_buffer
        
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.prometheus_metrics = PrometheusMetrics()
        
        # In-memory buffer for batching
        self._buffer: List[InferenceMetric] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        
        self.logger = logger.bind(component="metrics_collector")
    
    async def initialize(self) -> None:
        """Initialize database connections and background tasks"""
        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    'jit': 'off',
                    'application_name': 'metrics_collector'
                }
            )
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True
            )
            
            # Test connections
            await self._test_connections()
            
            # Start background flush task
            self._flush_task = asyncio.create_task(self._background_flush())
            
            self.logger.info("Metrics collector initialized successfully")
            
        except asyncpg.PostgresError as e:
            raise DatabaseConnectionError(f"PostgreSQL connection failed: {e}")
        except redis.RedisError as e:
            raise RedisConnectionError(f"Redis connection failed: {e}")
        except Exception as e:
            raise MetricsCollectorError(f"Initialization failed: {e}")
    
    async def _test_connections(self) -> None:
        """Test database connections"""
        # Test PostgreSQL
        async with self.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test Redis
        await self.redis_client.ping()
    
    async def collect_inference_metric(
        self,
        model_id: str,
        version: str,
        prediction_latency: float,
        confidence_score: float,
        input_features: Dict[str, Any],
        prediction: Any,
        ground_truth: Optional[Any] = None,
        request_id: str = ""
    ) -> str:
        """Collect single inference metric"""
        try:
            metric = InferenceMetric(
                model_id=model_id,
                version=version,
                timestamp=datetime.utcnow(),
                prediction_latency=prediction_latency,
                confidence_score=confidence_score,
                input_features=input_features,
                prediction=prediction,
                ground_truth=ground_truth,
                request_id=request_id
            )
            
            # Validate metric
            self._validate_metric(metric)
            
            # Record in Prometheus immediately
            self.prometheus_metrics.record_inference(metric)
            
            # Add to buffer
            await self._add_to_buffer(metric)
            
            # Store in Redis for real-time access
            await self._store_in_redis(metric)
            
            self.logger.debug(
                "Inference metric collected",
                model_id=model_id,
                version=version,
                latency=prediction_latency,
                confidence=confidence_score
            )
            
            return f"{model_id}:{version}:{metric.timestamp.isoformat()}"
            
        except Exception as e:
            self.logger.error("Failed to collect inference metric", error=str(e))
            raise InvalidMetricError(f"Failed to collect metric: {e}")
    
    def _validate_metric(self, metric: InferenceMetric) -> None:
        """Validate metric data"""
        if not metric.model_id or not metric.version:
            raise InvalidMetricError("Model ID and version are required")
        
        if metric.prediction_latency < 0:
            raise InvalidMetricError("Prediction latency must be non-negative")
        
        if not 0 <= metric.confidence_score <= 1:
            raise InvalidMetricError("Confidence score must be between 0 and 1")
        
        if not isinstance(metric.input_features, dict):
            raise InvalidMetricError("Input features must be a dictionary")
    
    async def _add_to_buffer(self, metric: InferenceMetric) -> None:
        """Add metric to in-memory buffer"""
        async with self._buffer_lock:
            self._buffer.append(metric)
            
            # If buffer is full, trigger immediate flush
            if len(self._buffer) >= self.max_memory_buffer:
                await self._flush_buffer()
    
    async def _store_in_redis(self, metric: InferenceMetric) -> None:
        """Store metric in Redis for real-time access"""
        try:
            # Store recent metrics for the model (sliding window)
            key = f"metrics:{metric.model_id}:{metric.version}"
            metric_data = json.dumps(metric.to_dict())
            
            pipe = self.redis_client.pipeline()
            pipe.lpush(key, metric_data)
            pipe.ltrim(key, 0, 999)  # Keep last 1000 metrics
            pipe.expire(key, 3600)  # 1 hour TTL
            await pipe.execute()
            
            # Store aggregated stats
            stats_key = f"stats:{metric.model_id}:{metric.version}"
            await self._update_redis_stats(stats_key, metric)
            
        except redis.RedisError as e:
            self.logger.warning("Failed to store metric in Redis", error=str(e))
    
    async def _update_redis_stats(self, key: str, metric: InferenceMetric) -> None:
        """Update aggregated statistics in Redis"""
        try:
            pipe = self.redis_client.pipeline()
            
            # Update counters and running averages
            pipe.hincrby(key, "total_inferences", 1)
            pipe.hincrbyfloat(key, "total_latency", metric.prediction_latency)
            pipe.hincrbyfloat(key, "total_confidence", metric.confidence_score)
            
            # Set last seen timestamp
            pipe.hset(key, "last_seen", metric.timestamp.isoformat())
            pipe.expire(key, 3600)
            
            await pipe.execute()
            
        except redis.RedisError as e:
            self.logger.warning("Failed to update Redis stats", error=str(e))
    
    async def _background_flush(self) -> None:
        """Background task to flush buffer periodically"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in background flush", error=str(e))
    
    async def _flush_buffer(self) -> None:
        """Flush buffered metrics to PostgreSQL"""
        if not self._buffer:
            return
        
        async with self._buffer_lock:
            if not self._buffer:
                return
            
            batch = self._buffer.copy()
            self._buffer.clear()
        
        try:
            await self._batch_insert_postgres(batch)
            self.logger.info("Flushed metrics batch", count=len(batch))
            
        except Exception as e:
            self.logger.error("Failed to flush metrics batch", error=str(e), count=len(batch))
            # Put metrics back in buffer for retry
            async with self._buffer_lock:
                self._buffer.extend(batch)
    
    async def _batch_insert_postgres(self, metrics: List[InferenceMetric]) -> None:
        """Batch insert metrics into PostgreSQL"""
        if not metrics:
            return
        
        insert_query = """
            INSERT INTO inference_metrics (
                model_id, version, timestamp, prediction_latency, confidence_score,
                input_features, prediction, ground_truth, feature_drift_score,
                concept_drift_score, request_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                for metric in metrics:
                    await conn.execute(
                        insert_query,
                        metric.model_id,
                        metric.version,
                        metric.timestamp,
                        metric.prediction_latency,
                        metric.confidence_score,
                        json.dumps(metric.input_features),
                        json.dumps(metric.prediction),
                        json.dumps(metric.ground_truth) if metric.ground_truth else None,
                        metric.feature_drift_score,
                        metric.concept_drift_score,
                        metric.request_id
                    )
    
    async def get_recent_metrics(
        self,
        model_id: str,
        version: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent metrics from Redis"""
        try:
            key = f"metrics:{model_id}:{version}"
            data = await self.redis_client.lrange(key, 0, limit - 1)
            
            return [json.loads(item) for item in data]
            
        except redis.RedisError as e:
            self.logger.error("Failed to get recent metrics from Redis", error=str(e))
            return []
    
    async def get_model_stats(
        self,
        model_id: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated model statistics from Redis"""
        try:
            key = f"stats:{model_id}:{version}"
            stats = await self.redis_client.hgetall(key)
            
            if not stats:
                return None
            
            total_inferences = int(stats.get("total_inferences", 0))
            if total_inferences == 0:
                return None
            
            return {
                "total_inferences": total_inferences,
                "avg_latency": float(stats["total_latency"]) / total_inferences,
                "avg_confidence": float(stats["total_confidence"]) / total_inferences,
                "last_seen": stats.get("last_seen")
            }
            
        except (redis.RedisError, ValueError, KeyError) as e:
            self.logger.error("Failed to get model stats from Redis", error=str(e))
            return None
    
    async def update_drift_scores(
        self,
        model_id: str,
        version: str,
        feature_drift_score: Optional[float] = None,
        concept_drift_score: Optional[float] = None
    ) -> None:
        """Update drift scores for a model version"""
        try:
            labels = [model_id, version]
            
            if feature_drift_score is not None:
                self.prometheus_metrics.drift_gauge.labels(*labels, "feature").set(feature_drift_score)
            
            if concept_drift_score is not None:
                self.prometheus_metrics.drift_gauge.labels(*labels, "concept").set(concept_drift_score)
            
            # Also store in Redis
            key = f"drift:{model_id}:{version}"
            updates = {}
            if feature_drift_score is not None:
                updates["feature_drift"] = feature_drift_score
            if concept_drift_score is not None:
                updates["concept_drift"] = concept_drift_score
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            await self.redis_client.hmset(key, updates)
            await self.redis_client.expire(key, 3600)
            
        except Exception as e:
            self.logger.error("Failed to update drift scores", error=str(e))
    
    async def update_performance_prediction(
        self,
        model_id: str,
        version: str,
        performance_score: float
    ) -> None:
        """Update performance degradation prediction"""
        try:
            self.prometheus_metrics.performance_gauge.labels(model_id, version).set(performance_score)
            
            # Store in Redis
            key = f"performance:{model_id}:{version}"
            await self.redis_client.hset(key, "score", performance_score)
            await self.redis_client.hset(key, "updated_at", datetime.utcnow().isoformat())
            await self.redis_client.expire(key, 3600)
            
        except Exception as e:
            self.logger.error("Failed to update performance prediction", error=str(e))
    
    @asynccontextmanager
    async def batch_context(self, model_id: str, version: str):
        """Context manager for batch metric collection"""
        batch_metrics = []
        
        def add_metric(
            prediction_latency: float,
            confidence_score: float,
            input_features: Dict[str, Any],
            prediction: Any,
            ground_truth: Optional[Any] = None,
            request_id: str = ""
        ):
            batch_metrics.append({
                'prediction_latency': prediction_latency,
                'confidence_score': confidence_score,
                'input_features': input_features,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'request_id': request_id
            })
        
        try:
            yield add_metric
        finally:
            # Process batch
            if batch_metrics:
                for metric_data in batch_metrics:
                    await self.collect_inference_metric(
                        model_id=model_id,
                        version=version,
                        **metric_data
                    )
    
    async def cleanup(self) -> None:
        """Cleanup resources and connections"""
        try:
            # Cancel background task
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush remaining buffer
            await self._flush_buffer()
            
            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            
            if self.pg_pool:
                await self.pg_pool.close()
            
            self.logger.info("Metrics collector cleaned up")
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))


async def create_metrics_collector(
    postgres_dsn: str,
    redis_url: str,
    **kwargs
) -> MetricsCollector:
    """Factory function to create and initialize metrics collector"""
    collector = MetricsCollector(postgres_dsn, redis_url, **kwargs)
    await collector.initialize()
    return collector