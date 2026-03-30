"""
Prometheus metrics exporter for ML model performance monitoring.

This module provides custom Prometheus metrics for tracking model performance,
drift detection, and prediction accuracy in real-time.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Enum, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import HTTPException
import asyncpg
from redis import Redis

from ..core.exceptions import MetricsExportError, DatabaseError
from ..cache.redis_client import RedisClient
from ..db.models import ModelPerformanceRecord, PredictionRecord


logger = structlog.get_logger(__name__)


class ModelPerformanceMetrics:
    """Custom Prometheus metrics for ML model performance monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics collectors."""
        self.registry = registry or CollectorRegistry()
        
        # Model inference metrics
        self.inference_requests_total = Counter(
            'ml_inference_requests_total',
            'Total number of inference requests',
            ['model_id', 'version', 'status'],
            registry=self.registry
        )
        
        self.inference_duration_seconds = Histogram(
            'ml_inference_duration_seconds',
            'Time spent processing inference requests',
            ['model_id', 'version'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy score',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'ml_model_precision',
            'Current model precision score',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'ml_model_recall',
            'Current model recall score',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'ml_model_f1_score',
            'Current model F1 score',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        # Drift detection metrics
        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            'Data drift detection score (0-1)',
            ['model_id', 'feature_name'],
            registry=self.registry
        )
        
        self.concept_drift_score = Gauge(
            'ml_concept_drift_score',
            'Concept drift detection score (0-1)',
            ['model_id'],
            registry=self.registry
        )
        
        self.drift_alerts_total = Counter(
            'ml_drift_alerts_total',
            'Total number of drift alerts triggered',
            ['model_id', 'drift_type', 'severity'],
            registry=self.registry
        )
        
        # Performance degradation metrics
        self.performance_degradation_score = Gauge(
            'ml_performance_degradation_score',
            'Predicted performance degradation score (0-1)',
            ['model_id'],
            registry=self.registry
        )
        
        self.model_health_status = Enum(
            'ml_model_health_status',
            'Current model health status',
            ['model_id'],
            states=['healthy', 'warning', 'critical', 'unknown'],
            registry=self.registry
        )
        
        # Resource utilization metrics
        self.model_memory_usage_bytes = Gauge(
            'ml_model_memory_usage_bytes',
            'Memory usage by model in bytes',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        self.model_cpu_usage_percent = Gauge(
            'ml_model_cpu_usage_percent',
            'CPU usage by model as percentage',
            ['model_id', 'version'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'ml_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'ml_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )


class PrometheusExporter:
    """Prometheus metrics exporter for ML model performance monitoring."""
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis_client: RedisClient,
        export_interval: int = 30,
        metrics_retention_hours: int = 24
    ):
        """
        Initialize Prometheus exporter.
        
        Args:
            db_pool: Database connection pool
            redis_client: Redis client for caching
            export_interval: Metrics export interval in seconds
            metrics_retention_hours: How long to keep metrics data
        """
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.export_interval = export_interval
        self.metrics_retention_hours = metrics_retention_hours
        
        self.metrics = ModelPerformanceMetrics()
        self._export_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            "Initialized Prometheus exporter",
            export_interval=export_interval,
            retention_hours=metrics_retention_hours
        )
    
    async def start_exporter(self) -> None:
        """Start the metrics export background task."""
        if self._running:
            logger.warning("Prometheus exporter already running")
            return
        
        self._running = True
        self._export_task = asyncio.create_task(self._export_loop())
        logger.info("Started Prometheus metrics exporter")
    
    async def stop_exporter(self) -> None:
        """Stop the metrics export background task."""
        if not self._running:
            return
        
        self._running = False
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped Prometheus metrics exporter")
    
    async def _export_loop(self) -> None:
        """Main export loop that runs in the background."""
        while self._running:
            try:
                await self._export_metrics()
                await asyncio.sleep(self.export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in metrics export loop",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _export_metrics(self) -> None:
        """Export all metrics to Prometheus."""
        try:
            # Export model performance metrics
            await self._export_model_performance()
            
            # Export drift detection metrics
            await self._export_drift_metrics()
            
            # Export resource utilization metrics
            await self._export_resource_metrics()
            
            # Export cache metrics
            await self._export_cache_metrics()
            
            logger.debug("Successfully exported metrics to Prometheus")
            
        except Exception as e:
            logger.error("Failed to export metrics", error=str(e))
            raise MetricsExportError(f"Failed to export metrics: {e}") from e
    
    async def _export_model_performance(self) -> None:
        """Export model performance metrics."""
        try:
            query = """
                SELECT 
                    model_id,
                    model_version,
                    accuracy,
                    precision_score,
                    recall,
                    f1_score,
                    performance_score,
                    status
                FROM model_performance_records
                WHERE created_at >= $1
                ORDER BY created_at DESC
            """
            
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, cutoff_time)
            
            # Group by model_id and get latest metrics
            latest_metrics: Dict[str, Any] = {}
            for row in rows:
                key = f"{row['model_id']}-{row['model_version']}"
                if key not in latest_metrics:
                    latest_metrics[key] = row
            
            # Update Prometheus metrics
            for row in latest_metrics.values():
                labels = [row['model_id'], row['model_version']]
                
                if row['accuracy'] is not None:
                    self.metrics.model_accuracy.labels(*labels).set(row['accuracy'])
                
                if row['precision_score'] is not None:
                    self.metrics.model_precision.labels(*labels).set(row['precision_score'])
                
                if row['recall'] is not None:
                    self.metrics.model_recall.labels(*labels).set(row['recall'])
                
                if row['f1_score'] is not None:
                    self.metrics.model_f1_score.labels(*labels).set(row['f1_score'])
                
                if row['performance_score'] is not None:
                    self.metrics.performance_degradation_score.labels(
                        row['model_id']
                    ).set(1.0 - row['performance_score'])
                
                # Set health status
                status = row['status'] or 'unknown'
                self.metrics.model_health_status.labels(
                    row['model_id']
                ).state(status)
            
        except Exception as e:
            logger.error("Failed to export model performance metrics", error=str(e))
            raise
    
    async def _export_drift_metrics(self) -> None:
        """Export drift detection metrics."""
        try:
            # Get data drift scores
            drift_query = """
                SELECT 
                    model_id,
                    feature_name,
                    drift_score,
                    drift_type,
                    severity
                FROM drift_detection_results
                WHERE created_at >= $1
                ORDER BY created_at DESC
            """
            
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            async with self.db_pool.acquire() as conn:
                drift_rows = await conn.fetch(drift_query, cutoff_time)
            
            # Update drift metrics
            for row in drift_rows:
                if row['drift_type'] == 'data_drift' and row['feature_name']:
                    self.metrics.data_drift_score.labels(
                        row['model_id'],
                        row['feature_name']
                    ).set(row['drift_score'])
                
                elif row['drift_type'] == 'concept_drift':
                    self.metrics.concept_drift_score.labels(
                        row['model_id']
                    ).set(row['drift_score'])
            
            # Count drift alerts
            alert_query = """
                SELECT 
                    model_id,
                    drift_type,
                    severity,
                    COUNT(*) as alert_count
                FROM drift_detection_results
                WHERE created_at >= $1 AND drift_score > 0.7
                GROUP BY model_id, drift_type, severity
            """
            
            async with self.db_pool.acquire() as conn:
                alert_rows = await conn.fetch(alert_query, cutoff_time)
            
            for row in alert_rows:
                self.metrics.drift_alerts_total.labels(
                    row['model_id'],
                    row['drift_type'],
                    row['severity']
                ).inc(row['alert_count'])
                
        except Exception as e:
            logger.error("Failed to export drift metrics", error=str(e))
            raise
    
    async def _export_resource_metrics(self) -> None:
        """Export resource utilization metrics from Redis cache."""
        try:
            # Get resource metrics from Redis
            resource_keys = await self.redis_client.keys("resource:*")
            
            for key in resource_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                parts = key.split(':')
                if len(parts) < 4:
                    continue
                
                model_id = parts[1]
                version = parts[2]
                metric_type = parts[3]
                
                value_str = await self.redis_client.get(key)
                if not value_str:
                    continue
                
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue
                
                labels = [model_id, version]
                
                if metric_type == 'memory':
                    self.metrics.model_memory_usage_bytes.labels(*labels).set(value)
                elif metric_type == 'cpu':
                    self.metrics.model_cpu_usage_percent.labels(*labels).set(value)
                    
        except Exception as e:
            logger.error("Failed to export resource metrics", error=str(e))
            raise
    
    async def _export_cache_metrics(self) -> None:
        """Export cache hit/miss metrics."""
        try:
            cache_stats = await self.redis_client.info('stats')
            
            if 'keyspace_hits' in cache_stats:
                self.metrics.cache_hits_total.labels('redis').inc(
                    cache_stats['keyspace_hits']
                )
            
            if 'keyspace_misses' in cache_stats:
                self.metrics.cache_misses_total.labels('redis').inc(
                    cache_stats['keyspace_misses']
                )
                
        except Exception as e:
            logger.error("Failed to export cache metrics", error=str(e))
            raise
    
    def record_inference_request(
        self,
        model_id: str,
        version: str,
        duration: float,
        status: str = "success"
    ) -> None:
        """Record an inference request metric."""
        labels = [model_id, version, status]
        self.metrics.inference_requests_total.labels(*labels).inc()
        self.metrics.inference_duration_seconds.labels(
            model_id, version
        ).observe(duration)
    
    def update_drift_score(
        self,
        model_id: str,
        drift_score: float,
        drift_type: str = "concept_drift",
        feature_name: Optional[str] = None
    ) -> None:
        """Update drift detection score."""
        if drift_type == "data_drift" and feature_name:
            self.metrics.data_drift_score.labels(
                model_id, feature_name
            ).set(drift_score)
        elif drift_type == "concept_drift":
            self.metrics.concept_drift_score.labels(model_id).set(drift_score)
    
    def trigger_drift_alert(
        self,
        model_id: str,
        drift_type: str,
        severity: str = "warning"
    ) -> None:
        """Record a drift alert."""
        self.metrics.drift_alerts_total.labels(
            model_id, drift_type, severity
        ).inc()
    
    def get_metrics_content(self) -> bytes:
        """Get Prometheus metrics in exposition format."""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST
    
    @asynccontextmanager
    async def timed_operation(self, model_id: str, version: str, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            logger.error(
                "Timed operation failed",
                model_id=model_id,
                version=version,
                operation=operation,
                error=str(e)
            )
            raise
        finally:
            duration = time.time() - start_time
            self.record_inference_request(model_id, version, duration, status)


class MetricsCollectionError(Exception):
    """Exception raised when metrics collection fails."""
    pass