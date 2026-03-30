from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, JSON, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class ModelStatus(str, Enum):
    """Model deployment status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    FAILED = "failed"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of model drift."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    TARGET_DRIFT = "target_drift"


class Model(Base):
    """ML model metadata and configuration."""
    
    __tablename__ = "models"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True)
    version = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False)
    framework = Column(String(100), nullable=False)  # sklearn, pytorch, tensorflow
    status = Column(String(20), nullable=False, default=ModelStatus.ACTIVE.value)
    
    # Model configuration
    input_schema = Column(JSONB, nullable=False)
    output_schema = Column(JSONB, nullable=False)
    feature_names = Column(JSONB, nullable=False)
    target_column = Column(String(255))
    
    # Performance thresholds
    accuracy_threshold = Column(Float, default=0.8)
    precision_threshold = Column(Float, default=0.8)
    recall_threshold = Column(Float, default=0.8)
    f1_threshold = Column(Float, default=0.8)
    drift_threshold = Column(Float, default=0.05)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")
    metrics = relationship("ModelMetric", back_populates="model", cascade="all, delete-orphan")
    drift_detections = relationship("DriftDetection", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='_model_name_version_uc'),
        CheckConstraint("status IN ('active', 'inactive', 'degraded', 'failed')", name='_model_status_check'),
        CheckConstraint("accuracy_threshold >= 0.0 AND accuracy_threshold <= 1.0", name='_accuracy_threshold_check'),
        CheckConstraint("drift_threshold >= 0.0 AND drift_threshold <= 1.0", name='_drift_threshold_check'),
        Index('idx_models_status_created', 'status', 'created_at'),
        Index('idx_models_name_version', 'name', 'version'),
    )
    
    def __repr__(self) -> str:
        return f"<Model(name={self.name}, version={self.version}, status={self.status})>"


class Prediction(Base):
    """Individual model predictions with features and metadata."""
    
    __tablename__ = "predictions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Prediction data
    prediction_value = Column(JSONB, nullable=False)
    prediction_probability = Column(Float)
    confidence_score = Column(Float)
    
    # Input features
    features = Column(JSONB, nullable=False)
    feature_hash = Column(String(64), nullable=False)  # SHA256 of features for deduplication
    
    # Actual outcome (for performance calculation)
    actual_value = Column(JSONB)
    is_correct = Column(Boolean)
    
    # Timing and metadata
    inference_time_ms = Column(Float, nullable=False)
    predicted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    feedback_received_at = Column(DateTime(timezone=True))
    
    # Request metadata
    request_id = Column(String(255))
    user_id = Column(String(255))
    session_id = Column(String(255))
    source = Column(String(100), default="api")
    
    # Relationships
    model = relationship("Model", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_predictions_model_predicted_at', 'model_id', 'predicted_at'),
        Index('idx_predictions_feature_hash', 'feature_hash'),
        Index('idx_predictions_request_id', 'request_id'),
        Index('idx_predictions_actual_feedback', 'model_id', 'actual_value', 'feedback_received_at'),
        CheckConstraint("inference_time_ms >= 0", name='_inference_time_positive_check'),
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0.0 AND confidence_score <= 1.0)", 
                       name='_confidence_score_check'),
    )
    
    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, model_id={self.model_id}, predicted_at={self.predicted_at})>"


class ModelMetric(Base):
    """Aggregated model performance metrics over time windows."""
    
    __tablename__ = "model_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Time window
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    window_size_minutes = Column(Integer, nullable=False)
    
    # Volume metrics
    total_predictions = Column(Integer, nullable=False, default=0)
    predictions_with_feedback = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Latency metrics
    avg_inference_time_ms = Column(Float)
    p50_inference_time_ms = Column(Float)
    p95_inference_time_ms = Column(Float)
    p99_inference_time_ms = Column(Float)
    
    # Distribution metrics
    avg_confidence = Column(Float)
    prediction_distribution = Column(JSONB)  # Distribution of prediction values
    feature_statistics = Column(JSONB)  # Min/max/mean/std for each feature
    
    # Error rates
    error_rate = Column(Float, default=0.0)
    timeout_rate = Column(Float, default=0.0)
    
    # Metadata
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model = relationship("Model", back_populates="metrics")
    
    __table_args__ = (
        UniqueConstraint('model_id', 'window_start', 'window_size_minutes', 
                        name='_model_metric_window_uc'),
        Index('idx_model_metrics_model_window', 'model_id', 'window_start', 'window_end'),
        Index('idx_model_metrics_calculated_at', 'calculated_at'),
        CheckConstraint("window_end > window_start", name='_window_end_after_start_check'),
        CheckConstraint("window_size_minutes > 0", name='_window_size_positive_check'),
        CheckConstraint("total_predictions >= 0", name='_total_predictions_positive_check'),
        CheckConstraint("predictions_with_feedback >= 0 AND predictions_with_feedback <= total_predictions", 
                       name='_feedback_predictions_check'),
        CheckConstraint("error_rate >= 0.0 AND error_rate <= 1.0", name='_error_rate_check'),
    )
    
    def __repr__(self) -> str:
        return f"<ModelMetric(model_id={self.model_id}, window_start={self.window_start}, accuracy={self.accuracy})>"


class DriftDetection(Base):
    """Drift detection results for models."""
    
    __tablename__ = "drift_detections"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Detection configuration
    drift_type = Column(String(50), nullable=False)
    detection_method = Column(String(100), nullable=False)  # ks_test, psi, wasserstein
    
    # Time window
    reference_start = Column(DateTime(timezone=True), nullable=False)
    reference_end = Column(DateTime(timezone=True), nullable=False)
    current_start = Column(DateTime(timezone=True), nullable=False)
    current_end = Column(DateTime(timezone=True), nullable=False)
    
    # Drift results
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float)
    is_drift_detected = Column(Boolean, nullable=False)
    confidence_level = Column(Float, default=0.95)
    
    # Feature-level drift (for data drift)
    feature_drift_scores = Column(JSONB)  # Per-feature drift scores
    drifted_features = Column(JSONB)  # List of features with significant drift
    
    # Statistical details
    reference_statistics = Column(JSONB)  # Reference distribution stats
    current_statistics = Column(JSONB)  # Current distribution stats
    test_statistics = Column(JSONB)  # Test-specific statistics
    
    # Metadata
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    sample_size_reference = Column(Integer, nullable=False)
    sample_size_current = Column(Integer, nullable=False)
    
    # Relationships
    model = relationship("Model", back_populates="drift_detections")
    
    __table_args__ = (
        Index('idx_drift_detections_model_detected_at', 'model_id', 'detected_at'),
        Index('idx_drift_detections_drift_type', 'drift_type', 'is_drift_detected'),
        CheckConstraint("drift_type IN ('data_drift', 'concept_drift', 'prediction_drift', 'target_drift')", 
                       name='_drift_type_check'),
        CheckConstraint("reference_end > reference_start", name='_reference_window_check'),
        CheckConstraint("current_end > current_start", name='_current_window_check'),
        CheckConstraint("drift_score >= 0.0", name='_drift_score_positive_check'),
        CheckConstraint("p_value IS NULL OR (p_value >= 0.0 AND p_value <= 1.0)", name='_p_value_check'),
        CheckConstraint("confidence_level > 0.0 AND confidence_level < 1.0", name='_confidence_level_check'),
        CheckConstraint("sample_size_reference > 0 AND sample_size_current > 0", name='_sample_size_check'),
    )
    
    def __repr__(self) -> str:
        return f"<DriftDetection(model_id={self.model_id}, drift_type={self.drift_type}, is_drift_detected={self.is_drift_detected})>"


class Alert(Base):
    """Model performance and drift alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Alert details
    alert_type = Column(String(100), nullable=False)  # performance_degradation, drift_detected, high_latency
    severity = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    
    # Alert conditions
    metric_name = Column(String(100))
    threshold_value = Column(Float)
    actual_value = Column(Float)
    comparison_operator = Column(String(10))  # >, <, >=, <=, ==, !=
    
    # Related entities
    related_metric_id = Column(PG_UUID(as_uuid=True), ForeignKey('model_metrics.id'))
    related_drift_id = Column(PG_UUID(as_uuid=True), ForeignKey('drift_detections.id'))
    
    # Alert lifecycle
    is_active = Column(Boolean, nullable=False, default=True)
    is_acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(String(255))
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Alert context
    context = Column(JSONB)  # Additional context data
    
    # Relationships
    model = relationship("Model", back_populates="alerts")
    related_metric = relationship("ModelMetric", foreign_keys=[related_metric_id])
    related_drift = relationship("DriftDetection", foreign_keys=[related_drift_id])
    
    __table_args__ = (
        Index('idx_alerts_model_created_at', 'model_id', 'created_at'),
        Index('idx_alerts_severity_active', 'severity', 'is_active'),
        Index('idx_alerts_type_active', 'alert_type', 'is_active'),
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='_alert_severity_check'),
        CheckConstraint("comparison_operator IN ('>', '<', '>=', '<=', '==', '!=')", 
                       name='_comparison_operator_check'),
        CheckConstraint("acknowledged_at IS NULL OR acknowledged_by IS NOT NULL", 
                       name='_acknowledgment_check'),
        CheckConstraint("resolved_at IS NULL OR resolved_at >= created_at", name='_resolution_time_check'),
    )
    
    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, model_id={self.model_id}, alert_type={self.alert_type}, severity={self.severity})>"


class ModelBaseline(Base):
    """Baseline performance metrics for comparison."""
    
    __tablename__ = "model_baselines"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PG_UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    # Baseline period
    baseline_start = Column(DateTime(timezone=True), nullable=False)
    baseline_end = Column(DateTime(timezone=True), nullable=False)
    
    # Baseline metrics
    baseline_accuracy = Column(Float, nullable=False)
    baseline_precision = Column(Float)
    baseline_recall = Column(Float)
    baseline_f1_score = Column(Float)
    baseline_auc_roc = Column(Float)
    
    # Baseline distribution statistics
    baseline_feature_stats = Column(JSONB, nullable=False)
    baseline_prediction_dist = Column(JSONB, nullable=False)
    
    # Metadata
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Sample information
    sample_size = Column(Integer, nullable=False)
    data_quality_score = Column(Float)
    
    # Relationships
    model = relationship("Model", foreign_keys=[model_id])
    
    __table_args__ = (
        Index('idx_baselines_model_active', 'model_id', 'is_active'),
        Index('idx_baselines_created_at', 'created_at'),
        CheckConstraint("baseline_end > baseline_start", name='_baseline_window_check'),
        CheckConstraint("sample_size > 0", name='_baseline_sample_size_check'),
        CheckConstraint("baseline_accuracy >= 0.0 AND baseline_accuracy <= 1.0", 
                       name='_baseline_accuracy_check'),
        CheckConstraint("data_quality_score IS NULL OR (data_quality_score >= 0.0 AND data_quality_score <= 1.0)", 
                       name='_data_quality_score_check'),
    )
    
    def __repr__(self) -> str:
        return f"<ModelBaseline(model_id={self.model_id}, baseline_accuracy={self.baseline_accuracy}, is_active={self.is_active})>"