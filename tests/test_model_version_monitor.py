"""Tests for model version monitoring."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.monitoring.model_version_monitor import ModelVersionMonitor, VersionMetrics

@pytest.fixture
def monitor():
    return ModelVersionMonitor(drift_threshold=0.1, window_minutes=60)

def test_record_version_metrics(monitor):
    """Test recording metrics for a version."""
    monitor.record_version_metrics(
        version="v1.0",
        latency=0.1,
        confidence=0.95,
        error_occurred=False
    )
    
    assert "v1.0" in monitor.version_metrics
    assert len(monitor.version_metrics["v1.0"]) == 1
    
    metrics = monitor.version_metrics["v1.0"][0]
    assert metrics.version == "v1.0"
    assert metrics.avg_latency == 0.1
    assert metrics.avg_confidence == 0.95
    assert metrics.error_rate == 0.0

def test_set_baseline_version(monitor):
    """Test setting baseline version."""
    monitor.set_baseline_version("v1.0")
    assert monitor.baseline_version == "v1.0"

def test_detect_version_drift(monitor):
    """Test drift detection between versions."""
    # Record baseline metrics
    monitor.set_baseline_version("v1.0")
    monitor.record_version_metrics("v1.0", 0.1, 0.95, False)
    
    # Record comparison metrics with drift
    monitor.record_version_metrics("v2.0", 0.15, 0.90, False)  # 50% latency increase
    
    drift_results = monitor.detect_version_drift()
    
    assert "v2.0" in drift_results
    assert drift_results["v2.0"]["latency_drift"] == 0.5  # 50% increase
    assert drift_results["v2.0"]["confidence_drift"] > 0
    
def test_get_active_versions(monitor):
    """Test getting active versions."""
    monitor.record_version_metrics("v1.0", 0.1, 0.95, False)
    monitor.record_version_metrics("v2.0", 0.1, 0.95, False)
    
    active = monitor.get_active_versions()
    assert active == {"v1.0", "v2.0"}
    
def test_metrics_window_cleanup(monitor):
    """Test that old metrics are cleaned up."""
    with patch('src.monitoring.model_version_monitor.datetime') as mock_datetime:
        # Set initial time
        base_time = datetime.utcnow()
        mock_datetime.utcnow.return_value = base_time
        
        monitor.record_version_metrics("v1.0", 0.1, 0.95, False)
        assert len(monitor.version_metrics["v1.0"]) == 1
        
        # Move time forward beyond window
        future_time = base_time + timedelta(minutes=120)
        mock_datetime.utcnow.return_value = future_time
        
        monitor.record_version_metrics("v1.0", 0.1, 0.95, False)
        # Old metric should be cleaned up
        assert len(monitor.version_metrics["v1.0"]) == 1