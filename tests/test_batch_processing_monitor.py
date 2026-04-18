"""Tests for batch processing monitor."""

import pytest
from unittest.mock import patch
from src.monitoring.batch_processing_monitor import BatchProcessingMonitor, BatchMetrics


class TestBatchProcessingMonitor:
    """Test batch processing monitor functionality."""
    
    def setup_method(self):
        """Setup test monitor."""
        self.monitor = BatchProcessingMonitor(window_size=5)
    
    def test_record_batch_updates_metrics(self):
        """Test that recording batch updates all metrics."""
        self.monitor.record_batch(
            batch_size=10,
            processing_time=0.5,
            queue_depth=25,
            error_count=0
        )
        
        metrics = self.monitor.get_metrics()
        assert metrics is not None
        assert metrics.batch_size == 10
        assert metrics.processing_time == 0.5
        assert metrics.queue_depth == 25
        assert metrics.throughput == 20.0  # 10 items / 0.5 seconds
        assert metrics.error_count == 0
    
    def test_throughput_calculation(self):
        """Test throughput calculation over multiple batches."""
        # Record multiple batches
        self.monitor.record_batch(10, 1.0, 5)
        self.monitor.record_batch(20, 2.0, 3)
        self.monitor.record_batch(15, 1.5, 2)
        
        metrics = self.monitor.get_metrics()
        assert metrics is not None
        # Total: 45 items in 4.5 seconds = 10 items/second
        assert abs(metrics.throughput - 10.0) < 0.1
    
    def test_error_counting(self):
        """Test error count tracking."""
        self.monitor.record_batch(10, 0.5, 5, error_count=2)
        self.monitor.record_batch(15, 0.8, 3, error_count=1)
        
        metrics = self.monitor.get_metrics()
        assert metrics is not None
        assert metrics.error_count == 3
    
    def test_window_size_limit(self):
        """Test that window size is respected."""
        # Record more batches than window size
        for i in range(10):
            self.monitor.record_batch(i + 1, 0.1, 1)
        
        assert len(self.monitor._batch_times) == 5
        assert len(self.monitor._batch_sizes) == 5
    
    def test_no_metrics_when_empty(self):
        """Test that no metrics returned when no data recorded."""
        metrics = self.monitor.get_metrics()
        assert metrics is None
    
    def test_zero_processing_time_handling(self):
        """Test handling of zero processing time."""
        self.monitor.record_batch(10, 0.0, 5)
        
        metrics = self.monitor.get_metrics()
        assert metrics is not None
        assert metrics.throughput == 0.0