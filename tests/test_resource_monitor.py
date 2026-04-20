import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.monitoring.resource_monitor import ResourceMonitor, ResourceMetrics


class TestResourceMonitor:
    
    @pytest.fixture
    def monitor(self):
        return ResourceMonitor(check_interval=0.1)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_collect_metrics_success(self, mock_net, mock_disk, mock_memory, mock_cpu, monitor):
        # Arrange
        mock_cpu.return_value = 45.5
        mock_memory.return_value = MagicMock(percent=67.8)
        mock_disk.return_value = MagicMock(read_bytes=1024, write_bytes=2048)
        mock_net.return_value = MagicMock(bytes_sent=5000, bytes_recv=3000)
        
        # Act
        metrics = monitor.collect_metrics()
        
        # Assert
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.8
        assert metrics.disk_io_read == 1024
        assert metrics.disk_io_write == 2048
        assert metrics.network_bytes_sent == 5000
        assert metrics.network_bytes_recv == 3000
        assert len(monitor._metrics_history) == 1
    
    def test_detect_resource_spike(self, monitor):
        # Arrange
        high_cpu_metrics = ResourceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=85.0,
            memory_percent=70.0,
            gpu_percent=95.0,
            disk_io_read=0, disk_io_write=0,
            network_bytes_sent=0, network_bytes_recv=0
        )
        
        # Act
        spikes = monitor.detect_resource_spike(high_cpu_metrics)
        
        # Assert
        assert spikes['cpu_spike'] is True
        assert spikes['memory_spike'] is False
        assert spikes['gpu_spike'] is True
    
    def test_get_resource_trend_insufficient_data(self, monitor):
        # Act
        trend = monitor.get_resource_trend()
        
        # Assert
        assert trend['cpu_trend'] == 0.0
        assert trend['memory_trend'] == 0.0
    
    def test_get_resource_trend_with_data(self, monitor):
        # Arrange
        base_time = datetime.now(timezone.utc)
        monitor._metrics_history = [
            ResourceMetrics(base_time, 20.0, 30.0, None, 0, 0, 0, 0),
            ResourceMetrics(base_time, 40.0, 50.0, None, 0, 0, 0, 0),
            ResourceMetrics(base_time, 60.0, 70.0, None, 0, 0, 0, 0)
        ]
        
        # Act
        trend = monitor.get_resource_trend()
        
        # Assert
        assert trend['cpu_trend'] > 0  # Increasing trend
        assert trend['memory_trend'] > 0  # Increasing trend
