import pytest
from unittest.mock import patch, MagicMock
from src.monitoring.memory_monitor import MemoryMonitor

@pytest.fixture
def memory_monitor():
    return MemoryMonitor(threshold_mb=100.0)

@patch('src.monitoring.memory_monitor.psutil.Process')
def test_start_monitoring_success(mock_process, memory_monitor):
    # Mock memory info
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 104857600  # 100MB
    mock_process.return_value.memory_info.return_value = mock_memory_info
    
    result = memory_monitor.start_monitoring('test_model', 'v1.0')
    
    assert 'current_memory_mb' in result
    assert result['current_memory_mb'] == 100.0
    assert 'threshold_exceeded' in result

@patch('src.monitoring.memory_monitor.psutil.Process')
def test_memory_threshold_exceeded(mock_process, memory_monitor):
    # Set baseline
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 52428800  # 50MB baseline
    mock_process.return_value.memory_info.return_value = mock_memory_info
    memory_monitor.start_monitoring('test_model', 'v1.0')
    
    # Simulate memory increase beyond threshold
    mock_memory_info.rss = 209715200  # 200MB (150MB increase > 100MB threshold)
    result = memory_monitor.start_monitoring('test_model', 'v1.0')
    
    assert result['threshold_exceeded'] is True
    assert result['memory_delta_mb'] == 150.0

@patch('src.monitoring.memory_monitor.psutil.virtual_memory')
def test_get_memory_stats(mock_virtual_memory, memory_monitor):
    # Mock system memory
    mock_memory = MagicMock()
    mock_memory.total = 1073741824  # 1GB
    mock_memory.available = 536870912  # 512MB
    mock_memory.percent = 50.0
    mock_virtual_memory.return_value = mock_memory
    
    stats = memory_monitor.get_memory_stats()
    
    assert stats['total_mb'] == 1024.0
    assert stats['available_mb'] == 512.0
    assert stats['used_percent'] == 50.0

def test_reset_baseline(memory_monitor):
    memory_monitor.baseline_memory = 100.0
    memory_monitor.reset_baseline()
    assert memory_monitor.baseline_memory is None