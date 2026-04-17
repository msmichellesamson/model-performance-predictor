import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.monitoring.data_quality_monitor import DataQualityMonitor


@pytest.fixture
def monitor():
    redis_client = Mock()
    return DataQualityMonitor(redis_client=redis_client)


def test_check_missing_values(monitor):
    # Test with missing values
    data = {'feature1': [1, 2, None, 4], 'feature2': [1, 2, 3, 4]}
    result = monitor.check_missing_values(data)
    
    assert result['has_issues'] is True
    assert result['missing_rate'] == 0.125  # 1 missing out of 8 total
    assert 'feature1' in result['affected_features']


def test_check_missing_values_clean(monitor):
    # Test with no missing values
    data = {'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]}
    result = monitor.check_missing_values(data)
    
    assert result['has_issues'] is False
    assert result['missing_rate'] == 0.0
    assert len(result['affected_features']) == 0


def test_check_outliers(monitor):
    # Test with outliers using IQR method
    data = {'feature1': [1, 2, 3, 4, 100]}  # 100 is clear outlier
    result = monitor.check_outliers(data)
    
    assert result['has_issues'] is True
    assert result['outlier_rate'] > 0
    assert 'feature1' in result['affected_features']


def test_check_outliers_normal(monitor):
    # Test with normal distribution
    data = {'feature1': [1, 2, 3, 4, 5]}
    result = monitor.check_outliers(data)
    
    assert result['has_issues'] is False
    assert result['outlier_rate'] == 0


def test_check_data_types(monitor):
    # Test with mixed types
    data = {'feature1': [1, 2, '3', 4], 'feature2': [1.0, 2.0, 3.0, 4.0]}
    result = monitor.check_data_types(data)
    
    assert result['has_issues'] is True
    assert 'feature1' in result['type_mismatches']


def test_check_data_types_consistent(monitor):
    # Test with consistent types
    data = {'feature1': [1, 2, 3, 4], 'feature2': [1.0, 2.0, 3.0, 4.0]}
    result = monitor.check_data_types(data)
    
    assert result['has_issues'] is False
    assert len(result['type_mismatches']) == 0


def test_validate_schema(monitor):
    expected_schema = {'feature1': 'int', 'feature2': 'float'}
    data = {'feature1': [1, 2, 3], 'feature2': [1.0, 2.0, 3.0]}
    
    result = monitor.validate_schema(data, expected_schema)
    assert result['valid'] is True
    assert len(result['errors']) == 0


def test_validate_schema_missing_features(monitor):
    expected_schema = {'feature1': 'int', 'feature2': 'float'}
    data = {'feature1': [1, 2, 3]}  # Missing feature2
    
    result = monitor.validate_schema(data, expected_schema)
    assert result['valid'] is False
    assert any('Missing required feature' in error for error in result['errors'])


@patch('src.monitoring.data_quality_monitor.time.time')
def test_monitor_stores_metrics(mock_time, monitor):
    mock_time.return_value = 1234567890
    
    data = {'feature1': [1, 2, 3, 4], 'feature2': [1.0, 2.0, 3.0, 4.0]}
    monitor.monitor(data)
    
    # Verify Redis calls were made
    assert monitor.redis_client.hset.called
    assert monitor.redis_client.expire.called