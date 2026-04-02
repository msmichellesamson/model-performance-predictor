import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.cache.redis_client import RedisClient


@pytest.fixture
def mock_redis():
    with patch('src.cache.redis_client.redis.Redis') as mock:
        redis_instance = Mock()
        mock.return_value = redis_instance
        yield redis_instance


@pytest.fixture
def redis_client(mock_redis):
    return RedisClient(host='localhost', port=6379, db=0)


class TestRedisClient:
    def test_init_creates_connection(self, mock_redis):
        client = RedisClient(host='test-host', port=1234, db=1)
        assert client.redis is not None
    
    def test_set_prediction_success(self, redis_client, mock_redis):
        mock_redis.setex.return_value = True
        
        result = redis_client.set_prediction('model_123', {'score': 0.95, 'drift': 0.1})
        
        assert result is True
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == 'pred:model_123'
        assert args[1] == 300  # default TTL
        assert json.loads(args[2]) == {'score': 0.95, 'drift': 0.1}
    
    def test_set_prediction_with_custom_ttl(self, redis_client, mock_redis):
        mock_redis.setex.return_value = True
        
        redis_client.set_prediction('model_456', {'score': 0.8}, ttl=600)
        
        args = mock_redis.setex.call_args[0]
        assert args[1] == 600
    
    def test_get_prediction_exists(self, redis_client, mock_redis):
        expected_data = {'score': 0.92, 'drift': 0.05}
        mock_redis.get.return_value = json.dumps(expected_data).encode()
        
        result = redis_client.get_prediction('model_789')
        
        assert result == expected_data
        mock_redis.get.assert_called_once_with('pred:model_789')
    
    def test_get_prediction_not_exists(self, redis_client, mock_redis):
        mock_redis.get.return_value = None
        
        result = redis_client.get_prediction('nonexistent')
        
        assert result is None
    
    def test_get_prediction_invalid_json(self, redis_client, mock_redis):
        mock_redis.get.return_value = b'invalid json'
        
        result = redis_client.get_prediction('model_bad')
        
        assert result is None
    
    def test_cache_metrics_batch(self, redis_client, mock_redis):
        mock_redis.pipeline.return_value.__enter__ = Mock(return_value=mock_redis)
        mock_redis.pipeline.return_value.__exit__ = Mock(return_value=None)
        mock_redis.execute.return_value = [True, True]
        
        metrics = [
            {'model_id': 'model_1', 'accuracy': 0.95, 'timestamp': 1234567890},
            {'model_id': 'model_2', 'accuracy': 0.88, 'timestamp': 1234567891}
        ]
        
        result = redis_client.cache_metrics_batch(metrics)
        
        assert result is True
        assert mock_redis.setex.call_count == 2
    
    def test_get_cached_metrics(self, redis_client, mock_redis):
        expected_metrics = [{'accuracy': 0.95}, {'accuracy': 0.88}]
        mock_redis.mget.return_value = [json.dumps(m).encode() for m in expected_metrics]
        
        result = redis_client.get_cached_metrics(['key1', 'key2'])
        
        assert result == expected_metrics
        mock_redis.mget.assert_called_once_with(['metrics:key1', 'metrics:key2'])