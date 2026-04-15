import pytest
from unittest.mock import Mock, patch
import json
from src.api.alerts import alerts_bp
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(alerts_bp, url_prefix='/api/v1')
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@patch('src.api.alerts.redis_client')
class TestAlertsAPI:
    
    def test_get_active_alerts_empty(self, mock_redis, client):
        mock_redis.get_active_alerts.return_value = []
        
        response = client.get('/api/v1/alerts/active')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['alerts'] == []
        assert data['total'] == 0

    def test_get_active_alerts_with_data(self, mock_redis, client):
        mock_alerts = [
            {'id': 'alert1', 'type': 'drift', 'severity': 'high', 'timestamp': '2024-01-01T00:00:00Z'},
            {'id': 'alert2', 'type': 'latency', 'severity': 'medium', 'timestamp': '2024-01-01T01:00:00Z'}
        ]
        mock_redis.get_active_alerts.return_value = mock_alerts
        
        response = client.get('/api/v1/alerts/active')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['alerts'] == mock_alerts
        assert data['total'] == 2

    def test_acknowledge_alert_success(self, mock_redis, client):
        mock_redis.acknowledge_alert.return_value = True
        
        response = client.post('/api/v1/alerts/alert123/acknowledge')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'acknowledged' in data['message']
        mock_redis.acknowledge_alert.assert_called_once_with('alert123')

    def test_acknowledge_alert_not_found(self, mock_redis, client):
        mock_redis.acknowledge_alert.return_value = False
        
        response = client.post('/api/v1/alerts/nonexistent/acknowledge')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'not found' in data['message']

    def test_get_alert_history(self, mock_redis, client):
        mock_history = [
            {'id': 'hist1', 'resolved_at': '2024-01-01T02:00:00Z', 'duration': 3600}
        ]
        mock_redis.get_alert_history.return_value = mock_history
        
        response = client.get('/api/v1/alerts/history?limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['alerts'] == mock_history
        mock_redis.get_alert_history.assert_called_once_with(limit=10)

    def test_redis_connection_error(self, mock_redis, client):
        mock_redis.get_active_alerts.side_effect = Exception('Redis connection failed')
        
        response = client.get('/api/v1/alerts/active')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'Redis connection failed' in data['message']
