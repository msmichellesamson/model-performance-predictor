# Model Performance Predictor

Real-time ML model performance degradation prediction using inference metrics and drift detection.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │───▶│  Metrics API    │───▶│   Predictor     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │◀───│  Drift Monitor  │    │     Alerts      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │      Redis      │    │   Alertmanager  │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## Core Features

### 🔍 Real-time Monitoring
- **Feature Drift Detection**: Statistical drift using Wasserstein distance
- **Model Performance Tracking**: Accuracy, latency, and confidence metrics
- **Data Quality Monitoring**: Missing values, outliers, schema validation
- **Memory & Resource Monitoring**: Container resource usage and limits

### 🚨 Intelligent Alerting
- **Threshold-based Alerts**: Configurable per-metric thresholds
- **Circuit Breaker**: Automatic model failover on degradation
- **Multi-channel Notifications**: Slack, PagerDuty, email integration
- **Alert Correlation**: Group related performance issues

### 📊 Performance Prediction
- **Degradation Forecasting**: ML-based performance trend prediction
- **Risk Assessment**: Quantify likelihood of model failure
- **Proactive Recommendations**: Suggest retraining, scaling, or rollback

## Quick Start

### Local Development
```bash
# Start infrastructure
docker-compose up -d redis prometheus

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.main:app --reload --port 8000

# Generate test data
python scripts/generate_test_data.py
```

### Production Deployment
```bash
# Deploy infrastructure
terraform -chdir=terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -l app=model-performance-predictor
```

## API Endpoints

### Core Monitoring
- `GET /health` - Service health check
- `GET /drift/status` - Current drift status across features
- `GET /predictions/{model_id}/risk` - Performance degradation risk score
- `POST /alerts/configure` - Configure alert thresholds

### Metrics & Analytics
- `GET /metrics` - Prometheus metrics endpoint
- `GET /drift/distributions/{feature}` - Feature distribution comparison
- `GET /predictions/{model_id}/performance` - Historical performance data

### Configuration
- `POST /drift/threshold` - Update drift detection thresholds
- `PUT /monitoring/config` - Update monitoring configuration

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_TIMEOUT=5

# Monitoring Settings
DRIFT_THRESHOLD=0.5
PERFORMANCE_WINDOW=1000
ALERT_COOLDOWN=300

# Database
DATABASE_URL=postgresql://user:pass@localhost/modelmon

# Prometheus
PROMETHEUS_URL=http://localhost:9090
```

### Drift Detection
```python
# Per-feature thresholds
DRIFT_THRESHOLDS = {
    "age": 0.3,
    "income": 0.5,
    "credit_score": 0.2
}

# Detection methods
DRIFT_METHODS = ["wasserstein", "ks_test", "psi"]
```

## Monitoring & Observability

### Key Metrics
- `model_accuracy_current`: Real-time model accuracy
- `drift_score_current`: Feature drift scores
- `prediction_latency_p99`: 99th percentile latency
- `circuit_breaker_state`: Circuit breaker status

### Dashboards
- **Grafana**: Model Performance Overview
- **Grafana**: Drift Detection Dashboard
- **Prometheus**: Raw metrics and alerting rules

### Alerts
- **Critical**: Model accuracy below 70%
- **Warning**: Feature drift score above threshold
- **Info**: Performance trend degradation detected

## Runbooks

Operational guides for common scenarios:
- [`runbooks/drift-detection.md`](runbooks/drift-detection.md) - Drift detection troubleshooting

## Development

### Project Structure
```
src/
├── api/           # FastAPI endpoints
├── core/          # Core business logic
├── monitoring/    # Monitoring components
├── cache/         # Redis caching layer
├── alerts/        # Alerting system
└── db/           # Database models

tests/
├── unit/         # Unit tests
└── integration/  # Integration tests

terraform/        # Infrastructure as code
k8s/             # Kubernetes manifests
runbooks/        # Operational runbooks
```

### Testing
```bash
# Unit tests
pytest tests/ -v

# Integration tests
pytest tests/integration/ -v

# Load testing
locust -f tests/load/locustfile.py
```

### Tech Stack
- **Backend**: Python 3.11, FastAPI, asyncio
- **Database**: PostgreSQL, Redis
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Terraform (GCP), Kubernetes
- **CI/CD**: GitHub Actions, Docker

## Contributing

1. Write tests for new features
2. Follow type hints and error handling patterns
3. Update README.md with any new endpoints or config
4. Ensure Prometheus metrics are properly labeled

## License

MIT License