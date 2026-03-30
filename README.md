# ML Model Performance Predictor

[![CI/CD](https://github.com/michellesamson/ml-performance-predictor/workflows/CI/badge.svg)](https://github.com/michellesamson/ml-performance-predictor/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time ML model performance degradation prediction using statistical drift detection and inference pattern analysis. Continuously monitors production ML models and predicts performance drops before they impact users through advanced time-series analysis and multi-dimensional drift detection.

## Skills Demonstrated

- **AI/ML Engineering**: Statistical drift detection, time-series prediction, feature engineering on inference metrics
- **Backend Systems**: High-throughput FastAPI service with async processing and proper error handling
- **SRE/Observability**: Custom Prometheus metrics, alerting, performance monitoring, reliability patterns
- **Database Engineering**: PostgreSQL time-series optimization, Redis caching strategies, query performance
- **Infrastructure**: Terraform-managed GCP infrastructure, auto-scaling, monitoring stack deployment
- **DevOps**: Multi-stage Docker builds, CI/CD with model validation, GitOps deployment patterns

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │───▶│  Inference      │───▶│   Performance   │
│   (External)    │    │  Metrics        │    │   Predictor     │
└─────────────────┘    │  Collection     │    │   (This System) │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │◀───│     Redis       │───▶│   Prometheus    │
│  Time-Series    │    │   Real-time     │    │    Metrics      │
│   Storage       │    │    Cache        │    │   & Alerting    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Drift         │    │   FastAPI       │    │    Grafana      │
│  Detection      │◀───│  REST API       │───▶│  Dashboards     │
│  Algorithms     │    │  (Async)        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/michellesamson/ml-performance-predictor.git
cd ml-performance-predictor

# Local development with Docker Compose
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src

# Start the service
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Generate synthetic test data
python scripts/generate_test_data.py --models 5 --days 30
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ml_predictor
REDIS_URL=redis://localhost:6379/0

# ML Configuration
DRIFT_THRESHOLD=0.15
PREDICTION_WINDOW_HOURS=24
MIN_SAMPLES_FOR_PREDICTION=100

# Monitoring
PROMETHEUS_PORT=8001
LOG_LEVEL=INFO

# GCP (for production)
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
```

### Model Configuration

```yaml
# config/models.yml
drift_detection:
  methods:
    - kolmogorov_smirnov
    - population_stability_index
    - jensen_shannon_divergence
  
performance_prediction:
  algorithm: isolation_forest
  contamination: 0.1
  window_size: 168  # hours
```

## Infrastructure

Deploy to GCP using Terraform:

```bash
# Initialize Terraform
cd terraform/
terraform init

# Plan deployment
terraform plan -var="project_id=your-gcp-project"

# Deploy infrastructure
terraform apply

# Get service URLs
terraform output service_url
terraform output grafana_url
```

### Infrastructure Components

- **Cloud Run**: Auto-scaling FastAPI service
- **Cloud SQL (PostgreSQL)**: Time-series metrics storage with read replicas
- **Memorystore (Redis)**: Sub-second prediction caching
- **Cloud Monitoring**: Prometheus metrics collection
- **Cloud Load Balancer**: High availability with health checks

## API Usage

### Submit Inference Metrics

```bash
curl -X POST "http://localhost:8000/api/v1/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "recommendation-engine-v2",
    "timestamp": "2024-01-20T15:30:00Z",
    "inference_latency": 45.2,
    "prediction_confidence": 0.87,
    "input_features": {
      "feature_1": 0.23,
      "feature_2": 1.45
    },
    "prediction": 0.92
  }'
```

### Get Performance Predictions

```bash
curl "http://localhost:8000/api/v1/models/recommendation-engine-v2/prediction"

# Response
{
  "model_id": "recommendation-engine-v2",
  "predicted_degradation": {
    "probability": 0.23,
    "confidence": 0.91,
    "time_to_degradation_hours": 18.5
  },
  "drift_detected": {
    "feature_drift": 0.12,
    "prediction_drift": 0.08,
    "methods_triggered": ["kolmogorov_smirnov"]
  },
  "recommendations": [
    "Monitor feature_2 closely - showing high drift",
    "Consider model retraining in next 12 hours"
  ]
}
```

### Health and Metrics

```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8001/metrics
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/test_predictor.py -v

# Integration tests
pytest tests/integration/ -v

# Load tests
pytest tests/load/ -v --timeout=300

# Coverage report
pytest --cov=src --cov-report=html
```

### Local Development

```bash
# Start dependencies
docker-compose up postgres redis prometheus -d

# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
python -m uvicorn src.main:app --reload

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Performance Testing

```bash
# Generate realistic test data
python scripts/generate_test_data.py --models 10 --requests-per-second 100

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Monitoring

### Key Metrics

- `ml_predictor_inference_latency`: Model inference response time
- `ml_predictor_drift_score`: Feature and prediction drift measurements  
- `ml_predictor_degradation_probability`: Predicted performance degradation
- `ml_predictor_cache_hit_ratio`: Redis cache efficiency
- `ml_predictor_db_query_duration`: PostgreSQL query performance

### Alerts

- Model degradation probability > 0.8
- Drift score > configured threshold
- API latency > 500ms (95th percentile)
- Cache hit ratio < 0.7
- Database connection pool exhaustion

## License

MIT