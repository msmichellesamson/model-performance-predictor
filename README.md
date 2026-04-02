# Model Performance Predictor

Real-time ML model performance degradation prediction using inference metrics and drift detection.

## Features
- **Performance Prediction**: ML-based degradation risk assessment
- **Drift Detection**: Statistical drift detection for features and predictions
- **Real-time Monitoring**: Prometheus metrics and alerting
- **Caching**: Redis for fast metric lookups
- **Production Ready**: Kubernetes deployment with proper observability

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│  API Layer   │───▶│ Predictor   │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    │
                           ▼                    ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ Drift Engine │    │ Metrics DB  │
                   └──────────────┘    └─────────────┘
                           │                    │
                           ▼                    ▼
                   ┌──────────────┐    ┌─────────────┐
                   │    Redis     │    │ Prometheus  │
                   └──────────────┘    └─────────────┘
```

## API Documentation

See [api/openapi.yaml](api/openapi.yaml) for complete API specification.

### Key Endpoints
- `POST /predictions` - Predict performance degradation
- `POST /drift/detect` - Detect data/model drift
- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check

## Quick Start

```bash
# Start services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Predict degradation
curl -X POST http://localhost:8000/predictions \
  -H "Content-Type: application/json" \
  -d '{"model_id": "model-123", "metrics": {"accuracy": 0.95, "latency_p95": 250, "error_rate": 0.02}}'
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:8000/metrics
- **Alerts**: Configured for degradation risk > 0.8

## Skills Demonstrated

- **AI/ML**: Performance prediction models, drift detection algorithms
- **Backend**: FastAPI, async processing, error handling
- **Database**: PostgreSQL for metrics, Redis for caching
- **Infrastructure**: Terraform, Kubernetes, Docker
- **SRE**: Prometheus monitoring, health checks, observability
- **DevOps**: CI/CD pipeline, containerization
- **Data**: Real-time metrics processing, feature engineering

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run locally
python src/main.py
```

## Deployment

```bash
# Deploy infrastructure
cd terraform && terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```