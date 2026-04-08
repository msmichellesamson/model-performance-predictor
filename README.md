# Model Performance Predictor

Real-time ML model performance degradation prediction using inference metrics and drift detection.

## Features

- **Performance Prediction**: Predicts model degradation using inference metrics
- **Drift Detection**: Real-time feature and concept drift monitoring
- **Circuit Breaker**: Automatic failover when models underperform
- **Multi-layer Monitoring**: Latency, accuracy, data quality tracking
- **Alert System**: Threshold-based alerting with multiple channels
- **Production Ready**: Kubernetes deployment, observability, caching

## Quick Start

```bash
# Start services
docker-compose up -d

# Run predictions
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{
    "model_id": "fraud-detector-v2",
    "metrics": {
      "latency_p99": 250.0,
      "error_rate": 0.02,
      "throughput": 1500
    }
  }'
```

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│  FastAPI     │───▶│ Predictor   │
│             │    │  Gateway     │    │ Engine      │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    │
                           ▼                    ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  Monitoring  │    │   Redis     │
                   │  Stack       │    │   Cache     │
                   └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ PostgreSQL   │
                   │ Metrics DB   │
                   └──────────────┘
```

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **Database**: PostgreSQL, Redis
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Docker, Kubernetes, Terraform
- **ML**: scikit-learn, numpy, pandas

## API Documentation

See [API_GUIDE.md](docs/API_GUIDE.md) for comprehensive usage examples and troubleshooting.

## Deployment

### Local Development
```bash
docker-compose up -d
python -m pytest tests/
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Terraform (GCP)
```bash
cd terraform/
terraform init
terraform plan
terraform apply
```

## Monitoring

- **Metrics**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)
- **API Health**: http://localhost:8000/api/v1/health

## Skills Demonstrated

- **ML**: Real-time model performance prediction, drift detection
- **Backend**: FastAPI, microservices, distributed systems
- **Database**: PostgreSQL optimization, Redis caching
- **Infrastructure**: Kubernetes, Terraform, Docker
- **SRE**: Prometheus monitoring, circuit breakers, alerting
- **DevOps**: CI/CD, GitOps, container orchestration