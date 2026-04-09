# Model Performance Predictor

Real-time ML model performance degradation prediction using inference metrics and drift detection.

## Architecture

- **Predictor Service**: Core ML model performance prediction
- **Metrics Collection**: Real-time inference metrics aggregation
- **Drift Detection**: Statistical drift detection across feature distributions
- **Alerting**: Threshold-based performance degradation alerts
- **Monitoring**: Prometheus metrics with Grafana dashboards

## Tech Stack

- **Backend**: Python FastAPI, PostgreSQL, Redis
- **Infrastructure**: Terraform (GCP), Kubernetes
- **Monitoring**: Prometheus, Grafana, custom ServiceMonitor
- **ML**: Scikit-learn drift detection, statistical analysis

## Quick Start

```bash
# Local development
docker-compose up -d

# Kubernetes deployment
kubectl apply -f k8s/

# Infrastructure
cd terraform && terraform apply
```

## API Endpoints

- `POST /api/v1/predictions/performance` - Predict model degradation
- `GET /api/v1/drift/features/{model_id}` - Feature drift analysis
- `GET /api/v1/alerts/active` - Active performance alerts
- `GET /metrics` - Prometheus metrics

## Monitoring

Prometheus ServiceMonitor automatically scrapes:
- Application metrics (`/metrics`)
- Custom model metrics (`/api/v1/predictions/metrics`)
- 30-second intervals with 10-second timeout

## Skills Demonstrated

- **ML**: Real-time model performance prediction, drift detection
- **Infrastructure**: Kubernetes ServiceMonitor, Prometheus integration
- **Backend**: FastAPI microservice, PostgreSQL data modeling
- **DevOps**: Container orchestration, monitoring configuration
- **Database**: Performance metrics storage, query optimization
- **SRE**: Observability, alerting, service monitoring