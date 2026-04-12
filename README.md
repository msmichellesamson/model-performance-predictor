# Model Performance Predictor

Real-time ML model performance degradation prediction using inference metrics and drift detection.

## Features

- **Performance Prediction**: Real-time model degradation detection
- **Drift Detection**: Feature and concept drift monitoring
- **Multi-Monitor System**: Accuracy, latency, confidence, data quality tracking
- **Circuit Breaker**: Fault tolerance for degraded models
- **Redis Caching**: Fast metric storage and retrieval
- **Prometheus Integration**: Comprehensive observability
- **Auto-scaling**: HPA-based scaling on CPU, memory, and custom metrics

## Tech Stack

- **Backend**: Python, FastAPI, asyncio
- **Database**: PostgreSQL, Redis
- **Infrastructure**: Kubernetes, Terraform (GCP)
- **Monitoring**: Prometheus, custom metrics
- **CI/CD**: GitHub Actions, Docker

## Architecture

```
API Layer (FastAPI)
├── Predictions API
├── Drift Detection API
├── Risk Assessment API
└── Health/Alerts API

Core Engine
├── Performance Predictor
├── Metrics Collector
└── Drift Detector

Monitoring Stack
├── Accuracy Monitor
├── Latency Monitor
├── Confidence Monitor
├── Feature Drift Monitor
├── Data Quality Monitor
└── Circuit Breaker

Infrastructure
├── PostgreSQL (metrics storage)
├── Redis (caching)
├── Prometheus (observability)
└── Kubernetes (orchestration)
```

## Quick Start

```bash
# Start with Docker Compose
docker-compose up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/

# Generate test data
python scripts/generate_test_data.py
```

## API Endpoints

- `POST /predictions/performance` - Predict model performance
- `GET /drift/status` - Check drift status
- `GET /risk/assessment` - Get risk assessment
- `GET /health` - Health check
- `GET /alerts` - Active alerts

## Monitoring

The system includes comprehensive monitoring:

- **Accuracy tracking** with configurable thresholds
- **Latency monitoring** with P95/P99 tracking
- **Feature drift detection** using statistical tests
- **Data quality checks** for completeness and validity
- **Circuit breaker** for fault tolerance
- **Auto-scaling** based on load and resource usage

## Infrastructure

### Kubernetes Deployment

- **HPA**: Auto-scales 2-10 pods based on CPU (70%), memory (80%), and RPS (100)
- **ServiceMonitor**: Prometheus scraping configuration
- **ConfigMaps**: Environment-specific configuration

### Terraform (GCP)

- GKE cluster with monitoring enabled
- Cloud SQL PostgreSQL instance
- Redis Memorystore instance
- IAM roles and service accounts

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
uvicorn src.main:app --reload
```

## Configuration

Key environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `PROMETHEUS_URL`: Prometheus endpoint
- `ALERT_THRESHOLDS`: JSON config for alerting

## Scaling

The HPA automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- HTTP requests per second (target: 100 RPS)

Scale-up: Max 50% increase or 2 pods per minute
Scale-down: Max 10% decrease every 5 minutes