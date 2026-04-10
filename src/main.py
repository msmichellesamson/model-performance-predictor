"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import health, predictions, drift, alerts, risk
from .monitoring.prometheus import PrometheusMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
prometheus = PrometheusMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Model Performance Predictor")
    # Initialize metrics collection
    prometheus.start_metrics_server()
    yield
    logger.info("Shutting down Model Performance Predictor")


app = FastAPI(
    title="Model Performance Predictor",
    description="Real-time ML model performance degradation prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(drift.router, prefix="/drift", tags=["drift"])
app.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
app.include_router(risk.router, prefix="/api/v1", tags=["risk"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Model Performance Predictor",
        "version": "1.0.0",
        "status": "running"
    }
