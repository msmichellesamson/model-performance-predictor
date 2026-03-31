from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from .api import health, drift
from .monitoring.prometheus import setup_metrics
from .cache.redis_client import RedisClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Model Performance Predictor")
    setup_metrics()
    
    # Initialize Redis connection
    redis_client = RedisClient()
    await redis_client.ping()
    
    yield
    
    # Shutdown
    logger.info("Shutting down gracefully")
    await redis_client.close()

app = FastAPI(
    title="Model Performance Predictor",
    description="Real-time ML model performance degradation prediction",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(drift.router)

@app.get("/")
async def root():
    return {"message": "Model Performance Predictor API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
