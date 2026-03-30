import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_client import make_asgi_app
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Settings, get_settings
from src.core.database import init_db, get_db_session
from src.core.redis_client import get_redis_client
from src.core.exceptions import ModelPerformanceException, ValidationError
from src.api.routes import model_metrics, drift_detection, predictions, health
from src.services.model_monitor import ModelMonitorService
from src.services.drift_detector import DriftDetectionService
from src.services.performance_predictor import PerformancePredictorService
from src.core.metrics import setup_metrics, REQUEST_COUNT, REQUEST_DURATION

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ServiceContainer:
    """Dependency injection container for application services."""
    
    def __init__(self):
        self.model_monitor: ModelMonitorService | None = None
        self.drift_detector: DriftDetectionService | None = None
        self.performance_predictor: PerformancePredictorService | None = None
        self._redis_client: redis.Redis | None = None
        
    async def initialize(self, settings: Settings) -> None:
        """Initialize all application services."""
        try:
            # Initialize Redis client
            self._redis_client = await get_redis_client(settings)
            
            # Initialize services
            self.model_monitor = ModelMonitorService(
                redis_client=self._redis_client,
                settings=settings
            )
            
            self.drift_detector = DriftDetectionService(
                redis_client=self._redis_client,
                settings=settings
            )
            
            self.performance_predictor = PerformancePredictorService(
                redis_client=self._redis_client,
                settings=settings
            )
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize services", error=str(e))
            raise ModelPerformanceException(f"Service initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all application services."""
        try:
            if self._redis_client:
                await self._redis_client.close()
                
            logger.info("Services cleaned up successfully")
            
        except Exception as e:
            logger.error("Error during service cleanup", error=str(e))


# Global service container
service_container = ServiceContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    settings = get_settings()
    
    try:
        logger.info("Starting application", version=settings.APP_VERSION)
        
        # Initialize database
        await init_db(settings)
        logger.info("Database initialized")
        
        # Initialize services
        await service_container.initialize(settings)
        
        # Setup metrics
        setup_metrics()
        logger.info("Metrics setup completed")
        
        # Start background tasks
        asyncio.create_task(start_background_monitoring())
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    finally:
        logger.info("Shutting down application")
        await service_container.cleanup()
        logger.info("Application shutdown completed")


async def start_background_monitoring() -> None:
    """Start background monitoring tasks."""
    try:
        # Start drift detection monitoring
        if service_container.drift_detector:
            asyncio.create_task(
                service_container.drift_detector.start_continuous_monitoring()
            )
        
        # Start performance monitoring
        if service_container.model_monitor:
            asyncio.create_task(
                service_container.model_monitor.start_monitoring()
            )
            
        logger.info("Background monitoring tasks started")
        
    except Exception as e:
        logger.error("Failed to start background monitoring", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Model Performance Predictor",
        description="Real-time ML model performance degradation prediction using inference metrics and drift detection",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )
    
    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Include API routes
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health"]
    )
    
    app.include_router(
        model_metrics.router,
        prefix="/api/v1/metrics",
        tags=["Model Metrics"]
    )
    
    app.include_router(
        drift_detection.router,
        prefix="/api/v1/drift",
        tags=["Drift Detection"]
    )
    
    app.include_router(
        predictions.router,
        prefix="/api/v1/predictions",
        tags=["Performance Predictions"]
    )
    
    # Global exception handlers
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request, exc: ValidationError):
        logger.warning("Validation error", error=str(exc), path=request.url.path)
        raise HTTPException(status_code=400, detail=str(exc))
    
    @app.exception_handler(ModelPerformanceException)
    async def model_performance_error_handler(request, exc: ModelPerformanceException):
        logger.error("Model performance error", error=str(exc), path=request.url.path)
        raise HTTPException(status_code=500, detail="Internal service error")
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        raise HTTPException(status_code=500, detail="Internal server error")
    
    # Middleware for request/response logging and metrics
    @app.middleware("http")
    async def logging_middleware(request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            duration = asyncio.get_event_loop().time() - start_time
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=f"{duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            
            # Update error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                duration=f"{duration:.3f}s",
                error=str(e)
            )
            
            raise
    
    return app


def get_service_container() -> ServiceContainer:
    """Dependency to get service container."""
    return service_container


# Create application instance
app = create_app()


async def main() -> None:
    """Main application entry point."""
    settings = get_settings()
    
    try:
        config = uvicorn.Config(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_config=None,  # Use structlog instead
            access_log=False,  # Handle via middleware
            loop="uvloop",
            http="httptools",
            workers=1,  # Use 1 worker for async app with background tasks
        )
        
        server = uvicorn.Server(config)
        logger.info(
            "Starting server",
            host=settings.HOST,
            port=settings.PORT,
            debug=settings.DEBUG
        )
        
        await server.serve()
        
    except Exception as e:
        logger.error("Server startup failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())