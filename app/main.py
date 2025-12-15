"""
Main Application Entry Point.

FastAPI application with middleware, startup events, and routing.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import router
from app.database import engine, Base
from app.monitoring import (
    PrometheusMiddleware,
    metrics_endpoint,
    set_app_info,
    set_model_info,
    set_model_loaded,
)
from ml.predict import get_predictor, ModelLoadError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Set app info metrics
    set_app_info(settings.app_name, settings.app_version)

    # Try to load model
    try:
        predictor = get_predictor(
            model_name=settings.model_name,
            model_stage=settings.model_stage,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
        )
        predictor.load_model()

        # Set model metrics
        set_model_info(
            model_name=settings.model_name,
            model_version=str(predictor.model_version),
            model_stage=settings.model_stage,
        )
        set_model_loaded(True)

        logger.info(
            f"Model loaded: {settings.model_name} "
            f"v{predictor.model_version} ({settings.model_stage})"
        )

    except ModelLoadError as e:
        logger.warning(f"Could not load model at startup: {e}")
        logger.warning("API will start but predictions will be unavailable")
        set_model_loaded(False)

    yield

    # Shutdown
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="""
        ## Credit Score Prediction API

        A production-ready API for credit scoring predictions powered by machine learning.

        ### Features
        - **Single Predictions**: Submit individual credit applications
        - **Batch Predictions**: Process multiple applications at once
        - **Model Versioning**: Track model versions via MLflow
        - **Monitoring**: Prometheus metrics for observability

        ### Model Information
        The API uses a Random Forest classifier trained on synthetic credit data.
        The model predicts whether a credit application should be APPROVED or REJECTED.

        ### Authentication
        Currently, the API is open for demonstration purposes.
        In production, implement proper authentication.
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/v1/health",
            "metrics": "/metrics",
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
