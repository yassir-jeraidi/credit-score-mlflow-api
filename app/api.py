"""
API Endpoints for Credit Scoring.

Defines FastAPI routes for predictions, health checks, and model info.
"""
import time
import uuid
import logging
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas import (
    CreditApplication,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from app.config import get_settings, Settings
from app.monitoring import record_prediction, record_batch_size
from ml.predict import get_predictor, CreditScorePredictor, PredictionError, ModelLoadError
from ml.config import ALL_FEATURES

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def get_model_predictor() -> CreditScorePredictor:
    """
    Dependency to get the model predictor.

    Returns:
        CreditScorePredictor instance
    """
    settings = get_settings()
    return get_predictor(
        model_name=settings.model_name,
        model_stage=settings.model_stage,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API service is running",
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns service status and version.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness check",
    description="Check if the API is ready to serve predictions",
)
async def readiness_check(
    predictor: CreditScorePredictor = Depends(get_model_predictor)
) -> ReadinessResponse:
    """
    Readiness check endpoint.

    Verifies that the model is loaded and MLflow is connected.
    """
    model_info = predictor.get_model_info()

    return ReadinessResponse(
        status="ready" if model_info["is_loaded"] else "not_ready",
        model_loaded=model_info["is_loaded"],
        model_version=model_info.get("model_version"),
        mlflow_connected=True,  # If we got here, MLflow connection works
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
    description="Get information about the currently loaded model",
)
async def get_model_info(
    predictor: CreditScorePredictor = Depends(get_model_predictor)
) -> ModelInfoResponse:
    """
    Get information about the loaded model.

    Returns model name, version, stage, and features.
    """
    model_info = predictor.get_model_info()

    return ModelInfoResponse(
        model_name=model_info["model_name"],
        model_stage=model_info["model_stage"],
        model_version=model_info.get("model_version"),
        is_loaded=model_info["is_loaded"],
        features=ALL_FEATURES,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Make a credit prediction",
    description="Submit a credit application and receive a prediction",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)
async def predict(
    application: CreditApplication,
    predictor: CreditScorePredictor = Depends(get_model_predictor)
) -> PredictionResponse:
    """
    Make a credit prediction for a single application.

    Args:
        application: Credit application data

    Returns:
        Prediction result with confidence scores
    """
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    try:
        start_time = time.time()

        # Convert to dict for prediction
        features = application.model_dump()

        # Make prediction
        result = predictor.predict(features)

        duration = time.time() - start_time

        # Record metrics
        record_prediction(
            result=result["prediction"],
            model_version=result["model_version"],
            duration=duration,
        )

        # Generate application ID
        application_id = str(uuid.uuid4())

        return PredictionResponse(
            application_id=application_id,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            approval_probability=result["approval_probability"],
            rejection_probability=result["rejection_probability"],
            model_version=result["model_version"],
            timestamp=datetime.utcnow(),
        )

    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Make batch predictions",
    description="Submit multiple credit applications and receive predictions",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: CreditScorePredictor = Depends(get_model_predictor)
) -> BatchPredictionResponse:
    """
    Make predictions for multiple applications.

    Args:
        request: Batch of credit applications

    Returns:
        List of prediction results
    """
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )

    try:
        start_time = time.time()

        # Convert applications to list of dicts
        features_list = [app.model_dump() for app in request.applications]

        # Record batch size
        record_batch_size(len(features_list))

        # Make batch predictions
        results = predictor.predict_batch(features_list)

        duration = time.time() - start_time

        # Create response with application IDs
        predictions = []
        for result in results:
            application_id = str(uuid.uuid4())

            # Record metrics for each prediction
            record_prediction(
                result=result["prediction"],
                model_version=result["model_version"],
                duration=duration / len(results),  # Average time per prediction
            )

            predictions.append(PredictionResponse(
                application_id=application_id,
                prediction=result["prediction"],
                confidence=result["confidence"],
                risk_score=result["risk_score"],
                approval_probability=result["approval_probability"],
                rejection_probability=result["rejection_probability"],
                model_version=result["model_version"],
                timestamp=datetime.utcnow(),
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=duration * 1000,
        )

    except PredictionError as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
