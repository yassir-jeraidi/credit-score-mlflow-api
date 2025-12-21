"""
API Endpoints for Credit Scoring.

Defines FastAPI routes for predictions, health checks, and model info.
"""

import logging
import time
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app import models, security
from app.config import get_settings
from app.database import get_db
from app.monitoring import record_batch_size, record_prediction
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditApplication,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    Token,
    UserCreate,
    UserResponse,
)
from ml.config import ALL_FEATURES
from ml.predict import CreditScorePredictor, PredictionError, get_predictor

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, get_settings().jwt_secret_key, algorithms=[get_settings().jwt_algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


@router.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=409, detail="Email already registered")
    hashed_password = security.get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/auth/login", response_model=Token, tags=["Authentication"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=get_settings().access_token_expire_minutes)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


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
    predictor: CreditScorePredictor = Depends(get_model_predictor),
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
    predictor: CreditScorePredictor = Depends(get_model_predictor),
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
    predictor: CreditScorePredictor = Depends(get_model_predictor),
    current_user: models.User = Depends(get_current_user),
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
    predictor: CreditScorePredictor = Depends(get_model_predictor),
    current_user: models.User = Depends(get_current_user),
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

            predictions.append(
                PredictionResponse(
                    application_id=application_id,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    risk_score=result["risk_score"],
                    approval_probability=result["approval_probability"],
                    rejection_probability=result["rejection_probability"],
                    model_version=result["model_version"],
                    timestamp=datetime.utcnow(),
                )
            )

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
