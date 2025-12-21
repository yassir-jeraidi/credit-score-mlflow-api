"""
Pydantic Schemas for Request/Response Validation.

Defines data models for the Credit Score API.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class CreditApplication(BaseModel):
    """Credit application input schema with validation."""

    age: int = Field(
        ..., ge=18, le=100, description="Age of the applicant (18-100 years)", examples=[35]
    )
    income: float = Field(
        ..., gt=0, description="Annual income in currency units", examples=[65000.0]
    )
    employment_length: int = Field(
        ..., ge=0, le=50, description="Years of employment", examples=[8]
    )
    loan_amount: float = Field(..., gt=0, description="Requested loan amount", examples=[15000.0])
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ] = Field(..., description="Purpose of the loan", examples=["PERSONAL"])
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        ..., description="Home ownership status", examples=["MORTGAGE"]
    )
    credit_history_length: int = Field(
        ..., ge=0, le=50, description="Years of credit history", examples=[10]
    )
    num_credit_lines: int = Field(
        ..., ge=0, le=50, description="Number of credit lines/accounts", examples=[5]
    )
    derogatory_marks: int = Field(
        ..., ge=0, le=20, description="Number of derogatory marks on credit report", examples=[0]
    )
    total_debt: float = Field(..., ge=0, description="Total existing debt", examples=[20000.0])

    @field_validator("employment_length")
    @classmethod
    def validate_employment_vs_age(cls, v, info):
        """Validate that employment length is reasonable given age."""
        # Note: Can't access age here in Pydantic v2 easily
        # This is kept as a placeholder for more complex validation
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 65000.0,
                "employment_length": 8,
                "loan_amount": 15000.0,
                "loan_intent": "PERSONAL",
                "home_ownership": "MORTGAGE",
                "credit_history_length": 10,
                "num_credit_lines": 5,
                "derogatory_marks": 0,
                "total_debt": 20000.0,
            }
        }


class PredictionResponse(BaseModel):
    """Credit prediction response schema."""

    application_id: str = Field(..., description="Unique identifier for this prediction request")
    prediction: Literal["APPROVED", "REJECTED"] = Field(..., description="Credit decision")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the prediction (0-1)"
    )
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (probability of rejection)")
    approval_probability: float = Field(..., ge=0, le=1, description="Probability of approval")
    rejection_probability: float = Field(..., ge=0, le=1, description="Probability of rejection")
    model_version: str = Field(..., description="Version of the model used for prediction")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the prediction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "application_id": "550e8400-e29b-41d4-a716-446655440000",
                "prediction": "APPROVED",
                "confidence": 0.85,
                "risk_score": 0.15,
                "approval_probability": 0.85,
                "rejection_probability": 0.15,
                "model_version": "1",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        },
        "protected_namespaces": (),
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    applications: List[CreditApplication] = Field(
        ..., min_length=1, max_length=100, description="List of credit applications (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of applications processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Current server timestamp"
    )


class ReadinessResponse(BaseModel):
    """Readiness check response schema."""

    status: Literal["ready", "not_ready"] = Field(..., description="Service readiness status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    mlflow_connected: bool = Field(..., description="Whether MLflow connection is established")
    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    model_name: str = Field(..., description="Name of the registered model")
    model_stage: str = Field(..., description="Model stage (Production, Staging, etc.)")
    model_version: Optional[str] = Field(None, description="Current model version")
    is_loaded: bool = Field(..., description="Whether the model is loaded")
    features: List[str] = Field(..., description="List of input features")
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class Token(BaseModel):
    """Token schema."""

    access_token: str
    token_type: str


class UserBase(BaseModel):
    """Base user schema."""

    email: str


class UserCreate(UserBase):
    """User creation schema."""

    password: str


class UserResponse(UserBase):
    """User response schema."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True
