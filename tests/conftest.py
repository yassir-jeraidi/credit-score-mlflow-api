"""
Pytest Configuration and Fixtures.

Provides shared fixtures for testing the Credit Score API.
"""

import os
import sys
from typing import Any, Dict, Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These imports must come after path modification
from app import security  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.

    Yields:
        TestClient instance
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_token(client: TestClient) -> str:
    """
    Create a test user and return authentication token.

    Returns:
        JWT access token for authenticated requests
    """
    # Create test user
    test_user = {"email": "test@example.com", "password": "testpassword123"}

    # Try to register (may fail if user exists)
    client.post("/api/v1/auth/register", json=test_user)

    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        data={"username": test_user["email"], "password": test_user["password"]},
    )

    if response.status_code == 200:
        return response.json()["access_token"]

    # Fallback: create token directly for testing
    from datetime import timedelta

    return security.create_access_token(
        data={"sub": "test@example.com"}, expires_delta=timedelta(minutes=30)
    )


@pytest.fixture
def auth_headers(auth_token: str) -> Dict[str, str]:
    """
    Create authorization headers with token.

    Returns:
        Headers dict with Bearer token
    """
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_predictor():
    """
    Create a mock predictor for testing without MLflow.

    Returns:
        Mock predictor with prediction capabilities
    """
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.model_version = "1"
    mock.model_name = "credit-score-model"
    mock.model_stage = "Production"

    mock.get_model_info.return_value = {
        "model_name": "credit-score-model",
        "model_stage": "Production",
        "model_version": "1",
        "is_loaded": True,
    }

    def predict_side_effect(features: Dict[str, Any]) -> Dict[str, Any]:
        """Mock prediction based on features."""
        # Simple mock logic: high income = approved
        risk_score = 0.3 if features.get("income", 0) > 50000 else 0.7
        return {
            "prediction": "APPROVED" if risk_score < 0.5 else "REJECTED",
            "prediction_code": 0 if risk_score < 0.5 else 1,
            "confidence": abs(0.5 - risk_score) + 0.5,
            "risk_score": risk_score,
            "approval_probability": 1 - risk_score,
            "rejection_probability": risk_score,
            "model_version": "1",
        }

    mock.predict.side_effect = predict_side_effect

    def predict_batch_side_effect(features_list):
        return [predict_side_effect(f) for f in features_list]

    mock.predict_batch.side_effect = predict_batch_side_effect

    return mock


@pytest.fixture
def sample_application() -> Dict[str, Any]:
    """
    Create a sample credit application for testing.

    Returns:
        Dictionary with sample application data
    """
    return {
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


@pytest.fixture
def high_risk_application() -> Dict[str, Any]:
    """
    Create a high-risk credit application for testing.

    Returns:
        Dictionary with high-risk application data
    """
    return {
        "age": 22,
        "income": 25000.0,
        "employment_length": 1,
        "loan_amount": 50000.0,
        "loan_intent": "VENTURE",
        "home_ownership": "RENT",
        "credit_history_length": 1,
        "num_credit_lines": 1,
        "derogatory_marks": 3,
        "total_debt": 15000.0,
    }


@pytest.fixture
def batch_applications(sample_application, high_risk_application):
    """
    Create a batch of applications for testing.

    Returns:
        List of application dictionaries
    """
    return [sample_application, high_risk_application]


@pytest.fixture
def invalid_application() -> Dict[str, Any]:
    """
    Create an invalid application for testing validation.

    Returns:
        Dictionary with invalid application data
    """
    return {
        "age": 15,  # Too young
        "income": -1000,  # Negative income
        "employment_length": 8,
        "loan_amount": 15000.0,
        "loan_intent": "INVALID_INTENT",  # Invalid category
        "home_ownership": "MORTGAGE",
        "credit_history_length": 10,
        "num_credit_lines": 5,
        "derogatory_marks": 0,
        "total_debt": 20000.0,
    }
