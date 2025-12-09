"""
API Integration Tests.

Tests for the Credit Score API endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_readiness_check_model_loaded(self, client: TestClient, mock_predictor):
        """Test readiness check when model is loaded."""
        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.get("/api/v1/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is True
            assert "model_version" in data

    def test_readiness_check_model_not_loaded(self, client: TestClient, mock_predictor):
        """Test readiness check when model is not loaded."""
        mock_predictor.is_loaded.return_value = False
        mock_predictor.get_model_info.return_value = {
            "model_name": "credit-score-model",
            "model_stage": "Production",
            "model_version": None,
            "is_loaded": False,
        }

        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.get("/api/v1/health/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"
            assert data["model_loaded"] is False


class TestModelEndpoints:
    """Tests for model information endpoints."""

    def test_get_model_info(self, client: TestClient, mock_predictor):
        """Test model info endpoint."""
        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.get("/api/v1/model/info")

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "credit-score-model"
            assert data["model_stage"] == "Production"
            assert data["is_loaded"] is True
            assert "features" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_success(
        self,
        client: TestClient,
        mock_predictor,
        sample_application
    ):
        """Test successful prediction."""
        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.post("/api/v1/predict", json=sample_application)

            assert response.status_code == 200
            data = response.json()
            assert "application_id" in data
            assert data["prediction"] in ["APPROVED", "REJECTED"]
            assert 0 <= data["confidence"] <= 1
            assert 0 <= data["risk_score"] <= 1
            assert "model_version" in data

    def test_predict_high_risk(
        self,
        client: TestClient,
        mock_predictor,
        high_risk_application
    ):
        """Test prediction for high-risk application."""
        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.post("/api/v1/predict", json=high_risk_application)

            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == "REJECTED"
            assert data["risk_score"] > 0.5

    def test_predict_validation_error_age(self, client: TestClient, sample_application):
        """Test prediction with invalid age."""
        sample_application["age"] = 15  # Too young

        response = client.post("/api/v1/predict", json=sample_application)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_predict_validation_error_income(self, client: TestClient, sample_application):
        """Test prediction with invalid income."""
        sample_application["income"] = -1000  # Negative

        response = client.post("/api/v1/predict", json=sample_application)

        assert response.status_code == 422

    def test_predict_validation_error_loan_intent(
        self,
        client: TestClient,
        sample_application
    ):
        """Test prediction with invalid loan intent."""
        sample_application["loan_intent"] = "INVALID"

        response = client.post("/api/v1/predict", json=sample_application)

        assert response.status_code == 422

    def test_predict_model_not_loaded(
        self,
        client: TestClient,
        mock_predictor,
        sample_application
    ):
        """Test prediction when model is not loaded."""
        mock_predictor.is_loaded.return_value = False

        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.post("/api/v1/predict", json=sample_application)

            assert response.status_code == 503

    def test_batch_predict_success(
        self,
        client: TestClient,
        mock_predictor,
        batch_applications
    ):
        """Test batch prediction."""
        with patch("app.api.get_predictor", return_value=mock_predictor):
            response = client.post(
                "/api/v1/predict/batch",
                json={"applications": batch_applications}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["total_processed"] == 2
            assert "processing_time_ms" in data

    def test_batch_predict_empty_list(self, client: TestClient):
        """Test batch prediction with empty list."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"applications": []}
        )

        assert response.status_code == 422

    def test_batch_predict_too_many(
        self,
        client: TestClient,
        sample_application
    ):
        """Test batch prediction with too many applications."""
        # Create 101 applications (over limit)
        applications = [sample_application.copy() for _ in range(101)]

        response = client.post(
            "/api/v1/predict/batch",
            json={"applications": applications}
        )

        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client: TestClient):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics(self, client: TestClient):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"] or \
               "text/plain" in response.headers.get("content-type", "")
