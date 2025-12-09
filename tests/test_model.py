"""
ML Model Tests.

Tests for data generation, model training, and prediction utilities.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    LOAN_INTENT_CATEGORIES,
    HOME_OWNERSHIP_CATEGORIES,
)
from ml.data_generator import generate_credit_data, split_data, _generate_target
from ml.train import create_preprocessing_pipeline, create_model_pipeline, evaluate_model


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def test_generate_credit_data_shape(self):
        """Test that generated data has correct shape."""
        n_samples = 1000
        df = generate_credit_data(n_samples=n_samples)

        assert len(df) == n_samples
        assert "target" in df.columns

    def test_generate_credit_data_columns(self):
        """Test that generated data has all required columns."""
        df = generate_credit_data(n_samples=100)

        expected_columns = ALL_FEATURES + ["target"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_generate_credit_data_numerical_ranges(self):
        """Test that numerical features are in valid ranges."""
        df = generate_credit_data(n_samples=1000)

        # Age should be 18-70
        assert df["age"].min() >= 18
        assert df["age"].max() <= 70

        # Income should be positive
        assert df["income"].min() > 0

        # Employment length should be non-negative
        assert df["employment_length"].min() >= 0

        # Loan amount should be positive
        assert df["loan_amount"].min() > 0

    def test_generate_credit_data_categorical_values(self):
        """Test that categorical features have valid values."""
        df = generate_credit_data(n_samples=1000)

        # Check loan intent values
        unique_intents = df["loan_intent"].unique()
        for intent in unique_intents:
            assert intent in LOAN_INTENT_CATEGORIES

        # Check home ownership values
        unique_ownership = df["home_ownership"].unique()
        for ownership in unique_ownership:
            assert ownership in HOME_OWNERSHIP_CATEGORIES

    def test_generate_credit_data_target_binary(self):
        """Test that target is binary."""
        df = generate_credit_data(n_samples=1000)

        unique_targets = df["target"].unique()
        assert set(unique_targets).issubset({0, 1})

    def test_generate_credit_data_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        df1 = generate_credit_data(n_samples=100, random_state=42)
        df2 = generate_credit_data(n_samples=100, random_state=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_split_data(self):
        """Test data splitting."""
        df = generate_credit_data(n_samples=1000)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

        # Check sizes
        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200

        # Check no target in features
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns


class TestModelPipeline:
    """Tests for model pipeline creation."""

    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        preprocessor = create_preprocessing_pipeline()

        assert preprocessor is not None
        assert hasattr(preprocessor, "fit")
        assert hasattr(preprocessor, "transform")

    def test_create_model_pipeline(self):
        """Test full model pipeline creation."""
        pipeline = create_model_pipeline()

        assert pipeline is not None
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_model_pipeline_fit(self):
        """Test that model pipeline can fit data."""
        df = generate_credit_data(n_samples=500)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

        pipeline = create_model_pipeline()
        pipeline.fit(X_train, y_train)

        # Check can make predictions
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_model_pipeline_predict_proba(self):
        """Test that model pipeline returns probabilities."""
        df = generate_credit_data(n_samples=500)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

        pipeline = create_model_pipeline()
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_model(self):
        """Test model evaluation metrics."""
        df = generate_credit_data(n_samples=500)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

        pipeline = create_model_pipeline()
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        # Check all metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1


class TestPredictionUtilities:
    """Tests for prediction utilities."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        from ml.predict import CreditScorePredictor

        predictor = CreditScorePredictor(
            model_name="test-model",
            model_stage="Production",
            mlflow_tracking_uri="http://localhost:5000"
        )

        assert predictor.model_name == "test-model"
        assert predictor.model_stage == "Production"
        assert not predictor.is_loaded()

    def test_get_model_info_not_loaded(self):
        """Test model info when not loaded."""
        from ml.predict import CreditScorePredictor

        predictor = CreditScorePredictor()
        info = predictor.get_model_info()

        assert info["is_loaded"] is False
        assert info["model_version"] is None

    def test_prediction_without_loading(self):
        """Test that prediction fails when model not loaded."""
        from ml.predict import CreditScorePredictor, PredictionError

        predictor = CreditScorePredictor()

        with pytest.raises(PredictionError):
            predictor.predict({"age": 35, "income": 50000})
