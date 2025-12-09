"""
Prediction Utilities for Credit Scoring Model.

Handles model loading from MLflow and prediction logic.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from ml.config import (
    MODEL_NAME,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class CreditScorePredictor:
    """
    Credit Score Prediction class.

    Handles model loading from MLflow and predictions.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_stage: str = "Production",
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_name: Name of registered model in MLflow
            model_stage: Model stage to load (Production, Staging, etc.)
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.model_name = model_name
        self.model_stage = model_stage
        self.model = None
        self.model_version = None

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def load_model(self) -> None:
        """Load model from MLflow Registry."""
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            logger.info(f"Loading model from: {model_uri}")

            self.model = mlflow.sklearn.load_model(model_uri)

            # Get model version
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(
                self.model_name,
                stages=[self.model_stage]
            )
            if versions:
                self.model_version = versions[0].version

            logger.info(
                f"Loaded model {self.model_name} "
                f"version {self.model_version} ({self.model_stage})"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Could not load model: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "model_version": self.model_version,
            "is_loaded": self.is_loaded(),
        }

    def predict(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a single prediction.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded():
            raise PredictionError("Model not loaded")

        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Ensure correct column order
            df = df[ALL_FEATURES]

            # Make prediction
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]

            # Get probability of rejection (class 1)
            risk_score = float(probabilities[1])
            confidence = float(max(probabilities))

            result = {
                "prediction": "REJECTED" if prediction == 1 else "APPROVED",
                "prediction_code": int(prediction),
                "confidence": round(confidence, 4),
                "risk_score": round(risk_score, 4),
                "approval_probability": round(float(probabilities[0]), 4),
                "rejection_probability": round(float(probabilities[1]), 4),
                "model_version": str(self.model_version),
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}")

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction results
        """
        if not self.is_loaded():
            raise PredictionError("Model not loaded")

        try:
            # Convert to DataFrame
            df = pd.DataFrame(features_list)

            # Ensure correct column order
            df = df[ALL_FEATURES]

            # Make predictions
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)

            results = []
            for i in range(len(predictions)):
                risk_score = float(probabilities[i][1])
                confidence = float(max(probabilities[i]))

                result = {
                    "prediction": "REJECTED" if predictions[i] == 1 else "APPROVED",
                    "prediction_code": int(predictions[i]),
                    "confidence": round(confidence, 4),
                    "risk_score": round(risk_score, 4),
                    "approval_probability": round(float(probabilities[i][0]), 4),
                    "rejection_probability": round(float(probabilities[i][1]), 4),
                    "model_version": str(self.model_version),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise PredictionError(f"Batch prediction failed: {e}")


# Global predictor instance
_predictor: Optional[CreditScorePredictor] = None


def get_predictor(
    model_name: str = MODEL_NAME,
    model_stage: str = "Production",
    mlflow_tracking_uri: Optional[str] = None,
) -> CreditScorePredictor:
    """
    Get or create global predictor instance.

    Args:
        model_name: Name of registered model
        model_stage: Model stage to load
        mlflow_tracking_uri: MLflow tracking URI

    Returns:
        CreditScorePredictor instance
    """
    global _predictor

    if _predictor is None:
        _predictor = CreditScorePredictor(
            model_name=model_name,
            model_stage=model_stage,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    return _predictor


def reset_predictor() -> None:
    """Reset the global predictor instance."""
    global _predictor
    _predictor = None


if __name__ == "__main__":
    # Test prediction
    predictor = CreditScorePredictor(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    try:
        predictor.load_model()

        test_features = {
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

        result = predictor.predict(test_features)
        print(f"\nPrediction Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    except ModelLoadError as e:
        print(f"Could not load model: {e}")
        print("Please ensure a model is trained and promoted to Production")
