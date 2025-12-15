"""
Model Training Script with MLflow Integration.

Trains a credit scoring model and logs everything to MLflow.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from ml.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    DEFAULT_SAMPLE_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
)
from ml.data_generator import generate_credit_data, split_data, load_from_csv, generate_and_save_data, RAW_DATA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.

    Returns:
        ColumnTransformer with numerical and categorical preprocessing
    """
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def create_model_pipeline(
    model_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Create full model pipeline with preprocessing and classifier.

    Args:
        model_params: Optional model hyperparameters

    Returns:
        Sklearn Pipeline with preprocessor and classifier
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    preprocessor = create_preprocessing_pipeline()
    classifier = GradientBoostingClassifier(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def train_model(
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    model_params: Optional[Dict[str, Any]] = None,
    mlflow_tracking_uri: Optional[str] = None,
    register_model: bool = True,
    data_path: Optional[str] = None,
    generate_new_data: bool = False,
) -> str:
    """
    Train credit scoring model with MLflow tracking.

    Args:
        n_samples: Number of training samples (used if generating new data)
        model_params: Optional model hyperparameters
        mlflow_tracking_uri: MLflow tracking server URI
        register_model: Whether to register model in MLflow Registry
        data_path: Path to CSV data file (uses default if None)
        generate_new_data: If True, generate new data instead of loading from CSV

    Returns:
        MLflow run ID
    """
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    # Set MLflow tracking URI
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set or create experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load or generate data
    if generate_new_data:
        logger.info(f"Generating {n_samples} new training samples...")
        df, saved_path = generate_and_save_data(n_samples=n_samples)
        logger.info(f"Data saved to {saved_path}")
    else:
        data_file = data_path if data_path else RAW_DATA_PATH
        try:
            logger.info(f"Loading training data from {data_file}...")
            df = load_from_csv(data_file)
        except FileNotFoundError:
            logger.warning(f"Data file not found at {data_file}. Generating new data...")
            df, saved_path = generate_and_save_data(n_samples=n_samples)
            logger.info(f"Data saved to {saved_path}")
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=TEST_SIZE)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("numerical_features", NUMERICAL_FEATURES)
        mlflow.log_param("categorical_features", CATEGORICAL_FEATURES)

        # Create and train model
        logger.info("Training model...")
        model = create_model_pipeline(model_params)
        model.fit(X_train, y_train)

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")

        # Log confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(
            str(cm),
            "confusion_matrix.txt"
        )

        # Log classification report
        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")

        # Log feature importances
        feature_names = (
            NUMERICAL_FEATURES +
            list(model.named_steps["preprocessor"]
                 .named_transformers_["cat"]
                 .get_feature_names_out(CATEGORICAL_FEATURES))
        )
        importances = model.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        mlflow.log_text(
            importance_df.to_string(),
            "feature_importances.txt"
        )

        # Log model
        logger.info("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME if register_model else None,
        )

        # Log dataset info
        mlflow.log_param("target_distribution", df["target"].value_counts().to_dict())

        logger.info(f"Model training complete. Run ID: {run_id}")

        return run_id


def get_latest_model_version(
    model_name: str = MODEL_NAME,
    mlflow_tracking_uri: Optional[str] = None,
) -> Optional[int]:
    """
    Get the latest version of a registered model.

    Args:
        model_name: Name of the registered model
        mlflow_tracking_uri: MLflow tracking server URI

    Returns:
        Latest version number or None if model not found
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(int(v.version) for v in versions)
    except Exception as e:
        logger.warning(f"Could not get model versions: {e}")

    return None


def promote_model_to_production(
    model_name: str = MODEL_NAME,
    version: Optional[int] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> None:
    """
    Promote a model version to Production stage.

    Args:
        model_name: Name of the registered model
        version: Version to promote (latest if None)
        mlflow_tracking_uri: MLflow tracking server URI
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.MlflowClient()

    if version is None:
        version = get_latest_model_version(model_name, mlflow_tracking_uri)

    if version is None:
        raise ValueError(f"No versions found for model: {model_name}")

    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    logger.info(f"Promoted {model_name} version {version} to Production")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train credit scoring model")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of training samples (used when generating new data)"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=True,
        help="Register model in MLflow Registry"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote model to Production after training"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data file (uses default data/raw/credit_data.csv if not specified)"
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate new data instead of loading from CSV"
    )

    args = parser.parse_args()

    run_id = train_model(
        n_samples=args.n_samples,
        mlflow_tracking_uri=args.mlflow_uri,
        register_model=args.register,
        data_path=args.data_path,
        generate_new_data=args.generate_data,
    )

    if args.promote:
        promote_model_to_production(mlflow_tracking_uri=args.mlflow_uri)
