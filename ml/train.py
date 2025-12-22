"""
Model Training Script with MLflow Integration.

Trains a credit scoring model and logs everything to MLflow.
"""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_SAMPLE_SIZE,
    "/Users/mohamed.hakim.dev@gmail.com/Credit Score",
    MODEL_NAME,
    NUMERICAL_FEATURES,
    TEST_SIZE,
)
from ml.data_generator import (
    RAW_DATA_PATH,
    generate_and_save_data,
    load_from_csv,
    split_data,
)

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
        drop="first", sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(model_params: Optional[Dict[str, Any]] = None) -> Pipeline:
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

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("databricks")

    # Set or create experiment
    mlflow.set_experiment("/Users/mohamed.hakim.dev@gmail.com/Credit Score")

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

        # Save metrics to file for CML report
        import json
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Log confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Log classification report
        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")

        # Log feature importances
        feature_names = NUMERICAL_FEATURES + list(
            model.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
        importances = model.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        mlflow.log_text(importance_df.to_string(), "feature_importances.txt")

        # Log model
        logger.info("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME if register_model else None,
        )

        # Log dataset info
        mlflow.log_param("target_distribution", df["target"].value_counts().to_dict())

        # Save plots and metrics history
        try:
            save_plots(model, X_test, y_test, X_train, y_train, run_id)
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

        logger.info(f"Model training complete. Run ID: {run_id}")

        return run_id


def save_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    run_id: str,
) -> None:
    """
    Generate and save training plots (confusion matrix, learning curves).

    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test labels
        X_train: Train features
        y_train: Train labels
        run_id: MLflow run ID
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # 2. Learning Curves (Loss, Accuracy, F1)
    # Get classifier and preprocessor
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Initialize history buffers
    train_loss = list(clf.train_score_)
    test_loss = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []

    # Calculate metrics for each boosting stage
    # Note: calculating full train metrics per stage can be slow for large datasets
    # limiting to chunks or subsets might be wise, but for verification it's fine.

    # Test Metrics per stage
    for i, y_pred in enumerate(clf.staged_predict(X_test_trans)):
        test_acc.append(accuracy_score(y_test, y_pred))
        test_f1.append(f1_score(y_test, y_pred))

    for i, y_proba in enumerate(clf.staged_predict_proba(X_test_trans)):
        test_loss.append(log_loss(y_test, y_proba))

    # Train Metrics per stage (Acc/F1 - Loss is already in train_score_)
    for i, y_pred in enumerate(clf.staged_predict(X_train_trans)):
        train_acc.append(accuracy_score(y_train, y_pred))
        train_f1.append(f1_score(y_train, y_pred))

    # Save history to CSV
    history_df = pd.DataFrame(
        {
            "stage": range(len(train_loss)),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        }
    )
    history_df.to_csv("training_history.csv", index=False)

    # Log artifacts to MLflow
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_history.csv")

    print(f"Saved plots and history for run {run_id}")


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
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "https://dbc-1391691d-cdf9.cloud.databricks.com"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--register", action="store_true", default=True, help="Register model in MLflow Registry"
    )
    parser.add_argument(
        "--promote", action="store_true", help="Promote model to Production after training"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data file (uses default data/raw/credit_data.csv if not specified)",
    )
    parser.add_argument(
        "--generate-data", action="store_true", help="Generate new data instead of loading from CSV"
    )

    args = parser.parse_args()

    run_id = train_model(
        mlflow_tracking_uri=args.mlflow_uri,
        register_model=args.register,
        data_path=args.data_path,
        generate_new_data=args.generate_data,
    )

    if args.promote:
        promote_model_to_production(mlflow_tracking_uri=args.mlflow_uri)
