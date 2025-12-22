"""
Model Training Script with MLflow Integration.

Trains a credit scoring model and logs everything to MLflow.
"""

import logging
import os
import argparse
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
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MODEL_PARAMS,
    NUMERICAL_FEATURES,
    TEST_SIZE,
)
from ml.data_generator import (
    RAW_DATA_PATH,
    load_from_csv,
    split_data,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "credit-score-catalog.credit-scoring.credit-score-model"


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Create preprocessing pipeline for features."""
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
    """Create full model pipeline with preprocessing and classifier."""
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS

    preprocessor = create_preprocessing_pipeline()
    classifier = GradientBoostingClassifier(**model_params)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_plots(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, 
               X_train: pd.DataFrame, y_train: pd.Series, run_id: str) -> None:
    """Generate and save training plots (confusion matrix, learning curves)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import log_loss

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

    # 2. Learning Curves
    clf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    train_loss = list(clf.train_score_)
    test_loss = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []

    # Calculate metrics for each boosting stage
    for y_pred in clf.staged_predict(X_test_trans):
        test_acc.append(accuracy_score(y_test, y_pred))
        test_f1.append(f1_score(y_test, y_pred))

    for y_proba in clf.staged_predict_proba(X_test_trans):
        test_loss.append(log_loss(y_test, y_proba))

    for y_pred in clf.staged_predict(X_train_trans):
        train_acc.append(accuracy_score(y_train, y_pred))
        train_f1.append(f1_score(y_train, y_pred))

    # Save history
    history_df = pd.DataFrame({
        "stage": range(len(train_loss)),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    })
    history_df.to_csv("training_history.csv", index=False)

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_history.csv")
    print(f"Saved plots and history for run {run_id}")


def train_model(
    mlflow_tracking_uri: Optional[str] = None,
    register_model: bool = True,
    data_path: Optional[str] = None,
) -> str:
    """Train credit scoring model with MLflow tracking."""

    # Set MLflow tracking URI and Experiment
    # Ensure env vars DATABRICKS_HOST and DATABRICKS_TOKEN are set in CML
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Users/mohamed.hakim.dev@gmail.com/credit-score")
    mlflow.set_tags({
        "project": "credit-scoring",
        "dataset": "dvc-managed",
        "framework": "sklearn",
        "ci": "cml",
    })

    mlflow.end_run()  # <-- ADD THIS LINE

    

    # Load Data - STRICTLY from DVC/Path
    data_file = data_path if data_path else RAW_DATA_PATH
    logger.info(f"Loading training data from {data_file}...")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Data file not found at {data_file}. Ensure DVC pull was successful."
        )
        
    df = load_from_csv(data_file)
    X_train, X_test, y_train, y_test = split_data(df, test_size=TEST_SIZE)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log parameters
        mlflow.log_params(DEFAULT_MODEL_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)

        # Train
        logger.info("Training model...")
        model = create_model_pipeline(DEFAULT_MODEL_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")

        # Save metrics locally for CML report
        import json
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Log artifacts
        y_pred = model.predict(X_test)
        mlflow.log_text(str(confusion_matrix(y_test, y_pred)), "confusion_matrix.txt")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

        # Log Model
        logger.info("Logging model to MLflow...")
        signature = infer_signature(X_test, y_pred)
        
        # First, log the model without registration
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
        )
        logger.info(f"Model logged to: {model_info.model_uri}")

        # Save Model

        # try:
        #     logger.info(f"Saving model as '{MODEL_NAME}'...")
        #     mlflow.sklearn.save_model(
        #         MODEL_NAME,
        #         "credit-score-catalog/credit-scoring/"
        #     )
        #     logger.info(f"Model saved successfully")
        # except Exception as e:
        #     logger.warning(
        #         f"Model Saving failed (model still logged): {e}\n"
        #         "You can manually register the model from the Databricks UI."
        #     )

        # Generate and Log Plots
        try:
            save_plots(model, X_test, y_test, X_train, y_train, run_id)
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit scoring model")

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--register", 
        action="store_true", 
        help="Register model in MLflow Registry (requires proper IAM permissions)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV data file",
    )

    args = parser.parse_args()

    train_model(
        mlflow_tracking_uri=args.mlflow_uri,
        register_model=args.register,
        data_path=args.data_path,
    )