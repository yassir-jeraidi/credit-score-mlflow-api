"""ML Configuration for Credit Scoring Model."""
from typing import List, Dict, Any

# Feature configuration
NUMERICAL_FEATURES: List[str] = [
    "age",
    "income",
    "employment_length",
    "loan_amount",
    "credit_history_length",
    "num_credit_lines",
    "derogatory_marks",
    "total_debt",
]

CATEGORICAL_FEATURES: List[str] = [
    "loan_intent",
    "home_ownership",
]

ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Loan intent categories
LOAN_INTENT_CATEGORIES: List[str] = [
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]

# Home ownership categories
HOME_OWNERSHIP_CATEGORIES: List[str] = [
    "RENT",
    "OWN",
    "MORTGAGE",
    "OTHER",
]

# Model hyperparameters - tuned for Gradient Boosting
DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "subsample": 0.8,
    "random_state": 42,
}

# MLflow configuration
MLFLOW_EXPERIMENT_NAME: str = "credit-scoring"
MODEL_NAME: str = "credit-score-model"

# Data generation settings
DEFAULT_SAMPLE_SIZE: int = 50000
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
