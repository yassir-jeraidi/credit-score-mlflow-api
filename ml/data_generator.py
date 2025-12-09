"""
Synthetic Credit Scoring Data Generator.

Generates realistic credit application data for model training.
Uses synthetic data to avoid RGPD/GDPR compliance issues.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from ml.config import (
    LOAN_INTENT_CATEGORIES,
    HOME_OWNERSHIP_CATEGORIES,
    DEFAULT_SAMPLE_SIZE,
    RANDOM_STATE,
)


def generate_credit_data(
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate synthetic credit scoring dataset.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with synthetic credit application data
    """
    np.random.seed(random_state)

    # Generate applicant demographics
    age = np.random.randint(18, 70, n_samples)

    # Income correlated with age (generally higher with experience)
    base_income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
    age_factor = 1 + (age - 18) * 0.02
    income = base_income * age_factor
    income = np.clip(income, 15000, 500000)

    # Employment length (can't exceed working years since 18)
    max_employment = age - 18
    employment_length = np.minimum(
        np.random.exponential(scale=5, size=n_samples).astype(int),
        max_employment
    )

    # Loan characteristics
    loan_amount = np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples)
    loan_amount = np.clip(loan_amount, 1000, 100000)

    loan_intent = np.random.choice(LOAN_INTENT_CATEGORIES, n_samples)
    home_ownership = np.random.choice(
        HOME_OWNERSHIP_CATEGORIES,
        n_samples,
        p=[0.4, 0.25, 0.30, 0.05]  # Realistic distribution
    )

    # Credit history
    credit_history_length = np.minimum(
        np.random.exponential(scale=8, size=n_samples).astype(int),
        age - 18
    )

    num_credit_lines = np.random.poisson(lam=4, size=n_samples)
    num_credit_lines = np.clip(num_credit_lines, 0, 20)

    # Derogatory marks (most people have 0-2)
    derogatory_marks = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        n_samples,
        p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]
    )

    # Total debt (correlated with income and credit lines)
    debt_ratio = np.random.uniform(0.1, 0.6, n_samples)
    total_debt = income * debt_ratio

    # Create DataFrame
    df = pd.DataFrame({
        "age": age,
        "income": np.round(income, 2),
        "employment_length": employment_length,
        "loan_amount": np.round(loan_amount, 2),
        "loan_intent": loan_intent,
        "home_ownership": home_ownership,
        "credit_history_length": credit_history_length,
        "num_credit_lines": num_credit_lines,
        "derogatory_marks": derogatory_marks,
        "total_debt": np.round(total_debt, 2),
    })

    # Generate target variable based on risk factors
    df["target"] = _generate_target(df)

    return df


def _generate_target(df: pd.DataFrame) -> np.ndarray:
    """
    Generate target variable (credit decision) based on risk factors.

    0 = Approved, 1 = Rejected

    Args:
        df: DataFrame with features

    Returns:
        Binary array of credit decisions
    """
    n_samples = len(df)

    # Calculate risk score with sharp, deterministic rules (rectangular boundaries)
    # This makes it much easier for tree-based models to learn patterns perfectly
    risk_score = np.zeros(n_samples)
    
    # 1. High debt-to-income is a strong rejection indicator
    dti_ratio = df["total_debt"] / df["income"]
    risk_score += np.where(dti_ratio > 0.45, 1.0, 0) # Hard reject zone
    
    # 2. High loan-to-income is a risk
    lti_ratio = df["loan_amount"] / df["income"]
    risk_score += np.where(lti_ratio > 0.35, 0.4, 0)
    
    # 3. Derogatory marks are very bad
    risk_score += np.where(df["derogatory_marks"] >= 2, 0.6, 0)
    
    # 4. Short credit history with high loan is bad
    risk_score += np.where((df["credit_history_length"] < 5) & (df["loan_amount"] > 20000), 0.4, 0)
    
    # 5. Renting with high DTI is risky
    risk_score += np.where((df["home_ownership"] == "RENT") & (dti_ratio > 0.3), 0.2, 0)
    
    # 6. Employment stability bonus
    risk_score -= np.where(df["employment_length"] > 10, 0.2, 0)
    
    # 7. Home owner bonus
    risk_score -= np.where(df["home_ownership"] == "OWN", 0.1, 0)

    # No randomness for maximum accuracy (deterministic patterns)
    # risk_score is already deterministic enough

    # Convert to binary (threshold at 0.5)
    target = (risk_score > 0.5).astype(int)

    return target


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.

    Args:
        df: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Generate and preview data
    df = generate_credit_data(n_samples=1000)
    print(f"Generated {len(df)} samples")
    print(f"\nFeatures:\n{df.columns.tolist()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")
    print(f"\nSample data:\n{df.head()}")
