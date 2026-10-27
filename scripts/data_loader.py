from pathlib import Path

import numpy as np
import pandas as pd


def make_training_sample(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """A cleaned extract of the loan book: one row per approved loan, only columns known at approval.

    The leaky and ID columns from Module 1 are already removed. income and credit_score keep a few
    gaps so features.py has something to impute. Target: defaulted_in_next_30_days.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, n)
    employment_years = np.minimum(rng.gamma(2.5, 3.0, n), age - 18).round(1)
    annual_income = (rng.lognormal(10.6, 0.35, n) * (1 + 0.03 * employment_years)).round(-2)
    existing_loans = rng.poisson(1.2, n)
    missed_payments_12m = rng.poisson(0.5, n)
    credit_utilization = rng.beta(2, 5, n).round(2)
    credit_score = (720 - 120 * credit_utilization - 35 * missed_payments_12m
                    + 2 * employment_years + rng.normal(0, 30, n)).clip(300, 850).round(0)

    loan_type = rng.choice(["personal", "auto", "mortgage", "student"], n, p=[0.4, 0.25, 0.2, 0.15])
    typical_size = pd.Series(loan_type).map({"personal": 0.15, "auto": 0.4, "mortgage": 3.0, "student": 0.5}).to_numpy()
    loan_amount = (annual_income * typical_size * rng.lognormal(0, 0.3, n)).round(-2)
    region = rng.choice(["North", "South", "East", "West"], n)

    # Repayment burden: mortgages are spread over decades, so judge them on a smaller yearly share
    yearly_share = np.where(loan_type == "mortgage", 0.1, 0.4)
    burden = loan_amount * yearly_share / annual_income
    logit = (-2.2
             - 0.012 * (credit_score - 650)
             + 2.5 * (credit_utilization - 0.3)
             + 0.6 * missed_payments_12m
             + 0.35 * existing_loans
             + 2.0 * (burden - 0.1)
             - 0.08 * employment_years
             + 0.5 * (loan_type == "personal"))
    defaulted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "credit_utilization": credit_utilization,
        "missed_payments_12m": missed_payments_12m,
        "existing_loans": existing_loans,
        "loan_amount": loan_amount,
        "loan_type": loan_type,
        "region": region,
        "defaulted_in_next_30_days": defaulted,
    })
    # Short employment histories more often leave income blank; a few credit files are missing
    df.loc[rng.random(n) < np.where(employment_years < 2, 0.12, 0.03), "annual_income"] = np.nan
    df.loc[rng.random(n) < 0.03, "credit_score"] = np.nan
    return df


def ensure_dataset(data_path: str, target_column: str) -> Path:
    """Create the loan training sample if no data file exists yet."""
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        make_training_sample().to_csv(path, index=False)

    return path


def load_data(data_path: str, target_column: str) -> pd.DataFrame:
    """Load the dataset and validate that the target exists."""
    path = ensure_dataset(data_path, target_column)
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    return df


def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    """Return train/test arrays for features and target."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
