"""Shared fixtures: a small synthetic dataset with the same schema as the IBM Telco CSV (no download needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    tenure = rng.randint(0, 73, n)
    monthly = np.round(rng.uniform(18, 119, n), 2)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    def addon():
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))
    logit = -0.04 * tenure + 0.01 * monthly + np.where(contract == "Month-to-month", 1.0, -1.0) - 0.3
    churn = np.where(rng.rand(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    total = np.round(tenure * monthly, 2).astype(object)
    total[tenure == 0] = " "  # blank string, as in the real file
    return pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)], "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]), "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n), "tenure": tenure, "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": internet, "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(), "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn})


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic()


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, raw_df):
    """Train a tiny model once per test session on synthetic data."""
    from src.train import train_from_dataframe
    d = tmp_path_factory.mktemp("models")
    train_from_dataframe(raw_df, models_dir=d, epochs=3, verbose=0)
    return d
