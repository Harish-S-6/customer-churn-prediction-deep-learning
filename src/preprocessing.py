"""Cleaning, target encoding, splitting and the scikit-learn preprocessing pipeline.

The same `clean_data` and fitted `ColumnTransformer` are used at training and inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, FEATURES, ID_COL, NUMERIC_FEATURES, SEED, TARGET

_SENIOR_MAP = {0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes",
               0.0: "No", 1.0: "Yes", True: "Yes", False: "No"}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    * strips whitespace in text columns and turns blank strings into missing values
    * converts numeric columns to numbers; a blank TotalCharges for a customer with
      tenure == 0 is set to 0.0 (no billing yet). Any other missing numeric value stays NaN
      and is imputed (median) inside the pipeline, fitted on the training set only.
    * maps SeniorCitizen 0/1 to "No"/"Yes" so it is treated as a categorical feature
    """
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(object)
            s = s.where(s.isna(), s.astype(str).str.strip())
            df[col] = s.replace("", np.nan)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"TotalCharges", "tenure"} <= set(df.columns):
        df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(_SENIOR_MAP).astype(object)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def encode_target(series: pd.Series) -> np.ndarray:
    """Convert Churn (Yes/No) into 1/0. Raises ValueError on any other value."""
    s = series.astype(str).str.strip()
    bad = sorted(set(s) - {"Yes", "No"})
    if bad:
        raise ValueError(f"Unexpected values in target column '{TARGET}': {bad}")
    return (s == "Yes").astype(int).to_numpy()


def split_features_target(df: pd.DataFrame):
    """Return (X, y). X holds only model features: no ID and never the target."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURES].copy()
    assert TARGET not in X.columns and ID_COL not in X.columns
    y = encode_target(df[TARGET]) if TARGET in df.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + standardize numeric columns; mode-impute + one-hot encode categoricals."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
                             remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(n) for n in preprocessor.get_feature_names_out()]


def stratified_split(X: pd.DataFrame, y: np.ndarray, seed: int = SEED,
                     val_size: float = 0.15, test_size: float = 0.15) -> dict:
    """Stratified train/validation/test split (default 70/15/15) with a fixed seed."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size), stratify=y_tmp, random_state=seed)
    return dict(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)


def prepare_splits(df_raw: pd.DataFrame, seed: int = SEED) -> dict:
    """clean -> split features/target -> stratified split. Nothing is fitted here."""
    X, y = split_features_target(clean_data(df_raw))
    return stratified_split(X, y, seed=seed)
