"""Load the saved artifacts and make validated predictions for new customers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (CATEGORICAL_FEATURES, FEATURES, METADATA_FILENAME, MODEL_FILENAME, MODELS_DIR,
                        NUMERIC_FEATURES, PREPROCESSOR_FILENAME)
from src.preprocessing import clean_data

TRAIN_HINT = "Train the model first with:  python -m src.train"


class ModelNotFoundError(FileNotFoundError):
    """Saved model artifacts are missing."""


class InputValidationError(ValueError):
    """Customer input is missing fields or contains invalid values."""


class ChurnPredictor:
    def __init__(self, models_dir: Path | str = MODELS_DIR):
        import tensorflow as tf
        d = Path(models_dir)
        paths = [d / MODEL_FILENAME, d / PREPROCESSOR_FILENAME, d / METADATA_FILENAME]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise ModelNotFoundError(f"Missing model artifacts in {d}: {missing}. {TRAIN_HINT}")
        self.model = tf.keras.models.load_model(paths[0])
        self.preprocessor = joblib.load(paths[1])
        self.metadata = json.loads(paths[2].read_text(encoding="utf-8"))
        self.threshold = float(self.metadata["threshold"]["value"])
        self.categories = self.metadata["categories"]
        self.numeric_ranges = self.metadata["numeric_ranges"]

    # ------------------------------------------------------------ validation
    def validate_input(self, record: dict) -> tuple[dict, list[str]]:
        """Return (clean record, warnings). Raises InputValidationError for unusable input."""
        if not isinstance(record, dict):
            raise InputValidationError("Customer record must be a dictionary of feature values.")
        unexpected = sorted(set(record) - set(FEATURES) - {"customerID"})
        if unexpected:
            raise InputValidationError(f"Unexpected fields: {unexpected}")
        errors, warnings, out = [], [], {}
        for col in NUMERIC_FEATURES:
            v = record.get(col)
            if v is None or (isinstance(v, str) and not v.strip()):
                if col == "TotalCharges":
                    out[col] = np.nan  # imputed by the pipeline (training median)
                    warnings.append("TotalCharges not given; the training-set median was used.")
                else:
                    errors.append(f"Missing required field: {col}")
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{col} must be a number, got {v!r}")
                continue
            if not math.isfinite(x) or x < 0:
                errors.append(f"{col} must be a finite, non-negative number, got {v!r}")
                continue
            lo, hi = self.numeric_ranges[col]
            if x < lo or x > hi:
                warnings.append(f"{col}={x:g} is outside the training range [{lo:g}, {hi:g}]; treat the result with caution.")
            out[col] = x
        for col in CATEGORICAL_FEATURES:
            v = record.get(col)
            if v is None or (isinstance(v, str) and not v.strip()):
                errors.append(f"Missing required field: {col}")
            elif str(v) not in self.categories[col]:
                errors.append(f"{col}: {v!r} is not one of {self.categories[col]}")
            else:
                out[col] = str(v)
        if errors:
            raise InputValidationError("; ".join(errors))
        return out, warnings

    # ------------------------------------------------------------ prediction
    def predict_proba_frame(self, X: pd.DataFrame) -> np.ndarray:
        """Churn probabilities for a dataframe of raw feature columns (identical preprocessing to training)."""
        X = clean_data(X)[FEATURES]
        Z = self.preprocessor.transform(X).astype("float32")
        return self.model.predict(Z, verbose=0).ravel()

    def predict_one(self, record: dict) -> dict:
        clean, warnings = self.validate_input(record)
        p = float(self.predict_proba_frame(pd.DataFrame([clean]))[0])
        churn = p >= self.threshold
        return {"churn_probability": p, "predicted_class": int(churn),
                "label": "Churn" if churn else "No churn", "threshold": self.threshold, "warnings": warnings}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise InputValidationError(f"Missing columns: {missing}")
        p = self.predict_proba_frame(df[FEATURES])
        return pd.DataFrame({"churn_probability": p, "predicted_class": (p >= self.threshold).astype(int)}, index=df.index)


def load_predictor(models_dir: Path | str = MODELS_DIR) -> ChurnPredictor:
    return ChurnPredictor(models_dir)
