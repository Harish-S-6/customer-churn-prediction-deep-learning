import pandas as pd
import pytest

from src.config import FEATURES
from src.predict import ChurnPredictor, InputValidationError, ModelNotFoundError
from src.preprocessing import clean_data

GOOD = {"gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "tenure": 12,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 55.5, "TotalCharges": 666.0}


@pytest.fixture(scope="module")
def predictor(trained_dir):
    return ChurnPredictor(trained_dir)


def test_prediction_output(predictor):
    r = predictor.predict_one(GOOD)
    assert 0.0 <= r["churn_probability"] <= 1.0
    assert r["predicted_class"] in (0, 1) and r["label"] in ("Churn", "No churn")
    assert r["predicted_class"] == int(r["churn_probability"] >= r["threshold"])


def test_missing_artifacts(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        ChurnPredictor(tmp_path)


@pytest.mark.parametrize("change", [{"Contract": "Lifetime"}, {"tenure": -3}, {"MonthlyCharges": "abc"},
                                    {"tenure": float("nan")}, {"gender": ""}, {"unknown_field": 1}])
def test_invalid_input_rejected(predictor, change):
    with pytest.raises(InputValidationError):
        predictor.predict_one({**GOOD, **change})


def test_missing_field_rejected(predictor):
    rec = dict(GOOD)
    rec.pop("Contract")
    with pytest.raises(InputValidationError, match="Contract"):
        predictor.predict_one(rec)


def test_missing_total_charges_allowed_with_warning(predictor):
    r = predictor.predict_one({**GOOD, "TotalCharges": None})
    assert 0 <= r["churn_probability"] <= 1 and r["warnings"]


def test_non_dict_rejected(predictor):
    with pytest.raises(InputValidationError):
        predictor.predict_one([1, 2, 3])


def test_batch_matches_single(predictor, raw_df):
    out = predictor.predict_batch(clean_data(raw_df).head(5)[FEATURES])
    assert len(out) == 5 and out["churn_probability"].between(0, 1).all()
    single = predictor.predict_one({k: v for k, v in clean_data(raw_df).iloc[0][FEATURES].to_dict().items()})
    assert abs(single["churn_probability"] - out["churn_probability"].iloc[0]) < 1e-5


def test_metadata_contents(predictor):
    m = predictor.metadata
    assert m["target"]["column"] == "Churn" and "customerID" not in m["raw_features"]
    assert m["n_features"] == len(m["feature_names"]) and 0 < m["threshold"]["value"] < 1
