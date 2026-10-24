import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.preprocessing import (build_preprocessor, clean_data, encode_target, get_feature_names, prepare_splits,
                               split_features_target)


def test_blank_total_charges_become_zero_for_new_customers(raw_df):
    df = clean_data(raw_df)
    assert df["TotalCharges"].notna().all()
    assert (df.loc[raw_df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_blank_total_charges_with_tenure_stay_missing():
    df = pd.DataFrame({"tenure": [5], "TotalCharges": [" "]})
    assert clean_data(df)["TotalCharges"].isna().all()  # imputed later, in the pipeline


def test_target_conversion():
    assert encode_target(pd.Series(["Yes", "No", "Yes"])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Maybe"]))


def test_identifier_and_target_not_in_features(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert list(X.columns) == FEATURES and len(y) == len(X)


def test_senior_citizen_mapped_to_labels(raw_df):
    assert set(clean_data(raw_df)["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_stratified_split_sizes_and_balance(raw_df):
    s = prepare_splits(raw_df)
    n = len(raw_df)
    assert len(s["y_train"]) + len(s["y_val"]) + len(s["y_test"]) == n
    assert abs(len(s["y_train"]) / n - 0.70) < 0.02 and abs(len(s["y_test"]) / n - 0.15) < 0.02
    rates = [s[k].mean() for k in ("y_train", "y_val", "y_test")]
    assert max(rates) - min(rates) < 0.03
    assert not set(s["X_train"].index) & set(s["X_test"].index)


def test_preprocessor_output_and_train_inference_consistency(raw_df):
    s = prepare_splits(raw_df)
    pre = build_preprocessor()
    Ztr = pre.fit_transform(s["X_train"])
    Zte = pre.transform(s["X_test"])
    assert Ztr.shape[1] == Zte.shape[1] == len(get_feature_names(pre))
    assert Ztr.dtype.kind == "f" and not np.isnan(Ztr).any() and not np.isnan(Zte).any()
    # scaler learned from training data only
    num_idx = [get_feature_names(pre).index(c) for c in NUMERIC_FEATURES]
    assert np.allclose(Ztr[:, num_idx].mean(axis=0), 0, atol=1e-6)


def test_unknown_category_ignored_at_inference(raw_df):
    s = prepare_splits(raw_df)
    pre = build_preprocessor().fit(s["X_train"])
    X = s["X_test"].head(3).copy()
    X["Contract"] = "Five year"
    Z = pre.transform(X)
    names = get_feature_names(pre)
    assert Z[:, [i for i, n in enumerate(names) if n.startswith("Contract_")]].sum() == 0


def test_missing_feature_column_raises(raw_df):
    with pytest.raises(ValueError):
        split_features_target(clean_data(raw_df).drop(columns=["tenure"]))
