"""Train the Keras MLP churn classifier. Run with:  python -m src.train"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (CATEGORICAL_FEATURES, FEATURES, METADATA_FILENAME, MODEL_FILENAME, MODELS_DIR,
                        NUMERIC_FEATURES, PREPROCESSOR_FILENAME, SEED, TARGET)
from src.data_loader import load_data
from src.preprocessing import build_preprocessor, get_feature_names, prepare_splits

IMBALANCE_CUTOFF = 0.40  # use class weights if the minority class is below this share of the training set


def set_seeds(seed: int = SEED) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_model(n_features: int, learning_rate: float = 0.001):
    """MLP: n_features -> Dense(64, relu) -> Dropout(0.30) -> Dense(32, relu) -> Dropout(0.20) -> Dense(1, sigmoid)."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="churn_mlp")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def decide_class_weights(y_train: np.ndarray, cutoff: float = IMBALANCE_CUTOFF):
    """Return (class_weight dict or None, info dict). Weights are used only when the minority share < cutoff."""
    minority_share = float(min(y_train.mean(), 1 - y_train.mean()))
    use = minority_share < cutoff
    info = {"minority_share_train": round(minority_share, 4), "cutoff": cutoff, "used": bool(use),
            "rationale": ("minority class share is below the cutoff, so 'balanced' class weights are applied to "
                          "the training loss only" if use else "classes are reasonably balanced; no weights applied")}
    if not use:
        return None, info
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    cw = {0: float(w[0]), 1: float(w[1])}
    info["weights"] = {"0": cw[0], "1": cw[1]}
    return cw, info


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing F1 for the churn class on VALIDATION data (never on test data)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def train_from_dataframe(df_raw: pd.DataFrame, models_dir: Path = MODELS_DIR, epochs: int = 100,
                         batch_size: int = 32, seed: int = SEED, learning_rate: float = 0.001,
                         patience: int = 10, verbose: int = 2) -> dict:
    """Full training workflow on a raw dataframe. Saves model, preprocessor and metadata; returns metadata."""
    import tensorflow as tf
    set_seeds(seed)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    s = prepare_splits(df_raw, seed=seed)
    print(f"Split sizes -> train: {len(s['y_train'])}, validation: {len(s['y_val'])}, test: {len(s['y_test'])}")
    for name in ("train", "val", "test"):
        print(f"  churn rate in {name}: {s['y_' + name].mean() * 100:.2f}%")
    leak = {"customerID", TARGET} & set(s["X_train"].columns)
    if leak:
        raise RuntimeError(f"Leakage: forbidden columns in features: {leak}")

    pre = build_preprocessor()
    Xtr = pre.fit_transform(s["X_train"]).astype("float32")  # fitted on TRAIN only
    Xva = pre.transform(s["X_val"]).astype("float32")
    feature_names = get_feature_names(pre)
    print(f"Transformed feature matrix: {Xtr.shape[1]} features, dtype {Xtr.dtype}")

    class_weight, cw_info = decide_class_weights(s["y_train"])
    print(f"Class weights: {cw_info}")

    model = build_model(Xtr.shape[1], learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(Xtr, s["y_train"], validation_data=(Xva, s["y_val"]), epochs=epochs, batch_size=batch_size,
                     class_weight=class_weight, callbacks=[es], verbose=verbose)
    H = {k: [float(x) for x in v] for k, v in hist.history.items()}
    best_epoch = int(np.argmin(H["val_loss"]) + 1)

    p_val = model.predict(Xva, verbose=0).ravel()
    threshold, val_f1 = select_threshold(s["y_val"], p_val)
    print(f"Best epoch {best_epoch}/{len(H['loss'])}; threshold {threshold:.2f} (validation F1 {val_f1:.3f})")

    model.save(models_dir / MODEL_FILENAME)
    joblib.dump(pre, models_dir / PREPROCESSOR_FILENAME)

    Xtrain_df = s["X_train"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {"column": TARGET, "positive_class": "Yes", "encoding": {"No": 0, "Yes": 1}},
        "seed": seed, "tensorflow_version": tf.__version__,
        "raw_features": FEATURES, "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": feature_names, "n_features": len(feature_names),
        "categories": {c: sorted(map(str, Xtrain_df[c].dropna().unique())) for c in CATEGORICAL_FEATURES},
        "numeric_ranges": {c: [float(Xtrain_df[c].min()), float(Xtrain_df[c].max())] for c in NUMERIC_FEATURES},
        "split": {"train": int(len(s["y_train"])), "validation": int(len(s["y_val"])), "test": int(len(s["y_test"])),
                  "stratified": True, "ratio": "70/15/15"},
        "architecture": "Input -> Dense(64,relu) -> Dropout(0.30) -> Dense(32,relu) -> Dropout(0.20) -> Dense(1,sigmoid)",
        "training": {"optimizer": "Adam", "learning_rate": learning_rate, "loss": "binary_crossentropy",
                     "batch_size": batch_size, "max_epochs": epochs, "early_stopping_patience": patience,
                     "epochs_run": len(H["loss"]), "best_epoch": best_epoch},
        "class_weights": cw_info,
        "threshold": {"value": threshold, "selection": "maximize churn-class F1 on the validation set",
                      "validation_f1": val_f1},
        "history": H,
    }
    (models_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model, preprocessor and metadata to {models_dir}")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the churn MLP.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--data", type=str, default=None, help="path to the CSV (default: data/raw/...)")
    a = ap.parse_args()
    train_from_dataframe(load_data(a.data), epochs=a.epochs, batch_size=a.batch_size, seed=a.seed)


if __name__ == "__main__":
    main()
