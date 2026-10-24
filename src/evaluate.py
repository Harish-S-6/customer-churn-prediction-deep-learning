"""Evaluate the saved model on the held-out test set. Run with:  python -m src.evaluate"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import FEATURES, FIGURES_DIR, METRICS_PATH, MODELS_DIR, REPORTS_DIR, SEED
from src.data_loader import load_data
from src.predict import ChurnPredictor
from src.preprocessing import prepare_splits
from src.train import decide_class_weights, select_threshold


def compute_metrics(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(y_true, pred, target_names=["No churn", "Churn"],
                                                       output_dict=True, zero_division=0),
    }


def permutation_importance(predictor: ChurnPredictor, X_test: pd.DataFrame, y_test, n_repeats: int = 10,
                           seed: int = SEED) -> pd.DataFrame:
    """Drop in test ROC-AUC when one RAW feature column is shuffled (one-hot columns are therefore grouped)."""
    rng = np.random.RandomState(seed)
    base = roc_auc_score(y_test, predictor.predict_proba_frame(X_test))
    rows = []
    for col in FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - roc_auc_score(y_test, predictor.predict_proba_frame(Xp)))
        rows.append((col, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["feature", "mean_auc_drop", "std"]).sort_values("mean_auc_drop", ascending=False)


# ------------------------------------------------------------------ plots
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: dict, path, threshold):
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14},
                xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"], ax=ax)
    ax.set(title=f"Confusion Matrix (test set, threshold {threshold:.2f})", xlabel="Predicted", ylabel="Actual")
    _save(fig, path)


def plot_roc(y, prob, path, baseline_prob=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    fpr, tpr, _ = roc_curve(y, prob)
    ax.plot(fpr, tpr, lw=2, label=f"MLP (AUC = {auc(fpr, tpr):.3f})")
    if baseline_prob is not None:
        f2, t2, _ = roc_curve(y, baseline_prob)
        ax.plot(f2, t2, lw=1.5, ls="-.", label=f"Logistic regression (AUC = {auc(f2, t2):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(title="ROC Curve (test set)", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr(y, prob, path):
    p, r, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(r, p, lw=2, label=f"MLP (PR-AUC = {average_precision_score(y, prob):.3f})")
    ax.axhline(np.mean(y), color="k", ls="--", label=f"No-skill (churn rate = {np.mean(y):.2f})")
    ax.set(title="Precision-Recall Curve (test set)", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_history(hist: dict, key: str, path, best_epoch: int):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = range(1, len(hist[key]) + 1)
    ax.plot(ep, hist[key], label=f"Training {key}")
    ax.plot(ep, hist["val_" + key], label=f"Validation {key}")
    ax.axvline(best_epoch, color="gray", ls="--", label=f"Best epoch = {best_epoch}")
    ax.set(title=f"Training vs Validation {key.capitalize()}", xlabel="Epoch", ylabel=key.capitalize())
    ax.legend()
    _save(fig, path)


def plot_importance(imp: pd.DataFrame, path, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d["mean_auc_drop"], xerr=d["std"], color="#2b6cb0")
    ax.set(title="Permutation Importance (drop in test ROC-AUC)", xlabel="Mean decrease in ROC-AUC")
    _save(fig, path)


# ------------------------------------------------------------------ main workflow
def evaluate_from_dataframe(df_raw: pd.DataFrame, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
                            n_repeats: int = 10) -> dict:
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred = ChurnPredictor(models_dir)
    meta = pred.metadata
    s = prepare_splits(df_raw, seed=meta["seed"])  # identical deterministic split as training
    if len(s["y_test"]) != meta["split"]["test"]:
        raise RuntimeError("Test split differs from training; retrain with `python -m src.train`.")

    p_test = pred.predict_proba_frame(s["X_test"])
    p_val = pred.predict_proba_frame(s["X_val"])
    thr = pred.threshold
    nn = compute_metrics(s["y_test"], p_test, thr)
    nn["validation_roc_auc"] = float(roc_auc_score(s["y_val"], p_val))
    p_train = pred.predict_proba_frame(s["X_train"])
    nn["train_accuracy_at_threshold"] = float(accuracy_score(s["y_train"], (p_train >= thr).astype(int)))

    # Logistic-regression baseline, same preprocessing, threshold chosen on validation data.
    Ztr, Zva, Zte = (pred.preprocessor.transform(s[k]) for k in ("X_train", "X_val", "X_test"))
    cw, _ = decide_class_weights(s["y_train"])
    lr = LogisticRegression(max_iter=2000, class_weight=cw, random_state=meta["seed"]).fit(Ztr, s["y_train"])
    lr_thr, _ = select_threshold(s["y_val"], lr.predict_proba(Zva)[:, 1])
    lr_prob = lr.predict_proba(Zte)[:, 1]
    baseline = compute_metrics(s["y_test"], lr_prob, lr_thr)

    imp = permutation_importance(pred, s["X_test"], s["y_test"], n_repeats=n_repeats, seed=meta["seed"])
    plot_confusion(nn["confusion_matrix"], fig_dir / "model_confusion_matrix.png", thr)
    plot_roc(s["y_test"], p_test, fig_dir / "model_roc_curve.png", lr_prob)
    plot_pr(s["y_test"], p_test, fig_dir / "model_precision_recall_curve.png")
    best = meta["training"]["best_epoch"]
    plot_history(meta["history"], "accuracy", fig_dir / "model_training_accuracy.png", best)
    plot_history(meta["history"], "loss", fig_dir / "model_training_loss.png", best)
    plot_importance(imp, fig_dir / "model_feature_importance.png")

    y_test = s["y_test"]
    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_samples": int(len(y_test)), "test_churn_rate": float(y_test.mean()),
        "majority_class_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        "neural_network": nn, "logistic_regression_baseline": baseline,
        "permutation_importance": imp.round(5).to_dict(orient="records"),
        "notes": ["All metrics are computed on the held-out test set; the threshold was chosen on validation data.",
                  "Permutation importance is computed on test data and reflects reliance of this trained model "
                  "on each raw feature (one-hot columns grouped); it is not a causal effect."],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = [("Neural network (MLP)", nn), ("Logistic regression", baseline)]
    md = ["# Model Results (held-out test set)", "",
          "| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {n} | {m['threshold']:.2f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
           f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |" for n, m in rows]
    md += ["", f"Majority-class accuracy on the test set: {results['majority_class_accuracy']:.4f}. "
           "Precision, recall and F1 refer to the churn class.", ""]
    (reports_dir / "model_results.md").write_text("\n".join(md), encoding="utf-8")
    return results


def main() -> None:
    res = evaluate_from_dataframe(load_data())
    nn = res["neural_network"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in nn.items()
                      if k != "classification_report"}, indent=2))
    print(f"Saved {METRICS_PATH} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
