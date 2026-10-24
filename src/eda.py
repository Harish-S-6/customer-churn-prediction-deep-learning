"""Exploratory data analysis: statistics, the five required charts, extra charts and summary report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CATEGORICAL_FEATURES, FIGURES_DIR, REPORTS_DIR, TARGET
from src.data_loader import inspect_data, load_data
from src.preprocessing import clean_data

sns.set_theme(style="whitegrid")
COLORS = {"No": "#2b6cb0", "Yes": "#c53030"}
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CATEGORICAL_FEATURES_FOR_APP = [c for c in CATEGORICAL_FEATURES]


# ---------------------------------------------------------------- statistics
def load_clean(path=None) -> pd.DataFrame:
    return clean_data(load_data(path))


def churn_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts()
    return pd.DataFrame({"customers": counts, "percentage": (counts / len(df) * 100).round(2)})


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, churned customers and churn rate (%) for each category of `col`."""
    g = df.groupby(col)[TARGET].agg(customers="size", churned=lambda s: int((s == "Yes").sum()))
    g["churn_rate_pct"] = (g["churned"] / g["customers"] * 100).round(2)
    return g.sort_values("churn_rate_pct", ascending=False)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    out["SeniorCitizen"] = (df["SeniorCitizen"] == "Yes").astype(int)
    out["Churn"] = (df[TARGET] == "Yes").astype(int)
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(df).corr()


def iqr_outliers(df: pd.DataFrame, cols=("tenure", "MonthlyCharges", "TotalCharges")) -> pd.DataFrame:
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(dict(feature=c, q1=q1, q3=q3, iqr=iqr, lower=lo, upper=hi,
                         flagged=int(((df[c] < lo) | (df[c] > hi)).sum())))
    return pd.DataFrame(rows).set_index("feature").round(2)


# ---------------------------------------------------------------- figures (each returns a Figure)
def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="#2b6cb0", ax=ax)
    ax.axvline(df["MonthlyCharges"].mean(), color="red", ls="--", label=f"Mean = {df['MonthlyCharges'].mean():.2f}")
    ax.axvline(df["MonthlyCharges"].median(), color="green", ls=":", label=f"Median = {df['MonthlyCharges'].median():.2f}")
    ax.set(title="Distribution of Monthly Charges", xlabel="Monthly Charges", ylabel="Number of Customers")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cnt = df[TARGET].value_counts().reindex(["No", "Yes"])
    bars = ax.bar(cnt.index, cnt.values, color=[COLORS["No"], COLORS["Yes"]])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + cnt.max() * 0.01, f"{v:,}\n({v / len(df) * 100:.2f}%)", ha="center")
    ax.set(title="Customer Churn Distribution", xlabel="Churn", ylabel="Number of Customers", ylim=(0, cnt.max() * 1.2))
    fig.tight_layout()
    return fig


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(correlation_matrix(df), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    fig.tight_layout()
    return fig


def plot_monthly_by_churn(df):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x=TARGET, y="MonthlyCharges", hue=TARGET, order=["No", "Yes"], hue_order=["No", "Yes"],
                palette=COLORS, legend=False, ax=ax)
    ax.set(title="Monthly Charges by Churn Status", xlabel="Churn", ylabel="Monthly Charges")
    fig.tight_layout()
    return fig


def plot_churn_by_contract(df):
    order = [c for c in CONTRACT_ORDER if c in set(df["Contract"])]
    counts = pd.crosstab(df["Contract"], df[TARGET]).loc[order]
    rates = churn_rate_by(df, "Contract").loc[order, "churn_rate_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    counts.plot(kind="bar", ax=axes[0], color=[COLORS["No"], COLORS["Yes"]], rot=0)
    axes[0].set(title="Customers by Contract Type and Churn", xlabel="Contract Type", ylabel="Number of Customers")
    axes[0].legend(title="Churn")
    bars = axes[1].bar(order, rates.values, color="#dd6b20")
    for b, v in zip(bars, rates.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v + rates.max() * 0.02, f"{v:.2f}%", ha="center")
    axes[1].set(title="Churn Rate by Contract Type", xlabel="Contract Type", ylabel="Churn Rate (%)", ylim=(0, rates.max() * 1.2))
    fig.tight_layout()
    return fig


def plot_tenure_by_churn(df):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(data=df, x="tenure", hue=TARGET, bins=36, multiple="layer", alpha=0.55, palette=COLORS, ax=ax)
    ax.set(title="Customer Tenure by Churn Status", xlabel="Tenure (months)", ylabel="Number of Customers")
    fig.tight_layout()
    return fig


def plot_churn_rate_by_category(df, col):
    rates = churn_rate_by(df, col)
    fig, ax = plt.subplots(figsize=(7, 4 + 0.15 * len(rates)))
    bars = ax.barh(rates.index.astype(str), rates["churn_rate_pct"], color="#dd6b20")
    for b, v in zip(bars, rates["churn_rate_pct"]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
    ax.invert_yaxis()
    ax.set(title=f"Churn Rate by {col}", xlabel="Churn Rate (%)", xlim=(0, max(rates["churn_rate_pct"].max() * 1.15, 1)))
    fig.tight_layout()
    return fig


FIGURES = {
    "eda_01_monthly_charges.png": plot_monthly_charges,
    "eda_02_churn_distribution.png": plot_churn_distribution,
    "eda_03_correlation_heatmap.png": plot_correlation,
    "eda_04_monthly_charges_by_churn.png": plot_monthly_by_churn,
    "eda_05_churn_by_contract.png": plot_churn_by_contract,
    "eda_06_tenure_by_churn.png": plot_tenure_by_churn,
}


def save_all_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fig = fn(df)
        p = out_dir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    fig = plot_churn_rate_by_category(df, "PaymentMethod")
    p = out_dir / "eda_07_churn_by_payment_method.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- written summary
def write_summary(df: pd.DataFrame, path: Path = REPORTS_DIR / "eda_summary.md", raw: pd.DataFrame | None = None) -> Path:
    """Write reports/eda_summary.md. Every number is computed from `df` (and `raw` for quality checks)."""
    raw = raw if raw is not None else df
    q = inspect_data(raw, verbose=False)
    cs = churn_summary(df)
    corr = correlation_matrix(df)["Churn"].drop("Churn")
    contract = churn_rate_by(df, "Contract")
    pay = churn_rate_by(df, "PaymentMethod")
    internet = churn_rate_by(df, "InternetService")
    med = df.groupby(TARGET)["MonthlyCharges"].median()
    ten = df.groupby(TARGET)["tenure"].median()
    out = iqr_outliers(df)
    top, low = contract.index[0], contract.index[-1]
    blank_mask = raw["TotalCharges"].astype(str).str.strip() == ""
    n_blank = int(blank_mask.sum())
    if n_blank:
        all_new = bool((pd.to_numeric(raw.loc[blank_mask, "tenure"], errors="coerce") == 0).all())
        blank_note = (f"`TotalCharges` is stored as text in the raw CSV and has {n_blank} blank entries. "
                      + ("All of them belong to customers with `tenure == 0` (no billing yet), so `clean_data` sets them to 0. "
                         if all_new else "Not all belong to customers with `tenure == 0`; those with tenure 0 are set to 0 and "
                         "the rest are median-imputed in the pipeline (training data only). ")
                      + "No rows are removed.")
    else:
        blank_note = "`TotalCharges` has no blank entries in this file."
    L = []
    L += ["# EDA Summary", "",
          "_Generated by `python -m src.eda` from the downloaded IBM Telco Customer Churn CSV. "
          "All figures below are computed from that file._", "",
          "## Dataset", "",
          f"- Rows: **{df.shape[0]:,}**, columns: **{df.shape[1]}**",
          f"- Duplicate rows: **{q['duplicate_rows']}**; duplicate customer IDs: **{q['duplicate_ids']}**",
          f"- Missing (NaN) values in the raw file: **{sum(q['missing_values'].values())}**",
          f"- Blank strings in raw file: **{q['blank_strings'] or 'none'}**",
          f"- Other data problems detected: {q['problems'] or 'none'}", "",
          blank_note, "",
          "## Target variable", "", cs.to_markdown(), "",
          f"{cs.loc['Yes', 'percentage']}% of customers churned, so the classes are imbalanced. A model that always "
          f"predicts \"No\" would be about {cs.loc['No', 'percentage']}% accurate, which is why precision, recall, F1 "
          "and ROC-AUC are reported alongside accuracy.", "",
          "## Numerical summary", "",
          df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2).to_markdown(), "",
          "### Potential outliers (IQR rule)", "", out.to_markdown(), "",
          "A flagged value is a candidate for inspection, not an automatic error. No observations are removed "
          "on the basis of this rule.", "",
          "## Chart interpretations", "",
          "### Figure 1 - Monthly charges",
          f"Mean {df['MonthlyCharges'].mean():.2f}, median {df['MonthlyCharges'].median():.2f}, standard deviation "
          f"{df['MonthlyCharges'].std():.2f}, range {df['MonthlyCharges'].min():.2f} to {df['MonthlyCharges'].max():.2f}. "
          "Compare the mean and median with the histogram to see whether the distribution is skewed or multi-modal.", "",
          "![](figures/eda_01_monthly_charges.png)", "",
          "### Figure 2 - Churn distribution",
          f"{int(cs.loc['Yes', 'customers']):,} churned and {int(cs.loc['No', 'customers']):,} stayed.", "",
          "![](figures/eda_02_churn_distribution.png)", "",
          "### Figure 3 - Correlation heatmap",
          "Pearson correlation of each numerical feature with churn (1 = churned): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items())
          + ". Correlation measures linear association only and does not show causation; "
            "tenure and TotalCharges also overlap strongly with each other.", "",
          "![](figures/eda_03_correlation_heatmap.png)", "",
          "### Figure 4 - Monthly charges by churn status",
          f"Median monthly charge: {med.get('No', float('nan')):.2f} for customers who stayed versus "
          f"{med.get('Yes', float('nan')):.2f} for customers who churned. Median tenure: "
          f"{ten.get('No', float('nan')):.0f} versus {ten.get('Yes', float('nan')):.0f} months.", "",
          "![](figures/eda_04_monthly_charges_by_churn.png)", "",
          "### Figure 5 - Churn rate by contract type", contract.to_markdown(), "",
          f"The highest churn rate is for **{top}** ({contract.loc[top, 'churn_rate_pct']}%) and the lowest for "
          f"**{low}** ({contract.loc[low, 'churn_rate_pct']}%).", "",
          "![](figures/eda_05_churn_by_contract.png)", "",
          "## Other observations", "",
          f"- Highest churn by payment method: **{pay.index[0]}** ({pay.iloc[0]['churn_rate_pct']}%); "
          f"lowest: **{pay.index[-1]}** ({pay.iloc[-1]['churn_rate_pct']}%).",
          f"- Highest churn by internet service type: **{internet.index[0]}** ({internet.iloc[0]['churn_rate_pct']}%); "
          f"lowest: **{internet.index[-1]}** ({internet.iloc[-1]['churn_rate_pct']}%).", "",
          "## Limitations", "",
          "- The dataset is an IBM sample dataset; its representativeness of real customer populations is unknown.",
          "- It is a single snapshot with no time dimension, so changes in behaviour over time cannot be studied.",
          "- The patterns above are associations. They do not show that changing a customer's contract, "
          "payment method or price would change whether the customer leaves.",
          "- EDA statistics are computed on the full dataset for description only; all model fitting uses training data only."]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    raw = load_data()
    inspect_data(raw)
    df = clean_data(raw)
    paths = save_all_figures(df)
    summary = write_summary(df, raw=raw)
    print(f"\nSaved {len(paths)} figures to {FIGURES_DIR}\nWrote {summary}")


if __name__ == "__main__":
    main()
