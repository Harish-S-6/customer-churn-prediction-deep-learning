"""Streamlit app: overview, data dashboard, churn prediction and model performance.

Run with:  streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import eda
from src.config import DATA_URL, FIGURES_DIR, METRICS_PATH, RAW_PATH
from src.data_loader import DataLoadError, load_data
from src.predict import InputValidationError, ModelNotFoundError, load_predictor
from src.preprocessing import clean_data

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")


# ----------------------------------------------------------------- cached loaders (no display code)
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset() -> pd.DataFrame:
    return clean_data(load_data())


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    return load_predictor()


@st.cache_data
def get_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------- pages
def page_overview():
    st.title("📉 Exploratory Data Analysis and Customer Churn Prediction Using Deep Learning")
    st.markdown("""
**Customer churn** is the loss of customers who stop using a company's service. Because keeping a customer is
usually cheaper than winning a new one, companies study which customers are likely to leave.

### Objectives
1. Explore and clean the IBM Telco Customer Churn dataset.
2. Visualise churn patterns by tenure, contract, charges and services.
3. Train a Multilayer Perceptron (TensorFlow/Keras) to estimate the probability that a customer churns.
4. Evaluate it on held-out data and explain which features it relies on.
5. Provide this interactive app for exploration and prediction.

### Technologies
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · TensorFlow/Keras · Streamlit · Pytest

### Workflow
Data loading & validation → EDA → cleaning & preprocessing (fit on training data only) → stratified 70/15/15 split →
MLP training with early stopping → test-set evaluation → permutation importance → this app.
""")
    st.info(f"Dataset: IBM Telco Customer Churn sample data — [{DATA_URL}]({DATA_URL}). "
            "It is a sample dataset; results do not describe any real company.")


def page_dashboard():
    st.header("Data Analysis Dashboard")
    try:
        df = get_dataset()
    except DataLoadError as exc:
        st.error("The dataset is not available.")
        st.code(str(exc))
        st.markdown("Run `python scripts/download_data.py` or place the CSV at "
                    f"`{RAW_PATH.relative_to(RAW_PATH.parents[2])}`.")
        return
    churn = df["Churn"].eq("Yes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn.mean() * 100:.2f}%")
    c3.metric("Mean tenure (months)", f"{df['tenure'].mean():.1f}")
    c4.metric("Mean monthly charge", f"{df['MonthlyCharges'].mean():.2f}")

    left, right = st.columns(2)
    with left:
        show_fig(eda.plot_churn_distribution(df))
        show_fig(eda.plot_correlation(df))
    with right:
        show_fig(eda.plot_monthly_charges(df))
        show_fig(eda.plot_monthly_by_churn(df))
    show_fig(eda.plot_churn_by_contract(df))

    st.subheader("Explore churn rate by any categorical field")
    col = st.selectbox("Field", [c for c in eda.CATEGORICAL_FEATURES_FOR_APP])
    st.dataframe(eda.churn_rate_by(df, col))
    show_fig(eda.plot_churn_rate_by_category(df, col))

    st.subheader("Numerical summary")
    st.dataframe(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))
    st.caption("Associations shown here do not prove that any feature causes churn.")


def page_prediction():
    st.header("Customer Churn Prediction")
    try:
        predictor = get_predictor()
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.code("python scripts/download_data.py\npython -m src.train\npython -m src.evaluate", language="bash")
        return
    cats = predictor.categories
    yn = lambda b: "Yes" if b else "No"  # noqa: E731

    a, b, c = st.columns(3)
    with a:
        st.subheader("Customer")
        gender = st.selectbox("Gender", cats["gender"])
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has a partner")
        dependents = st.checkbox("Has dependents")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
    with b:
        st.subheader("Services")
        phone = st.checkbox("Phone service", value=True)
        lines = st.selectbox("Multiple lines", ["No", "Yes"] if phone else ["No phone service"])
        internet = st.selectbox("Internet service", cats["InternetService"])
        has_net = internet != "No"
        addon = lambda label: st.selectbox(label, ["No", "Yes"] if has_net else ["No internet service"])  # noqa: E731
        security, backup, protection, support = (addon("Online security"), addon("Online backup"),
                                                 addon("Device protection"), addon("Tech support"))
        tv, movies = addon("Streaming TV"), addon("Streaming movies")
    with c:
        st.subheader("Contract & billing")
        contract = st.selectbox("Contract", cats["Contract"])
        paperless = st.checkbox("Paperless billing", value=True)
        payment = st.selectbox("Payment method", cats["PaymentMethod"])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0, step=0.5)
        estimate = st.checkbox("Estimate total charges as tenure × monthly charges", value=True)
        total = (tenure * monthly) if estimate else st.number_input("Total charges", min_value=0.0, value=800.0, step=10.0)
        st.caption(f"Total charges used: {total:,.2f}")

    record = {"gender": gender, "SeniorCitizen": yn(senior), "Partner": yn(partner), "Dependents": yn(dependents),
              "tenure": tenure, "PhoneService": yn(phone), "MultipleLines": lines, "InternetService": internet,
              "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
              "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
              "PaperlessBilling": yn(paperless), "PaymentMethod": payment, "MonthlyCharges": monthly,
              "TotalCharges": total}
    if st.button("Predict Churn", type="primary"):
        try:
            with st.spinner("Predicting..."):
                res = predictor.predict_one(record)
        except InputValidationError as exc:
            st.error(f"Invalid input: {exc}")
            return
        p = res["churn_probability"]
        m1, m2 = st.columns(2)
        m1.metric("Estimated churn probability", f"{p * 100:.1f}%")
        m2.metric("Decision threshold", f"{res['threshold']:.2f}")
        st.progress(min(max(p, 0.0), 1.0))
        (st.error if res["predicted_class"] else st.success)(
            f"Predicted class: **{res['label']}**")
        for w in res["warnings"]:
            st.warning(w)
        st.caption("This is a statistical estimate from a model trained on sample data. It is not a guarantee "
                   "of what any individual customer will do.")


def page_performance():
    st.header("Model Performance")
    m = get_metrics()
    if m is None:
        st.warning("No evaluation results found. Train and evaluate the model first:")
        st.code("python -m src.train\npython -m src.evaluate", language="bash")
        return
    nn, lr = m["neural_network"], m["logistic_regression_baseline"]
    st.caption(f"Computed on the held-out test set ({m['test_samples']:,} customers, "
               f"{m['test_churn_rate'] * 100:.1f}% churn). Threshold {nn['threshold']:.2f} was chosen on validation data. "
               f"Always-'No' accuracy would be {m['majority_class_accuracy'] * 100:.1f}%.")
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                        ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]):
        col.metric(label, f"{nn[key]:.3f}")
    table = pd.DataFrame({"Neural network (MLP)": {k: nn[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
                          "Logistic regression": {k: lr[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}}).round(4)
    st.dataframe(table)
    cm = nn["confusion_matrix"]
    st.write(f"Confusion matrix counts: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    figs = [("model_confusion_matrix.png", "Confusion matrix"), ("model_roc_curve.png", "ROC curve"),
            ("model_precision_recall_curve.png", "Precision-recall curve"), ("model_feature_importance.png", "Permutation importance"),
            ("model_training_loss.png", "Training vs validation loss"), ("model_training_accuracy.png", "Training vs validation accuracy")]
    grid = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        p = FIGURES_DIR / f
        if p.exists():
            grid[i % 2].image(str(p), caption=cap)
    st.caption("Permutation importance describes what this trained model relies on; it does not show causal effects.")


PAGES = {"Project Overview": page_overview, "Data Analysis Dashboard": page_dashboard,
         "Customer Churn Prediction": page_prediction, "Model Performance": page_performance}


def main():
    st.sidebar.title("Navigation")
    PAGES[st.sidebar.radio("Go to", list(PAGES))]()


main()
