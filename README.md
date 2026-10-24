# Exploratory Data Analysis and Customer Churn Prediction Using Deep Learning

End-to-end customer churn prediction project using exploratory data analysis, Python, TensorFlow/Keras, Scikit-learn, and Streamlit.

## Project overview
This project explores the IBM Telco Customer Churn sample dataset, cleans and preprocesses it without data leakage, trains a
Multilayer Perceptron (MLP) with TensorFlow/Keras to estimate each customer's probability of churning, evaluates it on a
held-out test set, and serves the model through an interactive Streamlit app.

## Problem statement
Only a minority of telecom customers leave, and it is not obvious which account, service and billing attributes are
associated with leaving. The goal is to describe those patterns (EDA) and to test whether a neural network can predict churn
from the available attributes.

## Project objectives
1. Load and validate the IBM Telco Customer Churn dataset.
2. Perform EDA with statistics and at least five charts.
3. Build a leak-free preprocessing pipeline (imputation, one-hot encoding, standardization).
4. Train and evaluate an MLP on stratified 70/15/15 splits.
5. Explain the model with permutation importance and compare it with a logistic-regression baseline.
6. Provide a tested, reproducible codebase and a Streamlit app.

## Features
- Dataset downloader with local caching, column validation and manual-download fallback
- Data-quality report (missing values, blanks, duplicates, invalid values, IQR outliers)
- EDA notebook (executed, with charts) and an auto-generated `reports/eda_summary.md`
- `ColumnTransformer` pipeline fitted on training data only
- Keras MLP with early stopping, class weighting (only when justified) and a validation-selected threshold
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves, training curves, permutation importance
- Validated single and batch prediction API (`src/predict.py`)
- 4-page Streamlit app, pytest suite and GitHub Actions workflow

## Technology stack
Python 3.12 (the version this project was tested with) - Pandas - NumPy - Matplotlib - Seaborn - Scikit-learn - TensorFlow/Keras - Streamlit - Joblib - Jupyter - Pytest

## Dataset source
IBM Telco Customer Churn sample data.
- Documentation: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
- CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The CSV is downloaded on demand into `data/raw/` and is **not** committed to this repository.

## Repository structure
```text
customer-churn-prediction-deep-learning/
├── app.py                      # Streamlit application
├── requirements.txt
├── pytest.ini
├── data/raw, data/processed    # CSV is downloaded here (git-ignored)
├── notebooks/01_customer_churn_eda.ipynb
├── src/
│   ├── config.py               # paths, column lists, seed
│   ├── data_loader.py          # download, load, validate, inspect
│   ├── preprocessing.py        # cleaning, target encoding, split, ColumnTransformer
│   ├── eda.py                  # statistics, charts, eda_summary.md
│   ├── train.py                # Keras MLP training
│   ├── evaluate.py             # test-set metrics, plots, permutation importance
│   └── predict.py              # validated inference
├── models/                     # churn_model.keras, preprocessor.joblib, model_metadata.json
├── reports/                    # eda_summary.md, metrics.json, model_results.md, figures/
├── scripts/download_data.py
├── tests/
└── .github/workflows/python-tests.yml
```

## Installation
```bash
git clone https://github.com/Harish-S-6/customer-churn-prediction-deep-learning.git
cd customer-churn-prediction-deep-learning
```

### Virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Dataset download
```bash
python scripts/download_data.py          # skipped automatically if the file already exists; --force to refresh
```
If the download fails, save the CSV manually as `data/raw/Telco-Customer-Churn.csv` (the error message prints the URL).

## EDA notebook
```bash
python -m jupyter notebook notebooks/01_customer_churn_eda.ipynb
python -m src.eda            # regenerates reports/figures/eda_*.png and reports/eda_summary.md
```
In Google Colab: clone the repo, run `pip install -r requirements.txt`, and open the notebook from the `notebooks/` folder.

## Model training
```bash
python -m src.train          # options: --epochs 100 --batch-size 32 --seed 42 --data PATH
```
Saves `models/churn_model.keras`, `models/preprocessor.joblib` and `models/model_metadata.json`.

## Model evaluation
```bash
python -m src.evaluate       # writes reports/metrics.json, reports/model_results.md and reports/figures/model_*.png
```

## Streamlit app
```bash
streamlit run app.py
```
Pages: Project Overview, Data Analysis Dashboard, Customer Churn Prediction, Model Performance.

## Example prediction workflow
```python
from src.predict import load_predictor
predictor = load_predictor()
result = predictor.predict_one({
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 240.0})
print(result)   # churn_probability, predicted_class, label, threshold, warnings
```
Invalid input (unknown category, negative number, missing field) raises `InputValidationError` with a clear message.

## Model evaluation metrics
Computed on the held-out test set (1,057 customers, 26.6% churn) by `python -m src.evaluate` using the artifacts in `models/` (seed 42, single run).
The decision threshold (0.54) was chosen on the validation set by maximizing churn-class F1.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Neural network (MLP) | 0.7739 | 0.5556 | 0.7473 | 0.6373 | 0.8460 | 0.6427 |
| Logistic regression (baseline) | 0.7625 | 0.5385 | 0.7473 | 0.6259 | 0.8448 | 0.6592 |

Confusion matrix (MLP): TN=608, FP=168, FN=71, TP=210. Always predicting "No churn" would be 73.4% accurate, so accuracy alone is not informative here.
Precision, recall and F1 refer to the churn class. Training stopped at epoch 19 (best epoch 9).

## Actual findings
- The MLP and the logistic-regression baseline perform very similarly (ROC-AUC 0.846 vs 0.845); the neural network gives no clear advantage on this tabular dataset.
- Churn precision (0.56) is much lower than recall (0.75): the model finds most churners but also flags many customers who stay.
- Permutation importance ranks **tenure**, **Contract** and **InternetService** highest. See `reports/eda_summary.md` for the EDA findings (e.g. churn rates by contract type).
- These are single-run, single-split results from one sample dataset. Associations do not imply causation.

## Running the tests
```bash
python -m pytest tests/
```
The tests use a small synthetic dataset, so no download is needed.

## Troubleshooting
- **`Dataset not found` / download error**: download the CSV manually (URL above) into `data/raw/`.
- **`Missing model artifacts`**: run `python -m src.train` (then `python -m src.evaluate`).
- **Model fails to load after upgrading TensorFlow or scikit-learn**: re-run `python -m src.train` to regenerate the artifacts.
- **`ModuleNotFoundError: src`**: run commands from the repository root (use `python -m src.train`, not `python src/train.py`).
- **Slow first start**: TensorFlow import takes several seconds; Streamlit caches the model afterwards.

## Limitations
- Sample dataset with unknown representativeness; no time dimension.
- Moderate class imbalance; low churn precision at the chosen threshold.
- One training run and split; no hyperparameter search or cross-validation.
- Permutation importance describes model reliance, not causes; correlated features can understate each other.
- The app output is a model estimate, not a guarantee of customer behaviour.

## Future improvements
Cross-validation and repeated seeds, tuning, gradient-boosted tree baselines, SHAP explanations, calibration and cost-based threshold selection, derived features, and evaluation on newer independent data.

## References
- IBM, Telco Customer Churn sample data (links above).
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12, 2825-2830.
- Abadi et al. (2016). TensorFlow: A system for large-scale machine learning. OSDI. https://arxiv.org/abs/1603.04467
- Kingma & Ba (2015). Adam: A method for stochastic optimization. https://arxiv.org/abs/1412.6980
- Srivastava et al. (2014). Dropout. JMLR 15, 1929-1958.
- Breiman (2001). Random forests. Machine Learning 45, 5-32 (permutation importance).

## License
MIT - see `LICENSE`.
