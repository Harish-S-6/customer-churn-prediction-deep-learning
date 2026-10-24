"""Central configuration: paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_FILENAME = "churn_model.keras"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
METADATA_FILENAME = "model_metadata.json"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
PREPROCESSOR_PATH = MODELS_DIR / PREPROCESSOR_FILENAME
METADATA_PATH = MODELS_DIR / METADATA_FILENAME
METRICS_PATH = REPORTS_DIR / "metrics.json"

SEED = 42
ID_COL = "customerID"
TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COL] + FEATURES + [TARGET]
