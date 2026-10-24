"""Dataset download, loading, validation and inspection."""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (CATEGORICAL_FEATURES, DATA_URL, ID_COL, RAW_PATH, REQUIRED_COLUMNS, TARGET)


class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be downloaded, read or validated."""


def manual_instructions(path: Path = RAW_PATH, url: str = DATA_URL) -> str:
    return (f"Download the CSV manually from:\n  {url}\nand save it as:\n  {path}\n"
            "Then re-run the command.")


def validate_columns(df: pd.DataFrame) -> None:
    """Raise DataLoadError if required columns are missing or the table is empty."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"Dataset is missing required columns: {missing}. "
                            "The file may not be the IBM Telco Customer Churn CSV.")
    if df.empty:
        raise DataLoadError("Dataset contains no rows.")


def _parse_csv(source) -> pd.DataFrame:
    try:
        df = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"File is not a valid CSV: {exc}") from exc
    validate_columns(df)
    return df


def download_dataset(dest: Path = RAW_PATH, url: str = DATA_URL, timeout: int = 30,
                     force: bool = False) -> Path:
    """Download the CSV unless it already exists locally (and force is False)."""
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # fixed https URL
            content = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataLoadError(f"Automatic download failed ({exc}).\n{manual_instructions(dest, url)}") from exc
    try:
        _parse_csv(io.BytesIO(content))  # validate before saving
    except DataLoadError as exc:
        raise DataLoadError(f"Downloaded file is invalid: {exc}\n{manual_instructions(dest, url)}") from exc
    dest.write_bytes(content)
    return dest


def load_data(path: Path | str | None = None, download: bool = True) -> pd.DataFrame:
    """Load the raw dataset (downloading it first if necessary) and validate its columns."""
    path = Path(path) if path else RAW_PATH
    if not path.exists():
        if not download:
            raise DataLoadError(f"Dataset not found at {path}.\n{manual_instructions(path)}")
        download_dataset(path)
    return _parse_csv(path)


def inspect_data(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run data-quality checks on the RAW dataframe; return results and optionally print them."""
    report: dict = {"shape": df.shape}
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    blanks = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in text_cols}
    report["blank_strings"] = {c: n for c, n in blanks.items() if n}
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_ids"] = int(df[ID_COL].duplicated().sum())
    report["unique_values"] = {c: sorted(df[c].dropna().astype(str).unique().tolist())
                               for c in CATEGORICAL_FEATURES + [TARGET]}
    problems = []
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad = int(tc.isna().sum() - df["TotalCharges"].isna().sum())
    if n_bad:
        problems.append(f"TotalCharges: {n_bad} non-numeric/blank entries")
    bad_target = sorted(set(df[TARGET].dropna().astype(str)) - {"Yes", "No"})
    if bad_target:
        problems.append(f"Churn contains unexpected values: {bad_target}")
    if df[TARGET].isna().any():
        problems.append("Churn contains missing values")
    if not set(df["SeniorCitizen"].dropna().unique()) <= {0, 1}:
        problems.append("SeniorCitizen has values other than 0/1")
    for col in ("tenure", "MonthlyCharges"):
        if (pd.to_numeric(df[col], errors="coerce") < 0).any():
            problems.append(f"{col} has negative values")
    report["problems"] = problems
    if verbose:
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\nFirst 5 rows:\n{df.head()}\n")
        print("Column types:\n" + df.dtypes.to_string() + "\n")
        print("Numerical summary:\n" + df.describe().round(2).to_string() + "\n")
        print(f"Missing values: {report['missing_values'] or 'none'}")
        print(f"Blank strings:  {report['blank_strings'] or 'none'}")
        print(f"Duplicate rows: {report['duplicate_rows']}, duplicate IDs: {report['duplicate_ids']}")
        print(f"Problems found: {problems or 'none'}")
    return report
