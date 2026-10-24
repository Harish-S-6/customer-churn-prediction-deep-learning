import urllib.error

import pandas as pd
import pytest

from src import data_loader
from src.data_loader import DataLoadError, download_dataset, inspect_data, load_data, validate_columns


def test_load_valid_csv(tmp_path, raw_df):
    p = tmp_path / "d.csv"
    raw_df.to_csv(p, index=False)
    df = load_data(p, download=False)
    assert df.shape == raw_df.shape


def test_missing_columns_rejected(tmp_path, raw_df):
    p = tmp_path / "bad.csv"
    raw_df.drop(columns=["Contract"]).to_csv(p, index=False)
    with pytest.raises(DataLoadError, match="Contract"):
        load_data(p, download=False)


def test_missing_file_without_download(tmp_path):
    with pytest.raises(DataLoadError, match="manually"):
        load_data(tmp_path / "nope.csv", download=False)


def test_invalid_csv_content(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(DataLoadError):
        load_data(p, download=False)


def test_download_failure_is_graceful(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(data_loader.urllib.request, "urlopen", boom)
    with pytest.raises(DataLoadError, match="Automatic download failed"):
        download_dataset(tmp_path / "x.csv")


def test_existing_file_not_downloaded_again(tmp_path, monkeypatch):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    monkeypatch.setattr(data_loader.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert download_dataset(p) == p


def test_inspect_detects_blank_total_charges(raw_df):
    rep = inspect_data(raw_df, verbose=False)
    assert rep["blank_strings"]["TotalCharges"] == int((raw_df["tenure"] == 0).sum())
    assert any("TotalCharges" in p for p in rep["problems"])


def test_validate_empty_dataframe(raw_df):
    with pytest.raises(DataLoadError):
        validate_columns(raw_df.iloc[0:0])
