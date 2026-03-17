"""
Data loading and train/valid/test split for weather forecasting.
Split: 12 years for training+validation, 3 years for test (chronological).
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_arima_data(data_dir: str = "data") -> pd.DataFrame:
    """Load unscaled dataset for ARIMA (and feature building)."""
    path = Path(data_dir) / "dataset_for_arima.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_normalized_data(data_dir: str = "data") -> pd.DataFrame:
    """Load normalized dataset for LSTM / logistic regression. Tries both possible filenames."""
    for name in ("dataset_normalized_lstm.csv", "dataset_normalized.csv"):
        path = Path(data_dir) / name
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df
    raise FileNotFoundError(f"No normalized dataset found in {data_dir}/")


def split_train_valid_test(
    df: pd.DataFrame,
    test_start: str = "2023-01-01",
    valid_start: str = "2021-01-01",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: 12 years train+valid, 3 years test.
    Within the 12 years: train before valid_start, valid from valid_start to test_start.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must have column '{date_col}'")
    df[date_col] = pd.to_datetime(df[date_col])
    test_start = pd.Timestamp(test_start)
    valid_start = pd.Timestamp(valid_start)

    train = df[df[date_col] < valid_start].copy()
    valid = df[(df[date_col] >= valid_start) & (df[date_col] < test_start)].copy()
    test = df[df[date_col] >= test_start].copy()

    return train, valid, test


def split_Xy_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = "date",
    valid_start: str = "2021-01-01",
    test_start: str = "2023-01-01",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split feature matrix X and target y by date. X must contain date_col."""
    valid_start = pd.Timestamp(valid_start)
    test_start = pd.Timestamp(test_start)
    dates = pd.to_datetime(X[date_col])
    train_mask = dates < valid_start
    valid_mask = (dates >= valid_start) & (dates < test_start)
    test_mask = dates >= test_start
    return (
        X.loc[train_mask], y.loc[train_mask],
        X.loc[valid_mask], y.loc[valid_mask],
        X.loc[test_mask], y.loc[test_mask],
    )
