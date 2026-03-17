"""
Logistic Regression for binary weather prediction (e.g. rain yes/no).
Uses lagged and rolling features from the preprocessed dataset.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Columns to use as base features (excluding target and date)
FEATURE_COLS = [
    "T_max", "T_min", "rain_mm", "RH_mean", "cloud", "MSLP",
    "DTR", "rain_freq", "MSLP_delta",
]


def build_rain_target(df: pd.DataFrame, rain_col: str = "rain_mm") -> pd.Series:
    """Binary target: 1 if rain > 0, else 0."""
    return (df[rain_col] > 0).astype(int)


def build_lag_rolling_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    window: int = 7,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add lag and rolling (mean) features. Drops rows with NaN from lags/rolling.
    """
    if lags is None:
        lags = [1, 2, 3, 7]
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    out = df[["date"] + feature_cols].copy()
    for c in feature_cols:
        for lag in lags:
            out[f"{c}_lag{lag}"] = df[c].shift(lag)
        out[f"{c}_roll_mean{window}"] = df[c].rolling(window, min_periods=1).mean().shift(1)
    out = out.drop(columns=feature_cols)  # keep only lags and roll to avoid leakage
    out["date"] = df["date"].values
    return out


def prepare_logistic_data(
    df: pd.DataFrame,
    target_col: str = "rain_mm",
    lags: Optional[List[int]] = None,
    window: int = 7,
) -> tuple:
    """
    Build feature matrix X (with date) and binary target y.
    Drops rows with NaN (from lags/rolling).
    """
    X = build_lag_rolling_features(df, lags=lags, window=window)
    y = build_rain_target(df, rain_col=target_col)
    feature_names = [c for c in X.columns if c != "date"]
    X = X.dropna()
    # Align y to remaining rows
    y = y.loc[X.index].copy()
    return X, y, feature_names


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train logistic regression with StandardScaler on features."""
    scaler = StandardScaler()
    X_tr = X_train[feature_names]
    X_tr_scaled = scaler.fit_transform(X_tr)
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_tr_scaled, y_train.values)
    return model, scaler


def predict_logistic(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Predict binary labels."""
    X = X[feature_names].copy()
    X = X.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
