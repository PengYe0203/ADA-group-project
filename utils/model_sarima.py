"""
SARIMA for univariate weather series (e.g. T_max).
Stationarity is handled via order (d, D) and seasonal order.
"""
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def check_stationarity(series: pd.Series, name: str = "series") -> dict:
    """ADF test; return dict with test stat and p-value (plain Python types for clean print)."""
    if not HAS_STATSMODELS:
        return {"adf_stat": None, "pvalue": None, "stationary": None}
    series = series.dropna()
    res = adfuller(series, autolag="AIC")
    return {
        "adf_stat": float(res[0]),
        "pvalue": float(res[1]),
        "stationary": bool(res[1] < 0.05),
    }


def _ensure_daily_freq(series: pd.Series) -> pd.Series:
    """Ensure series has a DatetimeIndex with daily frequency for statsmodels."""
    series = series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    freq = getattr(series.index, "freq", None) or pd.infer_freq(series.index)
    if freq is None:
        # Irregular or unknown: use a regular daily index (same length) so statsmodels gets freq
        start = series.index.min()
        series.index = pd.date_range(start=start, periods=len(series), freq="D")
    elif freq != "D" and freq != pd.offsets.Day():
        # Ensure it's daily for seasonal_order s=7
        series.index = pd.date_range(start=series.index.min(), periods=len(series), freq="D")
    return series


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 0, 1),
    seasonal_order: tuple = (1, 0, 1, 7),
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> "SARIMAX":
    """
    Fit SARIMA on the given univariate series.
    order=(p,d,q), seasonal_order=(P,D,Q,s).
    For daily data, s=7 (weekly) is common; s=365 is heavy.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")
    series = series.dropna()
    series = _ensure_daily_freq(series)
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    fitted = model.fit(disp=False, maxiter=500)
    return fitted


def forecast_sarima(
    fitted: "SARIMAX",
    steps: int = 1,
) -> np.ndarray:
    """Forecast next `steps` steps. Returns 1D array."""
    f = fitted.forecast(steps=steps)
    return np.asarray(f).ravel()


def fit_and_forecast_test(
    train_series: pd.Series,
    test_len: int,
    order: tuple = (1, 0, 1),
    seasonal_order: tuple = (1, 0, 1, 7),
) -> np.ndarray:
    """
    Fit SARIMA on train_series and forecast next test_len steps.
    Returns 1D array of length test_len for comparison with test set.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required.")
    train_series = train_series.dropna()
    if len(train_series) < 10:
        return np.full(test_len, np.nan)
    try:
        fitted = fit_sarima(train_series, order=order, seasonal_order=seasonal_order)
        return forecast_sarima(fitted, steps=test_len)
    except Exception:
        return np.full(test_len, np.nan)
