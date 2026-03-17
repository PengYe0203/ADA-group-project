"""
Plotting helpers for model comparison and prediction visualization.
"""
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions_vs_actual(
    dates: Union[np.ndarray, pd.DatetimeIndex],
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predictions vs Actual",
    ylabel: str = "Value",
    figsize: tuple = (12, 4),
    path: Optional[str] = None,
) -> None:
    """Time series plot: actual and predicted."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, actual, label="Actual", alpha=0.8)
    ax.plot(dates, predicted, label="Predicted", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    metrics_dict: dict,
    metric_names: Optional[List[str]] = None,
    title: str = "Model Comparison",
    path: Optional[str] = None,
) -> None:
    """
    metrics_dict: e.g. {"Logistic": {"Accuracy": 0.8, "F1": 0.7}, "LSTM": {"RMSE": 1.2}, ...}
    """
    if not metrics_dict:
        return
    models = list(metrics_dict.keys())
    if metric_names is None:
        metric_names = sorted(set(k for m in metrics_dict for k in metrics_dict[m]))
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [metrics_dict[model].get(m, np.nan) for m in metric_names]
        rects = ax.bar(x + i * width - (len(models) - 1) * width / 2, vals, width, label=model)
        # Label each bar with its value
        labels = [f"{v:.2f}" if np.isfinite(v) else "—" for v in vals]
        ax.bar_label(rects, labels=labels, fontsize=8, padding=2)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_future_forecast(
    hist_dates: Union[np.ndarray, pd.DatetimeIndex],
    hist_values: np.ndarray,
    future_dates: Union[np.ndarray, pd.DatetimeIndex],
    future_pred: np.ndarray,
    title: str = "Future Forecast",
    ylabel: str = "Value",
    figsize: tuple = (12, 5),
    path: Optional[str] = None,
) -> None:
    """Plot history and future forecast."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hist_dates, hist_values, label="Historical", color="C0", alpha=0.8)
    ax.plot(future_dates, future_pred, label="Forecast", color="C1", linestyle="--", marker="o", markersize=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
